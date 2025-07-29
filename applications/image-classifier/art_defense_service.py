#!/usr/bin/env python3
"""
IBM ART Defense Service
Demonstrates defensive AI security capabilities using IBM Adversarial Robustness Toolbox

This service provides robust defenses against adversarial attacks and can be used
to demonstrate MITRE ATLAS defensive strategies.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import time

# IBM ART imports for defenses
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import (
    GaussianNoise, 
    JpegCompression, 
    SpatialSmoothing,
    FeatureSqueezing,
    ThermometerEncoding
)
from art.defences.postprocessor import (
    HighConfidence,
    ReverseSigmoid,
    Rounded
)
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = '/app/uploads'
MODEL_FOLDER = '/app/models'
DEFENDED_FOLDER = '/app/defended'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DEFENDED_FOLDER, exist_ok=True)

# Global variables
classifier = None
defense_stats = {
    'total_predictions': 0,
    'defended_predictions': 0,
    'detected_adversarial': 0,
    'defense_methods_used': []
}

class ARTDefenseSystem:
    """
    Comprehensive defense system using IBM ART
    """
    
    def __init__(self):
        self.model = None
        self.classifier = None
        self.preprocessors = {}
        self.postprocessors = {}
        self.detection_threshold = 0.8
        
        self.initialize_defenses()
    
    def initialize_defenses(self):
        """Initialize all defense mechanisms"""
        logger.info("Initializing IBM ART defense system...")
        
        # Load and setup model
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        
        # Create ART classifier
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 224, 224),
            nb_classes=1000,
            preprocessing=(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        )
        
        # Initialize preprocessors (input defenses)
        self.preprocessors = {
            'gaussian_noise': GaussianNoise(sigma=0.1, apply_fit=False, apply_predict=True),
            'jpeg_compression': JpegCompression(quality=75, apply_fit=False, apply_predict=True),
            'spatial_smoothing': SpatialSmoothing(window_size=3, apply_fit=False, apply_predict=True),
            'feature_squeezing': FeatureSqueezing(bit_depth=8, apply_fit=False, apply_predict=True),
            'thermometer_encoding': ThermometerEncoding(num_space=10, apply_fit=False, apply_predict=True)
        }
        
        # Initialize postprocessors (output defenses)
        self.postprocessors = {
            'high_confidence': HighConfidence(cutoff=0.9, apply_fit=False, apply_predict=True),
            'reverse_sigmoid': ReverseSigmoid(beta=1.0, gamma=0.1, apply_fit=False, apply_predict=True),
            'rounded': Rounded(decimals=2, apply_fit=False, apply_predict=True)
        }
        
        logger.info("IBM ART defense system initialized successfully")
    
    def detect_adversarial_example(self, image_array: np.ndarray) -> dict:
        """
        Detect if input is likely an adversarial example
        """
        try:
            # Method 1: Prediction confidence analysis
            predictions = self.classifier.predict(image_array)
            max_confidence = np.max(predictions)
            confidence_entropy = -np.sum(predictions * np.log(predictions + 1e-10))
            
            # Method 2: Multiple defense comparison
            defense_predictions = {}
            for name, preprocessor in self.preprocessors.items():
                try:
                    defended_image = preprocessor(image_array)[0]
                    defended_pred = self.classifier.predict(defended_image.reshape(1, 3, 224, 224))
                    defense_predictions[name] = defended_pred
                except Exception as e:
                    logger.warning(f"Defense {name} failed: {e}")
                    continue
            
            # Calculate prediction consistency
            if defense_predictions:
                original_class = np.argmax(predictions)
                consistent_predictions = 0
                
                for name, pred in defense_predictions.items():
                    defense_class = np.argmax(pred)
                    if defense_class == original_class:
                        consistent_predictions += 1
                
                consistency_score = consistent_predictions / len(defense_predictions)
            else:
                consistency_score = 1.0
            
            # Adversarial detection logic
            is_adversarial = (
                max_confidence > 0.95 and consistency_score < 0.5
            ) or (
                max_confidence < 0.3 and confidence_entropy > 2.0
            )
            
            detection_result = {
                'is_adversarial': is_adversarial,
                'confidence_score': float(max_confidence),
                'entropy_score': float(confidence_entropy),
                'consistency_score': float(consistency_score),
                'detection_method': 'Multi-defense Consistency Analysis',
                'confidence_threshold': self.detection_threshold
            }
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Adversarial detection failed: {e}")
            return {'is_adversarial': False, 'error': str(e)}
    
    def apply_defense_suite(self, image_array: np.ndarray, defense_level: str = 'medium') -> tuple:
        """
        Apply defense mechanisms based on specified level
        """
        try:
            defended_image = image_array.copy()
            applied_defenses = []
            
            if defense_level == 'light':
                # Light defense - minimal processing
                defense = self.preprocessors['gaussian_noise']
                defended_image = defense(defended_image)[0]
                applied_defenses.append('gaussian_noise')
                
            elif defense_level == 'medium':
                # Medium defense - balanced approach
                for defense_name in ['gaussian_noise', 'jpeg_compression']:
                    if defense_name in self.preprocessors:
                        defense = self.preprocessors[defense_name]
                        defended_image = defense(defended_image.reshape(1, 3, 224, 224))[0]
                        applied_defenses.append(defense_name)
                        
            elif defense_level == 'heavy':
                # Heavy defense - comprehensive protection
                for defense_name in ['gaussian_noise', 'jpeg_compression', 'spatial_smoothing', 'feature_squeezing']:
                    if defense_name in self.preprocessors:
                        try:
                            defense = self.preprocessors[defense_name]
                            defended_image = defense(defended_image.reshape(1, 3, 224, 224))[0]
                            applied_defenses.append(defense_name)
                        except Exception as e:
                            logger.warning(f"Defense {defense_name} failed: {e}")
                            continue
            
            return defended_image, applied_defenses
            
        except Exception as e:
            logger.error(f"Defense application failed: {e}")
            return image_array, []
    
    def robust_prediction(self, image_array: np.ndarray, defense_level: str = 'medium') -> dict:
        """
        Make robust prediction with defensive measures
        """
        start_time = time.time()
        
        # Step 1: Adversarial detection
        detection_result = self.detect_adversarial_example(image_array)
        
        # Step 2: Apply defenses if needed
        if detection_result.get('is_adversarial', False) or defense_level != 'none':
            defended_image, applied_defenses = self.apply_defense_suite(image_array, defense_level)
        else:
            defended_image = image_array
            applied_defenses = []
        
        # Step 3: Make prediction
        predictions = self.classifier.predict(defended_image.reshape(1, 3, 224, 224))
        
        # Step 4: Apply postprocessing defenses
        if 'high_confidence' in self.postprocessors:
            postprocessor = self.postprocessors['high_confidence']
            predictions = postprocessor(predictions)[0]
        
        # Step 5: Format results
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_predictions = []
        
        for i, idx in enumerate(top5_indices):
            top5_predictions.append({
                'class_id': int(idx),
                'class_name': f"class_{idx}",
                'confidence': float(predictions[0][idx])
            })
        
        result = {
            'predictions': top5_predictions,
            'processing_time': time.time() - start_time,
            'defense_info': {
                'adversarial_detected': detection_result.get('is_adversarial', False),
                'detection_confidence': detection_result.get('confidence_score', 0.0),
                'defense_level': defense_level,
                'applied_defenses': applied_defenses,
                'consistency_score': detection_result.get('consistency_score', 1.0)
            },
            'model_version': 'IBM ART Defended',
            'framework': 'Adversarial Robustness Toolbox'
        }
        
        return result

# Initialize defense system
defense_system = ARTDefenseSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for defense system"""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        # Convert to numpy for ART
        numpy_image = tensor.numpy()
        return numpy_image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'IBM ART Defense Service',
        'defenses_available': len(defense_system.preprocessors) + len(defense_system.postprocessors),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/defense/info', methods=['GET'])
def defense_info():
    """Get information about available defenses"""
    return jsonify({
        'service': 'IBM ART Defense Service',
        'available_preprocessors': list(defense_system.preprocessors.keys()),
        'available_postprocessors': list(defense_system.postprocessors.keys()),
        'defense_levels': ['light', 'medium', 'heavy'],
        'adversarial_detection': True,
        'stats': defense_stats
    })

@app.route('/defense/predict', methods=['POST'])
def defended_predict():
    """Main defended prediction endpoint"""
    global defense_stats
    
    start_time = time.time()
    defense_stats['total_predictions'] += 1
    
    try:
        # Get defense level from request
        defense_level = request.form.get('defense_level', 'medium')
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file.save(filepath)
        
        elif 'image_path' in request.json:
            filepath = request.json['image_path']
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        input_array = preprocess_image(filepath)
        if input_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # Make robust prediction
        result = defense_system.robust_prediction(input_array, defense_level)
        
        # Update stats
        defense_stats['defended_predictions'] += 1
        if result['defense_info']['adversarial_detected']:
            defense_stats['detected_adversarial'] += 1
        
        # Track defense methods used
        for defense in result['defense_info']['applied_defenses']:
            if defense not in defense_stats['defense_methods_used']:
                defense_stats['defense_methods_used'].append(defense)
        
        # Add request metadata
        result['request_info'] = {
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Defended prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Defended prediction failed: {str(e)}'}), 500

@app.route('/defense/detect', methods=['POST'])
def detect_adversarial():
    """Adversarial example detection endpoint"""
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file.save(filepath)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        input_array = preprocess_image(filepath)
        if input_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # Detect adversarial example
        detection_result = defense_system.detect_adversarial_example(input_array)
        
        detection_result.update({
            'timestamp': datetime.now().isoformat(),
            'detection_service': 'IBM ART Defense System'
        })
        
        return jsonify(detection_result)
        
    except Exception as e:
        logger.error(f"Adversarial detection error: {e}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/defense/stats', methods=['GET'])
def get_defense_stats():
    """Get defense statistics"""
    return jsonify({
        'statistics': defense_stats,
        'uptime': time.time(),
        'service': 'IBM ART Defense Service'
    })

@app.route('/defense/benchmark', methods=['POST'])
def benchmark_defenses():
    """Benchmark different defense levels"""
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file.save(filepath)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        input_array = preprocess_image(filepath)
        if input_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # Test all defense levels
        defense_levels = ['none', 'light', 'medium', 'heavy']
        benchmark_results = {}
        
        for level in defense_levels:
            start_time = time.time()
            result = defense_system.robust_prediction(input_array, level)
            processing_time = time.time() - start_time
            
            benchmark_results[level] = {
                'top_prediction': result['predictions'][0],
                'processing_time': processing_time,
                'applied_defenses': result['defense_info']['applied_defenses'],
                'adversarial_detected': result['defense_info']['adversarial_detected']
            }
        
        return jsonify({
            'benchmark_results': benchmark_results,
            'test_image': filepath,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': f'Benchmark failed: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting IBM ART Defense Service")
    logger.info("This service provides robust AI security defenses using IBM Adversarial Robustness Toolbox")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True) 