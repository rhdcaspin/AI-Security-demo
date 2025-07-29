#!/usr/bin/env python3
"""
Vulnerable Image Classification Service
Demonstrates AI/ML security vulnerabilities for MITRE ATLAS demo
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
import pickle
import hashlib
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - Deliberately insecure for demo purposes
UPLOAD_FOLDER = '/app/uploads'
MODEL_FOLDER = '/app/models'
DATA_FOLDER = '/app/data'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Global variables for model and stats
model = None
class_names = []
prediction_stats = {'total_predictions': 0, 'successful_predictions': 0}
query_log = []

# VULNERABILITY: Insecure model loading from arbitrary paths
def load_model_unsafe(model_path):
    """VULNERABLE: Loads model from any path without validation"""
    try:
        if model_path.endswith('.pkl'):
            # VULNERABILITY: Arbitrary pickle deserialization
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            return torch.load(model_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the image classification model"""
    global model, class_names
    
    try:
        # Load pre-trained ResNet18 model
        model = models.resnet18(pretrained=True)
        model.eval()
        
        # Load ImageNet class names
        class_names = load_imagenet_classes()
        logger.info("Model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model = None

def load_imagenet_classes():
    """Load ImageNet class names"""
    try:
        # Simple class names for demo
        return [f"class_{i}" for i in range(1000)]
    except:
        return [f"unknown_class_{i}" for i in range(1000)]

def preprocess_image(image_path):
    """Preprocess image for model inference"""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

# VULNERABILITY: No authentication or authorization
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# VULNERABILITY: Information disclosure
@app.route('/model/info', methods=['GET'])
def model_info():
    """VULNERABLE: Exposes model architecture and parameters"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_type': type(model).__name__,
        'num_classes': len(class_names),
        'parameters_count': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        'architecture': str(model),
        'class_names': class_names[:10],  # First 10 classes
        'prediction_stats': prediction_stats
    }
    return jsonify(info)

# VULNERABILITY: Model parameter extraction
@app.route('/model/weights', methods=['GET'])
def get_model_weights():
    """VULNERABLE: Exposes model weights"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        layer_name = request.args.get('layer', 'conv1')
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if hasattr(layer, 'weight'):
                weights = layer.weight.data.numpy().tolist()
                return jsonify({
                    'layer': layer_name,
                    'weights': weights,
                    'shape': list(layer.weight.shape)
                })
        return jsonify({'error': f'Layer {layer_name} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# VULNERABILITY: Unrestricted file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    """VULNERABLE: Unrestricted file upload with path traversal"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # VULNERABILITY: Insufficient file validation
    filename = request.form.get('filename', file.filename)
    # VULNERABILITY: Path traversal
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'path': filepath,
            'size': os.path.getsize(filepath)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# VULNERABILITY: Command injection through model loading
@app.route('/model/load', methods=['POST'])
def load_custom_model():
    """VULNERABLE: Loads arbitrary model files"""
    data = request.get_json()
    if not data or 'model_path' not in data:
        return jsonify({'error': 'model_path required'}), 400
    
    model_path = data['model_path']
    
    # VULNERABILITY: No path validation
    global model
    try:
        model = load_model_unsafe(model_path)
        if model:
            return jsonify({'message': f'Model loaded from {model_path}'})
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - vulnerable to adversarial attacks"""
    global prediction_stats, query_log
    
    start_time = time.time()
    prediction_stats['total_predictions'] += 1
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
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
            # VULNERABILITY: Arbitrary file access
            filepath = request.json['image_path']
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess and predict
        input_tensor = preprocess_image(filepath)
        if input_tensor is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # VULNERABILITY: No input validation for adversarial examples
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Prepare response
        predictions = []
        for i in range(5):
            idx = top5_idx[0][i].item()
            prob = top5_prob[0][i].item()
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            predictions.append({
                'class_id': idx,
                'class_name': class_name,
                'confidence': float(prob)
            })
        
        # Log query for analysis (VULNERABILITY: Excessive logging)
        query_info = {
            'timestamp': datetime.now().isoformat(),
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'file_hash': hashlib.md5(open(filepath, 'rb').read()).hexdigest(),
            'predictions': predictions,
            'processing_time': time.time() - start_time
        }
        query_log.append(query_info)
        
        # Keep only last 1000 queries
        if len(query_log) > 1000:
            query_log = query_log[-1000:]
        
        prediction_stats['successful_predictions'] += 1
        
        response = {
            'predictions': predictions,
            'processing_time': time.time() - start_time,
            'model_version': '1.0',
            'query_id': len(query_log)
        }
        
        # VULNERABILITY: Include sensitive debug info
        if request.args.get('debug') == 'true':
            response['debug'] = {
                'raw_outputs': outputs.tolist(),
                'input_shape': list(input_tensor.shape),
                'file_path': filepath,
                'model_params': sum(p.numel() for p in model.parameters())
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# VULNERABILITY: Exposes internal logs
@app.route('/logs', methods=['GET'])
def get_logs():
    """VULNERABLE: Exposes prediction logs"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify({
        'total_queries': len(query_log),
        'recent_queries': query_log[-limit:] if query_log else [],
        'stats': prediction_stats
    })

# VULNERABILITY: Administrative functions without authentication
@app.route('/admin/reset', methods=['POST'])
def admin_reset():
    """VULNERABLE: Reset functionality without auth"""
    global query_log, prediction_stats
    query_log = []
    prediction_stats = {'total_predictions': 0, 'successful_predictions': 0}
    return jsonify({'message': 'System reset successfully'})

# VULNERABILITY: Data poisoning endpoint
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """VULNERABLE: Accepts arbitrary feedback data"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # VULNERABILITY: No validation of feedback data
    feedback_file = os.path.join(DATA_FOLDER, f"feedback_{int(time.time())}.json")
    try:
        with open(feedback_file, 'w') as f:
            json.dump(data, f)
        return jsonify({'message': 'Feedback saved', 'file': feedback_file})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# VULNERABILITY: File system access
@app.route('/files/<path:filename>')
def download_file(filename):
    """VULNERABLE: Arbitrary file download"""
    try:
        # VULNERABILITY: Path traversal
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        return send_file(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    logger.info("Starting Vulnerable Image Classification Service")
    logger.warning("WARNING: This service contains intentional vulnerabilities for demonstration purposes")
    
    # Initialize the model
    initialize_model()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 