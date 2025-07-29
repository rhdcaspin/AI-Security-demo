#!/usr/bin/env python3
"""
IBM ART-Enhanced Adversarial Attack Tool
Demonstrates advanced MITRE ATLAS T1551 techniques using IBM Adversarial Robustness Toolbox

This tool leverages IBM ART to generate sophisticated adversarial examples
that demonstrate enterprise-grade AI security threats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import requests
import io
import json
import argparse
import logging
import os
import time
from typing import Dict, List, Any, Tuple

# IBM ART imports
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    DeepFool,
    BoundaryAttack,
    HopSkipJump,
    SquareAttack,
    AutoProjectedGradientDescent
)
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import GaussianNoise, JpegCompression
from art.defences.postprocessor import HighConfidence, ReverseSigmoid
from art.utils import load_cifar10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARTAdversarialAttacker:
    """
    Advanced adversarial attacker using IBM ART framework
    """
    
    def __init__(self, target_url: str, model_endpoint: str = "/predict"):
        self.target_url = target_url.rstrip('/')
        self.model_endpoint = model_endpoint
        self.session = requests.Session()
        self.classifier = None
        self.surrogate_model = None
        
        # ART attack configurations
        self.attack_configs = {
            'fgsm': {'eps': 0.3, 'norm': np.inf, 'eps_step': 0.1},
            'pgd': {'eps': 0.3, 'eps_step': 0.01, 'max_iter': 40, 'norm': np.inf},
            'c_w': {'confidence': 0.0, 'targeted': False, 'learning_rate': 0.01, 'max_iter': 10},
            'deepfool': {'max_iter': 100, 'epsilon': 1e-6, 'nb_grads': 10},
            'boundary': {'targeted': False, 'delta': 0.01, 'epsilon': 0.01, 'step_adapt': 0.667},
            'hsj': {'targeted': False, 'norm': 2, 'max_iter': 50, 'max_eval': 10000},
            'square': {'norm': np.inf, 'eps': 0.3, 'max_iter': 5000},
            'auto_pgd': {'norm': np.inf, 'eps': 0.3, 'eps_step': 0.1, 'max_iter': 100}
        }
        
    def create_surrogate_model(self) -> PyTorchClassifier:
        """
        Create a surrogate model for generating adversarial examples
        """
        logger.info("Creating surrogate model for adversarial generation...")
        
        # Use ResNet18 as surrogate (similar to target)
        model = models.resnet18(pretrained=True)
        model.eval()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create ART classifier
        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 224, 224),
            nb_classes=1000,
            preprocessing=(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        )
        
        self.classifier = classifier
        logger.info("Surrogate model created successfully")
        return classifier
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for ART attacks"""
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
    
    def query_target_model(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Query the target model with an image array"""
        try:
            # Convert numpy array back to PIL for upload
            tensor = torch.from_numpy(image_array.squeeze(0))
            to_pil = transforms.ToPILImage()
            image = to_pil(tensor)
            
            # Save temporarily for upload
            temp_path = f"/tmp/query_{int(time.time())}_{np.random.randint(1000)}.jpg"
            image.save(temp_path)
            
            # Query model
            with open(temp_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.target_url}{self.model_endpoint}",
                    files=files,
                    timeout=30
                )
            
            # Clean up
            os.remove(temp_path)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Model query failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error querying target model: {e}")
            return None
    
    def fgsm_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fast Gradient Sign Method using IBM ART
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info("Executing FGSM attack using IBM ART...")
        
        config = self.attack_configs['fgsm']
        attack = FastGradientMethod(
            estimator=self.classifier,
            eps=config['eps'],
            norm=config['norm'],
            eps_step=config['eps_step']
        )
        
        # Generate adversarial example
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'FGSM (IBM ART)',
            'epsilon': config['eps'],
            'norm': config['norm'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def pgd_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Projected Gradient Descent using IBM ART
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info("Executing PGD attack using IBM ART...")
        
        config = self.attack_configs['pgd']
        attack = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=config['eps'],
            eps_step=config['eps_step'],
            max_iter=config['max_iter'],
            norm=config['norm']
        )
        
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'PGD (IBM ART)',
            'epsilon': config['eps'],
            'max_iterations': config['max_iter'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def carlini_wagner_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Carlini & Wagner L2 attack using IBM ART
        MITRE ATLAS T1551.004 - Adversarial Examples (Advanced)
        """
        logger.info("Executing Carlini & Wagner attack using IBM ART...")
        
        config = self.attack_configs['c_w']
        attack = CarliniL2Method(
            classifier=self.classifier,
            confidence=config['confidence'],
            targeted=config['targeted'],
            learning_rate=config['learning_rate'],
            max_iter=config['max_iter']
        )
        
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'Carlini & Wagner L2 (IBM ART)',
            'confidence': config['confidence'],
            'learning_rate': config['learning_rate'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def deepfool_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        DeepFool attack using IBM ART
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info("Executing DeepFool attack using IBM ART...")
        
        config = self.attack_configs['deepfool']
        attack = DeepFool(
            classifier=self.classifier,
            max_iter=config['max_iter'],
            epsilon=config['epsilon'],
            nb_grads=config['nb_grads']
        )
        
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'DeepFool (IBM ART)',
            'max_iterations': config['max_iter'],
            'epsilon': config['epsilon'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def boundary_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Boundary Attack using IBM ART (Black-box)
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info("Executing Boundary attack using IBM ART...")
        
        config = self.attack_configs['boundary']
        attack = BoundaryAttack(
            estimator=self.classifier,
            targeted=config['targeted'],
            delta=config['delta'],
            epsilon=config['epsilon'],
            step_adapt=config['step_adapt'],
            max_iter=10  # Reduced for demo performance
        )
        
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'Boundary Attack (IBM ART)',
            'attack_type': 'Black-box',
            'delta': config['delta'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def square_attack(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Square Attack using IBM ART (Query-efficient)
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info("Executing Square attack using IBM ART...")
        
        config = self.attack_configs['square']
        attack = SquareAttack(
            estimator=self.classifier,
            norm=config['norm'],
            eps=config['eps'],
            max_iter=100  # Reduced for demo
        )
        
        adversarial_image = attack.generate(x=image)
        
        # Test against target
        original_result = self.query_target_model(image)
        adversarial_result = self.query_target_model(adversarial_image)
        
        attack_info = {
            'method': 'Square Attack (IBM ART)',
            'attack_type': 'Query-efficient',
            'epsilon': config['eps'],
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_prediction': adversarial_result['predictions'][0] if adversarial_result else None,
            'attack_success': (original_result and adversarial_result and 
                             original_result['predictions'][0]['class_id'] != 
                             adversarial_result['predictions'][0]['class_id'])
        }
        
        return adversarial_image, attack_info
    
    def demonstrate_defenses(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Demonstrate IBM ART defensive techniques
        MITRE ATLAS Defense Strategies
        """
        logger.info("Demonstrating IBM ART defensive techniques...")
        
        defenses_results = {}
        
        # 1. Gaussian Noise Defense
        gaussian_defense = GaussianNoise(sigma=0.1, apply_fit=False, apply_predict=True)
        defended_image_gaussian = gaussian_defense(image)[0]
        
        # 2. JPEG Compression Defense
        jpeg_defense = JpegCompression(quality=50, apply_fit=False, apply_predict=True)
        defended_image_jpeg = jpeg_defense(image)[0]
        
        # Test defenses against FGSM
        fgsm_attack = FastGradientMethod(
            estimator=self.classifier,
            eps=0.3,
            norm=np.inf
        )
        
        # Generate adversarial example
        adversarial_clean = fgsm_attack.generate(x=image)
        adversarial_gaussian = fgsm_attack.generate(x=defended_image_gaussian.reshape(1, 3, 224, 224))
        adversarial_jpeg = fgsm_attack.generate(x=defended_image_jpeg.reshape(1, 3, 224, 224))
        
        # Test all variants
        original_result = self.query_target_model(image)
        adv_clean_result = self.query_target_model(adversarial_clean)
        adv_gaussian_result = self.query_target_model(adversarial_gaussian)
        adv_jpeg_result = self.query_target_model(adversarial_jpeg)
        
        defenses_results = {
            'original_prediction': original_result['predictions'][0] if original_result else None,
            'adversarial_clean': {
                'prediction': adv_clean_result['predictions'][0] if adv_clean_result else None,
                'attack_success': (original_result and adv_clean_result and 
                                 original_result['predictions'][0]['class_id'] != 
                                 adv_clean_result['predictions'][0]['class_id'])
            },
            'adversarial_with_gaussian_defense': {
                'prediction': adv_gaussian_result['predictions'][0] if adv_gaussian_result else None,
                'attack_success': (original_result and adv_gaussian_result and 
                                 original_result['predictions'][0]['class_id'] != 
                                 adv_gaussian_result['predictions'][0]['class_id'])
            },
            'adversarial_with_jpeg_defense': {
                'prediction': adv_jpeg_result['predictions'][0] if adv_jpeg_result else None,
                'attack_success': (original_result and adv_jpeg_result and 
                                 original_result['predictions'][0]['class_id'] != 
                                 adv_jpeg_result['predictions'][0]['class_id'])
            }
        }
        
        return defenses_results
    
    def run_comprehensive_art_attack_suite(self, image_path: str, output_dir: str = "./art_attack_results") -> Dict[str, Any]:
        """
        Run comprehensive IBM ART attack suite
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting comprehensive IBM ART attack suite on {image_path}")
        logger.info("MITRE ATLAS Technique: T1551 - Evade ML Model (Advanced)")
        
        # Initialize surrogate model
        self.create_surrogate_model()
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            logger.error("Failed to preprocess image")
            return {}
        
        # Store results
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_image': image_path,
            'attack_results': {},
            'defense_results': {},
            'summary': {}
        }
        
        # Define attack methods
        attack_methods = [
            ('fgsm', self.fgsm_attack),
            ('pgd', self.pgd_attack),
            ('carlini_wagner', self.carlini_wagner_attack),
            ('deepfool', self.deepfool_attack),
            ('boundary', self.boundary_attack),
            ('square', self.square_attack)
        ]
        
        successful_attacks = 0
        
        # Execute each attack
        for attack_name, attack_method in attack_methods:
            try:
                logger.info(f"Executing {attack_name} attack...")
                adversarial_image, attack_info = attack_method(image)
                
                # Save adversarial image
                adv_path = os.path.join(output_dir, f"{attack_name}_adversarial.jpg")
                tensor = torch.from_numpy(adversarial_image.squeeze(0))
                to_pil = transforms.ToPILImage()
                adv_pil = to_pil(tensor)
                adv_pil.save(adv_path)
                
                attack_info['adversarial_image_path'] = adv_path
                results['attack_results'][attack_name] = attack_info
                
                if attack_info.get('attack_success', False):
                    successful_attacks += 1
                    
                logger.info(f"{attack_name} attack completed - Success: {attack_info.get('attack_success', False)}")
                
            except Exception as e:
                logger.error(f"Failed to execute {attack_name} attack: {e}")
                results['attack_results'][attack_name] = {'error': str(e)}
        
        # Demonstrate defenses
        try:
            logger.info("Demonstrating defensive techniques...")
            defense_results = self.demonstrate_defenses(image)
            results['defense_results'] = defense_results
        except Exception as e:
            logger.error(f"Failed to demonstrate defenses: {e}")
            results['defense_results'] = {'error': str(e)}
        
        # Summary
        total_attacks = len(attack_methods)
        success_rate = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0
        
        results['summary'] = {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'success_rate_percent': success_rate,
            'framework': 'IBM Adversarial Robustness Toolbox (ART)',
            'mitre_atlas_technique': 'T1551 - Evade ML Model'
        }
        
        # Save results
        results_path = os.path.join(output_dir, f"art_attack_report_{int(time.time())}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"IBM ART attack suite completed: {successful_attacks}/{total_attacks} successful ({success_rate:.1f}%)")
        logger.info(f"Detailed results saved to {results_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="IBM ART Advanced Adversarial Attack Tool - MITRE ATLAS T1551")
    parser.add_argument("--target", required=True, help="Target model URL")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="./art_attack_results", help="Output directory")
    parser.add_argument("--attack", choices=['fgsm', 'pgd', 'carlini_wagner', 'deepfool', 'boundary', 'square', 'all'], 
                       default='all', help="Attack method")
    parser.add_argument("--defenses", action='store_true', help="Demonstrate defensive techniques")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return
    
    # Initialize attacker
    attacker = ARTAdversarialAttacker(args.target)
    
    # Test connectivity
    try:
        response = requests.get(f"{args.target}/health", timeout=10)
        if response.status_code != 200:
            logger.error(f"Target service not available: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"Cannot connect to target: {e}")
        return
    
    logger.info(f"Target: {args.target}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Framework: IBM Adversarial Robustness Toolbox (ART)")
    logger.info(f"MITRE ATLAS Technique: T1551 - Evade ML Model")
    
    if args.attack == 'all':
        # Run comprehensive attack suite
        results = attacker.run_comprehensive_art_attack_suite(args.image, args.output)
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\n=== IBM ART Attack Suite Results ===")
            print(f"Total attacks: {summary['total_attacks']}")
            print(f"Successful attacks: {summary['successful_attacks']}")
            print(f"Success rate: {summary['success_rate_percent']:.1f}%")
            print(f"Framework: {summary['framework']}")
    else:
        logger.info(f"Running specific attack: {args.attack}")
        # Individual attack execution would go here
        
    logger.info("IBM ART adversarial attack demonstration completed")

if __name__ == "__main__":
    main() 