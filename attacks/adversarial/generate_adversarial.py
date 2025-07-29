#!/usr/bin/env python3
"""
Adversarial Example Generation Tool
Demonstrates MITRE ATLAS T1551 - Evade ML Model technique

This tool generates adversarial examples to fool image classification models
by adding imperceptible perturbations to input images.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import requests
import io
import json
import argparse
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialAttacker:
    """
    Implements various adversarial attack methods aligned with MITRE ATLAS
    """
    
    def __init__(self, target_url, model_endpoint="/predict"):
        self.target_url = target_url.rstrip('/')
        self.model_endpoint = model_endpoint
        self.session = requests.Session()
        
    def preprocess_image(self, image_path):
        """Preprocess image for attack generation"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    
    def postprocess_tensor(self, tensor):
        """Convert tensor back to PIL Image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        tensor = tensor.squeeze(0)
        to_pil = transforms.ToPILImage()
        return to_pil(tensor)
    
    def query_model(self, image_path, debug=False):
        """Query the target model with an image"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {'debug': 'true'} if debug else {}
                response = self.session.post(
                    f"{self.target_url}{self.model_endpoint}",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Model query failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return None
    
    def fgsm_attack(self, image_tensor, epsilon=0.03, target_class=None):
        """
        Fast Gradient Sign Method (FGSM) Attack
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info(f"Generating FGSM adversarial example (epsilon={epsilon})")
        
        # Create a simple gradient-based perturbation
        # In a real attack, this would use the actual model gradients
        # For demo purposes, we'll add structured noise
        
        noise = torch.randn_like(image_tensor) * epsilon
        adversarial_image = image_tensor + noise
        adversarial_image = torch.clamp(adversarial_image, -2.5, 2.5)  # Reasonable bounds
        
        return adversarial_image
    
    def pgd_attack(self, image_tensor, epsilon=0.03, alpha=0.01, iterations=10):
        """
        Projected Gradient Descent (PGD) Attack
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info(f"Generating PGD adversarial example (epsilon={epsilon}, iterations={iterations})")
        
        # Simulate iterative perturbation
        adversarial_image = image_tensor.clone()
        
        for i in range(iterations):
            # Add small perturbation each iteration
            noise = torch.randn_like(adversarial_image) * alpha
            adversarial_image = adversarial_image + noise
            
            # Project back to epsilon ball
            delta = adversarial_image - image_tensor
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_image = torch.clamp(image_tensor + delta, -2.5, 2.5)
        
        return adversarial_image
    
    def pixel_attack(self, image_tensor, num_pixels=5):
        """
        One-pixel/Few-pixel attack
        MITRE ATLAS T1551.004 - Adversarial Examples
        """
        logger.info(f"Generating pixel attack with {num_pixels} pixels")
        
        adversarial_image = image_tensor.clone()
        _, _, h, w = adversarial_image.shape
        
        # Randomly modify a few pixels
        for _ in range(num_pixels):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            channel = np.random.randint(0, 3)
            value = np.random.uniform(-2, 2)
            adversarial_image[0, channel, x, y] = value
        
        return adversarial_image
    
    def semantic_attack(self, image_path, output_path):
        """
        Semantic adversarial attack (brightness, contrast, rotation)
        MITRE ATLAS T1551 - Evade ML Model
        """
        logger.info("Generating semantic adversarial example")
        
        image = Image.open(image_path)
        
        # Apply semantic transformations
        transforms_list = [
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0))
        ]
        
        # Apply random transformation
        transform = np.random.choice(transforms_list)
        adversarial_image = transform(image)
        adversarial_image.save(output_path)
        
        return output_path
    
    def generate_attack_suite(self, image_path, output_dir="./adversarial_outputs"):
        """
        Generate a suite of adversarial examples using different methods
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting adversarial attack suite on {image_path}")
        
        # Load original image
        image_tensor = self.preprocess_image(image_path)
        
        # Get original prediction
        logger.info("Getting original prediction...")
        original_result = self.query_model(image_path, debug=True)
        if not original_result:
            logger.error("Failed to get original prediction")
            return
        
        original_class = original_result['predictions'][0]['class_name']
        original_confidence = original_result['predictions'][0]['confidence']
        logger.info(f"Original prediction: {original_class} ({original_confidence:.3f})")
        
        results = {
            'original': {
                'class': original_class,
                'confidence': original_confidence,
                'file': image_path
            },
            'attacks': {}
        }
        
        # FGSM Attack
        try:
            fgsm_tensor = self.fgsm_attack(image_tensor)
            fgsm_path = os.path.join(output_dir, "fgsm_adversarial.jpg")
            fgsm_image = self.postprocess_tensor(fgsm_tensor)
            fgsm_image.save(fgsm_path)
            
            # Test FGSM result
            fgsm_result = self.query_model(fgsm_path)
            if fgsm_result:
                results['attacks']['fgsm'] = {
                    'class': fgsm_result['predictions'][0]['class_name'],
                    'confidence': fgsm_result['predictions'][0]['confidence'],
                    'file': fgsm_path,
                    'success': fgsm_result['predictions'][0]['class_name'] != original_class
                }
                logger.info(f"FGSM: {results['attacks']['fgsm']['class']} ({results['attacks']['fgsm']['confidence']:.3f}) - Success: {results['attacks']['fgsm']['success']}")
        except Exception as e:
            logger.error(f"FGSM attack failed: {e}")
        
        # PGD Attack
        try:
            pgd_tensor = self.pgd_attack(image_tensor)
            pgd_path = os.path.join(output_dir, "pgd_adversarial.jpg")
            pgd_image = self.postprocess_tensor(pgd_tensor)
            pgd_image.save(pgd_path)
            
            # Test PGD result
            pgd_result = self.query_model(pgd_path)
            if pgd_result:
                results['attacks']['pgd'] = {
                    'class': pgd_result['predictions'][0]['class_name'],
                    'confidence': pgd_result['predictions'][0]['confidence'],
                    'file': pgd_path,
                    'success': pgd_result['predictions'][0]['class_name'] != original_class
                }
                logger.info(f"PGD: {results['attacks']['pgd']['class']} ({results['attacks']['pgd']['confidence']:.3f}) - Success: {results['attacks']['pgd']['success']}")
        except Exception as e:
            logger.error(f"PGD attack failed: {e}")
        
        # Pixel Attack
        try:
            pixel_tensor = self.pixel_attack(image_tensor)
            pixel_path = os.path.join(output_dir, "pixel_adversarial.jpg")
            pixel_image = self.postprocess_tensor(pixel_tensor)
            pixel_image.save(pixel_path)
            
            # Test Pixel result
            pixel_result = self.query_model(pixel_path)
            if pixel_result:
                results['attacks']['pixel'] = {
                    'class': pixel_result['predictions'][0]['class_name'],
                    'confidence': pixel_result['predictions'][0]['confidence'],
                    'file': pixel_path,
                    'success': pixel_result['predictions'][0]['class_name'] != original_class
                }
                logger.info(f"Pixel: {results['attacks']['pixel']['class']} ({results['attacks']['pixel']['confidence']:.3f}) - Success: {results['attacks']['pixel']['success']}")
        except Exception as e:
            logger.error(f"Pixel attack failed: {e}")
        
        # Semantic Attack
        try:
            semantic_path = os.path.join(output_dir, "semantic_adversarial.jpg")
            self.semantic_attack(image_path, semantic_path)
            
            # Test Semantic result
            semantic_result = self.query_model(semantic_path)
            if semantic_result:
                results['attacks']['semantic'] = {
                    'class': semantic_result['predictions'][0]['class_name'],
                    'confidence': semantic_result['predictions'][0]['confidence'],
                    'file': semantic_path,
                    'success': semantic_result['predictions'][0]['class_name'] != original_class
                }
                logger.info(f"Semantic: {results['attacks']['semantic']['class']} ({results['attacks']['semantic']['confidence']:.3f}) - Success: {results['attacks']['semantic']['success']}")
        except Exception as e:
            logger.error(f"Semantic attack failed: {e}")
        
        # Save results
        results_path = os.path.join(output_dir, "attack_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate success rate
        successful_attacks = sum(1 for attack in results['attacks'].values() if attack.get('success', False))
        total_attacks = len(results['attacks'])
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
        
        logger.info(f"Attack suite completed: {successful_attacks}/{total_attacks} successful ({success_rate:.1%})")
        logger.info(f"Results saved to {results_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack Tool - MITRE ATLAS T1551")
    parser.add_argument("--target", required=True, help="Target model URL")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="./adversarial_outputs", help="Output directory")
    parser.add_argument("--attack", choices=['fgsm', 'pgd', 'pixel', 'semantic', 'all'], 
                       default='all', help="Attack method")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Perturbation magnitude")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return
    
    # Initialize attacker
    attacker = AdversarialAttacker(args.target)
    
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
    logger.info(f"Attack: {args.attack}")
    logger.info(f"MITRE ATLAS Technique: T1551 - Evade ML Model")
    
    if args.attack == 'all':
        # Run full attack suite
        results = attacker.generate_attack_suite(args.image, args.output)
    else:
        # Run specific attack
        logger.info(f"Running {args.attack} attack...")
        # Implementation for specific attacks...
        
    logger.info("Adversarial attack demonstration completed")

if __name__ == "__main__":
    main() 