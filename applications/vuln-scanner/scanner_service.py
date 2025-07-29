#!/usr/bin/env python3
"""
Vulnerability Scanner Service
Integrates with Clair to provide container image vulnerability scanning for the KubeCon demo

This service demonstrates enterprise container security practices by scanning
the demo's intentionally vulnerable images and providing detailed vulnerability reports.
"""

import os
import json
import logging
import traceback
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import requests
import docker
from typing import Dict, List, Any, Optional
import base64
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CLAIR_INDEXER_URL = os.getenv('CLAIR_INDEXER_URL', 'http://clair-indexer.security.svc.cluster.local:6060')
CLAIR_MATCHER_URL = os.getenv('CLAIR_MATCHER_URL', 'http://clair-matcher.security.svc.cluster.local:6060')
SCAN_RESULTS_DIR = '/app/scan_results'
DOCKER_REGISTRY = os.getenv('DOCKER_REGISTRY', 'docker.io')

# Create directories
os.makedirs(SCAN_RESULTS_DIR, exist_ok=True)

# Global scan cache
scan_cache = {}
scan_history = []

class VulnerabilityScanner:
    """
    Vulnerability scanner using Clair backend
    """
    
    def __init__(self):
        self.indexer_url = CLAIR_INDEXER_URL
        self.matcher_url = CLAIR_MATCHER_URL
        self.session = requests.Session()
        self.session.timeout = 30
        
    def _get_manifest(self, image: str, tag: str = 'latest') -> Dict[str, Any]:
        """
        Get image manifest from registry
        """
        try:
            # For demo purposes, we'll create a simulated manifest
            # In production, this would query the actual registry
            manifest_digest = hashlib.sha256(f"{image}:{tag}".encode()).hexdigest()
            
            return {
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "schemaVersion": 2,
                "config": {
                    "mediaType": "application/vnd.docker.container.image.v1+json",
                    "size": 1469,
                    "digest": f"sha256:{manifest_digest[:64]}"
                },
                "layers": [
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "size": 32654,
                        "digest": f"sha256:{hashlib.sha256(f'layer1-{image}'.encode()).hexdigest()}"
                    },
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "size": 16724,
                        "digest": f"sha256:{hashlib.sha256(f'layer2-{image}'.encode()).hexdigest()}"
                    },
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "size": 73109,
                        "digest": f"sha256:{hashlib.sha256(f'layer3-{image}'.encode()).hexdigest()}"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get manifest for {image}:{tag}: {e}")
            return None
    
    def submit_image_for_indexing(self, image: str, tag: str = 'latest') -> str:
        """
        Submit image to Clair for indexing
        """
        try:
            manifest = self._get_manifest(image, tag)
            if not manifest:
                raise Exception("Failed to get image manifest")
            
            image_ref = f"{DOCKER_REGISTRY}/{image}:{tag}"
            manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            
            # Submit to Clair indexer
            index_payload = {
                "manifest": manifest,
                "layer": {
                    "uri": f"docker://{image_ref}",
                    "headers": {}
                }
            }
            
            logger.info(f"Submitting {image_ref} for indexing...")
            response = self.session.post(
                f"{self.indexer_url}/indexer/api/v1/index_report",
                json=index_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Image {image_ref} submitted for indexing successfully")
                return manifest_hash
            else:
                logger.error(f"Failed to submit {image_ref} for indexing: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting image for indexing: {e}")
            return None
    
    def get_vulnerability_report(self, manifest_hash: str) -> Dict[str, Any]:
        """
        Get vulnerability report for indexed image
        """
        try:
            logger.info(f"Fetching vulnerability report for manifest {manifest_hash[:12]}...")
            
            # For demo purposes, we'll generate simulated vulnerability data
            # In production, this would query Clair's vulnerability database
            demo_vulnerabilities = self._generate_demo_vulnerabilities()
            
            response = {
                "manifest_hash": manifest_hash,
                "state": "IndexFinished",
                "packages": demo_vulnerabilities['packages'],
                "distributions": demo_vulnerabilities['distributions'],
                "repository": demo_vulnerabilities['repository'],
                "environments": demo_vulnerabilities['environments'],
                "vulnerabilities": demo_vulnerabilities['vulnerabilities'],
                "success": True,
                "err": ""
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting vulnerability report: {e}")
            return {
                "manifest_hash": manifest_hash,
                "state": "IndexError",
                "success": False,
                "err": str(e)
            }
    
    def _generate_demo_vulnerabilities(self) -> Dict[str, Any]:
        """
        Generate demonstration vulnerability data for the demo
        """
        return {
            "packages": {
                "1": {
                    "id": "1",
                    "name": "openssl",
                    "version": "1.1.1k-1+deb11u1",
                    "kind": "binary",
                    "source": {
                        "id": "2",
                        "name": "openssl",
                        "version": "1.1.1k-1+deb11u1"
                    }
                },
                "2": {
                    "id": "2",
                    "name": "python3",
                    "version": "3.9.2-3",
                    "kind": "binary"
                },
                "3": {
                    "id": "3",
                    "name": "curl",
                    "version": "7.74.0-1.3+deb11u1",
                    "kind": "binary"
                },
                "4": {
                    "id": "4",
                    "name": "pytorch",
                    "version": "2.0.1",
                    "kind": "python"
                }
            },
            "distributions": {
                "1": {
                    "id": "1",
                    "did": "debian",
                    "name": "Debian GNU/Linux",
                    "version": "11 (bullseye)",
                    "version_code_name": "bullseye",
                    "version_id": "11",
                    "arch": "amd64"
                }
            },
            "repository": {
                "1": {
                    "id": "1",
                    "name": "debian",
                    "key": "debian",
                    "uri": "https://deb.debian.org/debian"
                }
            },
            "environments": {
                "1": {
                    "package_db": "var/lib/dpkg/status",
                    "introduced_in": "sha256:abc123...",
                    "distribution_id": "1"
                }
            },
            "vulnerabilities": {
                "CVE-2023-0286": {
                    "id": "CVE-2023-0286",
                    "updater": "debian",
                    "name": "CVE-2023-0286",
                    "description": "There is a type confusion vulnerability relating to X.400 address processing inside an X.509 GeneralName.",
                    "issued": "2023-02-08T00:00:00Z",
                    "links": "https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-0286",
                    "severity": "High",
                    "normalized_severity": "High",
                    "package": {
                        "id": "1",
                        "name": "openssl",
                        "version": "1.1.1k-1+deb11u1"
                    },
                    "dist": {
                        "id": "1",
                        "did": "debian",
                        "name": "Debian GNU/Linux",
                        "version": "11"
                    },
                    "repo": {
                        "id": "1",
                        "name": "debian"
                    },
                    "fixed_in_version": "1.1.1n-0+deb11u4"
                },
                "CVE-2023-23931": {
                    "id": "CVE-2023-23931",
                    "updater": "python",
                    "name": "CVE-2023-23931",
                    "description": "cryptography vulnerable to memory corruption via immutable objects",
                    "issued": "2023-02-07T00:00:00Z",
                    "links": "https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-23931",
                    "severity": "Medium",
                    "normalized_severity": "Medium",
                    "package": {
                        "id": "2",
                        "name": "python3",
                        "version": "3.9.2-3"
                    },
                    "dist": {
                        "id": "1",
                        "did": "debian",
                        "name": "Debian GNU/Linux",
                        "version": "11"
                    },
                    "fixed_in_version": "3.9.2-4"
                },
                "CVE-2023-27533": {
                    "id": "CVE-2023-27533",
                    "updater": "debian",
                    "name": "CVE-2023-27533",
                    "description": "curl TELNET option IAC injection vulnerability",
                    "issued": "2023-03-20T00:00:00Z",
                    "links": "https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-27533",
                    "severity": "Low",
                    "normalized_severity": "Low",
                    "package": {
                        "id": "3",
                        "name": "curl",
                        "version": "7.74.0-1.3+deb11u1"
                    },
                    "dist": {
                        "id": "1",
                        "did": "debian",
                        "name": "Debian GNU/Linux",
                        "version": "11"
                    },
                    "fixed_in_version": "7.74.0-1.3+deb11u2"
                },
                "DEMO-ML-001": {
                    "id": "DEMO-ML-001",
                    "updater": "demo",
                    "name": "DEMO-ML-001",
                    "description": "PyTorch contains known vulnerabilities in model deserialization that can lead to arbitrary code execution",
                    "issued": "2023-01-15T00:00:00Z",
                    "links": "https://pytorch.org/blog/pytorch-malicious-dependency-attack/",
                    "severity": "Critical",
                    "normalized_severity": "Critical",
                    "package": {
                        "id": "4",
                        "name": "pytorch",
                        "version": "2.0.1"
                    },
                    "dist": {
                        "id": "1",
                        "did": "python",
                        "name": "Python Package",
                        "version": "3.9"
                    },
                    "fixed_in_version": "2.1.0"
                }
            }
        }
    
    def scan_image(self, image: str, tag: str = 'latest') -> Dict[str, Any]:
        """
        Complete image scanning workflow
        """
        start_time = time.time()
        image_ref = f"{image}:{tag}"
        
        logger.info(f"Starting vulnerability scan for {image_ref}")
        
        # Check cache first
        cache_key = f"{image}:{tag}"
        if cache_key in scan_cache:
            cached_result = scan_cache[cache_key]
            if datetime.now() - cached_result['timestamp'] < timedelta(hours=1):
                logger.info(f"Returning cached scan result for {image_ref}")
                return cached_result['result']
        
        # Submit for indexing
        manifest_hash = self.submit_image_for_indexing(image, tag)
        if not manifest_hash:
            return {
                'image': image_ref,
                'status': 'error',
                'error': 'Failed to submit image for indexing',
                'timestamp': datetime.now().isoformat()
            }
        
        # Wait a bit for indexing to complete (in demo)
        time.sleep(2)
        
        # Get vulnerability report
        vuln_report = self.get_vulnerability_report(manifest_hash)
        
        # Process and format results
        scan_result = self._process_scan_results(image_ref, vuln_report, start_time)
        
        # Cache results
        scan_cache[cache_key] = {
            'result': scan_result,
            'timestamp': datetime.now()
        }
        
        # Add to scan history
        scan_history.append({
            'image': image_ref,
            'timestamp': datetime.now().isoformat(),
            'status': scan_result['status'],
            'vulnerabilities': scan_result.get('summary', {}).get('total_vulnerabilities', 0)
        })
        
        return scan_result
    
    def _process_scan_results(self, image_ref: str, vuln_report: Dict, start_time: float) -> Dict[str, Any]:
        """
        Process and format vulnerability scan results
        """
        try:
            if not vuln_report.get('success', False):
                return {
                    'image': image_ref,
                    'status': 'error',
                    'error': vuln_report.get('err', 'Unknown error'),
                    'timestamp': datetime.now().isoformat(),
                    'scan_duration': time.time() - start_time
                }
            
            vulnerabilities = vuln_report.get('vulnerabilities', {})
            
            # Categorize vulnerabilities by severity
            severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
            vuln_details = []
            
            for vuln_id, vuln in vulnerabilities.items():
                severity = vuln.get('normalized_severity', 'Unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                vuln_details.append({
                    'id': vuln_id,
                    'severity': severity,
                    'description': vuln.get('description', ''),
                    'package': vuln.get('package', {}).get('name', 'unknown'),
                    'current_version': vuln.get('package', {}).get('version', 'unknown'),
                    'fixed_version': vuln.get('fixed_in_version', 'Not available'),
                    'links': vuln.get('links', ''),
                    'issued': vuln.get('issued', '')
                })
            
            # Sort vulnerabilities by severity
            severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Unknown': 4}
            vuln_details.sort(key=lambda x: severity_order.get(x['severity'], 4))
            
            # Calculate risk score
            risk_score = (
                severity_counts['Critical'] * 10 +
                severity_counts['High'] * 7 +
                severity_counts['Medium'] * 4 +
                severity_counts['Low'] * 1
            )
            
            # Determine overall risk level
            if severity_counts['Critical'] > 0:
                risk_level = 'Critical'
            elif severity_counts['High'] > 0:
                risk_level = 'High'
            elif severity_counts['Medium'] > 0:
                risk_level = 'Medium'
            elif severity_counts['Low'] > 0:
                risk_level = 'Low'
            else:
                risk_level = 'Clean'
            
            return {
                'image': image_ref,
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'scan_duration': time.time() - start_time,
                'summary': {
                    'total_vulnerabilities': len(vulnerabilities),
                    'severity_counts': severity_counts,
                    'risk_score': risk_score,
                    'risk_level': risk_level
                },
                'vulnerabilities': vuln_details,
                'packages': vuln_report.get('packages', {}),
                'distributions': vuln_report.get('distributions', {}),
                'manifest_hash': vuln_report.get('manifest_hash', '')
            }
            
        except Exception as e:
            logger.error(f"Error processing scan results: {e}")
            return {
                'image': image_ref,
                'status': 'error',
                'error': f'Processing error: {str(e)}',
                'timestamp': datetime.now().isoformat(),
                'scan_duration': time.time() - start_time
            }

# Initialize scanner
scanner = VulnerabilityScanner()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test Clair connectivity
        indexer_health = requests.get(f"{CLAIR_INDEXER_URL}/healthz", timeout=5)
        matcher_health = requests.get(f"{CLAIR_MATCHER_URL}/healthz", timeout=5)
        
        return jsonify({
            'status': 'healthy',
            'service': 'Vulnerability Scanner Service',
            'clair_indexer': 'healthy' if indexer_health.status_code == 200 else 'unhealthy',
            'clair_matcher': 'healthy' if matcher_health.status_code == 200 else 'unhealthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/scan', methods=['POST'])
def scan_image():
    """Scan a container image for vulnerabilities"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Image name is required'}), 400
        
        image = data['image']
        tag = data.get('tag', 'latest')
        
        logger.info(f"Received scan request for {image}:{tag}")
        
        # Perform scan
        result = scanner.scan_image(image, tag)
        
        # Save scan result
        result_file = os.path.join(SCAN_RESULTS_DIR, f"scan_{int(time.time())}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Scan failed: {str(e)}'}), 500

@app.route('/scan/results/<image_name>', methods=['GET'])
def get_scan_results(image_name):
    """Get scan results for a specific image"""
    try:
        tag = request.args.get('tag', 'latest')
        cache_key = f"{image_name}:{tag}"
        
        if cache_key in scan_cache:
            return jsonify(scan_cache[cache_key]['result'])
        else:
            return jsonify({'error': 'No scan results found for this image'}), 404
            
    except Exception as e:
        logger.error(f"Error retrieving scan results: {e}")
        return jsonify({'error': f'Failed to retrieve results: {str(e)}'}), 500

@app.route('/scan/history', methods=['GET'])
def get_scan_history():
    """Get scan history"""
    try:
        limit = int(request.args.get('limit', 50))
        return jsonify({
            'history': scan_history[-limit:],
            'total_scans': len(scan_history)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get history: {str(e)}'}), 500

@app.route('/scan/demo', methods=['POST'])
def scan_demo_images():
    """Scan all demo images"""
    try:
        demo_images = [
            'kubecon-demo/image-classifier',
            'kubecon-demo/art-defense-service',
            'python:3.10-slim',
            'postgres:13-alpine'
        ]
        
        results = {}
        for image in demo_images:
            logger.info(f"Scanning demo image: {image}")
            result = scanner.scan_image(image.replace('kubecon-demo/', ''), 'latest')
            results[image] = result
        
        # Generate summary
        total_vulns = sum(r.get('summary', {}).get('total_vulnerabilities', 0) for r in results.values())
        critical_images = [img for img, r in results.items() 
                          if r.get('summary', {}).get('severity_counts', {}).get('Critical', 0) > 0]
        
        return jsonify({
            'demo_scan_results': results,
            'summary': {
                'total_images_scanned': len(demo_images),
                'total_vulnerabilities_found': total_vulns,
                'images_with_critical_vulns': len(critical_images),
                'critical_images': critical_images
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Demo scan error: {e}")
        return jsonify({'error': f'Demo scan failed: {str(e)}'}), 500

@app.route('/scan/stats', methods=['GET'])
def get_scan_stats():
    """Get scanning statistics"""
    try:
        return jsonify({
            'total_scans': len(scan_history),
            'cached_results': len(scan_cache),
            'service_uptime': time.time(),
            'clair_endpoints': {
                'indexer': CLAIR_INDEXER_URL,
                'matcher': CLAIR_MATCHER_URL
            }
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Vulnerability Scanner Service")
    logger.info(f"Clair Indexer: {CLAIR_INDEXER_URL}")
    logger.info(f"Clair Matcher: {CLAIR_MATCHER_URL}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002, debug=True) 