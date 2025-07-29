#!/bin/bash

# KubeCon 2025 Demo - Attack Execution Script
# Execute various MITRE ATLAS attack scenarios for demonstration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEMO_NAMESPACE="ai-demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ATTACK_POD="attack-tools"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_attack() {
    echo -e "${RED}[ATTACK]${NC} $1"
}

log_detect() {
    echo -e "${CYAN}[DETECT]${NC} $1"
}

# Show banner
show_banner() {
    echo -e "${RED}"
    echo "████████████████████████████████████████████████████████"
    echo "█  MITRE ATLAS Attack Simulation - KubeCon 2025 Demo  █"
    echo "█  WARNING: This is a controlled demonstration         █"
    echo "████████████████████████████████████████████████████████"
    echo -e "${NC}"
}

# Check if demo environment is ready
check_environment() {
    log_info "Checking demo environment..."
    
    # Check if AI application is running
    if ! kubectl get deployment image-classifier -n "$DEMO_NAMESPACE" &> /dev/null; then
        log_error "AI application not deployed. Run ./scripts/setup-demo.sh first"
        exit 1
    fi
    
    # Check if attack tools pod exists
    if ! kubectl get pod "$ATTACK_POD" -n "$DEMO_NAMESPACE" &> /dev/null; then
        log_error "Attack tools pod not found. Run ./scripts/setup-demo.sh first"
        exit 1
    fi
    
    # Check if attack pod is ready
    if [[ $(kubectl get pod "$ATTACK_POD" -n "$DEMO_NAMESPACE" -o jsonpath='{.status.phase}') != "Running" ]]; then
        log_error "Attack tools pod is not running"
        exit 1
    fi
    
    log_info "Environment check passed ✓"
}

# Get AI service endpoints
get_ai_endpoint() {
    AI_SERVICE_IP=$(kubectl get svc image-classifier-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    AI_ENDPOINT="http://$AI_SERVICE_IP"
    echo "$AI_ENDPOINT"
}

get_defense_endpoint() {
    DEFENSE_SERVICE_IP=$(kubectl get svc art-defense-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    DEFENSE_ENDPOINT="http://$DEFENSE_SERVICE_IP"
    echo "$DEFENSE_ENDPOINT"
}

# Execute adversarial attack scenario
run_adversarial_attack() {
    log_attack "Executing MITRE ATLAS T1551 - Evade ML Model (Adversarial Examples)"
    echo ""
    
    local ai_endpoint=$(get_ai_endpoint)
    
    log_info "Target: $ai_endpoint"
    log_info "Technique: Adversarial Example Generation"
    echo ""
    
    # Step 1: Test normal model behavior
    log_info "Step 1: Testing normal model behavior..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

endpoint = '$ai_endpoint'
health_resp = requests.get(f'{endpoint}/health', timeout=10)
print(f'Health Check: {health_resp.status_code}')

if health_resp.status_code == 200:
    info_resp = requests.get(f'{endpoint}/model/info', timeout=10)
    if info_resp.status_code == 200:
        info = info_resp.json()
        print(f'Model Type: {info.get(\"model_type\", \"Unknown\")}')
        print(f'Classes: {info.get(\"num_classes\", \"Unknown\")}')
        print(f'Parameters: {info.get(\"parameters_count\", \"Unknown\")}')
    else:
        print('Model info not accessible')
else:
    print('Service not accessible')
"
    
    echo ""
    log_detect "Expected Detection: Falco should detect model information disclosure"
    echo ""
    
    # Step 2: Generate adversarial examples
    log_info "Step 2: Generating adversarial examples..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 /attacks/adversarial/generate_adversarial.py \
        --target "$ai_endpoint" \
        --image "/attacks/sample_image.jpg" \
        --attack "all" \
        --output "/tmp/adversarial_results" || true
    
    echo ""
    log_detect "Expected Detection: KubeArmor should detect suspicious process execution"
    echo ""
    
    # Step 3: High-frequency queries (DoS simulation)
    log_info "Step 3: Simulating high-frequency queries (DoS pattern)..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

endpoint = '$ai_endpoint'
query_count = 50
print(f'Sending {query_count} rapid queries to {endpoint}/predict...')

def send_query(i):
    try:
        # Simulate file upload
        files = {'file': ('test.txt', b'fake image data', 'image/jpeg')}
        resp = requests.post(f'{endpoint}/predict', files=files, timeout=5)
        if i % 10 == 0:
            print(f'Query {i}: {resp.status_code}')
    except Exception as e:
        if i % 10 == 0:
            print(f'Query {i}: Error - {str(e)[:50]}')

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(send_query, i) for i in range(query_count)]
    for future in futures:
        future.result()

print('High-frequency query attack completed')
"
    
    echo ""
    log_detect "Expected Detection: Falco should detect DoS attack pattern"
    log_info "Adversarial attack scenario completed"
}

# Execute model extraction attack
run_extraction_attack() {
    log_attack "Executing MITRE ATLAS T1552 - Steal ML Model (Model Extraction)"
    echo ""
    
    local ai_endpoint=$(get_ai_endpoint)
    
    log_info "Target: $ai_endpoint"
    log_info "Technique: Model Parameter and Architecture Extraction"
    echo ""
    
    # Step 1: Information gathering
    log_info "Step 1: Gathering model information..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

endpoint = '$ai_endpoint'
endpoints_to_probe = ['/model/info', '/model/weights', '/logs', '/admin/reset']

for ep in endpoints_to_probe:
    try:
        resp = requests.get(f'{endpoint}{ep}', timeout=10)
        print(f'{ep}: Status {resp.status_code}')
        if resp.status_code == 200 and 'json' in resp.headers.get('content-type', ''):
            data = resp.json()
            if ep == '/model/info':
                print(f'  Model: {data.get(\"model_type\", \"Unknown\")}')
                print(f'  Size: {data.get(\"model_size_mb\", \"Unknown\")} MB')
            elif ep == '/logs':
                print(f'  Log entries: {len(data.get(\"recent_queries\", []))}')
    except Exception as e:
        print(f'{ep}: Error - {str(e)[:50]}')
"
    
    echo ""
    log_detect "Expected Detection: Falco should detect model information disclosure attempts"
    echo ""
    
    # Step 2: Weight extraction attempts
    log_info "Step 2: Attempting weight extraction..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests

endpoint = '$ai_endpoint'
common_layers = ['conv1', 'conv2', 'fc', 'classifier', 'features', 'layer1']

for layer in common_layers:
    try:
        resp = requests.get(f'{endpoint}/model/weights', params={'layer': layer}, timeout=10)
        print(f'Layer {layer}: Status {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            if 'weights' in data:
                print(f'  Shape: {data.get(\"shape\", \"Unknown\")}')
                print(f'  Weights extracted: {len(str(data[\"weights\"]))} characters')
        elif resp.status_code == 404:
            print(f'  Layer not found')
    except Exception as e:
        print(f'Layer {layer}: Error - {str(e)[:50]}')
"
    
    echo ""
    log_detect "Expected Detection: KubeArmor should detect suspicious file access patterns"
    echo ""
    
    # Step 3: Automated extraction with Python script
    log_info "Step 3: Running automated model extraction..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 /attacks/extraction/model_extraction.py \
        --target "$ai_endpoint" \
        --queries 100 \
        --output "/tmp/extraction_results" || true
    
    echo ""
    log_detect "Expected Detection: Multiple security tools should detect automated extraction"
    log_info "Model extraction attack scenario completed"
}

# Execute data poisoning attack
run_poisoning_attack() {
    log_attack "Executing MITRE ATLAS T1565 - Data Manipulation (Data Poisoning)"
    echo ""
    
    local ai_endpoint=$(get_ai_endpoint)
    
    log_info "Target: $ai_endpoint"
    log_info "Technique: Training Data Poisoning"
    echo ""
    
    # Step 1: Upload malicious feedback data
    log_info "Step 1: Uploading poisoned feedback data..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

endpoint = '$ai_endpoint'

# Simulate poisoned feedback data
poisoned_data = {
    'image_id': 'malicious_001',
    'correct_label': 'airplane',
    'confidence': 0.95,
    'user_feedback': 'This is definitely an airplane',
    'metadata': {
        'source': 'automated_script',
        'poisoning_type': 'label_flip',
        'target_class': 'bird'
    },
    'training_data': [
        {'image': 'fake_airplane_1.jpg', 'label': 'airplane'},
        {'image': 'fake_airplane_2.jpg', 'label': 'airplane'},
        # Malicious entries
        {'image': 'bird_001.jpg', 'label': 'airplane'},  # Mislabeled
        {'image': 'bird_002.jpg', 'label': 'airplane'},  # Mislabeled
    ]
}

try:
    resp = requests.post(f'{endpoint}/feedback', json=poisoned_data, timeout=10)
    print(f'Feedback upload: Status {resp.status_code}')
    if resp.status_code == 200:
        data = resp.json()
        print(f'Poisoned data saved to: {data.get(\"file\", \"unknown\")}')
except Exception as e:
    print(f'Feedback upload error: {e}')
"
    
    echo ""
    
    # Step 2: Upload malicious files
    log_info "Step 2: Uploading malicious training files..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests

endpoint = '$ai_endpoint'

# Create fake file content
fake_files = [
    ('malicious_model.pkl', b'fake pickle data with backdoor'),
    ('../../../etc/passwd', b'attempting path traversal'),
    ('large_file.bin', b'X' * 10000),  # Large file
]

for filename, content in fake_files:
    try:
        files = {'file': (filename, content, 'application/octet-stream')}
        data = {'filename': filename}
        resp = requests.post(f'{endpoint}/upload', files=files, data=data, timeout=10)
        print(f'Upload {filename}: Status {resp.status_code}')
        if resp.status_code == 200:
            result = resp.json()
            print(f'  Saved to: {result.get(\"path\", \"unknown\")}')
    except Exception as e:
        print(f'Upload {filename}: Error - {str(e)[:50]}')
"
    
    echo ""
    log_detect "Expected Detection: KubeArmor should detect malicious file uploads"
    echo ""
    
    # Step 3: Attempt model replacement
    log_info "Step 3: Attempting model replacement..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

endpoint = '$ai_endpoint'

# Attempt to load malicious model
malicious_model_paths = [
    '/app/uploads/malicious_model.pkl',
    '/tmp/backdoor_model.pt',
    '../models/compromised.pth',
    '/app/data/poisoned_model.pkl'
]

for model_path in malicious_model_paths:
    try:
        payload = {'model_path': model_path}
        resp = requests.post(f'{endpoint}/model/load', json=payload, timeout=10)
        print(f'Model load attempt {model_path}: Status {resp.status_code}')
        if resp.status_code == 200:
            print(f'  SUCCESS: Model loaded from {model_path}')
        else:
            print(f'  Failed: {resp.text[:100]}')
    except Exception as e:
        print(f'Model load {model_path}: Error - {str(e)[:50]}')
"
    
    echo ""
    log_detect "Expected Detection: Multiple alerts for unauthorized model loading"
    log_info "Data poisoning attack scenario completed"
}

# Execute IBM ART advanced attack scenario
run_art_attack() {
    log_attack "Executing IBM ART Advanced Attack Suite - MITRE ATLAS T1551 (Enhanced)"
    echo ""
    
    local ai_endpoint=$(get_ai_endpoint)
    local defense_endpoint=$(get_defense_endpoint)
    
    log_info "Vulnerable Target: $ai_endpoint"
    log_info "Defended Target: $defense_endpoint"
    log_info "Framework: IBM Adversarial Robustness Toolbox (ART)"
    echo ""
    
    # Step 1: Compare vulnerable vs defended predictions
    log_info "Step 1: Baseline comparison - vulnerable vs defended model..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

vulnerable_endpoint = '$ai_endpoint'
defended_endpoint = '$defense_endpoint'

# Test with a normal image
print('=== Baseline Prediction Comparison ===')
test_endpoints = [
    ('Vulnerable Model', f'{vulnerable_endpoint}/predict'),
    ('IBM ART Defended', f'{defended_endpoint}/defense/predict')
]

for name, endpoint in test_endpoints:
    try:
        # Create a simple test image
        with open('/tmp/test.txt', 'w') as f:
            f.write('test image data')
        
        files = {'file': ('test.jpg', open('/tmp/test.txt', 'rb'), 'image/jpeg')}
        resp = requests.post(endpoint, files=files, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            top_pred = data['predictions'][0] if 'predictions' in data else 'Unknown'
            defense_info = data.get('defense_info', {})
            print(f'{name}: {top_pred}')
            if defense_info:
                print(f'  Defense Level: {defense_info.get(\"defense_level\", \"none\")}')
                print(f'  Adversarial Detected: {defense_info.get(\"adversarial_detected\", False)}')
        else:
            print(f'{name}: Error {resp.status_code}')
    except Exception as e:
        print(f'{name}: Error - {str(e)[:50]}')
"
    
    echo ""
    log_detect "Expected Detection: Defense service should show applied protections"
    echo ""
    
    # Step 2: Run IBM ART advanced attacks
    log_info "Step 2: Executing IBM ART advanced adversarial attacks..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 /attacks/adversarial/art_attacks.py \
        --target "$ai_endpoint" \
        --image "/attacks/sample_image.jpg" \
        --attack "all" \
        --output "/tmp/art_results" || true
    
    echo ""
    log_detect "Expected Detection: Multiple sophisticated attack techniques detected"
    echo ""
    
    # Step 3: Test defense effectiveness
    log_info "Step 3: Testing IBM ART defense effectiveness..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests
import json

defended_endpoint = '$defense_endpoint'

print('=== IBM ART Defense Benchmark ===')
defense_levels = ['none', 'light', 'medium', 'heavy']

for level in defense_levels:
    try:
        # Create test file
        with open('/tmp/test.txt', 'w') as f:
            f.write('test adversarial image data')
        
        files = {'file': ('adv_test.jpg', open('/tmp/test.txt', 'rb'), 'image/jpeg')}
        data = {'defense_level': level}
        
        resp = requests.post(f'{defended_endpoint}/defense/predict', files=files, data=data, timeout=15)
        
        if resp.status_code == 200:
            result = resp.json()
            defense_info = result.get('defense_info', {})
            print(f'Defense Level {level.upper()}:')
            print(f'  Applied Defenses: {defense_info.get(\"applied_defenses\", [])}')
            print(f'  Processing Time: {result.get(\"processing_time\", 0):.3f}s')
            print(f'  Adversarial Detected: {defense_info.get(\"adversarial_detected\", False)}')
        else:
            print(f'Defense Level {level}: Error {resp.status_code}')
    except Exception as e:
        print(f'Defense Level {level}: Error - {str(e)[:50]}')
    print()
"
    
    echo ""
    log_detect "Expected Detection: Graduated defense effectiveness shown"
    echo ""
    
    # Step 4: Adversarial detection demonstration
    log_info "Step 4: IBM ART adversarial detection capabilities..."
    kubectl exec -n "$DEMO_NAMESPACE" "$ATTACK_POD" -- python3 -c "
import requests

defended_endpoint = '$defense_endpoint'

print('=== Adversarial Detection Test ===')
try:
    # Test detection endpoint
    with open('/tmp/suspicious.txt', 'w') as f:
        f.write('potentially adversarial image data with anomalous patterns')
    
    files = {'file': ('suspicious.jpg', open('/tmp/suspicious.txt', 'rb'), 'image/jpeg')}
    resp = requests.post(f'{defended_endpoint}/defense/detect', files=files, timeout=15)
    
    if resp.status_code == 200:
        detection = resp.json()
        print(f'Adversarial Detected: {detection.get(\"is_adversarial\", False)}')
        print(f'Confidence Score: {detection.get(\"confidence_score\", 0):.3f}')
        print(f'Consistency Score: {detection.get(\"consistency_score\", 1):.3f}')
        print(f'Detection Method: {detection.get(\"detection_method\", \"Unknown\")}')
    else:
        print(f'Detection test failed: {resp.status_code}')
except Exception as e:
    print(f'Detection test error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: Advanced adversarial detection capabilities demonstrated"
    log_info "IBM ART advanced attack scenario completed"
}

# Show security monitoring status
show_security_status() {
    echo ""
    echo -e "${CYAN}=== Security Monitoring Status ===${NC}"
    
    # Falco alerts
    echo -e "${YELLOW}Recent Falco Alerts:${NC}"
    kubectl logs --tail=20 daemonset/falco -n security | grep -E "(WARNING|ERROR|CRITICAL)" | tail -5 || echo "No recent alerts"
    
    echo ""
    
    # KubeArmor logs
    echo -e "${YELLOW}Recent KubeArmor Events:${NC}"
    kubectl logs --tail=20 daemonset/kubearmor -n security | grep -E "(WARN|ERROR|ALERT)" | tail -5 || echo "No recent events"
    
    echo ""
    
    # Pod events
    echo -e "${YELLOW}Recent Pod Events:${NC}"
    kubectl get events -n "$DEMO_NAMESPACE" --sort-by='.lastTimestamp' | tail -5 || echo "No recent events"
}

# Main help function
show_help() {
    echo "Usage: $0 <attack_type> [options]"
    echo ""
    echo "Attack Types:"
    echo "  adversarial  - MITRE ATLAS T1551: Evade ML Model (Adversarial Examples)"
    echo "  extraction   - MITRE ATLAS T1552: Steal ML Model (Model Extraction)"
    echo "  poisoning    - MITRE ATLAS T1565: Data Manipulation (Data Poisoning)"
    echo "  art          - IBM ART Advanced Attack Suite (Enhanced T1551)"
    echo "  all          - Run all attack scenarios sequentially"
    echo "  status       - Show security monitoring status"
    echo ""
    echo "Options:"
    echo "  --help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 adversarial                 # Run adversarial attack"
    echo "  $0 extraction                  # Run model extraction"
    echo "  $0 art                         # Run IBM ART advanced attacks"
    echo "  $0 all                         # Run all attacks"
    echo "  $0 status                      # Check security status"
}

# Main execution
main() {
    if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    show_banner
    check_environment
    
    local attack_type="$1"
    
    case "$attack_type" in
        "adversarial")
            run_adversarial_attack
            show_security_status
            ;;
        "extraction")
            run_extraction_attack
            show_security_status
            ;;
        "poisoning")
            run_poisoning_attack
            show_security_status
            ;;
        "art")
            run_art_attack
            show_security_status
            ;;
        "all")
            log_info "Running complete attack scenario suite..."
            echo ""
            run_adversarial_attack
            echo ""
            echo "==========================================="
            echo ""
            run_extraction_attack
            echo ""
            echo "==========================================="
            echo ""
            run_poisoning_attack
            echo ""
            echo "==========================================="
            echo ""
            run_art_attack
            show_security_status
            ;;
        "status")
            show_security_status
            ;;
        *)
            log_error "Unknown attack type: $attack_type"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}Attack demonstration completed!${NC}"
    echo -e "${CYAN}Check Grafana dashboard for detailed security metrics.${NC}"
}

# Execute main function
main "$@" 