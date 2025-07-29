#!/bin/bash

# KubeCon 2025 Demo Setup Script
# Aligning Enterprise AI Security with MITRE ATLAS Using Open Source Technologies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEMO_NAMESPACE="ai-demo"
SECURITY_NAMESPACE="security"
MONITORING_NAMESPACE="monitoring"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "  KubeCon 2025 Demo: AI Security with MITRE ATLAS"
    echo "  Aligning Enterprise AI Security Using Open Source Technologies"
    echo "=================================================================="
    echo -e "${NC}"
}

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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/README.md" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Create namespaces
create_namespaces() {
    log_step "Creating namespaces..."
    
    kubectl apply -f "$PROJECT_ROOT/kubernetes/base/namespace.yaml"
    
    # Wait for namespaces to be ready
    kubectl wait --for=condition=Active namespace/$DEMO_NAMESPACE --timeout=60s
    kubectl wait --for=condition=Active namespace/$SECURITY_NAMESPACE --timeout=60s
    kubectl wait --for=condition=Active namespace/$MONITORING_NAMESPACE --timeout=60s
    
    log_info "Namespaces created successfully ✓"
}

# Build and push container images
build_images() {
    log_step "Building container images..."
    
    # Build image classifier
    log_info "Building image classifier..."
    cd "$PROJECT_ROOT/applications/image-classifier"
    docker build -t kubecon-demo/image-classifier:latest .
    
    # Build IBM ART defense service
    log_info "Building IBM ART defense service..."
    docker build -f Dockerfile.art-defense -t kubecon-demo/art-defense-service:latest .
    
    # Build vulnerability scanner service
    log_info "Building vulnerability scanner service..."
    cd "$PROJECT_ROOT/applications/vuln-scanner"
    docker build -t kubecon-demo/vuln-scanner-service:latest .
    
    # Build LLM service
    log_info "Building vulnerable LLM service..."
    cd "$PROJECT_ROOT/applications/llm-service"
    docker build -t kubecon-demo/llm-service:latest .
    
    # Build Garak scanner service
    log_info "Building Garak scanner service..."
    cd "$PROJECT_ROOT/applications/garak-scanner"
    docker build -t kubecon-demo/garak-scanner-service:latest .
    
    # Tag for local registry (if using kind/minikube)
    if kubectl config current-context | grep -E "(kind|minikube)" &> /dev/null; then
        log_info "Detected local Kubernetes cluster, loading images..."
        if command -v kind &> /dev/null && kind get clusters 2>/dev/null | grep -q "kind"; then
            kind load docker-image kubecon-demo/image-classifier:latest
            kind load docker-image kubecon-demo/art-defense-service:latest
            kind load docker-image kubecon-demo/vuln-scanner-service:latest
            kind load docker-image kubecon-demo/llm-service:latest
            kind load docker-image kubecon-demo/garak-scanner-service:latest
        elif command -v minikube &> /dev/null; then
            minikube image load kubecon-demo/image-classifier:latest
            minikube image load kubecon-demo/art-defense-service:latest
            minikube image load kubecon-demo/vuln-scanner-service:latest
            minikube image load kubecon-demo/llm-service:latest
            minikube image load kubecon-demo/garak-scanner-service:latest
        fi
    fi
    
    cd "$PROJECT_ROOT"
    log_info "Container images built successfully ✓"
}

# Install Helm repositories
setup_helm_repos() {
    log_step "Setting up Helm repositories..."
    
    # Add security tool repositories
    helm repo add falcosecurity https://falcosecurity.github.io/charts
    helm repo add kubearmor https://kubearmor.github.io/charts
    
    # Add monitoring repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    
    # Update repositories
    helm repo update
    
    log_info "Helm repositories configured ✓"
}

# Install Falco for runtime security monitoring
install_falco() {
    log_step "Installing Falco for runtime security monitoring..."
    
    # Create Falco configuration
    cat > /tmp/falco-values.yaml << EOF
falco:
  grpc:
    enabled: true
  grpcOutput:
    enabled: true
  jsonOutput: true
  jsonIncludeOutputProperty: true
  
falcosidekick:
  enabled: true
  config:
    webhook:
      address: "http://webhook-logger:8080/falco"
  
driver:
  enabled: true
  kind: ebpf

customRules:
  ai-security.yaml: |-
$(cat "$PROJECT_ROOT/security/falco/ai-security-rules.yaml" | sed 's/^/    /')
EOF
    
    # Install Falco
    helm upgrade --install falco falcosecurity/falco \
        --namespace "$SECURITY_NAMESPACE" \
        --values /tmp/falco-values.yaml \
        --wait
    
    log_info "Falco installed successfully ✓"
}

# Install KubeArmor for runtime security enforcement
install_kubearmor() {
    log_step "Installing KubeArmor for runtime security enforcement..."
    
    # Install KubeArmor
    helm upgrade --install kubearmor kubearmor/kubearmor \
        --namespace "$SECURITY_NAMESPACE" \
        --set enableStdOutLogs=true \
        --wait
    
    # Wait for KubeArmor to be ready
    kubectl wait --for=condition=ready pod -l app=kubearmor -n "$SECURITY_NAMESPACE" --timeout=300s
    
    log_info "KubeArmor installed successfully ✓"
}

# Install Clair vulnerability scanner
install_clair() {
    log_step "Installing Clair vulnerability scanner..."
    
    # Deploy Clair components
    kubectl apply -f "$PROJECT_ROOT/kubernetes/security/clair-deployment.yaml"
    
    # Wait for Clair components to be ready
    log_info "Waiting for Clair PostgreSQL..."
    kubectl wait --for=condition=ready pod -l app=clair,component=database -n "$SECURITY_NAMESPACE" --timeout=300s
    
    log_info "Waiting for Clair indexer..."
    kubectl wait --for=condition=ready pod -l app=clair,component=indexer -n "$SECURITY_NAMESPACE" --timeout=300s
    
    log_info "Waiting for Clair matcher..."
    kubectl wait --for=condition=ready pod -l app=clair,component=matcher -n "$SECURITY_NAMESPACE" --timeout=300s
    
    # Deploy vulnerability scanner service
    kubectl apply -f "$PROJECT_ROOT/kubernetes/security/vuln-scanner-deployment.yaml"
    
    # Deploy Garak scanner service
    kubectl apply -f "$PROJECT_ROOT/kubernetes/security/garak-scanner-deployment.yaml"
    
    # Wait for vulnerability scanner to be ready
    log_info "Waiting for vulnerability scanner service..."
    kubectl wait --for=condition=ready pod -l app=vuln-scanner-service -n "$SECURITY_NAMESPACE" --timeout=300s
    
    # Wait for Garak scanner to be ready
    log_info "Waiting for Garak scanner service..."
    kubectl wait --for=condition=ready pod -l app=garak-scanner-service -n "$SECURITY_NAMESPACE" --timeout=300s
    
    log_info "Clair vulnerability scanner and Garak LLM scanner installed successfully ✓"
}

# Install monitoring stack
install_monitoring() {
    log_step "Installing monitoring stack (Prometheus + Grafana)..."
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --set grafana.adminPassword=admin \
        --set grafana.service.type=NodePort \
        --set prometheus.service.type=NodePort \
        --wait
    
    log_info "Monitoring stack installed successfully ✓"
}

# Deploy the vulnerable AI application
deploy_ai_application() {
    log_step "Deploying vulnerable AI application..."
    
    # Deploy the vulnerable application
    kubectl apply -f "$PROJECT_ROOT/kubernetes/base/image-classifier-deployment.yaml"
    
        # Deploy IBM ART defense service
    kubectl apply -f "$PROJECT_ROOT/kubernetes/base/art-defense-deployment.yaml"
    
    # Deploy LLM service
    kubectl apply -f "$PROJECT_ROOT/kubernetes/base/llm-service-deployment.yaml"

    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment/image-classifier -n "$DEMO_NAMESPACE" --timeout=300s
    kubectl wait --for=condition=available deployment/art-defense-service -n "$DEMO_NAMESPACE" --timeout=300s
    kubectl wait --for=condition=available deployment/llm-service -n "$DEMO_NAMESPACE" --timeout=300s
    
    log_info "AI applications (vulnerable + defense + LLM) deployed successfully ✓"
}

# Apply security policies
apply_security_policies() {
    log_step "Applying security policies..."
    
    # Apply KubeArmor policies
    kubectl apply -f "$PROJECT_ROOT/security/kubearmor/ai-security-policies.yaml"
    
    # Wait a bit for policies to be processed
    sleep 10
    
    log_info "Security policies applied successfully ✓"
}

# Create sample data for testing
create_sample_data() {
    log_step "Creating sample data for testing..."
    
    # Create sample images directory
    mkdir -p "$PROJECT_ROOT/sample-data/images"
    
    # Create a simple test image using Python (if available)
    if command -v python3 &> /dev/null; then
        python3 << 'EOF'
from PIL import Image
import numpy as np
import os

# Create a simple test image
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
os.makedirs("sample-data/images", exist_ok=True)
img.save("sample-data/images/test_image.jpg")
print("Test image created: sample-data/images/test_image.jpg")
EOF
    fi
    
    log_info "Sample data created ✓"
}

# Setup attack tools
setup_attack_tools() {
    log_step "Setting up attack tools..."
    
    # Create attack tools pod
    cat > /tmp/attack-tools-pod.yaml << EOF
apiVersion: v1
kind: Pod
metadata:
  name: attack-tools
  namespace: $DEMO_NAMESPACE
  labels:
    app: attack-tools
    role: demo-attacker
spec:
  containers:
  - name: attack-tools
    image: python:3.10-slim
    command: ["/bin/bash", "-c", "apt-get update && apt-get install -y curl wget && pip install torch torchvision pillow numpy requests && sleep infinity"]
    workingDir: /attacks
    volumeMounts:
    - name: attack-scripts
      mountPath: /attacks
  volumes:
  - name: attack-scripts
    configMap:
      name: attack-scripts
      defaultMode: 0755
  restartPolicy: Never
EOF
    
    # Create attack scripts configmap
    kubectl create configmap attack-scripts -n "$DEMO_NAMESPACE" \
        --from-file="$PROJECT_ROOT/attacks/" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy attack tools pod
    kubectl apply -f /tmp/attack-tools-pod.yaml
    
    log_info "Attack tools setup completed ✓"
}

# Configure dashboard and dashboards
setup_dashboards() {
    log_step "Setting up monitoring dashboards..."
    
    # Create MITRE ATLAS dashboard configuration
    cat > /tmp/mitre-atlas-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "MITRE ATLAS - AI Security Dashboard",
    "panels": [
      {
        "title": "AI Security Events",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(falco_events_total[5m])",
            "legendFormat": "Security Events"
          }
        ]
      },
      {
        "title": "Model Query Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"image-classifier\"}[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      }
    ]
  }
}
EOF
    
    log_info "Dashboards configured ✓"
}

# Display access information
show_access_info() {
    log_step "Demo deployment completed! Access information:"
    
    echo ""
    echo -e "${GREEN}=== AI Application ===${NC}"
    AI_SERVICE_IP=$(kubectl get svc image-classifier-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    DEFENSE_SERVICE_IP=$(kubectl get svc art-defense-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    LLM_SERVICE_IP=$(kubectl get svc llm-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    SCANNER_SERVICE_IP=$(kubectl get svc vuln-scanner-service -n "$SECURITY_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    GARAK_SERVICE_IP=$(kubectl get svc garak-scanner-service -n "$SECURITY_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    echo "Vulnerable AI Service: http://$AI_SERVICE_IP"
    echo "IBM ART Defense Service: http://$DEFENSE_SERVICE_IP"
    echo "Vulnerable LLM Service: http://$LLM_SERVICE_IP"
    echo "Container Vuln Scanner: http://$SCANNER_SERVICE_IP"
    echo "Garak LLM Scanner: http://$GARAK_SERVICE_IP"
    echo "Health Check (AI): kubectl port-forward svc/image-classifier-service 8080:80 -n $DEMO_NAMESPACE"
    echo "Health Check (Defense): kubectl port-forward svc/art-defense-service 8081:80 -n $DEMO_NAMESPACE"
    echo "Health Check (LLM): kubectl port-forward svc/llm-service 8083:80 -n $DEMO_NAMESPACE"
    echo "Health Check (Container Scanner): kubectl port-forward svc/vuln-scanner-service 8082:80 -n $SECURITY_NAMESPACE"
    echo "Health Check (LLM Scanner): kubectl port-forward svc/garak-scanner-service 8084:80 -n $SECURITY_NAMESPACE"
    
    echo ""
    echo -e "${GREEN}=== Monitoring ===${NC}"
    echo "Grafana: kubectl port-forward svc/prometheus-grafana 3000:80 -n $MONITORING_NAMESPACE"
    echo "Then visit: http://localhost:3000 (admin/admin)"
    echo "Prometheus: kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n $MONITORING_NAMESPACE"
    
    echo ""
    echo -e "${GREEN}=== Security Monitoring ===${NC}"
    echo "Falco Logs: kubectl logs -f daemonset/falco -n $SECURITY_NAMESPACE"
    echo "KubeArmor Logs: kubectl logs -f daemonset/kubearmor -n $SECURITY_NAMESPACE"
    
    echo ""
    echo -e "${GREEN}=== Demo Commands ===${NC}"
    echo "Run adversarial attack: ./scripts/run-attack.sh adversarial"
    echo "Run model extraction: ./scripts/run-attack.sh extraction"
    echo "Run data poisoning: ./scripts/run-attack.sh poisoning"
    echo "Run IBM ART attacks: ./scripts/run-attack.sh art"
    echo "Run all attacks: ./scripts/run-attack.sh all"
    echo "Scan demo images: ./scripts/scan-images.sh"
    echo "Test LLM security: ./scripts/run-garak-tests.sh comprehensive"
    echo "View security events: kubectl get events -n $DEMO_NAMESPACE --sort-by='.lastTimestamp'"
    
    echo ""
    echo -e "${BLUE}=== MITRE ATLAS Techniques Demonstrated ===${NC}"
    echo "T1551 - Evade ML Model (Adversarial Examples + IBM ART Enhanced)"
    echo "T1552 - Steal ML Model (Model Extraction)"
    echo "T1565 - Data Manipulation (Data Poisoning)"
    echo "T1574 - Hijack Execution Flow"
    echo "T1036 - Masquerading"
    echo ""
    echo -e "${CYAN}=== IBM ART Framework Integration ===${NC}"
    echo "✓ Advanced adversarial attack methods (Carlini & Wagner, DeepFool, Boundary)"
    echo "✓ Real-time adversarial detection capabilities"
    echo "✓ Multi-layer defensive preprocessing"
    echo "✓ Enterprise-grade attack sophistication"
    echo ""
    echo -e "${CYAN}=== Clair Vulnerability Scanning ===${NC}"
    echo "✓ Container image vulnerability assessment"
    echo "✓ CVE database scanning and risk analysis"
    echo "✓ ML framework security validation"
    echo "✓ Supply chain security monitoring"
    echo ""
    echo -e "${CYAN}=== Garak LLM Security Testing ===${NC}"
    echo "✓ Comprehensive LLM vulnerability scanning"
    echo "✓ Prompt injection and jailbreaking detection"
    echo "✓ Data leakage and PII exposure testing"
    echo "✓ Bias and toxicity assessment"
    
    echo ""
    echo -e "${YELLOW}Ready for KubeCon demo! 🎉${NC}"
}

# Cleanup function
cleanup() {
    if [[ "${1:-}" == "--cleanup" ]]; then
        log_step "Cleaning up demo environment..."
        
        kubectl delete namespace "$DEMO_NAMESPACE" --ignore-not-found=true
        helm uninstall falco -n "$SECURITY_NAMESPACE" || true
        helm uninstall kubearmor -n "$SECURITY_NAMESPACE" || true
        helm uninstall prometheus -n "$MONITORING_NAMESPACE" || true
        
        # Clean up Clair components
        kubectl delete -f "$PROJECT_ROOT/kubernetes/security/clair-deployment.yaml" --ignore-not-found=true || true
        kubectl delete -f "$PROJECT_ROOT/kubernetes/security/vuln-scanner-deployment.yaml" --ignore-not-found=true || true
        
        # Clean up Garak components
        kubectl delete -f "$PROJECT_ROOT/kubernetes/security/garak-scanner-deployment.yaml" --ignore-not-found=true || true
        kubectl delete -f "$PROJECT_ROOT/kubernetes/base/llm-service-deployment.yaml" --ignore-not-found=true || true
        
        kubectl delete namespace "$SECURITY_NAMESPACE" --ignore-not-found=true
        kubectl delete namespace "$MONITORING_NAMESPACE" --ignore-not-found=true
        
        # Clean up temporary files
        rm -f /tmp/falco-values.yaml /tmp/attack-tools-pod.yaml /tmp/mitre-atlas-dashboard.json
        
        log_info "Cleanup completed ✓"
        exit 0
    fi
}

# Main execution
main() {
    print_banner
    
    # Check for cleanup flag
    cleanup "$@"
    
    # Execute setup steps
    check_prerequisites
    create_namespaces
    build_images
    setup_helm_repos
    install_falco
    install_kubearmor
    install_clair
    install_monitoring
    deploy_ai_application
    apply_security_policies
    create_sample_data
    setup_attack_tools
    setup_dashboards
    show_access_info
}

# Handle script arguments
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--cleanup] [--help]"
    echo ""
    echo "Options:"
    echo "  --cleanup    Remove all demo components"
    echo "  --help       Show this help message"
    exit 0
fi

# Run main function with all arguments
main "$@" 