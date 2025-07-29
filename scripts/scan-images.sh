#!/bin/bash

# KubeCon 2025 Demo - Vulnerability Scanning Script
# Uses Clair to demonstrate container security scanning capabilities

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEMO_NAMESPACE=${DEMO_NAMESPACE:-"ai-demo"}
SECURITY_NAMESPACE=${SECURITY_NAMESPACE:-"security"}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Logging functions
log_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Show banner
show_banner() {
    echo ""
    echo -e "${CYAN}======================================================${NC}"
    echo -e "${CYAN}🔍 CONTAINER VULNERABILITY SCANNING DEMONSTRATION 🔍${NC}"
    echo -e "${CYAN}======================================================${NC}"
    echo -e "${YELLOW}🚨 Using Clair for Enterprise Container Security 🚨${NC}"
    echo -e "${CYAN}======================================================${NC}"
    echo ""
}

# Check environment
check_environment() {
    log_step "Checking demo environment..."
    
    # Check if vulnerability scanner is running
    SCANNER_POD=$(kubectl get pods -n "$SECURITY_NAMESPACE" -l app=vuln-scanner-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$SCANNER_POD" ]]; then
        log_error "Vulnerability scanner service is not running!"
        log_info "Please run './scripts/setup-demo.sh' first to deploy the scanner"
        exit 1
    fi
    
    # Check if Clair is running
    CLAIR_INDEXER_POD=$(kubectl get pods -n "$SECURITY_NAMESPACE" -l app=clair,component=indexer --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    CLAIR_MATCHER_POD=$(kubectl get pods -n "$SECURITY_NAMESPACE" -l app=clair,component=matcher --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$CLAIR_INDEXER_POD" ]] || [[ -z "$CLAIR_MATCHER_POD" ]]; then
        log_error "Clair services are not running!"
        log_info "Please run './scripts/setup-demo.sh' first to deploy Clair"
        exit 1
    fi
    
    log_success "Environment check passed ✓"
    log_info "Scanner Pod: $SCANNER_POD"
    log_info "Clair Indexer: $CLAIR_INDEXER_POD"
    log_info "Clair Matcher: $CLAIR_MATCHER_POD"
    echo ""
}

# Get scanner service endpoint
get_scanner_endpoint() {
    SCANNER_SERVICE_IP=$(kubectl get svc vuln-scanner-service -n "$SECURITY_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    SCANNER_ENDPOINT="http://$SCANNER_SERVICE_IP"
    echo "$SCANNER_ENDPOINT"
}

# Scan a single image
scan_single_image() {
    local image="$1"
    local tag="${2:-latest}"
    local scanner_endpoint="$3"
    
    log_info "Scanning image: $image:$tag"
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$SCANNER_POD" -- python3 -c "
import requests
import json
import sys

scanner_endpoint = '$scanner_endpoint'
image = '$image'
tag = '$tag'

try:
    # Initiate scan
    payload = {'image': image, 'tag': tag}
    response = requests.post(f'{scanner_endpoint}/scan', json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        # Display results
        print(f'\\n=== Scan Results for {image}:{tag} ===')
        print(f'Status: {result.get(\"status\", \"unknown\")}')
        print(f'Scan Duration: {result.get(\"scan_duration\", 0):.2f}s')
        
        if result.get('status') == 'completed':
            summary = result.get('summary', {})
            print(f'\\nVulnerability Summary:')
            print(f'  Total Vulnerabilities: {summary.get(\"total_vulnerabilities\", 0)}')
            print(f'  Risk Level: {summary.get(\"risk_level\", \"Unknown\")}')
            print(f'  Risk Score: {summary.get(\"risk_score\", 0)}')
            
            severity_counts = summary.get('severity_counts', {})
            print(f'\\nSeverity Breakdown:')
            for severity in ['Critical', 'High', 'Medium', 'Low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f'  {severity}: {count}')
            
            # Show top 5 vulnerabilities
            vulnerabilities = result.get('vulnerabilities', [])[:5]
            if vulnerabilities:
                print(f'\\nTop Vulnerabilities:')
                for vuln in vulnerabilities:
                    print(f'  [{vuln[\"severity\"]}] {vuln[\"id\"]} - {vuln[\"package\"]}')
                    print(f'    {vuln[\"description\"][:80]}...')
        
        elif result.get('status') == 'error':
            print(f'Scan failed: {result.get(\"error\", \"Unknown error\")}')
    else:
        print(f'Scan request failed: {response.status_code} - {response.text}')

except Exception as e:
    print(f'Error during scan: {e}')
    sys.exit(1)
"
}

# Run demo vulnerability scanning
run_demo_scan() {
    log_step "Running comprehensive vulnerability scan on demo images..."
    echo ""
    
    local scanner_endpoint=$(get_scanner_endpoint)
    
    log_info "Scanner Endpoint: $scanner_endpoint"
    echo ""
    
    # Demo images to scan
    demo_images=(
        "image-classifier"
        "art-defense-service"
        "vuln-scanner-service"
        "python:3.10-slim"
        "postgres:13-alpine"
    )
    
    log_info "Scanning ${#demo_images[@]} demo images..."
    echo ""
    
    for image in "${demo_images[@]}"; do
        scan_single_image "$image" "latest" "$scanner_endpoint"
        echo ""
        log_info "Press Enter to continue to next image scan..."
        read -r
    done
    
    log_success "Individual image scans completed!"
}

# Run batch demo scan
run_batch_demo_scan() {
    log_step "Running batch scan on all demo images..."
    echo ""
    
    local scanner_endpoint=$(get_scanner_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$SCANNER_POD" -- python3 -c "
import requests
import json

scanner_endpoint = '$scanner_endpoint'

try:
    # Run demo batch scan
    response = requests.post(f'{scanner_endpoint}/scan/demo', timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        
        print('\\n=== Batch Demo Scan Results ===')
        summary = result.get('summary', {})
        print(f'Total Images Scanned: {summary.get(\"total_images_scanned\", 0)}')
        print(f'Total Vulnerabilities: {summary.get(\"total_vulnerabilities_found\", 0)}')
        print(f'Images with Critical Vulnerabilities: {summary.get(\"images_with_critical_vulns\", 0)}')
        
        critical_images = summary.get('critical_images', [])
        if critical_images:
            print(f'\\nCritical Images:')
            for img in critical_images:
                print(f'  - {img}')
        
        # Show scan results per image
        scan_results = result.get('demo_scan_results', {})
        print(f'\\n=== Per-Image Results ===')
        for image, scan_result in scan_results.items():
            if scan_result.get('status') == 'completed':
                summary = scan_result.get('summary', {})
                print(f'\\n{image}:')
                print(f'  Risk Level: {summary.get(\"risk_level\", \"Unknown\")}')
                print(f'  Total Vulnerabilities: {summary.get(\"total_vulnerabilities\", 0)}')
                
                severity_counts = summary.get('severity_counts', {})
                critical = severity_counts.get('Critical', 0)
                high = severity_counts.get('High', 0)
                medium = severity_counts.get('Medium', 0)
                low = severity_counts.get('Low', 0)
                
                if critical > 0:
                    print(f'  Critical: {critical}')
                if high > 0:
                    print(f'  High: {high}')
                if medium > 0:
                    print(f'  Medium: {medium}')
                if low > 0:
                    print(f'  Low: {low}')
            else:
                print(f'\\n{image}: {scan_result.get(\"status\", \"unknown\")}')
    else:
        print(f'Batch scan failed: {response.status_code} - {response.text}')

except Exception as e:
    print(f'Error during batch scan: {e}')
"
    
    echo ""
    log_success "Batch demo scan completed!"
}

# Show scan history
show_scan_history() {
    log_step "Showing vulnerability scan history..."
    echo ""
    
    local scanner_endpoint=$(get_scanner_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$SCANNER_POD" -- python3 -c "
import requests
import json

scanner_endpoint = '$scanner_endpoint'

try:
    # Get scan history
    response = requests.get(f'{scanner_endpoint}/scan/history?limit=10', timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print('=== Recent Scan History ===')
        history = result.get('history', [])
        total_scans = result.get('total_scans', 0)
        
        print(f'Total Scans Performed: {total_scans}')
        print(f'\\nRecent Scans:')
        
        for scan in history[-10:]:  # Show last 10
            print(f'  {scan[\"timestamp\"]} - {scan[\"image\"]} ({scan[\"status\"]}) - {scan[\"vulnerabilities\"]} vulns')
    else:
        print(f'Failed to get scan history: {response.status_code}')

except Exception as e:
    print(f'Error getting scan history: {e}')
"
    
    echo ""
}

# Show scanner stats
show_scanner_stats() {
    log_step "Showing vulnerability scanner statistics..."
    echo ""
    
    local scanner_endpoint=$(get_scanner_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$SCANNER_POD" -- python3 -c "
import requests
import json

scanner_endpoint = '$scanner_endpoint'

try:
    # Get scanner stats
    response = requests.get(f'{scanner_endpoint}/scan/stats', timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print('=== Vulnerability Scanner Statistics ===')
        print(f'Total Scans: {result.get(\"total_scans\", 0)}')
        print(f'Cached Results: {result.get(\"cached_results\", 0)}')
        print(f'Service Uptime: {result.get(\"service_uptime\", 0):.0f} seconds')
        
        endpoints = result.get('clair_endpoints', {})
        print(f'\\nClair Endpoints:')
        print(f'  Indexer: {endpoints.get(\"indexer\", \"unknown\")}')
        print(f'  Matcher: {endpoints.get(\"matcher\", \"unknown\")}')
    else:
        print(f'Failed to get scanner stats: {response.status_code}')

except Exception as e:
    print(f'Error getting scanner stats: {e}')
"
    
    echo ""
}

# Show help
show_help() {
    echo "Usage: $0 <scan_type> [options]"
    echo ""
    echo "Scan Types:"
    echo "  demo         - Interactive demo scan of individual images"
    echo "  batch        - Batch scan of all demo images"
    echo "  history      - Show recent scan history"
    echo "  stats        - Show scanner statistics"
    echo "  health       - Check scanner service health"
    echo ""
    echo "Options:"
    echo "  --help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 demo                        # Interactive demo scan"
    echo "  $0 batch                       # Batch scan all demo images"
    echo "  $0 history                     # Show scan history"
    echo "  $0 stats                       # Show scanner statistics"
}

# Check scanner health
check_scanner_health() {
    log_step "Checking vulnerability scanner health..."
    echo ""
    
    local scanner_endpoint=$(get_scanner_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$SCANNER_POD" -- python3 -c "
import requests
import json

scanner_endpoint = '$scanner_endpoint'

try:
    # Check health
    response = requests.get(f'{scanner_endpoint}/health', timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print('=== Vulnerability Scanner Health ===')
        print(f'Status: {result.get(\"status\", \"unknown\")}')
        print(f'Service: {result.get(\"service\", \"unknown\")}')
        print(f'Clair Indexer: {result.get(\"clair_indexer\", \"unknown\")}')
        print(f'Clair Matcher: {result.get(\"clair_matcher\", \"unknown\")}')
        print(f'Timestamp: {result.get(\"timestamp\", \"unknown\")}')
        
        if result.get('status') == 'healthy':
            print('\\n✅ All services are healthy!')
        else:
            print('\\n❌ Some services are unhealthy!')
    else:
        print(f'Health check failed: {response.status_code}')

except Exception as e:
    print(f'Error checking health: {e}')
"
    
    echo ""
}

# Main function
main() {
    if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    show_banner
    check_environment
    
    local scan_type="$1"
    
    case "$scan_type" in
        "demo")
            run_demo_scan
            ;;
        "batch")
            run_batch_demo_scan
            ;;
        "history")
            show_scan_history
            ;;
        "stats")
            show_scanner_stats
            ;;
        "health")
            check_scanner_health
            ;;
        *)
            log_error "Unknown scan type: $scan_type"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Vulnerability scanning demonstration completed!"
    echo -e "${CYAN}Check Grafana dashboard for detailed security metrics.${NC}"
}

# Execute main function
main "$@" 