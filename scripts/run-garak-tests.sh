#!/bin/bash

# KubeCon 2025 Demo - Garak LLM Security Testing Script
# Uses Garak to demonstrate LLM vulnerability scanning capabilities

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
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

log_attack() {
    echo -e "${PURPLE}🎯 $1${NC}"
}

log_detect() {
    echo -e "${YELLOW}🚨 $1${NC}"
}

# Show banner
show_banner() {
    echo ""
    echo -e "${PURPLE}============================================================${NC}"
    echo -e "${PURPLE}🎯 LLM SECURITY TESTING WITH GARAK DEMONSTRATION 🎯${NC}"
    echo -e "${PURPLE}============================================================${NC}"
    echo -e "${YELLOW}🚨 Testing for Prompt Injection, Jailbreaking & More 🚨${NC}"
    echo -e "${PURPLE}============================================================${NC}"
    echo ""
}

# Check environment
check_environment() {
    log_step "Checking Garak demo environment..."
    
    # Check if LLM service is running
    LLM_POD=$(kubectl get pods -n "$DEMO_NAMESPACE" -l app=llm-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$LLM_POD" ]]; then
        log_error "LLM service is not running!"
        log_info "Please run './scripts/setup-demo.sh' first to deploy the LLM service"
        exit 1
    fi
    
    # Check if Garak scanner is running
    GARAK_POD=$(kubectl get pods -n "$SECURITY_NAMESPACE" -l app=garak-scanner-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -z "$GARAK_POD" ]]; then
        log_error "Garak scanner service is not running!"
        log_info "Please run './scripts/setup-demo.sh' first to deploy Garak scanner"
        exit 1
    fi
    
    log_success "Environment check passed ✓"
    log_info "LLM Service Pod: $LLM_POD"
    log_info "Garak Scanner Pod: $GARAK_POD"
    echo ""
}

# Get service endpoints
get_llm_endpoint() {
    LLM_SERVICE_IP=$(kubectl get svc llm-service -n "$DEMO_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    LLM_ENDPOINT="http://$LLM_SERVICE_IP"
    echo "$LLM_ENDPOINT"
}

get_garak_endpoint() {
    GARAK_SERVICE_IP=$(kubectl get svc garak-scanner-service -n "$SECURITY_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    GARAK_ENDPOINT="http://$GARAK_SERVICE_IP"
    echo "$GARAK_ENDPOINT"
}

# Test LLM service health
test_llm_health() {
    log_step "Testing LLM service health..."
    
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

llm_endpoint = '$llm_endpoint'

try:
    # Test health endpoint
    response = requests.get(f'{llm_endpoint}/health', timeout=10)
    
    if response.status_code == 200:
        health_data = response.json()
        print('=== LLM Service Health ===')
        print(f'Status: {health_data.get(\"status\", \"unknown\")}')
        print(f'Model: {health_data.get(\"model\", \"unknown\")}')
        print(f'Service: {health_data.get(\"service\", \"unknown\")}')
        
        vulnerabilities = health_data.get('vulnerabilities_enabled', {})
        print(f'\\nVulnerabilities Enabled for Demo:')
        for vuln, enabled in vulnerabilities.items():
            status = '✓' if enabled else '✗'
            print(f'  {status} {vuln.replace(\"_\", \" \").title()}')
        
        print(f'\\n✅ LLM service is healthy and ready for testing!')
    else:
        print(f'❌ LLM service health check failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Failed to connect to LLM service: {e}')
"
    
    echo ""
}

# Run quick vulnerability test
run_quick_test() {
    log_attack "Running Quick LLM Vulnerability Test..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'
llm_endpoint = '$llm_endpoint'

try:
    # Run quick vulnerability test
    response = requests.post(
        f'{garak_endpoint}/scan/quick',
        json={'target_endpoint': llm_endpoint},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print('=== Quick LLM Vulnerability Test Results ===')
        summary = result.get('summary', {})
        print(f'Total Tests: {summary.get(\"total_tests\", 0)}')
        print(f'Vulnerabilities Found: {summary.get(\"vulnerabilities_found\", 0)}')
        print(f'Vulnerability Rate: {summary.get(\"vulnerability_rate\", 0):.1f}%')
        print(f'Overall Risk: {summary.get(\"overall_risk\", \"Unknown\")}')
        
        print(f'\\n=== Individual Test Results ===')
        for test in result.get('quick_test_results', []):
            vuln_status = '🚨 VULNERABLE' if test.get('vulnerability_detected') else '✅ SAFE'
            risk = test.get('risk_level', 'Unknown')
            print(f'\\nPrompt: {test[\"prompt\"][:60]}...')
            print(f'Status: {vuln_status} (Risk: {risk})')
            
            if test.get('vulnerability_detected'):
                vulns = test.get('vulnerabilities_triggered', [])
                print(f'Triggered: {', '.join(vulns)}')
                response_preview = test.get('model_response', '')[:100]
                print(f'Response: {response_preview}...')
    else:
        print(f'❌ Quick test failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Quick test error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: Multiple vulnerability types should be identified"
    echo ""
}

# Run prompt injection test
run_prompt_injection_test() {
    log_attack "Executing Prompt Injection Security Test..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'
llm_endpoint = '$llm_endpoint'

try:
    # Run prompt injection scan
    response = requests.post(
        f'{garak_endpoint}/scan/start',
        json={
            'scan_type': 'prompt_injection',
            'target_endpoint': llm_endpoint,
            'model_name': 'vulnerable-demo-llm'
        },
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        scan_results = result.get('results', {})
        
        print('=== Prompt Injection Test Results ===')
        print(f'Scan ID: {result.get(\"scan_id\", \"unknown\")}')
        print(f'Status: {scan_results.get(\"status\", \"unknown\")}')
        print(f'Target: {scan_results.get(\"target_endpoint\", \"unknown\")}')
        print(f'Vulnerabilities Found: {scan_results.get(\"vulnerabilities_found\", 0)}')
        
        critical = scan_results.get('critical_issues', 0)
        high = scan_results.get('high_issues', 0)
        medium = scan_results.get('medium_issues', 0)
        
        print(f'\\nSeverity Breakdown:')
        if critical > 0:
            print(f'  🔴 Critical: {critical}')
        if high > 0:
            print(f'  🟠 High: {high}')
        if medium > 0:
            print(f'  🟡 Medium: {medium}')
        
        findings = scan_results.get('findings', [])
        if findings:
            print(f'\\n=== Key Findings ===')
            for finding in findings[:3]:  # Show top 3
                severity = finding.get('severity', 'Unknown')
                description = finding.get('description', 'No description')
                risk_score = finding.get('risk_score', 0)
                
                print(f'\\n[{severity}] Risk Score: {risk_score}/10')
                print(f'Description: {description}')
                
                if 'example_prompt' in finding:
                    print(f'Test Prompt: {finding[\"example_prompt\"][:80]}...')
                if 'model_response' in finding:
                    print(f'Vulnerable Response: {finding[\"model_response\"][:80]}...')
    else:
        print(f'❌ Prompt injection test failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Prompt injection test error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: Critical prompt injection vulnerabilities"
    echo ""
}

# Run jailbreak test
run_jailbreak_test() {
    log_attack "Executing Jailbreak Resistance Test..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'
llm_endpoint = '$llm_endpoint'

try:
    # Run jailbreak scan
    response = requests.post(
        f'{garak_endpoint}/scan/start',
        json={
            'scan_type': 'jailbreak',
            'target_endpoint': llm_endpoint,
            'model_name': 'vulnerable-demo-llm'
        },
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        scan_results = result.get('results', {})
        
        print('=== Jailbreak Resistance Test Results ===')
        print(f'Vulnerabilities Found: {scan_results.get(\"vulnerabilities_found\", 0)}')
        print(f'Critical Issues: {scan_results.get(\"critical_issues\", 0)}')
        print(f'High Issues: {scan_results.get(\"high_issues\", 0)}')
        
        findings = scan_results.get('findings', [])
        if findings:
            print(f'\\n=== Jailbreak Vulnerabilities ===')
            for finding in findings:
                probe = finding.get('probe', 'Unknown')
                severity = finding.get('severity', 'Unknown')
                description = finding.get('description', 'No description')
                risk_score = finding.get('risk_score', 0)
                
                print(f'\\n[{severity}] {probe}')
                print(f'Risk Score: {risk_score}/10')
                print(f'Issue: {description}')
                
                if 'example_prompt' in finding:
                    prompt_preview = finding['example_prompt'][:60]
                    print(f'Attack Prompt: {prompt_preview}...')
                if 'model_response' in finding:
                    response_preview = finding['model_response'][:80]
                    print(f'Vulnerable Response: {response_preview}...')
    else:
        print(f'❌ Jailbreak test failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Jailbreak test error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: Jailbreak vulnerabilities and safety bypasses"
    echo ""
}

# Run data leakage test
run_data_leakage_test() {
    log_attack "Executing Data Leakage Security Test..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'
llm_endpoint = '$llm_endpoint'

try:
    # Run data leakage scan
    response = requests.post(
        f'{garak_endpoint}/scan/start',
        json={
            'scan_type': 'data_leakage',
            'target_endpoint': llm_endpoint,
            'model_name': 'vulnerable-demo-llm'
        },
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        scan_results = result.get('results', {})
        
        print('=== Data Leakage Test Results ===')
        print(f'Vulnerabilities Found: {scan_results.get(\"vulnerabilities_found\", 0)}')
        print(f'Critical Issues: {scan_results.get(\"critical_issues\", 0)}')
        print(f'High Issues: {scan_results.get(\"high_issues\", 0)}')
        
        findings = scan_results.get('findings', [])
        if findings:
            print(f'\\n=== Data Leakage Vulnerabilities ===')
            for finding in findings:
                probe = finding.get('probe', 'Unknown')
                severity = finding.get('severity', 'Unknown')
                description = finding.get('description', 'No description')
                risk_score = finding.get('risk_score', 0)
                
                print(f'\\n[{severity}] {probe}')
                print(f'Risk Score: {risk_score}/10')
                print(f'Issue: {description}')
                
                # Show specific leaked data
                if 'detected_keys' in finding:
                    keys = finding['detected_keys']
                    print(f'Leaked API Keys: {len(keys)} detected')
                    for key in keys[:2]:  # Show first 2
                        print(f'  - {key[:15]}...')
                
                if 'leaked_info' in finding:
                    print(f'Leaked Information: {finding[\"leaked_info\"]}')
                
                if 'pii_types' in finding:
                    pii_types = finding['pii_types']
                    print(f'PII Types Exposed: {', '.join(pii_types)}')
    else:
        print(f'❌ Data leakage test failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Data leakage test error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: API keys, system prompts, and PII exposure"
    echo ""
}

# Run comprehensive scan
run_comprehensive_scan() {
    log_attack "Executing Comprehensive LLM Security Assessment..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    local llm_endpoint=$(get_llm_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'

try:
    # Run comprehensive demo scan
    response = requests.post(
        f'{garak_endpoint}/scan/demo',
        json={'target_endpoint': '$llm_endpoint'},
        timeout=180
    )
    
    if response.status_code == 200:
        result = response.json()
        demo_results = result.get('demo_scan_results', {})
        summary = result.get('summary', {})
        
        print('=== Comprehensive LLM Security Assessment ===')
        print(f'Target: {summary.get(\"target_endpoint\", \"unknown\")}')
        print(f'Total Vulnerabilities: {summary.get(\"total_vulnerabilities\", 0)}')
        print(f'Critical Issues: {summary.get(\"critical_issues\", 0)}')
        print(f'High Issues: {summary.get(\"high_issues\", 0)}')
        print(f'Medium Issues: {summary.get(\"medium_issues\", 0)}')
        print(f'Overall Risk Score: {summary.get(\"overall_risk_score\", 0)}/10')
        print(f'Security Grade: {summary.get(\"security_grade\", \"Unknown\")}')
        
        # Show vulnerability categories
        vuln_categories = demo_results.get('vulnerability_categories', {})
        if vuln_categories:
            print(f'\\n=== Vulnerability Categories ===')
            for category, count in vuln_categories.items():
                if count > 0:
                    print(f'  {category.replace(\"_\", \" \").title()}: {count}')
        
        # Show recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f'\\n=== Security Recommendations ===')
            for i, rec in enumerate(recommendations, 1):
                print(f'  {i}. {rec}')
        
        print(f'\\n🎯 Scan completed in {demo_results.get(\"scan_duration\", 0):.1f} seconds')
        print(f'📊 Garak Version: {demo_results.get(\"garak_version\", \"unknown\")}')
    else:
        print(f'❌ Comprehensive scan failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Comprehensive scan error: {e}')
"
    
    echo ""
    log_detect "Expected Detection: Complete security assessment with actionable recommendations"
    echo ""
}

# Show scan history
show_scan_history() {
    log_step "Showing Garak scan history..."
    echo ""
    
    local garak_endpoint=$(get_garak_endpoint)
    
    kubectl exec -n "$SECURITY_NAMESPACE" "$GARAK_POD" -- python3 -c "
import requests
import json

garak_endpoint = '$garak_endpoint'

try:
    # Get scan history
    response = requests.get(f'{garak_endpoint}/scan/history?limit=10', timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print('=== Garak Scan History ===')
        history = result.get('scan_history', [])
        total_scans = result.get('total_scans', 0)
        
        print(f'Total Scans Performed: {total_scans}')
        print(f'\\nRecent Scans:')
        
        for scan in history[-5:]:  # Show last 5
            scan_id = scan.get('scan_id', 'unknown')
            scan_type = scan.get('scan_type', 'unknown')
            status = scan.get('status', 'unknown')
            timestamp = scan.get('timestamp', 'unknown')
            
            print(f'  {timestamp} - {scan_type} ({scan_id}) - {status}')
    else:
        print(f'Failed to get scan history: {response.status_code}')

except Exception as e:
    print(f'Error getting scan history: {e}')
"
    
    echo ""
}

# Show help
show_help() {
    echo "Usage: $0 <test_type> [options]"
    echo ""
    echo "Test Types:"
    echo "  health           - Check LLM service health and vulnerability status"
    echo "  quick            - Quick vulnerability test with sample prompts"
    echo "  prompt-injection - Comprehensive prompt injection testing"
    echo "  jailbreak        - Jailbreak resistance testing"
    echo "  data-leakage     - Data leakage and information disclosure testing"
    echo "  comprehensive    - Complete security assessment"
    echo "  history          - Show recent scan history"
    echo "  all              - Run all test scenarios sequentially"
    echo ""
    echo "Options:"
    echo "  --help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick                       # Quick vulnerability test"
    echo "  $0 prompt-injection            # Test for prompt injection"
    echo "  $0 comprehensive               # Full security assessment"
    echo "  $0 all                         # Run all tests"
}

# Main function
main() {
    if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    show_banner
    check_environment
    
    local test_type="$1"
    
    case "$test_type" in
        "health")
            test_llm_health
            ;;
        "quick")
            run_quick_test
            ;;
        "prompt-injection")
            run_prompt_injection_test
            ;;
        "jailbreak")
            run_jailbreak_test
            ;;
        "data-leakage")
            run_data_leakage_test
            ;;
        "comprehensive")
            run_comprehensive_scan
            ;;
        "history")
            show_scan_history
            ;;
        "all")
            log_info "Running complete LLM security test suite..."
            echo ""
            test_llm_health
            echo "=========================================="
            echo ""
            run_quick_test
            echo "=========================================="
            echo ""
            run_prompt_injection_test
            echo "=========================================="
            echo ""
            run_jailbreak_test
            echo "=========================================="
            echo ""
            run_data_leakage_test
            echo "=========================================="
            echo ""
            run_comprehensive_scan
            echo "=========================================="
            echo ""
            show_scan_history
            ;;
        *)
            log_error "Unknown test type: $test_type"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Garak LLM security testing completed!"
    echo -e "${CYAN}Check Grafana dashboard for detailed security metrics.${NC}"
}

# Execute main function
main "$@" 