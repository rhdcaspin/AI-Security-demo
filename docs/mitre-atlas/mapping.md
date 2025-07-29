# MITRE ATLAS Mapping for KubeCon 2025 Demo

This document maps the attack scenarios in our demo to specific MITRE ATLAS tactics and techniques, providing a comprehensive understanding of how enterprise AI security aligns with the ATLAS framework.

## Overview

The MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) framework is a comprehensive knowledge base for understanding adversarial threats against AI/ML systems. Our demo demonstrates real-world attack scenarios mapped to specific ATLAS techniques.

## Demo Attack Scenarios → MITRE ATLAS Mapping

### 1. Adversarial Examples Attack

**MITRE ATLAS Technique**: T1551 - Evade ML Model

#### Sub-techniques Demonstrated:

##### T1551.004 - Adversarial Examples
- **Description**: Crafting inputs that cause ML models to make incorrect predictions
- **Demo Implementation**: 
  - Fast Gradient Sign Method (FGSM) attack
  - Projected Gradient Descent (PGD) attack
  - One-pixel/Few-pixel attacks
  - Semantic adversarial transformations

**Attack Flow**:
```
1. Reconnaissance → Query model endpoints for information
2. Input Crafting → Generate adversarial perturbations
3. Evasion → Submit adversarial examples to fool classifier
4. Impact → Demonstrate misclassification results
```

**Detections in Demo**:
- Falco detects high-frequency model queries
- KubeArmor monitors suspicious process execution
- Prometheus tracks query patterns and anomalies

---

### 2. Model Extraction Attack

**MITRE ATLAS Technique**: T1552 - Steal ML Model

#### Sub-techniques Demonstrated:

##### T1552.001 - Model Architecture Extraction
- **Description**: Extracting model architecture through information disclosure
- **Demo Implementation**: Querying `/model/info` endpoint to gather:
  - Model type and architecture details
  - Number of parameters and classes
  - Model size and metadata

##### T1552.002 - Parameter Extraction  
- **Description**: Extracting model weights and parameters
- **Demo Implementation**: Accessing `/model/weights` endpoint to extract:
  - Layer-specific weight matrices
  - Bias vectors and activation parameters
  - Model state dictionaries

##### T1552.003 - Model Inversion
- **Description**: Reconstructing training data from model outputs
- **Demo Implementation**: 
  - Systematic querying with synthetic inputs
  - Decision boundary analysis
  - Training data reconstruction attempts

##### T1552.004 - Model Stealing via API
- **Description**: Recreating model functionality through API abuse
- **Demo Implementation**:
  - High-volume automated queries
  - Input-output pair collection
  - Substitute model training

**Attack Flow**:
```
1. Reconnaissance → Information gathering from exposed endpoints
2. Architecture Extraction → Download model structure details
3. Parameter Extraction → Extract weights and biases  
4. Behavioral Analysis → Query model with crafted inputs
5. Model Reconstruction → Build substitute model
```

**Detections in Demo**:
- Falco alerts on model information disclosure
- KubeArmor detects unauthorized file access
- Rate limiting triggers on excessive API usage

---

### 3. Data Poisoning Attack

**MITRE ATLAS Technique**: T1565 - Data Manipulation

#### Sub-techniques Demonstrated:

##### T1565.001 - Stored Data Manipulation
- **Description**: Modifying stored training data
- **Demo Implementation**:
  - Uploading mislabeled training samples
  - Injecting backdoor triggers in datasets
  - Corrupting validation data

##### T1565.002 - Runtime Data Manipulation
- **Description**: Manipulating data during model training/inference
- **Demo Implementation**:
  - Real-time feedback data poisoning
  - Malicious model replacement attempts
  - Training pipeline contamination

##### T1565.003 - Compromised Data Sources
- **Description**: Using compromised external data sources
- **Demo Implementation**:
  - Uploading malicious files with path traversal
  - Feedback system abuse
  - External dataset poisoning simulation

**Attack Flow**:
```
1. Initial Access → Gain access to training pipeline
2. Data Injection → Upload poisoned training samples
3. Model Poisoning → Replace or corrupt model files
4. Persistence → Maintain poisoned state
5. Impact → Demonstrate compromised predictions
```

**Detections in Demo**:
- KubeArmor file upload monitoring
- Falco detects suspicious file operations
- Integrity checking on model files

---

## Complete MITRE ATLAS Tactics Coverage

### Reconnaissance (T1595)
**Demonstrated in**: All attack scenarios
- **T1595.001** - Active Scanning: Probing AI service endpoints
- **T1595.002** - Vulnerability Scanning: Testing for exposed APIs

### Resource Development (T1587)
**Demonstrated in**: Model extraction attack
- **T1587.001** - Develop Capabilities: Creating extraction tools
- **T1587.002** - Acquire Infrastructure: Setting up attack platform

### Initial Access (T1199)
**Demonstrated in**: All attack scenarios  
- **T1199.001** - Trusted Relationship: Using legitimate API access
- **T1199.002** - Supply Chain Compromise: Simulated in data poisoning

### ML Model Access (T1552)
**Primary focus**: Model extraction scenario
- Full coverage as described above

### Execution (T1569)
**Demonstrated in**: All scenarios
- **T1569.001** - System Services: Using container execution
- **T1569.002** - Service Execution: Python script execution

### Persistence (T1546)
**Demonstrated in**: Data poisoning attack
- **T1546.001** - Model Backdoors: Persistent malicious model state
- **T1546.002** - Scheduled Tasks: Automated poisoning scripts

### Defense Evasion (T1551)
**Primary focus**: Adversarial examples scenario
- Full coverage as described above

### Collection (T1005)
**Demonstrated in**: Model extraction attack
- **T1005.001** - Data from Local System: Extracting model data
- **T1005.002** - Automated Collection: Scripted data gathering

### Impact (T1485)
**Demonstrated in**: All scenarios
- **T1485.001** - Model Corruption: Poisoning attack results
- **T1485.002** - Service Degradation: DoS simulation effects

---

## Security Controls Alignment

### Detection Controls

| MITRE ATLAS Technique | Detection Method | Tool Used | Alert Type |
|----------------------|------------------|-----------|------------|
| T1551 (Evasion) | Query pattern analysis | Falco | High-frequency requests |
| T1552 (Model Theft) | File access monitoring | KubeArmor | Unauthorized model access |
| T1565 (Data Poisoning) | File upload inspection | Both | Malicious file uploads |
| T1595 (Reconnaissance) | API monitoring | Prometheus | Abnormal endpoint access |

### Prevention Controls

| MITRE ATLAS Technique | Prevention Method | Implementation |
|----------------------|-------------------|----------------|
| T1551 | Input validation | Adversarial training |
| T1552 | Access controls | API rate limiting |
| T1565 | Data integrity | File validation |
| T1574 | Execution controls | Process restrictions |

### Response Controls

| Alert Level | Response Action | Automation |
|-------------|----------------|------------|
| CRITICAL | Block traffic | Automatic |
| HIGH | Log and alert | Semi-automatic |
| WARNING | Monitor | Manual review |
| INFO | Record metrics | Automatic |

---

## Metrics and KPIs

### Security Effectiveness Metrics

1. **Detection Rate**: Percentage of attacks detected
2. **False Positive Rate**: Legitimate activities flagged as suspicious  
3. **Response Time**: Time from attack to detection/response
4. **Coverage**: Percentage of ATLAS techniques covered

### Demo Success Metrics

1. **Attack Success Rate**: Attacks that achieved objectives
2. **Detection Accuracy**: Correct identification of techniques
3. **Alert Quality**: Relevance and actionability of alerts
4. **Mitigation Effectiveness**: Prevention of attack progression

---

## Integration with Enterprise Security

### SIEM Integration

```yaml
# Example SIEM rule for MITRE ATLAS T1552
rule_name: "ATLAS_T1552_Model_Extraction"
detection_logic:
  - High-frequency API queries to /model/* endpoints
  - Unusual data exfiltration patterns
  - Unauthorized access to model files
severity: HIGH
mitre_atlas_mapping:
  technique: "T1552"
  sub_technique: "T1552.002"
  tactic: "Collection"
```

### Threat Intelligence Integration

- Map detected attacks to ATLAS TTPs
- Update threat models based on new techniques
- Share indicators with threat intelligence platforms
- Correlate with external threat feeds

### Risk Assessment Framework

| Risk Factor | Description | ATLAS Mapping |
|-------------|-------------|---------------|
| Model Value | Business criticality of AI model | All techniques |
| Exposure | Public API availability | T1595, T1552 |
| Data Sensitivity | Training data classification | T1565 |
| Access Controls | Authentication/authorization | T1199 |

---

## Lessons Learned and Best Practices

### Key Takeaways

1. **Layered Defense**: Multiple security tools provide better coverage
2. **AI-Specific Monitoring**: Traditional tools need AI security extensions
3. **Behavioral Analysis**: Anomaly detection crucial for AI attacks
4. **Rapid Response**: Quick containment prevents attack progression

### Implementation Recommendations

1. **Deploy multiple detection tools** (Falco + KubeArmor + custom rules)
2. **Implement rate limiting** on ML API endpoints
3. **Monitor model file integrity** continuously
4. **Use adversarial training** to improve model robustness
5. **Establish incident response procedures** for AI-specific attacks

### Future Considerations

1. **Emerging ATLAS techniques** as the framework evolves
2. **Zero-day AI vulnerabilities** not yet catalogued
3. **Cross-domain attacks** spanning multiple AI systems
4. **Supply chain security** for AI model dependencies

---

This mapping demonstrates how enterprise AI security can be effectively aligned with the MITRE ATLAS framework using open source technologies, providing a comprehensive defense strategy against adversarial AI threats. 