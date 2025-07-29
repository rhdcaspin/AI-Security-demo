# AI Security Demo: Aligning Enterprise AI Security with MITRE ATLAS

![Demo Architecture](./docs/images/architecture-overview.png)

This repository contains a comprehensive end-to-end demonstration showcasing how to align enterprise AI security with the MITRE ATLAS framework using open source technologies in Kubernetes.

## 🎯 Demo Overview

This demo simulates realistic AI security threats and demonstrates how to detect, analyze, and mitigate them using:

- **MITRE ATLAS Framework**: Structured approach to AI/ML security threats
- **Open Source Security Tools**: KubeArmor, Falco, Clair, Garak, Grafana, Prometheus
- **Kubernetes**: Cloud-native deployment and orchestration
- **Real Attack Scenarios**: Live demonstrations of adversarial attacks

## 🏗️ Architecture

The demo consists of several key components:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Vulnerable AI  │  │  Attack Tools   │  │ Security Stack  │
│   Application   │  │   & Scenarios   │  │  & Monitoring   │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • Image Classifier│ │ • Adversarial   │  │ • KubeArmor     │
│ • Model Serving │  │   Examples      │  │ • Falco         │
│ • LLM Service   │  │ • Model Extract │  │ • Clair Scanner │
│ • API Endpoints │  │ • Data Poison   │  │ • Garak LLM     │
│ • File Storage  │  │ • IBM ART Suite │  │ • Grafana       │
│ • Vuln Scanner  │  │ • Garak Tests   │  │ • Prometheus    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 📁 Project Structure

```
├── applications/           # Vulnerable AI applications
│   ├── image-classifier/   # Main ML application
│   ├── model-server/      # Model serving endpoint
│   └── data-store/        # Training data storage
├── attacks/               # Attack scenarios and tools
│   ├── adversarial/       # Adversarial example generation
│   ├── extraction/        # Model extraction attacks
│   └── poisoning/         # Data poisoning scenarios
├── security/              # Security tools and policies
│   ├── kubearmor/         # Runtime security policies
│   ├── falco/            # Runtime detection rules
│   └── policies/         # Network and security policies
├── monitoring/            # Observability stack
│   ├── prometheus/        # Metrics collection
│   ├── grafana/          # Dashboards and visualization
│   └── alerting/         # Alert rules and webhooks
├── kubernetes/            # K8s deployment manifests
│   ├── base/             # Base configurations
│   ├── overlays/         # Environment-specific configs
│   └── operators/        # Custom operators
├── docs/                 # Documentation and guides
│   ├── mitre-atlas/      # MITRE ATLAS mapping
│   ├── attack-scenarios/ # Detailed attack descriptions
│   └── presentation/     # KubeCon presentation materials
└── scripts/              # Automation and demo scripts
```

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.x installed
- **Container Runtime**: Podman Desktop (recommended) OR Docker
  - [Podman Desktop](https://podman-desktop.io/) - Secure, daemonless container engine
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) - Traditional container platform

#### Why Podman Desktop for AI Security?

**🔒 Enhanced Security:**
- **Rootless containers** - No daemon running as root
- **No background daemon** - Reduced attack surface
- **Fork-exec model** - Better process isolation

**🏢 Enterprise Benefits:**
- **OCI compliant** - Works with all container registries
- **Docker compatibility** - Drop-in replacement for Docker commands
- **Red Hat backed** - Enterprise support and security updates
- **Air-gapped environments** - Better support for disconnected deployments

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rhdcaspin/AI-Security-demo
   cd AI-Security-demo
   ```

2. **Set up container runtime**
   
   **Option A: Podman Desktop (Recommended)**
   ```bash
   # Install Podman Desktop from https://podman-desktop.io/
   # Verify installation
   podman --version
   ```
   
   **Option B: Docker Desktop**
   ```bash
   # Install Docker Desktop from https://docker.com/
   # Verify installation
   docker --version
   ```

3. **Set up the demo environment**
   ```bash
   ./scripts/setup-demo.sh
   ```

3. **Deploy the vulnerable AI application**
   ```bash
   kubectl apply -k kubernetes/overlays/demo
   ```

4. **Install security monitoring stack**
   ```bash
   ./scripts/install-security-stack.sh
   ```

5. **Access the demo dashboard**
   ```bash
   kubectl port-forward svc/grafana 3000:3000 -n monitoring
   # Open http://localhost:3000 (admin/admin)
   ```

## 🎭 Demo Scenarios

### Scenario 1: Adversarial Examples Attack
- **MITRE ATLAS Technique**: T1551 (Evade ML Model)
- **Description**: Generate adversarial examples to fool image classifier
- **Detection**: Runtime behavioral analysis with KubeArmor
- **Mitigation**: Input validation and adversarial training

### Scenario 2: Model Extraction Attack
- **MITRE ATLAS Technique**: T1552 (Steal ML Model)
- **Description**: Extract model parameters through API queries
- **Detection**: Anomalous query patterns via Falco rules
- **Mitigation**: Rate limiting and query monitoring

### Scenario 3: Data Poisoning Attack
- **MITRE ATLAS Technique**: T1565 (Data Manipulation)
- **Description**: Inject malicious data into training pipeline
- **Detection**: Data integrity monitoring
- **Mitigation**: Data validation and provenance tracking

### Scenario 4: IBM ART Advanced Attack Suite
- **MITRE ATLAS Technique**: T1551+ (Enhanced Evade ML Model)
- **Description**: Enterprise-grade adversarial attacks using IBM Adversarial Robustness Toolbox
- **Attack Methods**: 
  - Carlini & Wagner L2 attacks
  - DeepFool sophisticated perturbations  
  - Boundary attacks (black-box)
  - Square attacks (query-efficient)
- **Defense Capabilities**: Real-time adversarial detection and multi-layer defensive preprocessing
- **Detection**: Advanced consistency analysis and entropy-based detection
- **Mitigation**: Graduated defense levels with performance/security trade-offs

### Scenario 5: Container Vulnerability Scanning
- **MITRE ATLAS Context**: Supply Chain Security and ML Pipeline Protection
- **Description**: Comprehensive container image vulnerability scanning using Clair
- **Scanning Capabilities**:
  - Automated vulnerability detection across OS packages and application dependencies
  - Critical/High/Medium/Low severity classification
  - ML-specific vulnerability identification (PyTorch, TensorFlow vulnerabilities)
  - Supply chain risk assessment
- **Detection**: Real-time CVE scanning and security policy enforcement
- **Integration**: Kubernetes admission control and CI/CD pipeline integration

### Scenario 6: LLM Security Testing with Garak
- **MITRE ATLAS Technique**: T1200 (Hardware Additions), T1566 (Phishing), T1078 (Valid Accounts)
- **Description**: Comprehensive LLM vulnerability scanning using Garak framework
- **Testing Capabilities**:
  - Prompt injection attack detection and prevention
  - Jailbreaking and safety bypass testing
  - Data leakage and information disclosure assessment
  - Bias, toxicity, and harmful content evaluation
  - PII exposure and privacy violation detection
- **Detection**: Advanced LLM-specific threat identification and risk scoring
- **Integration**: Real-time LLM security monitoring and automated testing pipelines

## 📊 MITRE ATLAS Mapping

This demo covers the following MITRE ATLAS tactics and techniques:

| Tactic | Technique | Demo Component |
|--------|-----------|----------------|
| Initial Access | Exploit Public-Facing Application | Vulnerable ML API |
| ML Model Access | Inference API Access | Model serving endpoint |
| Execution | User Execution | Adversarial example generation |
| Defense Evasion | Adversarial Examples | Image perturbation attacks |
| Collection | Model Extraction | API-based model stealing |
| Impact | Data Manipulation | Training data poisoning |

## 🛡️ Security Controls

The demo implements multiple layers of security controls:

### Runtime Security (KubeArmor)
- Process execution monitoring
- File access controls
- Network policy enforcement
- Syscall filtering

### Detection Rules (Falco)
- Anomalous API usage patterns
- Suspicious file modifications
- Unexpected network connections
- Container behavior analysis

### Monitoring & Alerting
- Real-time threat detection
- MITRE ATLAS technique mapping
- Incident response automation
- Security metrics and KPIs

## 🎪 Running the Demo

### Demo Script
Follow the step-by-step demo script: [Demo Script](./docs/demo-script.md)

### Key Demo Commands
```bash
# Start clean environment
./scripts/reset-demo.sh

# Launch attack scenario 1
./scripts/run-attack.sh adversarial

# Test LLM security vulnerabilities
./scripts/run-garak-tests.sh comprehensive

# Scan container images for vulnerabilities
./scripts/scan-images.sh batch

# View detection alerts
kubectl logs -f -l app=falco -n security

# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

## 📚 Documentation

- [MITRE ATLAS Mapping](./docs/mitre-atlas/mapping.md)
- [Attack Scenarios](./docs/attack-scenarios/)
- [Security Architecture](./docs/security-architecture.md)
- [IBM ART Integration](./docs/ibm-art-integration.md)
- [Clair Integration](./docs/clair-integration.md)
- [Garak Integration](./docs/garak-integration.md)
- [Podman Desktop Setup](./docs/podman-setup.md)
- [Troubleshooting Guide](./docs/troubleshooting.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./docs/contributing.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MITRE Corporation for the ATLAS framework
- CNCF and the Kubernetes community
- Open source security tool maintainers
- KubeCon organizers and attendees

---

**🎤 Ready for KubeCon? Let's secure AI together!** 
