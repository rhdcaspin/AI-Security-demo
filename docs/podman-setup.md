# Podman Desktop Setup Guide

## Overview

This guide covers setting up and using **Podman Desktop** with the AI Security demo. Podman Desktop is the recommended container runtime for enterprise environments due to its enhanced security features and daemonless architecture.

## Why Podman Desktop?

### 🔒 Security Advantages

**Rootless Containers**
- Containers run without root privileges by default
- Reduced security risk and attack surface
- Better compliance with enterprise security policies

**Daemonless Architecture**
- No background daemon running as root
- Fork-exec model provides better process isolation
- Eliminates daemon-related vulnerabilities

**Enhanced Isolation**
- Each container runs in its own user namespace
- Better separation between containers and host
- Improved security for multi-tenant environments

### 🏢 Enterprise Benefits

**OCI Compliance**
- Full compatibility with Open Container Initiative standards
- Works with all major container registries
- Future-proof container technology

**Drop-in Docker Replacement**
- Compatible with Docker CLI commands
- Seamless migration from Docker environments
- Existing scripts and workflows continue to work

**Air-Gapped Support**
- Better support for disconnected environments
- No external daemon dependencies
- Improved security for classified/restricted networks

## Installation

### 1. Install Podman Desktop

**macOS**
```bash
# Using Homebrew
brew install podman-desktop

# Or download from https://podman-desktop.io/
```

**Windows**
```bash
# Download installer from https://podman-desktop.io/
# Run the installer and follow setup wizard
```

**Linux**
```bash
# Fedora/RHEL/CentOS
sudo dnf install podman-desktop

# Ubuntu/Debian
# Download from https://podman-desktop.io/ or use flatpak
flatpak install flathub io.podman_desktop.PodmanDesktop
```

### 2. Verify Installation

```bash
# Check Podman version
podman --version

# Check Podman Desktop
podman-desktop --version

# Test basic functionality
podman run hello-world
```

### 3. Configure for Kubernetes Integration

**For kind clusters:**
```bash
# Podman works with kind using image archives
# The setup script handles this automatically
```

**For minikube clusters:**
```bash
# Configure minikube to use Podman
minikube config set driver podman
minikube config set container-runtime containerd

# Start minikube with Podman
minikube start --driver=podman
```

## Using with AI Security Demo

### Automatic Detection

The setup script automatically detects your container runtime:

```bash
./scripts/setup-demo.sh
```

**Expected output:**
```
🔧 Building container images...
ℹ️  Using Podman as container runtime
ℹ️  Building image classifier...
...
```

### Manual Podman Commands

If you prefer to build images manually:

```bash
# Build all services
cd applications/image-classifier
podman build -t kubecon-demo/image-classifier:latest .

cd ../llm-service  
podman build -t kubecon-demo/llm-service:latest .

cd ../garak-scanner
podman build -t kubecon-demo/garak-scanner-service:latest .

cd ../vuln-scanner
podman build -t kubecon-demo/vuln-scanner-service:latest .

# Build ART defense service
cd ../image-classifier
podman build -f Dockerfile.art-defense -t kubecon-demo/art-defense-service:latest .
```

### Image Management

```bash
# List built images
podman images | grep kubecon-demo

# Check image details
podman inspect kubecon-demo/image-classifier:latest

# Remove images if needed
podman rmi kubecon-demo/image-classifier:latest
```

## Integration with Kubernetes

### kind Integration

The script automatically handles Podman + kind integration:

```bash
# Images are saved as archives and loaded into kind
podman save kubecon-demo/image-classifier:latest | kind load image-archive /dev/stdin
```

### minikube Integration

For minikube with Podman:

```bash
# Configure minikube to use Podman
eval $(minikube podman-env)

# Build images (they'll be available in minikube automatically)
./scripts/setup-demo.sh
```

### Manual Image Loading

If you need to manually load images:

```bash
# For kind
podman save kubecon-demo/image-classifier:latest | kind load image-archive /dev/stdin

# For minikube (if not using podman-env)
podman save kubecon-demo/image-classifier:latest -o /tmp/image.tar
minikube image load /tmp/image.tar
```

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
# Ensure user is in podman group (Linux)
sudo usermod -aG podman $USER
newgrp podman
```

**Machine Not Started (macOS/Windows)**
```bash
# Initialize and start Podman machine
podman machine init
podman machine start
```

**Image Loading Issues with kind**
```bash
# Verify image exists
podman images | grep kubecon-demo

# Manual load with verbose output
podman save kubecon-demo/image-classifier:latest | kind load image-archive /dev/stdin --verbosity=1
```

**Minikube Connection Issues**
```bash
# Reset Podman environment
eval $(minikube podman-env)

# Verify connection
podman ps
```

### Debugging Commands

```bash
# Check Podman system info
podman info

# Check running containers
podman ps

# Check Podman events
podman events

# Check machine status (macOS/Windows)
podman machine list
```

## Performance Considerations

### Build Performance

Podman typically provides:
- **Faster builds** due to no daemon overhead
- **Lower memory usage** during build process
- **Better resource isolation** between builds

### Runtime Performance

- **Equivalent performance** to Docker for most workloads
- **Lower system overhead** due to daemonless architecture
- **Better security** with rootless execution

## Migration from Docker

### Automatic Migration

The setup script supports both runtimes:
- Detects available container runtime automatically
- Uses Podman if available, falls back to Docker
- No changes needed to existing workflows

### Manual Migration Commands

```bash
# Replace docker commands with podman
docker build → podman build
docker run → podman run
docker ps → podman ps
docker images → podman images

# Podman-specific enhancements
podman generate systemd → Generate systemd service files
podman pod → Manage pods (Docker Compose alternative)
```

## Advanced Features

### Rootless Pods

```bash
# Create multi-container pods (rootless)
podman pod create --name ai-security-pod
podman run --pod ai-security-pod kubecon-demo/image-classifier:latest
podman run --pod ai-security-pod kubecon-demo/llm-service:latest
```

### Systemd Integration

```bash
# Generate systemd service files
podman generate systemd --name ai-security-pod --files

# Enable and start service
systemctl --user enable pod-ai-security-pod.service
systemctl --user start pod-ai-security-pod.service
```

### Security Scanning

```bash
# Built-in vulnerability scanning
podman run --security-opt label=disable --rm -v /var/lib/containers/storage:/var/lib/containers/storage:ro quay.io/projectquay/clair:latest

# Check image security
podman inspect --format='{{.RootFS.Layers}}' kubecon-demo/image-classifier:latest
```

## Best Practices

### Security

1. **Always use rootless mode** (default in Podman Desktop)
2. **Enable SELinux/AppArmor** when available
3. **Use minimal base images** for reduced attack surface
4. **Regularly update Podman Desktop** for security patches

### Performance

1. **Use multi-stage builds** to reduce image size
2. **Leverage build cache** for faster rebuilds
3. **Clean up unused images** regularly
4. **Use .containerignore** files to exclude unnecessary files

### Enterprise Deployment

1. **Configure image registries** for your organization
2. **Set up image signing** for supply chain security
3. **Implement scanning policies** for vulnerability management
4. **Use pods** for multi-container applications

## Conclusion

Podman Desktop provides enhanced security and enterprise features that make it ideal for AI security demonstrations. The automatic detection in our setup script ensures seamless operation regardless of your container runtime choice, while providing the security benefits of rootless, daemonless container execution.

For additional support, visit:
- [Podman Desktop Documentation](https://podman-desktop.io/docs)
- [Podman GitHub](https://github.com/containers/podman)
- [Red Hat Podman Guide](https://docs.podman.io/) 