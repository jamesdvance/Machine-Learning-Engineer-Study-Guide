# Container Runtimes

## Summary

Container runtimes are the software components that actually execute containers on a host system. Understanding the runtime landscape helps ML engineers choose appropriate runtimes for different environments, troubleshoot container issues, and optimize performance for training and inference workloads.

Key points to remember:

- Container runtimes operate at two levels: high-level (containerd, CRI-O) and low-level (runc, crun)
- Kubernetes uses the Container Runtime Interface (CRI) to interact with runtimes
- Different runtimes offer tradeoffs between security, performance, and compatibility
- GPU workloads require runtime support through NVIDIA Container Toolkit
- Rootless and sandboxed runtimes improve security for multi-tenant environments

## Runtime Architecture

### Two-Level Runtime Model

Container execution involves two runtime levels:

```
┌─────────────────────────────────────────────────┐
│           Container Orchestration               │
│              (Kubernetes, Docker)               │
└─────────────────────┬───────────────────────────┘
                      │ CRI / Docker API
┌─────────────────────▼───────────────────────────┐
│            High-Level Runtime                   │
│         (containerd, CRI-O, dockerd)            │
│  - Image management                             │
│  - Container lifecycle                          │
│  - Network setup                                │
│  - Storage management                           │
└─────────────────────┬───────────────────────────┘
                      │ OCI Runtime Spec
┌─────────────────────▼───────────────────────────┐
│            Low-Level Runtime                    │
│           (runc, crun, youki)                   │
│  - Create namespaces                            │
│  - Set up cgroups                               │
│  - Execute container process                    │
└─────────────────────────────────────────────────┘
```

### High-Level Runtimes

Manage the complete container lifecycle:

- Pull and store images
- Create container filesystems
- Configure networking
- Manage container state
- Delegate execution to low-level runtime

### Low-Level Runtimes

Execute containers according to OCI spec:

- Create Linux namespaces
- Configure cgroups for resource limits
- Set up security contexts
- Execute the container process

## High-Level Runtimes

### containerd

Industry-standard runtime used by Docker and Kubernetes:

```bash
# Check containerd status
sudo systemctl status containerd

# List containers
sudo ctr containers list

# Pull image
sudo ctr images pull docker.io/library/python:3.11

# Run container
sudo ctr run --rm docker.io/library/python:3.11 test python --version
```

Features:
- Kubernetes CRI support
- Image transfer and storage
- Container execution and supervision
- Snapshot management
- Plugin architecture

Configuration (`/etc/containerd/config.toml`):

```toml
version = 2

[plugins."io.containerd.grpc.v1.cri"]
  sandbox_image = "registry.k8s.io/pause:3.9"

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
```

### CRI-O

Lightweight Kubernetes-specific runtime:

```bash
# Check CRI-O status
sudo systemctl status crio

# List pods
sudo crictl pods

# List containers
sudo crictl ps
```

Features:
- Minimal footprint
- Kubernetes-focused
- OCI-compliant
- Security-focused design

Configuration (`/etc/crio/crio.conf`):

```toml
[crio.runtime]
default_runtime = "runc"

[crio.runtime.runtimes.runc]
runtime_path = "/usr/bin/runc"
runtime_type = "oci"

[crio.runtime.runtimes.nvidia]
runtime_path = "/usr/bin/nvidia-container-runtime"
runtime_type = "oci"
```

### Docker Engine (dockerd)

Full-featured container platform:

```bash
# Docker uses containerd internally
docker info | grep -i runtime

# Run container
docker run --rm python:3.11 python --version

# With GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Architecture:
```
Docker CLI → Docker Daemon (dockerd) → containerd → runc
```

Docker adds:
- Build capabilities
- Compose orchestration
- Swarm mode
- User-friendly CLI

## Low-Level Runtimes

### runc

Reference OCI runtime implementation:

```bash
# Create OCI bundle
mkdir -p bundle/rootfs
docker export $(docker create python:3.11) | tar -C bundle/rootfs -xf -
cd bundle
runc spec  # Generate config.json

# Run container
sudo runc run mycontainer
```

Features:
- Written in Go
- Reference implementation
- Battle-tested stability
- Full OCI compliance

### crun

Fast, lightweight OCI runtime:

```bash
# Use crun instead of runc
sudo crun run mycontainer
```

Features:
- Written in C
- Lower memory footprint
- Faster startup times
- Full OCI compliance

Performance comparison:
| Metric | runc | crun |
|--------|------|------|
| Startup time | ~100ms | ~50ms |
| Memory overhead | ~10MB | ~1MB |
| Binary size | ~10MB | ~1MB |

Configure containerd to use crun:

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.crun]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.crun.options]
  BinaryName = "/usr/bin/crun"
```

### youki

OCI runtime written in Rust:

```bash
# Run with youki
sudo youki run mycontainer
```

Features:
- Memory safety through Rust
- Active development
- Growing feature set

## Specialized Runtimes

### gVisor (runsc)

Application kernel for sandboxed containers:

```bash
# Run with gVisor
docker run --runtime=runsc python:3.11 python --version
```

Architecture:
```
Container Process
      ↓
   Sentry (user-space kernel)
      ↓
   Host Kernel (limited syscalls)
```

Features:
- System call interception
- Reduced attack surface
- Compatibility with most applications

Limitations for ML:
- No direct GPU access
- Performance overhead for I/O
- Some syscalls not supported

### Kata Containers

Lightweight VMs for container isolation:

```bash
# Run with Kata
docker run --runtime=kata python:3.11 python --version
```

Architecture:
```
Container Process
      ↓
   Guest Kernel (in VM)
      ↓
   Hypervisor (QEMU/Firecracker)
      ↓
   Host Kernel
```

Features:
- Hardware-level isolation
- Separate kernel per container
- Strong security boundaries

Considerations for ML:
- GPU passthrough possible but complex
- Higher overhead than standard containers
- Better for untrusted workloads

### NVIDIA Container Runtime

GPU-enabled container execution:

```bash
# Direct usage
nvidia-container-runtime run mycontainer

# Via Docker
docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Via containerd
ctr run --gpus 0 --rm docker.io/nvidia/cuda:12.1.0-base test nvidia-smi
```

Architecture:
```
nvidia-container-runtime
      ↓
   runc (with prestart hook)
      ↓
   nvidia-container-cli (GPU setup)
      ↓
   Container with GPU access
```

Configuration (`/etc/nvidia-container-runtime/config.toml`):

```toml
[nvidia-container-cli]
root = "/run/nvidia/driver"
load-kmods = true

[nvidia-container-runtime]
debug = "/var/log/nvidia-container-runtime.log"
```

## Kubernetes Integration

### Container Runtime Interface (CRI)

Kubernetes communicates with runtimes via CRI:

```protobuf
service RuntimeService {
    rpc RunPodSandbox(RunPodSandboxRequest) returns (RunPodSandboxResponse);
    rpc CreateContainer(CreateContainerRequest) returns (CreateContainerResponse);
    rpc StartContainer(StartContainerRequest) returns (StartContainerResponse);
    rpc StopContainer(StopContainerRequest) returns (StopContainerResponse);
    rpc RemoveContainer(RemoveContainerRequest) returns (RemoveContainerResponse);
}
```

### Configuring Kubernetes Runtime

kubelet configuration:

```yaml
# /var/lib/kubelet/config.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
containerRuntimeEndpoint: unix:///run/containerd/containerd.sock
```

Or via kubelet flags:

```bash
kubelet --container-runtime-endpoint=unix:///run/containerd/containerd.sock
```

### Runtime Classes

Use different runtimes for different workloads:

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
---
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
```

Use in pods:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-model
spec:
  runtimeClassName: nvidia
  containers:
    - name: model
      image: mymodel:v1
      resources:
        limits:
          nvidia.com/gpu: 1
```

## GPU Runtime Configuration

### NVIDIA Container Toolkit Setup

Install NVIDIA Container Toolkit:

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Configure containerd
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### Verify GPU Access

```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# containerd (via crictl)
sudo crictl pull nvcr.io/nvidia/cuda:12.1.0-base
```

### GPU Runtime Options

Control GPU allocation:

```bash
# All GPUs
docker run --gpus all myimage

# Specific GPUs
docker run --gpus '"device=0,1"' myimage

# GPU capabilities
docker run --gpus 'all,capabilities=compute,utility' myimage
```

## Performance Considerations

### Startup Time

For inference services, startup time matters:

| Runtime | Cold Start | Warm Start |
|---------|------------|------------|
| runc | ~100ms | ~50ms |
| crun | ~50ms | ~25ms |
| gVisor | ~200ms | ~100ms |
| Kata | ~500ms | ~200ms |

### Resource Overhead

Runtime memory overhead:

| Runtime | Memory Overhead |
|---------|-----------------|
| runc | Minimal |
| crun | Minimal |
| gVisor | ~50-100MB |
| Kata | ~100-200MB |

### I/O Performance

For data-intensive ML workloads:

- **runc/crun**: Native I/O performance
- **gVisor**: Reduced I/O throughput
- **Kata**: Near-native with virtio

## Rootless Containers

### Benefits

Run containers without root privileges:

- Reduced security risk
- Multi-tenant safety
- No privileged daemon

### Rootless Docker

```bash
# Install rootless Docker
dockerd-rootless-setuptool.sh install

# Use rootless Docker
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker run --rm python:3.11 python --version
```

### Rootless Podman

```bash
# Podman is rootless by default
podman run --rm python:3.11 python --version

# With GPU (requires additional setup)
podman run --rm --device nvidia.com/gpu=all myimage
```

### Limitations

Rootless containers have restrictions:

- No privileged operations
- Limited network options
- GPU support requires additional configuration
- Some volume mount restrictions

## Best Practices

### Choose Runtime by Use Case

| Use Case | Recommended Runtime |
|----------|---------------------|
| Standard ML inference | containerd + runc |
| GPU workloads | containerd + nvidia-runtime |
| Fast startup needed | containerd + crun |
| Untrusted code | gVisor or Kata |
| Multi-tenant clusters | Rootless or gVisor |

### Monitor Runtime Health

```bash
# Check containerd
sudo ctr version
sudo ctr plugins ls

# Check CRI-O
sudo crictl version
sudo crictl info

# Check Docker
docker info
docker system df
```

### Keep Runtimes Updated

Runtime vulnerabilities can compromise all containers:

```bash
# Update containerd
sudo apt-get update && sudo apt-get upgrade containerd.io

# Update NVIDIA runtime
sudo apt-get upgrade nvidia-container-toolkit
```

## Common Pitfalls

### Runtime Mismatch

Ensure Kubernetes and runtime configurations align:

```bash
# Check kubelet runtime endpoint
ps aux | grep kubelet | grep container-runtime-endpoint

# Verify runtime is running
sudo systemctl status containerd
```

### GPU Runtime Not Configured

Symptoms: GPU not visible in containers

```bash
# Verify NVIDIA runtime is default
sudo cat /etc/docker/daemon.json
# Should show: "default-runtime": "nvidia"

# Or check containerd config
sudo cat /etc/containerd/config.toml | grep nvidia
```

### cgroup v2 Compatibility

Some older runtimes need cgroup v1:

```bash
# Check cgroup version
mount | grep cgroup

# Force cgroup v1 if needed (kernel parameter)
systemd.unified_cgroup_hierarchy=0
```

### Insufficient Permissions

Rootless containers may fail on volume mounts:

```bash
# Check user namespace mapping
cat /etc/subuid
cat /etc/subgid

# Ensure proper ownership
podman unshare chown -R 1000:1000 /data/models
```
