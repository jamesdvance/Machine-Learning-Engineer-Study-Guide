# Docker for ML

## Summary

Docker provides containerization for ML workloads, packaging models, dependencies, and runtime environments into portable, reproducible units. Containers solve the "works on my machine" problem by ensuring consistent execution across development, training, and serving environments.

Key points to remember:

- Containers package code, dependencies, and environment configuration together
- Base images for ML include NVIDIA CUDA, PyTorch, TensorFlow, and cloud-specific variants
- Layer caching dramatically speeds up builds; order Dockerfile instructions from least to most frequently changed
- GPU access requires NVIDIA Container Toolkit and appropriate runtime configuration
- Image size matters for deployment speed; optimize with multi-stage builds and minimal base images

## Container Fundamentals for ML

### Why Containers for ML

ML systems have complex dependency stacks:

- Python version requirements
- Deep learning frameworks (PyTorch, TensorFlow)
- CUDA and cuDNN versions for GPU support
- Scientific computing libraries (NumPy, SciPy)
- Custom native extensions

These dependencies interact in subtle ways. A model trained with PyTorch 2.0 and CUDA 11.8 may fail to load with different versions. Containers capture the exact environment.

Benefits for ML:

- Reproducibility: Same container produces same results
- Portability: Run on laptop, cluster, or cloud
- Isolation: Dependencies do not conflict between projects
- Scalability: Containers orchestrate easily with Kubernetes

### Docker Architecture

Docker uses a client-server architecture:

- Docker daemon: Manages containers and images
- Docker CLI: Command-line interface to daemon
- Docker images: Immutable templates for containers
- Docker containers: Running instances of images
- Docker registries: Storage for images

Images are built from Dockerfiles, stored in registries, and instantiated as containers.

## Dockerfile Best Practices for ML

### Base Image Selection

Choose base images appropriate for your workload:

Official Python images:
```dockerfile
FROM python:3.11-slim
```

NVIDIA GPU images:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

Framework-specific images:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM tensorflow/tensorflow:2.15.0-gpu
```

Cloud provider images:
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3
```

Framework images include pre-configured dependencies and are often the best starting point.

### Layer Ordering

Docker caches layers. Order instructions from least to most frequently changed:

```dockerfile
# Base image (rarely changes)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# System dependencies (occasionally changes)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (changes with requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (changes frequently)
COPY . /app
WORKDIR /app

# Entry point
CMD ["python", "serve.py"]
```

This ordering means code changes do not invalidate dependency cache.

### Dependency Installation

Install Python dependencies efficiently:

```dockerfile
# Copy only requirements first
COPY requirements.txt .

# Install without cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Or use pip wheel for better caching
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels \
    && pip install --no-cache-dir /wheels/*
```

For complex dependencies with native extensions:

```dockerfile
# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
```

### Model Inclusion

Include trained models in images:

```dockerfile
# Copy model files
COPY models/model.pt /app/models/

# Or download at build time
RUN wget https://storage.example.com/models/model.pt -O /app/models/model.pt
```

For large models, consider:
- Downloading at container start (slower startup, smaller image)
- Model-specific base images
- Volume mounts for shared model storage

### Security Practices

Run as non-root user:

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser . .
```

Avoid secrets in images:

```dockerfile
# Bad: secrets in image layer
ENV API_KEY=secret123

# Good: use runtime environment variables or secrets management
# API_KEY should be passed at runtime
```

## GPU Support

### NVIDIA Container Toolkit

GPU access requires NVIDIA Container Toolkit on the host:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Running GPU Containers

Enable GPU access at runtime:

```bash
# All GPUs
docker run --gpus all myimage

# Specific GPUs
docker run --gpus '"device=0,1"' myimage

# GPU memory limit (for some drivers)
docker run --gpus all --memory 8g myimage
```

### CUDA Version Compatibility

Match CUDA versions between image and host driver:

| Host Driver | Supported CUDA |
|-------------|----------------|
| 525.xx      | Up to 12.0     |
| 535.xx      | Up to 12.2     |
| 545.xx      | Up to 12.3     |

Forward compatibility allows newer drivers with older CUDA images.

## Building and Running

### Build Commands

```bash
# Basic build
docker build -t mymodel:v1 .

# Build with build arguments
docker build --build-arg MODEL_VERSION=v2 -t mymodel:v2 .

# Build with specific Dockerfile
docker build -f Dockerfile.gpu -t mymodel:gpu .

# Build for specific platform
docker build --platform linux/amd64 -t mymodel:amd64 .
```

### Build Optimization

Use BuildKit for faster builds:

```bash
export DOCKER_BUILDKIT=1
docker build -t mymodel:v1 .
```

Cache pip downloads:

```dockerfile
# Mount pip cache from host
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### Run Commands

```bash
# Basic run
docker run mymodel:v1

# Interactive with GPU
docker run -it --gpus all mymodel:v1 bash

# With port mapping
docker run -p 8080:8080 mymodel:v1

# With volume mount
docker run -v /data/models:/app/models mymodel:v1

# With environment variables
docker run -e MODEL_PATH=/app/models/v2 mymodel:v1
```

### Resource Limits

Constrain container resources:

```bash
# Memory limit
docker run --memory 16g mymodel:v1

# CPU limit
docker run --cpus 4 mymodel:v1

# Combined
docker run --memory 16g --cpus 4 --gpus all mymodel:v1
```

## Docker Compose for ML

Docker Compose manages multi-container applications:

```yaml
version: '3.8'

services:
  model:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/v1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

Compose simplifies development environments with multiple services.

## Image Size Optimization

### Slim Base Images

Use minimal base images:

```dockerfile
# Standard Python (1GB+)
FROM python:3.11

# Slim variant (150MB)
FROM python:3.11-slim

# Alpine (50MB, but compatibility issues with scientific libraries)
FROM python:3.11-alpine
```

For ML, slim variants work well. Alpine often causes problems with NumPy and PyTorch.

### Cleanup in Same Layer

Remove temporary files in same layer:

```dockerfile
# Bad: cleanup in separate layer
RUN apt-get update && apt-get install -y build-essential
RUN pip install package
RUN apt-get remove -y build-essential

# Good: cleanup in same layer
RUN apt-get update && apt-get install -y build-essential \
    && pip install package \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
```

### .dockerignore

Exclude unnecessary files:

```
# .dockerignore
__pycache__
*.pyc
.git
.env
notebooks/
tests/
data/
*.md
.coverage
.pytest_cache
```

Smaller build contexts mean faster builds.

## Debugging

### Interactive Access

Debug running containers:

```bash
# Shell in running container
docker exec -it container_name bash

# View logs
docker logs container_name

# Follow logs
docker logs -f container_name
```

### Image Inspection

Inspect image contents:

```bash
# View image layers
docker history mymodel:v1

# Inspect metadata
docker inspect mymodel:v1

# Run temporary container with shell
docker run -it --rm mymodel:v1 bash
```

### Common Issues

CUDA not found:
- Check NVIDIA driver installation on host
- Verify NVIDIA Container Toolkit installed
- Confirm --gpus flag used at runtime

Package import errors:
- Version mismatch between development and container
- Missing system libraries
- Incompatible Python version

Permission denied:
- Running as non-root user without proper permissions
- Volume mounts with wrong ownership

## Best Practices Summary

1. Use framework-specific base images for GPU workloads
2. Order Dockerfile layers for optimal caching
3. Install dependencies before copying code
4. Use multi-stage builds for production images
5. Run as non-root user
6. Never include secrets in images
7. Use .dockerignore to reduce context size
8. Tag images with version or commit hash
9. Test GPU access in containers before deployment
10. Document required runtime flags (GPU, ports, volumes)
