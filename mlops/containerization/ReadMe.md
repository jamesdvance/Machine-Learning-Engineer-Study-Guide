# Containerization for ML

## Summary

Containerization packages ML models with their dependencies and runtime environments into portable, reproducible units. Docker has become the standard for ML deployment, enabling consistent execution from development through production while simplifying dependency management, scaling, and orchestration.

Key points to remember:

- Containers solve dependency conflicts by isolating environments completely
- GPU support requires NVIDIA Container Toolkit and appropriate base images
- Multi-stage builds dramatically reduce image sizes for production
- Container registries provide versioned storage with security scanning
- Image optimization matters: large ML images slow deployment and increase costs

## Why Containerize ML Workloads

ML systems have complex dependency requirements:

Framework dependencies:
- Deep learning frameworks (PyTorch, TensorFlow, JAX)
- CUDA and cuDNN for GPU acceleration
- Specific Python versions

Scientific computing stack:
- NumPy, SciPy with BLAS/LAPACK backends
- Pandas, scikit-learn
- Domain-specific libraries

System dependencies:
- Image processing libraries (OpenCV, PIL)
- Audio/video codecs
- Native extensions and compilers

These dependencies interact in subtle ways. A model trained with specific versions may fail to load or produce different results with different versions. Containers capture the entire environment.

### Container Benefits for ML

Reproducibility:
- Same container produces identical results
- Training and serving use exactly matching environments
- Historical experiments can be re-run years later

Portability:
- Run on any machine with Docker
- Move between cloud providers
- Local development matches production

Isolation:
- Multiple projects with conflicting dependencies
- Different CUDA versions for different models
- No system-wide package pollution

Scalability:
- Kubernetes orchestration
- Horizontal scaling
- Resource management and quotas

## Containerization vs Virtual Environments

Python virtual environments (venv, conda) isolate Python packages but have limitations:

Virtual environments:
- Isolate Python packages only
- Share system libraries and Python interpreter
- Cannot isolate CUDA versions
- Lighter weight, faster to create

Containers:
- Isolate entire filesystem
- Independent system libraries and tools
- Can package specific CUDA versions
- Heavier weight but more complete isolation

For production ML deployment, containers provide stronger guarantees. Virtual environments remain useful for development when container overhead is impractical.

## Container Architecture for ML

### Base Image Selection

Choose base images appropriate for your workload:

For CPU-only workloads:
- `python:3.11-slim`: Minimal Python environment
- Framework-specific images optimized for CPU

For GPU workloads:
- NVIDIA CUDA base images: Start from nvidia/cuda with specific CUDA version
- Framework images: pytorch/pytorch, tensorflow/tensorflow with GPU support
- Cloud provider images: Optimized for specific platforms

For production:
- Prefer runtime images over development images
- Use slim variants when possible
- Consider distroless images for security

### Layer Structure

Docker images are built from layers. Optimal layer structure for ML:

1. Base image (system libraries, Python)
2. System dependencies (apt packages)
3. Python dependencies (pip/conda packages)
4. Model artifacts
5. Application code

Order from least to most frequently changed. This maximizes cache reuse.

### GPU Container Support

GPU access in containers requires:

On the host:
- NVIDIA GPU drivers
- NVIDIA Container Toolkit
- Docker configured to use nvidia runtime

In containers:
- Base image with CUDA libraries matching driver compatibility
- Framework compiled for matching CUDA version

Runtime configuration:
```bash
docker run --gpus all myimage
```

CUDA driver forward compatibility means newer drivers support older CUDA versions, but not vice versa.

## Image Size Optimization

ML images tend to be large:

- Full PyTorch image: 8+ GB
- TensorFlow GPU image: 6+ GB
- Common base + dependencies: 4-5 GB

Large images cause problems:
- Slow deployment and autoscaling
- Increased storage costs
- Longer CI/CD pipelines
- More bandwidth usage

Optimization strategies:

Multi-stage builds:
Separate build-time dependencies (compilers, headers) from runtime requirements. Final image contains only what is needed to run.

Minimal base images:
Use slim or alpine variants. Distroless images provide maximum reduction.

Dependency pruning:
Install only production dependencies. Exclude development tools, tests, documentation.

Layer optimization:
Combine commands to reduce layers. Clean up temporary files in same layer.

See the Multi-Stage Builds chapter for detailed patterns.

## Registry Strategy

Container registries store and distribute images:

Cloud provider registries:
- ECR (AWS), GCR/Artifact Registry (GCP), ACR (Azure)
- Native integration with respective cloud ML platforms
- IAM-based access control

Self-hosted registries:
- Harbor for enterprise features
- Docker Registry for simple deployments

Image management:
- Use specific tags, not just latest
- Consider immutable tags or digest references
- Implement retention policies for cleanup
- Enable vulnerability scanning

See the Container Registries chapter for detailed coverage.

## Development Workflow

### Local Development

Development containers maintain consistency:

```yaml
# docker-compose.yml for development
version: '3.8'
services:
  dev:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - model-cache:/root/.cache
    environment:
      - PYTHONDONTWRITEBYTECODE=1
    ports:
      - "8080:8080"
volumes:
  model-cache:
```

Volume mounts enable code changes without rebuilding.

### CI/CD Integration

Container builds integrate with CI/CD:

1. Build image on PR or push
2. Run tests in container
3. Push to registry
4. Deploy to staging/production

Use BuildKit for faster builds:
```bash
DOCKER_BUILDKIT=1 docker build -t myimage .
```

Cache layers between builds to speed up pipelines.

### Production Deployment

Production containers should be:

Immutable:
- No volume mounts for code
- All configuration via environment variables
- Image fully self-contained

Secure:
- Run as non-root user
- No unnecessary packages
- Scanned for vulnerabilities
- Signed if required

Observable:
- Health check endpoints
- Structured logging to stdout
- Metrics endpoints

## Orchestration

Containers typically deploy to orchestration platforms:

Kubernetes:
- Standard for container orchestration
- Handles scheduling, scaling, and networking
- GPU scheduling with device plugins

Docker Compose:
- Development and simple deployments
- Multi-container applications
- Not for production at scale

Cloud-managed services:
- ECS/Fargate (AWS)
- Cloud Run (GCP)
- Azure Container Instances

Orchestration handles:
- Service discovery
- Load balancing
- Auto-scaling
- Health monitoring
- Rolling updates

## Common Pitfalls

### Version Mismatches

CUDA version in image must be compatible with host driver. PyTorch/TensorFlow versions must match training environment. Document version requirements clearly.

### Image Bloat

Including development dependencies, multiple framework versions, or unnecessary files. Audit image contents regularly.

### Hardcoded Configuration

Baking environment-specific configuration into images. Use environment variables or config files mounted at runtime.

### Missing Health Checks

Containers without health checks may serve traffic before ready or continue running while unhealthy. Implement proper health check endpoints.

### Inadequate Logging

Logging to files inside containers. Log to stdout/stderr for container runtime collection.

## Comparison: Docker vs Alternatives

Docker remains dominant, but alternatives exist:

Podman:
- Docker-compatible, daemonless
- Rootless by default
- Better for security-sensitive environments

Singularity/Apptainer:
- HPC-focused
- Better multi-user security model
- Common in research computing

containerd/nerdctl:
- Kubernetes runtime
- Docker-compatible CLI
- Lighter weight than full Docker

For most ML teams, Docker provides the best ecosystem support and tooling.
