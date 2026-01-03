# Multi-Stage Docker Builds

## Summary

Multi-stage Docker builds use multiple FROM statements to create lean production images by separating build-time dependencies from runtime requirements. For ML systems, this dramatically reduces image size by excluding compilers, build tools, and development packages from final images.

Key points to remember:

- Multi-stage builds separate build environment from runtime environment
- Only artifacts needed at runtime are copied to the final image
- ML images often shrink from several gigabytes to hundreds of megabytes
- Named stages allow selective copying and parallel builds
- Build caching still works within and across stages

## Why Multi-Stage Builds Matter for ML

ML dependencies often require compilation:

- PyTorch and TensorFlow have native extensions
- NumPy, SciPy require BLAS/LAPACK
- Some packages need gcc, g++, and development headers

Traditional Dockerfiles include all build tools in the final image:

```dockerfile
# Traditional approach: 5GB+ image
FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && pip install torch numpy scipy scikit-learn

COPY . /app
CMD ["python", "serve.py"]
```

This produces large images with unnecessary tools. Multi-stage builds solve this.

## Basic Multi-Stage Pattern

Separate build and runtime stages:

```dockerfile
# Build stage
FROM python:3.11 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake

COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Runtime stage
FROM python:3.11-slim

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . /app
WORKDIR /app
CMD ["python", "serve.py"]
```

The final image contains only the slim Python base and pre-built wheels, not build tools.

## ML-Specific Patterns

### PyTorch Image Optimization

PyTorch images can be large. Optimize by copying only needed components:

```dockerfile
# Build stage with full PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS builder

COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Runtime with minimal PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . /app
WORKDIR /app
CMD ["python", "serve.py"]
```

The runtime variant excludes development headers and build tools.

### Custom Native Extensions

When building custom CUDA kernels or C++ extensions:

```dockerfile
# Build stage with CUDA development tools
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential

COPY requirements.txt .
RUN pip3 wheel --no-cache-dir -r requirements.txt -w /wheels

# Build custom extension
COPY src/extension /extension
WORKDIR /extension
RUN python3 setup.py bdist_wheel -d /wheels

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip3 install --no-cache-dir /wheels/*

COPY app /app
WORKDIR /app
CMD ["python3", "serve.py"]
```

Native extensions are compiled in the build stage but only the compiled artifacts reach production.

### Model Packaging

Include model artifacts efficiently:

```dockerfile
# Build and train stage
FROM python:3.11 AS trainer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train/ /train
WORKDIR /train
RUN python train.py --output /model

# Serving stage
FROM python:3.11-slim AS server

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --from=trainer /model /app/model
COPY serve/ /app
WORKDIR /app
CMD ["python", "serve.py"]
```

Training dependencies stay in the build stage; only the model and serving code reach production.

## Advanced Patterns

### Parallel Stage Builds

Docker BuildKit builds independent stages in parallel:

```dockerfile
# These run in parallel
FROM python:3.11 AS python-deps
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM node:18 AS frontend
COPY frontend/ /frontend
WORKDIR /frontend
RUN npm install && npm run build

# Final stage depends on both
FROM python:3.11-slim

COPY --from=python-deps /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY --from=frontend /frontend/dist /app/static

COPY app/ /app
WORKDIR /app
CMD ["python", "serve.py"]
```

Parallel builds reduce total build time.

### Conditional Stages

Use build arguments to select stages:

```dockerfile
FROM python:3.11 AS base
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS development
RUN pip install --no-cache-dir pytest ipython jupyter
COPY . /app
CMD ["python", "-m", "pytest"]

FROM base AS production
COPY app/ /app
WORKDIR /app
CMD ["python", "serve.py"]
```

Build specific targets:
```bash
docker build --target development -t myapp:dev .
docker build --target production -t myapp:prod .
```

### Testing in Build

Run tests as a build stage:

```dockerfile
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

# Test stage
FROM builder AS test
RUN pip install pytest
RUN pytest tests/

# Production stage (only reached if tests pass)
FROM python:3.11-slim AS production
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY app/ /app
CMD ["python", "serve.py"]
```

Tests run during build; failures prevent image creation.

### Caching Dependencies Across Builds

Use a dedicated cache stage:

```dockerfile
FROM python:3.11 AS dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM python:3.11-slim AS runtime
COPY --from=dependencies /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

FROM runtime AS final
COPY app/ /app
WORKDIR /app
CMD ["python", "serve.py"]
```

The dependencies stage is cached unless requirements.txt changes.

## Size Comparison

Typical size reductions for ML images:

| Approach | Image Size |
|----------|------------|
| Full development image | 8-12 GB |
| Single-stage with cleanup | 4-6 GB |
| Multi-stage (runtime base) | 2-3 GB |
| Multi-stage (slim base) | 1-2 GB |
| Multi-stage (distroless) | 500 MB - 1 GB |

Smaller images mean:
- Faster deployments
- Reduced storage costs
- Quicker autoscaling
- Smaller attack surface

## Best Practices

### Name All Stages

Use descriptive names:

```dockerfile
FROM python:3.11 AS builder
# ... build steps ...

FROM python:3.11-slim AS runtime
# ... runtime setup ...

FROM runtime AS production
# ... final image ...
```

Named stages clarify intent and enable targeted builds.

### Minimize Final Stage Layers

Keep the final stage simple:

```dockerfile
FROM python:3.11-slim AS production

# Single layer for dependencies
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Single layer for application
COPY app/ /app

WORKDIR /app
CMD ["python", "serve.py"]
```

Fewer layers mean smaller images and faster pulls.

### Use Consistent Base Images

Match base images between build and runtime stages:

```dockerfile
# Good: same Python version
FROM python:3.11 AS builder
FROM python:3.11-slim AS runtime

# Risky: different Python versions
FROM python:3.11 AS builder
FROM python:3.10-slim AS runtime  # May have compatibility issues
```

### Document Stage Purposes

Comment stage intentions:

```dockerfile
# Stage 1: Compile native extensions
FROM python:3.11 AS builder
# ...

# Stage 2: Run unit tests
FROM builder AS test
# ...

# Stage 3: Minimal production image
FROM python:3.11-slim AS production
# ...
```

### Consider Distroless Images

For maximum security and minimal size:

```dockerfile
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

FROM gcr.io/distroless/python3
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY app/ /app
WORKDIR /app
CMD ["serve.py"]
```

Distroless images have no shell or package manager, reducing attack surface.

## Common Pitfalls

### Missing Runtime Dependencies

Some packages need runtime libraries not in slim images:

```dockerfile
# May fail at runtime
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
# ImportError: libgomp.so.1 not found

# Fixed: install runtime libraries
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
```

Test images thoroughly before deployment.

### Broken Caching

Avoid invalidating cache unnecessarily:

```dockerfile
# Bad: any code change invalidates dependency cache
COPY . /app
RUN pip install -r /app/requirements.txt

# Good: only requirements.txt changes invalidate dependency cache
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
```

### Forgotten Artifacts

Ensure all needed files are copied:

```dockerfile
# Missing model files
FROM python:3.11-slim AS production
COPY --from=builder /wheels /wheels
COPY app/ /app
# Forgot: COPY --from=builder /model /app/model
```

Test that the final image runs correctly.
