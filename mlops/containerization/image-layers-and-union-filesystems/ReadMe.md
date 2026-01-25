# Image Layers and Union Filesystems

## Summary

Container images are built from stacked layers that combine into a single filesystem view using union filesystems. Understanding how layers work is essential for optimizing ML container images, reducing build times, minimizing storage costs, and debugging image-related issues.

Key points to remember:

- Each Dockerfile instruction creates a new image layer
- Layers are immutable and shared between images to save storage
- Union filesystems merge layers into a single coherent view
- Layer ordering significantly impacts build cache efficiency
- Large ML dependencies require careful layer management to keep images practical

## Layer Fundamentals

### What Are Layers?

A container image consists of stacked filesystem layers:

```
┌─────────────────────────────────────┐
│  Layer 4: Application code (10 MB)  │  COPY . /app
├─────────────────────────────────────┤
│  Layer 3: Python packages (2 GB)    │  RUN pip install -r requirements.txt
├─────────────────────────────────────┤
│  Layer 2: System packages (500 MB)  │  RUN apt-get install ...
├─────────────────────────────────────┤
│  Layer 1: Base image (100 MB)       │  FROM python:3.11-slim
└─────────────────────────────────────┘
```

Each layer contains only the filesystem changes (delta) from the previous layer.

### Layer Creation

Dockerfile instructions that create layers:

```dockerfile
FROM python:3.11-slim          # Base layer(s)
RUN apt-get update             # New layer
RUN apt-get install -y vim     # New layer
COPY requirements.txt .        # New layer
RUN pip install -r req.txt     # New layer
COPY . /app                    # New layer
```

Instructions that don't create layers:
- `ENV` (metadata only)
- `LABEL` (metadata only)
- `EXPOSE` (metadata only)
- `WORKDIR` (metadata only, unless directory doesn't exist)
- `ARG` (build-time only)
- `CMD` (metadata only)
- `ENTRYPOINT` (metadata only)

### Layer Immutability

Once created, layers never change:

```dockerfile
# Layer 1: Creates file
RUN echo "hello" > /data/file.txt

# Layer 2: "Deletes" file (actually creates whiteout)
RUN rm /data/file.txt
```

The file still exists in Layer 1; Layer 2 just marks it as deleted. Both layers contribute to image size.

### Content Addressing

Layers are identified by content hash:

```
sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4
```

Benefits:
- Deduplication across images
- Integrity verification
- Deterministic builds

## Union Filesystems

### How Union FS Works

Union filesystems overlay multiple directories into a single view:

```
        Unified View (Container sees this)
                    ↓
    ┌───────────────────────────────────┐
    │  /app/model.py (from Layer 4)     │
    │  /app/utils.py (from Layer 4)     │
    │  /usr/bin/python (from Layer 1)   │
    │  /lib/... (from Layers 1-3)       │
    └───────────────────────────────────┘
                    ↑
    Layer 4 (read-only) ─┬─ /app/model.py
    Layer 3 (read-only) ─┼─ /usr/local/lib/python3.11/
    Layer 2 (read-only) ─┼─ /usr/bin/...
    Layer 1 (read-only) ─┴─ /lib/x86_64-linux-gnu/
```

### Copy-on-Write

When a container modifies a file from a lower layer:

1. File is copied to container's writable layer
2. Modifications happen on the copy
3. Original layer remains unchanged

```
Container Write Layer (read-write)
    └── /app/model.py (modified copy)
         ↑ copied on first write
Layer 4 (read-only)
    └── /app/model.py (original)
```

### Union FS Implementations

#### OverlayFS

Default on modern Linux systems:

```bash
# Check current storage driver
docker info | grep "Storage Driver"
# Storage Driver: overlay2

# OverlayFS mount structure
mount | grep overlay
# overlay on /var/lib/docker/overlay2/abc.../merged type overlay
# (lowerdir=...,upperdir=...,workdir=...)
```

Components:
- **lowerdir**: Read-only image layers
- **upperdir**: Container's writable layer
- **workdir**: Temporary work directory
- **merged**: Unified view

#### Other Implementations

| Filesystem | Status | Use Case |
|------------|--------|----------|
| overlay2 | Default | Most Linux systems |
| fuse-overlayfs | Active | Rootless containers |
| btrfs | Supported | Btrfs filesystems |
| zfs | Supported | ZFS filesystems |
| devicemapper | Deprecated | Legacy systems |
| aufs | Deprecated | Old Ubuntu/Debian |

### Whiteout Files

Union FS handles deletions with whiteout markers:

```dockerfile
# Layer 1
RUN mkdir /data && echo "content" > /data/file.txt

# Layer 2
RUN rm /data/file.txt
# Creates whiteout: /data/.wh.file.txt
```

Whiteout types:
- **Standard whiteout**: `.wh.<filename>` hides specific file
- **Opaque whiteout**: `.wh..wh..opq` hides entire directory

## Layer Optimization for ML

### The ML Image Size Problem

Typical ML image layer breakdown:

```
┌─────────────────────────────────────────────┐
│ Application code                    (~50 MB) │
├─────────────────────────────────────────────┤
│ Model weights                       (~500 MB)│
├─────────────────────────────────────────────┤
│ PyTorch/TensorFlow                  (~2 GB)  │
├─────────────────────────────────────────────┤
│ CUDA libraries                      (~3 GB)  │
├─────────────────────────────────────────────┤
│ Base OS                             (~200 MB)│
└─────────────────────────────────────────────┘
Total: ~6 GB
```

### Layer Ordering Strategy

Order from least to most frequently changed:

```dockerfile
# Good: Optimal layer ordering
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 1. System dependencies (rarely change)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (change occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Model files (change with model updates)
COPY models/ /app/models/

# 4. Application code (changes frequently)
COPY src/ /app/src/
```

### Combining Commands

Reduce layers and avoid intermediate artifacts:

```dockerfile
# Bad: Multiple layers, leftover cache
RUN apt-get update
RUN apt-get install -y build-essential
RUN pip install numpy
RUN apt-get remove -y build-essential
RUN apt-get autoremove -y
RUN rm -rf /var/lib/apt/lists/*

# Good: Single layer, clean result
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install numpy && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

### Separating Dependencies

Split requirements for better caching:

```dockerfile
# Base requirements (stable)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# ML framework (occasionally updated)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Development requirements (frequently updated)
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
```

## Build Cache

### How Cache Works

Docker caches layers based on:
1. Parent layer hash
2. Instruction text
3. File checksums (for COPY/ADD)

```dockerfile
FROM python:3.11              # Cache hit if same base
RUN apt-get update            # Cache hit if parent matches
COPY requirements.txt .       # Cache hit if file unchanged
RUN pip install -r req.txt    # Cache hit if parent matches
COPY . /app                   # Cache miss if any file changed
```

### Cache Invalidation

A cache miss invalidates all subsequent layers:

```
Layer 1: FROM python:3.11     ✓ Cached
Layer 2: RUN apt-get...       ✓ Cached
Layer 3: COPY requirements.txt ✗ File changed - INVALIDATED
Layer 4: RUN pip install...   ✗ Must rebuild (parent changed)
Layer 5: COPY . /app          ✗ Must rebuild (parent changed)
```

### BuildKit Cache Mounts

Persist cache between builds:

```dockerfile
# syntax=docker/dockerfile:1.4

# Cache pip downloads
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Cache apt packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y python3
```

### Conditional Cache

Use build arguments for cache control:

```dockerfile
ARG CACHEBUST=1
RUN echo "Cache bust: $CACHEBUST" && \
    pip install --upgrade mypackage
```

```bash
# Force cache invalidation
docker build --build-arg CACHEBUST=$(date +%s) .
```

## Layer Inspection

### View Image Layers

```bash
# Docker history
docker history mymodel:v1

# Detailed layer info
docker inspect mymodel:v1 | jq '.[0].RootFS.Layers'

# Layer sizes with dive
dive mymodel:v1
```

### Analyze Layer Contents

Using `dive` tool:

```bash
# Install dive
brew install dive  # macOS
# or
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest mymodel:v1
```

Dive shows:
- Layer-by-layer filesystem changes
- Wasted space from deleted files
- Image efficiency score

### Export and Inspect

```bash
# Export image filesystem
docker save mymodel:v1 -o image.tar
tar -tvf image.tar

# Extract specific layer
tar -xf image.tar sha256:abc123.../layer.tar
tar -tvf layer.tar
```

## Storage Optimization

### Deduplication

Shared base images save storage:

```
Image A (PyTorch model)          Image B (TensorFlow model)
├── Layer: App code (unique)     ├── Layer: App code (unique)
├── Layer: PyTorch (unique)      ├── Layer: TensorFlow (unique)
├── Layer: Python (shared) ──────├── Layer: Python (shared)
└── Layer: Ubuntu (shared) ──────└── Layer: Ubuntu (shared)

Storage: Only one copy of shared layers
```

### Registry Layer Sharing

Registries store layers once:

```bash
# Push first image
docker push myregistry/model-a:v1  # Uploads all layers

# Push second image with shared base
docker push myregistry/model-b:v1  # Only uploads unique layers
```

### Squashing Layers

Combine all layers into one (use sparingly):

```bash
# Docker squash
docker build --squash -t mymodel:v1 .

# BuildKit
docker buildx build --squash -t mymodel:v1 .
```

Tradeoffs:
- Reduces layer count
- Loses layer sharing benefits
- Loses build cache benefits

## Advanced Patterns

### Layer Extraction for Model Serving

Separate model weights from code:

```dockerfile
# Base image with code and dependencies
FROM myregistry/model-base:v1 AS base

# Model image just adds weights
FROM base
COPY models/weights.pt /app/models/
```

Update models without rebuilding dependencies.

### Lazy Layer Loading

For very large images, use lazy loading:

```bash
# Stargz snapshotter enables lazy pulling
# Only fetches layers as needed

# Convert image to stargz format
ctr-remote image optimize mymodel:v1 mymodel:v1-stargz
```

### Layer Caching in CI/CD

```yaml
# GitHub Actions with layer caching
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: myregistry/model:${{ github.sha }}
    cache-from: type=registry,ref=myregistry/model:cache
    cache-to: type=registry,ref=myregistry/model:cache,mode=max
```

## Best Practices

### Keep Base Layers Stable

```dockerfile
# Pin base image digest for reproducibility
FROM python:3.11-slim@sha256:abc123...
```

### Minimize Layer Count for Final Image

```dockerfile
# Development: many layers for caching
FROM python:3.11 AS builder
RUN pip install build-tools
RUN pip install dependencies
RUN build-application

# Production: minimal layers
FROM python:3.11-slim
COPY --from=builder /app /app
```

### Clean Up in Same Layer

```dockerfile
# Bad: cleanup in separate layer
RUN apt-get install -y build-essential
RUN pip install package-needing-build
RUN apt-get remove -y build-essential  # Original still in previous layers!

# Good: cleanup in same layer
RUN apt-get install -y build-essential && \
    pip install package-needing-build && \
    apt-get remove -y build-essential && \
    rm -rf /var/lib/apt/lists/*
```

## Common Pitfalls

### Accidental Cache Invalidation

```dockerfile
# Bad: COPY . invalidates cache even for unrelated changes
COPY . /app
RUN pip install -r /app/requirements.txt

# Good: Copy only what's needed
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
```

### Large Layers from Model Files

```dockerfile
# Bad: Model in image layer (can't update independently)
COPY large-model.pt /app/models/

# Better: Download at runtime or use separate model image
RUN --mount=type=secret,id=aws \
    aws s3 cp s3://models/large-model.pt /app/models/
```

### Ignoring .dockerignore

```
# .dockerignore
.git
*.pyc
__pycache__
data/
notebooks/
*.pt
*.pth
.env
```

Without `.dockerignore`, COPY commands include everything, bloating layers.

### Misunderstanding Delete Behavior

```dockerfile
# This doesn't reduce image size!
RUN wget https://example.com/large-file.tar.gz
RUN tar -xzf large-file.tar.gz
RUN rm large-file.tar.gz  # File still exists in layer 1

# Fix: Single layer
RUN wget https://example.com/large-file.tar.gz && \
    tar -xzf large-file.tar.gz && \
    rm large-file.tar.gz
```
