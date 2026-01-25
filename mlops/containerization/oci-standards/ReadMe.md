# OCI Standards

## Summary

The Open Container Initiative (OCI) defines industry standards for container formats and runtimes, ensuring interoperability across the container ecosystem. Understanding OCI specifications helps ML engineers work with containers at a deeper level, troubleshoot compatibility issues, and make informed decisions about container tooling.

Key points to remember:

- OCI defines three specifications: Runtime, Image, and Distribution
- OCI Image spec standardizes how container images are built and stored
- OCI Runtime spec defines how containers execute on a host system
- OCI compliance ensures tools like Docker, Podman, and containerd work interchangeably
- Understanding OCI helps debug image builds and container runtime issues

## OCI Overview

### What is OCI?

The Open Container Initiative was established in 2015 by Docker and other industry leaders to create open standards for containers. Before OCI, container formats were proprietary, limiting portability and tooling choices.

OCI maintains three specifications:

1. **Runtime Specification (runtime-spec)**: How to run a container
2. **Image Specification (image-spec)**: How to package a container image
3. **Distribution Specification (distribution-spec)**: How to distribute container images

### Why OCI Matters for ML

ML containers benefit from OCI standards:

- **Portability**: Train on one platform, deploy anywhere
- **Tool flexibility**: Use Docker, Podman, or Buildah interchangeably
- **Registry compatibility**: Push to any OCI-compliant registry
- **Long-term stability**: Standards ensure images remain usable

## OCI Image Specification

### Image Components

An OCI image consists of:

```
OCI Image
├── Image Index (manifest list)
│   └── Platform-specific manifests
├── Image Manifest
│   ├── Config blob reference
│   └── Layer blob references
├── Image Configuration
│   ├── Environment variables
│   ├── Entry point
│   ├── Working directory
│   └── Layer history
└── Filesystem Layers
    ├── Layer 1 (base OS)
    ├── Layer 2 (dependencies)
    └── Layer N (application)
```

### Image Manifest

The manifest describes image contents:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:abc123...",
    "size": 7023
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:def456...",
      "size": 32654321
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:ghi789...",
      "size": 16724053
    }
  ]
}
```

### Image Configuration

The config blob contains runtime settings:

```json
{
  "architecture": "amd64",
  "os": "linux",
  "config": {
    "Env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "PYTHONUNBUFFERED=1"
    ],
    "Entrypoint": ["python"],
    "Cmd": ["serve.py"],
    "WorkingDir": "/app",
    "Labels": {
      "model.version": "v1.2.3"
    }
  },
  "rootfs": {
    "type": "layers",
    "diff_ids": [
      "sha256:layer1...",
      "sha256:layer2..."
    ]
  },
  "history": [
    {
      "created": "2024-01-15T10:00:00Z",
      "created_by": "/bin/sh -c apt-get update"
    }
  ]
}
```

### Image Index (Multi-Architecture)

Support multiple platforms with an image index:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.index.v1+json",
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "digest": "sha256:amd64...",
      "size": 7143,
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    },
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "digest": "sha256:arm64...",
      "size": 7682,
      "platform": {
        "architecture": "arm64",
        "os": "linux"
      }
    }
  ]
}
```

This enables single image references that work across architectures.

### Content Addressability

OCI uses content-addressable storage:

```
digest = sha256(content)
```

Benefits:
- **Integrity**: Content verified by hash
- **Deduplication**: Identical layers stored once
- **Immutability**: Digest guarantees exact content

```bash
# Reference by digest for immutability
docker pull myregistry/model@sha256:abc123...
```

## OCI Runtime Specification

### Runtime Bundle

A runtime bundle is an unpacked container ready to run:

```
bundle/
├── config.json    # Runtime configuration
└── rootfs/        # Container filesystem
    ├── bin/
    ├── lib/
    ├── usr/
    └── app/
```

### Runtime Configuration

The `config.json` defines container execution:

```json
{
  "ociVersion": "1.0.2",
  "process": {
    "terminal": false,
    "user": {
      "uid": 1000,
      "gid": 1000
    },
    "args": ["python", "serve.py"],
    "env": [
      "PATH=/usr/local/bin:/usr/bin",
      "MODEL_PATH=/models/fraud"
    ],
    "cwd": "/app"
  },
  "root": {
    "path": "rootfs",
    "readonly": false
  },
  "mounts": [
    {
      "destination": "/models",
      "type": "bind",
      "source": "/data/models",
      "options": ["rbind", "ro"]
    }
  ],
  "linux": {
    "namespaces": [
      {"type": "pid"},
      {"type": "network"},
      {"type": "mount"},
      {"type": "ipc"},
      {"type": "uts"}
    ],
    "resources": {
      "memory": {
        "limit": 8589934592
      },
      "cpu": {
        "shares": 1024
      }
    }
  }
}
```

### Container Lifecycle

OCI defines container states:

1. **Creating**: Container is being created
2. **Created**: Container created but not started
3. **Running**: Container process is executing
4. **Stopped**: Container process has exited

Lifecycle operations:
- `create`: Create container from bundle
- `start`: Start created container
- `kill`: Send signal to container
- `delete`: Remove container

### Linux-Specific Features

OCI runtime supports Linux kernel features:

Namespaces:
- `pid`: Process isolation
- `network`: Network stack isolation
- `mount`: Filesystem isolation
- `ipc`: Inter-process communication isolation
- `uts`: Hostname isolation
- `user`: User ID mapping

Cgroups:
- Memory limits
- CPU shares and quotas
- Device access
- I/O bandwidth

## OCI Distribution Specification

### Registry API

OCI Distribution defines the registry HTTP API:

```
GET  /v2/                                    # API version check
GET  /v2/<name>/manifests/<reference>        # Pull manifest
PUT  /v2/<name>/manifests/<reference>        # Push manifest
GET  /v2/<name>/blobs/<digest>               # Pull blob
POST /v2/<name>/blobs/uploads/               # Initiate blob upload
```

### Pull Workflow

```
1. GET /v2/mymodel/manifests/v1.0
   ← Returns manifest with layer digests

2. For each layer:
   GET /v2/mymodel/blobs/sha256:abc...
   ← Returns layer tarball

3. Assemble layers into rootfs
```

### Push Workflow

```
1. For each layer:
   POST /v2/mymodel/blobs/uploads/
   ← Returns upload URL

   PUT <upload-url>?digest=sha256:abc...
   ← Upload layer content

2. PUT /v2/mymodel/manifests/v1.0
   ← Upload manifest referencing layers
```

### Authentication

OCI registries use token-based auth:

```
1. GET /v2/mymodel/manifests/v1.0
   ← 401 Unauthorized
   ← WWW-Authenticate: Bearer realm="...",service="...",scope="..."

2. GET <auth-realm>?service=...&scope=...
   ← Returns bearer token

3. GET /v2/mymodel/manifests/v1.0
   Authorization: Bearer <token>
   ← Returns manifest
```

## OCI Artifacts

### Beyond Container Images

OCI can store arbitrary artifacts:

- Helm charts
- ML models
- WASM modules
- Security signatures

### ML Model as OCI Artifact

Store models directly in OCI registries:

```bash
# Push model using ORAS
oras push myregistry/models/fraud:v1.0 \
    --artifact-type application/vnd.ml.model \
    model.pt:application/octet-stream \
    config.json:application/json
```

Benefits:
- Unified storage with container images
- Same access control and distribution
- Content-addressable model versioning

### Custom Media Types

Define custom types for ML artifacts:

```
application/vnd.ml.model.pytorch.v1
application/vnd.ml.model.tensorflow.v1
application/vnd.ml.model.onnx.v1
application/vnd.ml.tokenizer.v1
```

## Working with OCI Images

### Inspecting Images

View image manifest:

```bash
# Using skopeo
skopeo inspect docker://myregistry/model:v1

# Using crane
crane manifest myregistry/model:v1

# Using docker
docker manifest inspect myregistry/model:v1
```

### Copying Images

Copy between registries:

```bash
# Using skopeo
skopeo copy \
    docker://source-registry/model:v1 \
    docker://dest-registry/model:v1

# Using crane
crane copy source-registry/model:v1 dest-registry/model:v1
```

### Building OCI Images

Tools for building OCI-compliant images:

```bash
# Buildah (daemonless)
buildah bud -t mymodel:v1 .

# Kaniko (in-cluster builds)
/kaniko/executor --dockerfile=Dockerfile --destination=myregistry/model:v1

# BuildKit
buildctl build --frontend dockerfile.v0 --local context=. --local dockerfile=.
```

## OCI Tools Ecosystem

### Image Tools

| Tool | Purpose |
|------|---------|
| skopeo | Copy and inspect images |
| crane | Manipulate images and registries |
| ORAS | Push/pull OCI artifacts |
| umoci | Unpack and manipulate images |

### Runtime Tools

| Tool | Purpose |
|------|---------|
| runc | Reference OCI runtime |
| crun | Fast OCI runtime in C |
| youki | OCI runtime in Rust |
| gVisor | Sandboxed OCI runtime |

### Build Tools

| Tool | Purpose |
|------|---------|
| BuildKit | Advanced image builder |
| Buildah | Daemonless builds |
| Kaniko | In-cluster builds |
| ko | Go application images |

## Best Practices

### Use Content Digests

For reproducibility, reference by digest:

```yaml
# Mutable tag (avoid in production)
image: myregistry/model:latest

# Immutable digest (preferred)
image: myregistry/model@sha256:abc123...
```

### Leverage Multi-Architecture

Build for multiple platforms:

```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t myregistry/model:v1 \
    --push .
```

### Minimize Layer Count

Each layer adds overhead:

```dockerfile
# Bad: multiple layers
RUN apt-get update
RUN apt-get install -y python3
RUN pip install torch

# Good: single layer
RUN apt-get update && \
    apt-get install -y python3 && \
    pip install torch && \
    rm -rf /var/lib/apt/lists/*
```

## Common Pitfalls

### Media Type Confusion

Different tools may use different media types:

```
# Docker media types
application/vnd.docker.distribution.manifest.v2+json

# OCI media types
application/vnd.oci.image.manifest.v1+json
```

Most registries accept both, but be aware of differences.

### Missing Platform Information

Multi-arch images need proper platform specs:

```json
"platform": {
  "architecture": "amd64",
  "os": "linux",
  "variant": "v8"  // For ARM variants
}
```

### Ignoring Layer Order

Layer order affects caching and size:

```dockerfile
# Put frequently changing content last
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # Changes often, should be last
```
