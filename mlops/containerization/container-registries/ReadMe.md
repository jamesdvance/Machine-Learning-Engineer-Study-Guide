# Container Registries

## Summary

Container registries store and distribute Docker images. For ML teams, registries provide versioned storage for training and serving images, access control for sensitive models, and integration with CI/CD pipelines and orchestration platforms.

Key points to remember:

- Registries store images with tags for versioning; use content-addressable digests for immutability
- Cloud provider registries integrate well with their respective ML platforms
- Private registries require authentication; configure credentials securely
- Image scanning detects vulnerabilities in base images and dependencies
- Caching strategies reduce pull times for large ML images

## Registry Fundamentals

### Image Naming

Container images follow a naming convention:

```
[registry/][namespace/]repository:tag
```

Examples:
- `python:3.11-slim` (Docker Hub official)
- `myorg/mymodel:v1.0` (Docker Hub user namespace)
- `gcr.io/myproject/mymodel:latest` (Google Container Registry)
- `123456789.dkr.ecr.us-east-1.amazonaws.com/mymodel:v1` (AWS ECR)

### Tags vs Digests

Tags are mutable pointers:
```
mymodel:latest    # Changes with each push
mymodel:v1.0      # Should be stable but can be overwritten
```

Digests are immutable content hashes:
```
mymodel@sha256:abc123...    # Always refers to exact image
```

For production deployments, use digests or strict tag policies to ensure reproducibility.

### Image Layers

Images consist of layers:
- Each Dockerfile instruction creates a layer
- Layers are cached and shared between images
- Registries store layers independently

Layer sharing reduces storage costs when images share base layers.

## Cloud Provider Registries

### Amazon ECR (Elastic Container Registry)

AWS-native registry with IAM integration:

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Push
docker tag mymodel:v1 123456789.dkr.ecr.us-east-1.amazonaws.com/mymodel:v1
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/mymodel:v1
```

Features:
- IAM-based access control
- Integration with ECS, EKS, and SageMaker
- Image scanning with Amazon Inspector
- Lifecycle policies for cleanup
- Cross-region replication

ECR is the natural choice for AWS-based ML workloads.

### Google Container Registry / Artifact Registry

GCP container storage:

```bash
# Authenticate
gcloud auth configure-docker

# Push
docker tag mymodel:v1 gcr.io/myproject/mymodel:v1
docker push gcr.io/myproject/mymodel:v1
```

Artifact Registry is the successor to Container Registry:
```bash
# Artifact Registry
gcloud auth configure-docker us-docker.pkg.dev
docker tag mymodel:v1 us-docker.pkg.dev/myproject/myrepo/mymodel:v1
docker push us-docker.pkg.dev/myproject/myrepo/mymodel:v1
```

Features:
- IAM integration
- Vulnerability scanning
- Integration with Cloud Run, GKE, Vertex AI
- Multi-region support

### Azure Container Registry

Azure-native registry:

```bash
# Authenticate
az acr login --name myregistry

# Push
docker tag mymodel:v1 myregistry.azurecr.io/mymodel:v1
docker push myregistry.azurecr.io/mymodel:v1
```

Features:
- Azure Active Directory integration
- Geo-replication
- Content trust with image signing
- Integration with AKS and Azure ML

### NVIDIA NGC

NVIDIA GPU Cloud provides optimized ML images:

```bash
# Pull optimized PyTorch
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Authenticate for private content
docker login nvcr.io
```

NGC provides:
- Pre-optimized deep learning frameworks
- Model zoo with pretrained models
- Helm charts for Kubernetes deployment
- Regular updates with latest CUDA optimizations

NGC images are often the best starting point for GPU workloads.

## Self-Hosted Registries

### Docker Registry

Open-source registry for simple deployments:

```yaml
# docker-compose.yml
version: '3'
services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    volumes:
      - registry-data:/var/lib/registry
volumes:
  registry-data:
```

Limited features but simple to run.

### Harbor

Enterprise-grade open-source registry:

- Role-based access control
- Vulnerability scanning (Trivy, Clair)
- Image signing and verification
- Replication between registries
- Garbage collection

Harbor is suitable for organizations needing self-hosted with enterprise features.

### GitLab Container Registry

Integrated with GitLab:

```bash
# Authenticate
docker login registry.gitlab.com

# Push
docker tag mymodel:v1 registry.gitlab.com/mygroup/myproject:v1
docker push registry.gitlab.com/mygroup/myproject:v1
```

Tight integration with GitLab CI/CD makes it convenient for GitLab users.

### GitHub Container Registry

Integrated with GitHub:

```bash
# Authenticate
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push
docker tag mymodel:v1 ghcr.io/myorg/mymodel:v1
docker push ghcr.io/myorg/mymodel:v1
```

Package visibility can be tied to repository visibility.

## Authentication and Access Control

### Credential Management

Store credentials securely:

```bash
# Docker credential helpers
# Store credentials in system keychain
docker login  # Credentials stored encrypted

# For CI/CD, use environment variables or secrets
docker login -u $REGISTRY_USER -p $REGISTRY_PASSWORD registry.example.com
```

### Kubernetes Integration

Create image pull secrets:

```bash
kubectl create secret docker-registry regcred \
    --docker-server=registry.example.com \
    --docker-username=user \
    --docker-password=password \
    --docker-email=email@example.com
```

Reference in pod spec:

```yaml
spec:
  imagePullSecrets:
    - name: regcred
  containers:
    - name: mymodel
      image: registry.example.com/mymodel:v1
```

### Service Account Authentication

For cloud registries, use service accounts:

AWS:
```yaml
# EKS pod identity or IRSA
spec:
  serviceAccountName: ecr-access
  containers:
    - name: mymodel
      image: 123456789.dkr.ecr.us-east-1.amazonaws.com/mymodel:v1
```

GCP:
```yaml
# Workload Identity
spec:
  serviceAccountName: gcr-access
  containers:
    - name: mymodel
      image: gcr.io/myproject/mymodel:v1
```

## Image Security

### Vulnerability Scanning

Scan images for known vulnerabilities:

```bash
# Trivy (standalone)
trivy image mymodel:v1

# Integrated scanning (ECR, GCR, ACR)
# Automatically scans on push
```

Address vulnerabilities:
- Update base images regularly
- Pin and update dependencies
- Use minimal base images to reduce attack surface

### Image Signing

Sign images for verification:

```bash
# Docker Content Trust
export DOCKER_CONTENT_TRUST=1
docker push mymodel:v1  # Automatically signs

# Cosign (Sigstore)
cosign sign mymodel:v1
cosign verify mymodel:v1
```

Signing ensures images were not tampered with.

### Admission Control

Enforce signed images in Kubernetes:

```yaml
# Kyverno policy
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-images
spec:
  validationFailureAction: Enforce
  rules:
    - name: verify-signature
      match:
        resources:
          kinds:
            - Pod
      verifyImages:
        - image: "registry.example.com/*"
          key: |-
            -----BEGIN PUBLIC KEY-----
            ...
            -----END PUBLIC KEY-----
```

## Image Lifecycle Management

### Tagging Strategy

Develop consistent tagging:

```
mymodel:latest           # Current development (mutable)
mymodel:v1.2.3           # Semantic version
mymodel:abc123           # Git commit hash
mymodel:2024-01-15       # Date-based
mymodel:v1.2.3-gpu       # Variant suffix
```

Recommendations:
- Always tag with specific version or commit
- Avoid relying on `latest` in production
- Include variant information (gpu, cpu, slim)

### Retention Policies

Clean up old images:

ECR lifecycle policy:
```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
```

Balance retention for rollback capability against storage costs.

### Multi-Architecture Images

Support different architectures:

```bash
# Build and push multi-arch manifest
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t registry.example.com/mymodel:v1 --push .
```

Multi-arch images enable running on different hardware (x86, ARM).

## Performance Optimization

### Caching Strategies

Speed up image pulls:

Registry-side:
- Use regional registries close to compute
- Enable geo-replication for global deployments

Client-side:
- Pre-pull common base images
- Use layer caching in CI/CD
- Consider registry mirrors or proxies

### Large Image Handling

ML images can be several gigabytes. Optimize:

- Use multi-stage builds to minimize final image
- Share base layers across models
- Consider lazy loading (stargz, nydus)
- Pre-warm nodes with common images

## Best Practices

1. Use cloud-native registries for cloud workloads
2. Implement vulnerability scanning in CI/CD
3. Use immutable tags or digests for production
4. Configure retention policies to manage storage
5. Set up authentication with service accounts, not user credentials
6. Tag images with meaningful versions and commit hashes
7. Sign images for sensitive workloads
8. Use regional registries to reduce latency
9. Monitor registry usage and access patterns
10. Document image variants and their purposes
