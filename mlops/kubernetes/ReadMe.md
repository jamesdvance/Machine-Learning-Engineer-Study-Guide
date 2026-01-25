# Kubernetes for ML

## Summary

Kubernetes provides the orchestration layer for deploying, scaling, and managing containerized ML workloads. It handles scheduling of training jobs and inference services, manages GPU resources, enables auto-scaling based on demand, and provides the foundation for production ML infrastructure at scale.

Key points to remember:

- Kubernetes abstracts infrastructure, enabling portable ML deployments across clouds
- Deployments manage stateless inference services with rolling updates and scaling
- Services provide stable networking and load balancing for model endpoints
- ConfigMaps and Secrets separate configuration from container images
- GPU scheduling requires device plugins and proper resource requests
- Horizontal Pod Autoscaler dynamically scales based on CPU, memory, or custom metrics

## Core Concepts for ML

### Pods and Containers

Pods are the smallest deployable units in Kubernetes. For ML:

- Inference servers run as containers within pods
- Sidecar containers handle logging, metrics, and feature retrieval
- Init containers download models before the main container starts
- Resource requests and limits ensure predictable GPU and memory allocation

### Workload Types

Deployments: Long-running inference services
- Rolling updates for model version changes
- Replica management for high availability
- Integration with HPA for auto-scaling

Jobs: One-time training runs
- Completion tracking
- Retry on failure
- TTL-based cleanup

CronJobs: Scheduled retraining
- Periodic model updates
- Batch inference pipelines

StatefulSets: Stateful ML workloads
- Distributed training with stable network identities
- Model servers with persistent storage

### Networking

Services expose pods to clients:
- ClusterIP for internal model-to-model communication
- LoadBalancer for external inference endpoints
- Ingress for HTTP routing and TLS termination

### Storage

PersistentVolumeClaims provide storage for:
- Training datasets
- Model checkpoints
- Feature caches

## ML Platform Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Ingress Controller             │
                    └─────────────────┬───────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼─────────┐   ┌────────▼────────┐   ┌─────────▼─────────┐
    │  Model Service A  │   │  Model Service B │   │  Model Service C  │
    │   (Deployment)    │   │   (Deployment)   │   │   (Deployment)    │
    └─────────┬─────────┘   └────────┬────────┘   └─────────┬─────────┘
              │                      │                      │
    ┌─────────▼─────────┐   ┌────────▼────────┐   ┌─────────▼─────────┐
    │   HPA (CPU/RPS)   │   │   HPA (Latency)  │   │   HPA (GPU Util)  │
    └───────────────────┘   └──────────────────┘   └───────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │       Cluster Autoscaler        │
                    │    (GPU and CPU Node Pools)     │
                    └─────────────────────────────────┘
```

## Chapter Overview

### [Deployments](deployments/ReadMe.md)

Managing model serving workloads:
- Deployment specifications for ML containers
- Rolling update strategies for model versions
- Resource management for CPU, memory, and GPU
- Health probes for model readiness
- ML-specific patterns like sidecars and init containers

### [Services](services/ReadMe.md)

Networking for ML endpoints:
- Service types and when to use each
- Load balancing across model replicas
- Ingress configuration for HTTP routing
- gRPC services for efficient inference
- A/B testing and canary deployment patterns

### [ConfigMaps and Secrets](configmaps-and-secrets/ReadMe.md)

Configuration management:
- Storing model paths and hyperparameters
- Managing API keys and credentials
- Environment-specific configuration
- External secret management integration
- Configuration update strategies

### [Horizontal Pod Autoscaler](horizontal-pod-autoscaler/ReadMe.md)

Dynamic scaling for ML workloads:
- CPU and memory-based scaling
- Custom metrics (latency, RPS, queue depth)
- Scaling behavior tuning
- ML-specific patterns for warmup and GPU
- Integration with KEDA for advanced triggers

### [GPU Scheduling](gpu-scheduling/ReadMe.md)

GPU resource management:
- NVIDIA device plugin setup
- Requesting and allocating GPUs
- Node selection for GPU types
- GPU sharing with time-slicing and MIG
- Cost optimization with spot instances

## Common Patterns

### Blue-Green Deployment

```yaml
# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
      version: blue
---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
      version: green
---
# Service switches between blue and green
apiVersion: v1
kind: Service
metadata:
  name: model
spec:
  selector:
    app: model
    version: blue  # Change to green to switch
```

### Model-per-Pod vs Router

Model-per-pod:
- Each pod serves one model
- Simple scaling and isolation
- Higher resource usage

Router pattern:
- Single router pod selects model
- Dynamic model loading
- More complex but flexible

### Feature Store Integration

```yaml
spec:
  containers:
    - name: model-server
      env:
        - name: FEATURE_STORE_URL
          valueFrom:
            configMapKeyRef:
              name: feature-config
              key: FEATURE_STORE_URL
    - name: feature-proxy
      image: feature-proxy:v1
      ports:
        - containerPort: 8081
```

## Best Practices

### Resource Management

- Always set resource requests and limits
- Profile models to determine actual usage
- Use GPU requests only when needed
- Implement pod disruption budgets for availability

### Observability

- Configure liveness and readiness probes
- Export metrics to Prometheus
- Use structured logging to stdout
- Trace requests through the inference pipeline

### Security

- Run containers as non-root
- Use network policies to restrict traffic
- Store secrets in external secret managers
- Scan images for vulnerabilities

### Cost Optimization

- Use spot/preemptible nodes for training
- Scale GPU nodes to zero when idle
- Right-size GPU selection for workloads
- Implement resource quotas per team

## Tools and Extensions

### KServe

Serverless inference with:
- Auto-scaling including scale-to-zero
- Canary rollouts
- Model explainability
- Multi-model serving

### Kubeflow

Complete ML platform on Kubernetes:
- Notebooks
- Training operators
- Pipelines
- Model serving

### NVIDIA GPU Operator

Automated GPU management:
- Driver installation
- Device plugin deployment
- GPU monitoring
- MIG configuration
