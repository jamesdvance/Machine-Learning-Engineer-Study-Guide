# GPU Scheduling in Kubernetes

## Summary

GPU scheduling in Kubernetes enables running ML workloads that require GPU acceleration for training and inference. The NVIDIA device plugin exposes GPUs as schedulable resources, while proper configuration ensures efficient utilization, isolation, and cost management of expensive GPU hardware.

Key points to remember:

- GPUs are scheduled as extended resources (`nvidia.com/gpu`) via device plugins
- GPUs cannot be shared between pods by default; each GPU is allocated entirely to one pod
- Node selectors and taints direct GPU workloads to appropriate nodes
- Time-slicing and MIG (Multi-Instance GPU) enable GPU sharing for smaller workloads
- GPU nodes are expensive; use node autoscaling and proper resource management

## GPU Device Plugin

### NVIDIA Device Plugin Installation

Deploy the NVIDIA device plugin DaemonSet:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

Or with Helm:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
    --namespace nvidia-device-plugin \
    --create-namespace
```

### Verify GPU Detection

Check that GPUs are discovered:

```bash
# View node GPU capacity
kubectl describe node gpu-node-1 | grep nvidia.com/gpu

# Should show:
# Capacity:
#   nvidia.com/gpu: 4
# Allocatable:
#   nvidia.com/gpu: 4
```

### Device Plugin Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false
        resources:
          - name: nvidia.com/gpu
            replicas: 4  # Allow 4 pods to share each GPU
```

## Requesting GPUs

### Basic GPU Request

Request GPU resources in pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-inference
spec:
  containers:
    - name: model-server
      image: myregistry/gpu-model:v1
      resources:
        limits:
          nvidia.com/gpu: 1  # Request 1 GPU
```

GPUs must be specified in limits, and requests equal limits automatically.

### Multiple GPUs

For distributed training or large models:

```yaml
spec:
  containers:
    - name: trainer
      image: myregistry/distributed-trainer:v1
      resources:
        limits:
          nvidia.com/gpu: 4  # Request 4 GPUs
      env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
```

### GPU in Deployments

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpu-model
  template:
    metadata:
      labels:
        app: gpu-model
    spec:
      containers:
        - name: model-server
          image: myregistry/gpu-model:v1
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
            limits:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: 1
          ports:
            - containerPort: 8080
```

## Node Selection

### Node Selectors

Target specific GPU node pools:

```yaml
spec:
  nodeSelector:
    gpu-type: nvidia-a100
  containers:
    - name: model-server
      resources:
        limits:
          nvidia.com/gpu: 1
```

### Node Affinity

More flexible GPU node selection:

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                  - NVIDIA-A100-SXM4-40GB
                  - NVIDIA-A100-SXM4-80GB
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - key: nvidia.com/gpu.memory
                operator: Gt
                values:
                  - "40000"  # Prefer GPUs with >40GB memory
```

### Taints and Tolerations

Prevent non-GPU workloads from running on GPU nodes:

Node taint:

```bash
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
```

Pod toleration:

```yaml
spec:
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: present
      effect: NoSchedule
  containers:
    - name: model-server
      resources:
        limits:
          nvidia.com/gpu: 1
```

## GPU Sharing

### Time-Slicing

Share GPU across multiple pods with time-slicing:

```yaml
# Device plugin config
version: v1
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 4  # 4 pods can share each physical GPU
```

Pods request the shared resource:

```yaml
spec:
  containers:
    - name: model-server
      resources:
        limits:
          nvidia.com/gpu: 1  # Gets 1/4 of a GPU
```

Time-slicing limitations:
- No memory isolation between pods
- All pods see full GPU memory
- Context switching overhead
- Best for inference, not training

### Multi-Instance GPU (MIG)

NVIDIA A100 and H100 support hardware partitioning:

```yaml
# Request specific MIG profile
spec:
  containers:
    - name: model-server
      resources:
        limits:
          nvidia.com/mig-1g.5gb: 1  # 1 GPU instance with 5GB
```

MIG profiles (A100 40GB):
- `nvidia.com/mig-1g.5gb`: 1/7 GPU, 5GB memory
- `nvidia.com/mig-2g.10gb`: 2/7 GPU, 10GB memory
- `nvidia.com/mig-3g.20gb`: 3/7 GPU, 20GB memory
- `nvidia.com/mig-7g.40gb`: Full GPU, 40GB memory

MIG advantages:
- Hardware-level isolation
- Guaranteed memory allocation
- Independent failure domains

### GPU Sharing Comparison

| Method | Isolation | Memory | Use Case |
|--------|-----------|--------|----------|
| Exclusive | Full | Full | Training |
| Time-slicing | None | Shared | Small inference |
| MIG | Hardware | Partitioned | Mixed workloads |

## ML Workload Patterns

### Training Jobs

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: trainer
          image: myregistry/trainer:v1
          resources:
            limits:
              nvidia.com/gpu: 8
              memory: "128Gi"
              cpu: "32"
          volumeMounts:
            - name: dataset
              mountPath: /data
            - name: checkpoints
              mountPath: /checkpoints
      volumes:
        - name: dataset
          persistentVolumeClaim:
            claimName: training-data
        - name: checkpoints
          persistentVolumeClaim:
            claimName: model-checkpoints
      nodeSelector:
        gpu-type: nvidia-a100
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### Inference Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 4
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
        - name: model-server
          image: myregistry/llm-server:v1
          resources:
            requests:
              memory: "32Gi"
              cpu: "8"
            limits:
              memory: "64Gi"
              cpu: "16"
              nvidia.com/gpu: 1
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 120  # Model loading time
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 180
            periodSeconds: 30
      nodeSelector:
        gpu-type: nvidia-a10g
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### Multi-GPU Distributed Training

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
spec:
  parallelism: 4  # 4 pods
  completions: 4
  template:
    spec:
      containers:
        - name: trainer
          image: myregistry/distributed-trainer:v1
          resources:
            limits:
              nvidia.com/gpu: 8  # 8 GPUs per pod = 32 total
          env:
            - name: WORLD_SIZE
              value: "32"
            - name: MASTER_ADDR
              value: "distributed-training-0"
            - name: MASTER_PORT
              value: "29500"
          command:
            - torchrun
            - --nproc_per_node=8
            - --nnodes=4
            - train.py
```

## GPU Monitoring

### DCGM Exporter

Deploy NVIDIA DCGM exporter for GPU metrics:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      containers:
        - name: dcgm-exporter
          image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
          ports:
            - containerPort: 9400
          securityContext:
            privileged: true
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
      nodeSelector:
        nvidia.com/gpu: "true"
```

### GPU Metrics

Key metrics from DCGM:
- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization percentage
- `DCGM_FI_DEV_MEM_COPY_UTIL`: Memory bandwidth utilization
- `DCGM_FI_DEV_FB_USED`: GPU memory used
- `DCGM_FI_DEV_FB_FREE`: GPU memory free
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature
- `DCGM_FI_DEV_POWER_USAGE`: Power consumption

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
```

## Node Autoscaling

### Cluster Autoscaler Configuration

Scale GPU node pools automatically:

```yaml
# GKE example
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
data:
  config: |
    {
      "nodePools": [
        {
          "name": "gpu-pool-a100",
          "minNodes": 0,
          "maxNodes": 10,
          "machineType": "a2-highgpu-8g"
        }
      ]
    }
```

### Karpenter for GPU Nodes

```yaml
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: gpu-provisioner
spec:
  requirements:
    - key: node.kubernetes.io/instance-type
      operator: In
      values:
        - p4d.24xlarge
        - p3.16xlarge
    - key: nvidia.com/gpu
      operator: Exists
  limits:
    resources:
      nvidia.com/gpu: 100  # Max 100 GPUs
  ttlSecondsAfterEmpty: 300  # Scale down after 5 min idle
  ttlSecondsUntilExpired: 86400  # Recycle nodes daily
```

## Cost Optimization

### Scale to Zero

For batch workloads, scale GPU nodes to zero:

```yaml
# Node pool configuration (cloud-specific)
minNodes: 0
maxNodes: 10
autoscaling: true
```

### Spot/Preemptible GPUs

Use spot instances for training:

```yaml
spec:
  nodeSelector:
    cloud.google.com/gke-spot: "true"
  tolerations:
    - key: cloud.google.com/gke-spot
      operator: Equal
      value: "true"
      effect: NoSchedule
```

Handle preemption gracefully:

```yaml
spec:
  containers:
    - name: trainer
      lifecycle:
        preStop:
          exec:
            command:
              - /bin/sh
              - -c
              - "save_checkpoint.py && sleep 30"
  terminationGracePeriodSeconds: 120
```

### Right-Sizing GPU Selection

Match GPU to workload:

| Workload | GPU | Memory | Cost Tier |
|----------|-----|--------|-----------|
| Small inference | T4 | 16GB | Low |
| Medium inference | A10G | 24GB | Medium |
| Large inference | A100 | 40-80GB | High |
| Training | A100, H100 | 40-80GB | High |

## Best Practices

### Use Resource Quotas

Limit GPU usage per namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-team
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
```

### Label GPU Nodes

```bash
kubectl label nodes gpu-node-1 \
    gpu-type=nvidia-a100 \
    gpu-memory=40gb \
    gpu-count=8
```

### Prioritize GPU Workloads

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-critical
value: 1000000
globalDefault: false
description: "Critical GPU workloads"
```

```yaml
spec:
  priorityClassName: gpu-critical
  containers:
    - name: model-server
      resources:
        limits:
          nvidia.com/gpu: 1
```

## Common Pitfalls

### Missing CUDA Libraries

Container must include CUDA libraries:

```dockerfile
# Use NVIDIA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Or install CUDA in existing image
RUN apt-get update && apt-get install -y cuda-toolkit-12-1
```

### GPU Memory OOM

Models exceeding GPU memory:

```python
# Check available memory before loading
import torch
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

### Incorrect Node Selection

Pods pending due to no matching nodes:

```bash
# Check why pod is pending
kubectl describe pod gpu-pod

# Common issues:
# - No nodes with requested GPU type
# - Insufficient GPU capacity
# - Missing tolerations
```

### Not Cleaning Up Failed Jobs

GPU resources held by failed pods:

```yaml
spec:
  ttlSecondsAfterFinished: 3600  # Clean up after 1 hour
  backoffLimit: 3
```

### Ignoring GPU Utilization

Monitor and right-size:

```bash
# Check GPU utilization in pod
kubectl exec -it gpu-pod -- nvidia-smi

# Prometheus query for underutilized GPUs
DCGM_FI_DEV_GPU_UTIL < 30
```
