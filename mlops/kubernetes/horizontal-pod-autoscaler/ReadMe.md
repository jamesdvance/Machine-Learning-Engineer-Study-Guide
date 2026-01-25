# Horizontal Pod Autoscaler

## Summary

Horizontal Pod Autoscaler (HPA) automatically adjusts the number of pod replicas based on observed metrics like CPU utilization, memory usage, or custom metrics. For ML serving, HPA enables cost-effective scaling by adding capacity during traffic spikes and reducing replicas during quiet periods.

Key points to remember:

- HPA scales Deployments, ReplicaSets, and StatefulSets horizontally
- Built-in metrics (CPU, memory) work out of the box with metrics-server
- Custom metrics enable scaling on inference latency, queue depth, or GPU utilization
- Scaling behavior can be tuned to prevent thrashing and ensure stability
- HPA works with Cluster Autoscaler to provision new nodes when needed

## HPA Basics

### Simple CPU-Based HPA

Scale based on CPU utilization:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

When average CPU exceeds 70%, HPA adds replicas. When it drops below, replicas decrease.

### Memory-Based Scaling

Scale on memory utilization:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Multiple Metrics

Combine CPU and memory:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

HPA scales to satisfy all metrics; the highest replica count wins.

## Custom Metrics for ML

### Prometheus Adapter Setup

Deploy prometheus-adapter to expose custom metrics:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
data:
  config.yaml: |
    rules:
      - seriesQuery: 'inference_latency_seconds{namespace!="",pod!=""}'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          matches: "^(.*)_seconds$"
          as: "${1}_seconds_avg"
        metricsQuery: 'avg(rate(<<.Series>>[2m])) by (<<.GroupBy>>)'
```

### Scaling on Inference Latency

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: inference_latency_seconds_avg
        target:
          type: AverageValue
          averageValue: 100m  # 100ms target latency
```

### Scaling on Request Queue Depth

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: pending_requests
        target:
          type: AverageValue
          averageValue: "10"  # Max 10 pending per pod
```

### Scaling on Requests Per Second

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"  # 100 RPS per pod
```

## External Metrics

### Scaling on Message Queue Length

Scale based on Kafka lag or SQS queue depth:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: batch-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: batch-inference
  minReplicas: 1
  maxReplicas: 50
  metrics:
    - type: External
      external:
        metric:
          name: sqs_queue_messages
          selector:
            matchLabels:
              queue: inference-requests
        target:
          type: AverageValue
          averageValue: "100"  # 100 messages per pod
```

### Scaling on Cloud Metrics

Using Stackdriver (GCP) metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: pubsub.googleapis.com|subscription|num_undelivered_messages
          selector:
            matchLabels:
              resource.labels.subscription_id: inference-requests
        target:
          type: AverageValue
          averageValue: "50"
```

## Scaling Behavior

### Controlling Scale-Up

Prevent aggressive scaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Wait before scaling up
      policies:
        - type: Percent
          value: 100  # Double at most
          periodSeconds: 60
        - type: Pods
          value: 4  # Add max 4 pods
          periodSeconds: 60
      selectPolicy: Min  # Use conservative policy
```

### Controlling Scale-Down

Prevent rapid scale-down:

```yaml
spec:
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
        - type: Percent
          value: 10  # Remove max 10%
          periodSeconds: 60
        - type: Pods
          value: 2  # Remove max 2 pods
          periodSeconds: 60
      selectPolicy: Min
```

### ML-Optimized Behavior

For ML serving with model warmup time:

```yaml
spec:
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300  # Slow scale down
      policies:
        - type: Pods
          value: 1  # One pod at a time
          periodSeconds: 120  # Every 2 minutes
```

## ML Scaling Patterns

### Predictive Scaling

Combine HPA with scheduled scaling for known traffic patterns:

```yaml
# Base HPA for reactive scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 5  # Updated by CronJob
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

CronJob to adjust minReplicas:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-up-morning
spec:
  schedule: "0 8 * * 1-5"  # 8 AM weekdays
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: kubectl
              image: bitnami/kubectl
              command:
                - kubectl
                - patch
                - hpa
                - fraud-model-hpa
                - --patch
                - '{"spec":{"minReplicas":10}}'
          restartPolicy: OnFailure
```

### Scaling with Model Warmup

Account for cold start time:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50  # Lower threshold for headroom
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600  # 10 min stabilization
```

### GPU Workload Scaling

For GPU-based inference:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-model
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "70"  # 70% GPU utilization
```

Requires DCGM exporter and prometheus-adapter configuration.

### Batch Processing Scale-to-Zero

For batch inference that can scale to zero:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: batch-processor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: batch-processor
  minReplicas: 0  # Scale to zero when idle
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: queue_length
        target:
          type: Value
          value: "1"  # Scale up when queue has items
```

Note: Scale-to-zero requires additional configuration or KEDA.

## KEDA Integration

### KEDA ScaledObject

KEDA provides advanced scaling triggers:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: fraud-model-scaler
spec:
  scaleTargetRef:
    name: fraud-model
  minReplicaCount: 0
  maxReplicaCount: 20
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: inference_requests_total
        threshold: "100"
        query: sum(rate(inference_requests_total{deployment="fraud-model"}[2m]))
```

### Kafka Trigger

Scale based on Kafka consumer lag:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-consumer-scaler
spec:
  scaleTargetRef:
    name: inference-consumer
  minReplicaCount: 1
  maxReplicaCount: 50
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: kafka:9092
        consumerGroup: inference-group
        topic: inference-requests
        lagThreshold: "100"
```

## Monitoring HPA

### Check HPA Status

```bash
# View HPA status
kubectl get hpa fraud-model-hpa

# Detailed status
kubectl describe hpa fraud-model-hpa

# Watch scaling events
kubectl get hpa fraud-model-hpa -w
```

### HPA Events

```bash
kubectl get events --field-selector involvedObject.name=fraud-model-hpa
```

### Metrics Availability

```bash
# Check if metrics are available
kubectl get --raw "/apis/metrics.k8s.io/v1beta1/pods" | jq .

# Custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq .
```

## Best Practices

### Set Appropriate Resource Requests

HPA calculations depend on resource requests:

```yaml
# Deployment
spec:
  containers:
    - resources:
        requests:
          cpu: "500m"  # HPA uses this for calculations
          memory: "1Gi"
```

Without requests, CPU utilization cannot be calculated.

### Use Multiple Metrics

Combine resource and custom metrics:

```yaml
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_latency_p99
      target:
        type: AverageValue
        averageValue: 200m  # 200ms
```

### Buffer for Warmup

Set lower utilization targets to account for warmup:

```yaml
# Instead of 80%, use 60% to allow headroom
target:
  type: Utilization
  averageUtilization: 60
```

### Test Scaling Behavior

Load test to verify scaling:

```bash
# Generate load
kubectl run load-generator --image=busybox --rm -it -- \
    /bin/sh -c "while true; do wget -q -O- http://fraud-model/predict; done"

# Watch HPA
kubectl get hpa -w
```

## Common Pitfalls

### Missing Metrics Server

HPA requires metrics-server for CPU/memory metrics:

```bash
# Check if metrics-server is running
kubectl get pods -n kube-system | grep metrics-server

# Install if missing
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Insufficient Resource Requests

```yaml
# Bad: no requests defined
containers:
  - name: model
    resources: {}  # HPA cannot calculate utilization

# Good: requests defined
containers:
  - name: model
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
```

### Thrashing

Pods scaling up and down rapidly:

```yaml
# Fix with stabilization windows
behavior:
  scaleUp:
    stabilizationWindowSeconds: 60
  scaleDown:
    stabilizationWindowSeconds: 300
```

### Ignoring Pod Startup Time

New pods need time to be ready:

```yaml
# Slow down scale-up for models with long initialization
behavior:
  scaleUp:
    policies:
      - type: Pods
        value: 2  # Add few pods at a time
        periodSeconds: 120  # Wait 2 min between scale-ups
```

### Conflicting with Cluster Autoscaler

HPA scales pods; Cluster Autoscaler scales nodes. Ensure:
- Node provisioning is faster than HPA stabilization
- Max replicas account for cluster capacity
- Resource requests match actual usage
