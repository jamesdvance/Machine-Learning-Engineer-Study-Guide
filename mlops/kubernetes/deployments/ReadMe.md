# Kubernetes Deployments

## Summary

Deployments manage the lifecycle of stateless applications in Kubernetes, ensuring the desired number of pod replicas are running and handling updates gracefully. For ML serving, Deployments provide the foundation for running model inference endpoints with high availability and rolling updates.

Key points to remember:

- Deployments manage ReplicaSets which manage Pods
- Rolling updates enable zero-downtime deployments
- Deployment strategies control how updates are applied
- Resource requests and limits ensure predictable scheduling
- Probes verify pod health and readiness

## Deployment Basics

### Deployment Specification

A basic ML model serving Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-model
  labels:
    app: fraud-model
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-model
  template:
    metadata:
      labels:
        app: fraud-model
        version: v1
    spec:
      containers:
        - name: model-server
          image: myregistry/fraud-model:v1.2.3
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          env:
            - name: MODEL_PATH
              value: /models/fraud_detector
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
```

### Key Components

Selector: Identifies which pods belong to this deployment

```yaml
selector:
  matchLabels:
    app: fraud-model
```

Template: Defines the pod specification

```yaml
template:
  metadata:
    labels:
      app: fraud-model  # Must match selector
```

Replicas: Desired number of pod instances

```yaml
replicas: 3
```

## Update Strategies

### Rolling Update

Default strategy; gradually replaces old pods with new:

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Extra pods during update
      maxUnavailable: 0  # Never reduce below desired
```

For ML models:
- `maxUnavailable: 0` ensures capacity is never reduced
- `maxSurge: 1` limits additional resource usage
- Gradual rollout allows monitoring for errors

### Recreate

Terminates all pods before creating new ones:

```yaml
spec:
  strategy:
    type: Recreate
```

Use when:
- New version is incompatible with old
- Shared resources cannot be accessed concurrently
- Downtime is acceptable

Avoid for production ML serving unless necessary.

## Resource Management

### Requests and Limits

Requests: Minimum resources guaranteed

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
```

Limits: Maximum resources allowed

```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "2"
```

For ML models:
- Set requests based on observed steady-state usage
- Set limits with headroom for peak loads
- Memory limits prevent OOM from affecting other pods

### GPU Resources

Request GPU resources:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

GPUs must be requested and limited equally; they cannot be shared.

## Health Probes

### Liveness Probe

Determines if container is alive:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60  # Time for model loading
  periodSeconds: 10
  failureThreshold: 3
```

If liveness fails, Kubernetes restarts the container.

For ML models:
- Allow sufficient initialDelay for model loading
- Check that inference engine is responsive
- Do not check model accuracy (that is readiness)

### Readiness Probe

Determines if container can receive traffic:

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 5
  failureThreshold: 3
```

If readiness fails, pod is removed from service endpoints.

For ML models:
- Verify model is loaded and warmed up
- Check dependencies (feature store, etc.) are accessible
- Can verify a sample prediction succeeds

### Startup Probe

For slow-starting containers:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8080
  failureThreshold: 30
  periodSeconds: 10
```

Use for large models that take minutes to load. Startup probe must succeed before liveness/readiness probes begin.

## Deployment Operations

### Creating and Updating

```bash
# Create deployment
kubectl apply -f deployment.yaml

# Update (triggers rolling update)
kubectl set image deployment/fraud-model \
    model-server=myregistry/fraud-model:v1.2.4

# Or edit directly
kubectl edit deployment fraud-model
```

### Monitoring Rollouts

```bash
# Watch rollout status
kubectl rollout status deployment/fraud-model

# View rollout history
kubectl rollout history deployment/fraud-model

# Pause rollout (for investigation)
kubectl rollout pause deployment/fraud-model

# Resume rollout
kubectl rollout resume deployment/fraud-model
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/fraud-model

# Rollback to specific revision
kubectl rollout undo deployment/fraud-model --to-revision=2
```

## ML-Specific Patterns

### Model Version Labels

Track model versions:

```yaml
metadata:
  labels:
    app: fraud-model
    model-version: v1.2.3
    model-date: "2024-01-15"
  annotations:
    model-uri: s3://models/fraud/v1.2.3/model.pt
    training-run: mlflow://runs/abc123
```

Labels enable version-based queries and routing.

### Sidecar Containers

Add supporting containers:

```yaml
spec:
  containers:
    - name: model-server
      image: myregistry/fraud-model:v1.2.3
      ports:
        - containerPort: 8080

    - name: logging-agent
      image: fluent/fluent-bit
      volumeMounts:
        - name: logs
          mountPath: /var/log

    - name: metrics-exporter
      image: prometheus/node-exporter
      ports:
        - containerPort: 9100
```

Common sidecars for ML:
- Logging agents
- Metrics exporters
- Feature retrieval proxies
- Authentication proxies

### Init Containers

Prepare before main container starts:

```yaml
spec:
  initContainers:
    - name: download-model
      image: amazon/aws-cli
      command:
        - aws
        - s3
        - cp
        - s3://models/fraud/v1.2.3/
        - /models/
        - --recursive
      volumeMounts:
        - name: model-storage
          mountPath: /models

  containers:
    - name: model-server
      image: myregistry/model-server:latest
      volumeMounts:
        - name: model-storage
          mountPath: /models
```

Use init containers to:
- Download models from object storage
- Run database migrations
- Wait for dependencies
- Prepare configuration

## Scaling

### Manual Scaling

```bash
kubectl scale deployment/fraud-model --replicas=5
```

### Declarative Scaling

Update deployment spec:

```yaml
spec:
  replicas: 5
```

### Horizontal Pod Autoscaler Integration

Deployments work with HPA:

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
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Best Practices

### Immutable Images

Never use :latest tags in production:

```yaml
# Bad
image: myregistry/fraud-model:latest

# Good
image: myregistry/fraud-model:v1.2.3
image: myregistry/fraud-model@sha256:abc123...
```

### Resource Sizing

Profile your model to determine resources:

```yaml
resources:
  requests:
    memory: "2Gi"    # Observed steady state
    cpu: "500m"
  limits:
    memory: "4Gi"    # Peak + buffer
    cpu: "2"
```

### Pod Disruption Budget

Protect availability during maintenance:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: fraud-model-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: fraud-model
```

### Anti-Affinity

Spread pods across nodes:

```yaml
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchLabels:
                app: fraud-model
            topologyKey: kubernetes.io/hostname
```

## Common Pitfalls

### Insufficient Probe Delays

Model loading takes time. Set initialDelaySeconds appropriately:

```yaml
livenessProbe:
  initialDelaySeconds: 120  # Allow 2 minutes for model load
```

### Missing Resource Limits

Without limits, pods can consume unbounded resources. Always set limits.

### Ignoring Graceful Shutdown

Handle SIGTERM properly in your model server:

```yaml
spec:
  terminationGracePeriodSeconds: 60  # Allow time to finish requests
```

### Too Aggressive Rolling Updates

Updating too fast can overwhelm the cluster:

```yaml
rollingUpdate:
  maxSurge: 25%      # Not too fast
  maxUnavailable: 0  # Maintain capacity
```
