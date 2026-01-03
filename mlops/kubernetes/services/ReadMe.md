# Kubernetes Services

## Summary

Services provide stable networking for pods, abstracting the dynamic nature of pod IPs behind consistent endpoints. For ML serving, Services expose model endpoints to clients, enable load balancing across replicas, and provide service discovery within the cluster.

Key points to remember:

- Services provide stable IPs and DNS names for pod groups
- ClusterIP for internal access, NodePort and LoadBalancer for external access
- Labels and selectors connect Services to Pods
- Ingress controllers route external HTTP traffic
- Service mesh adds advanced traffic management

## Service Types

### ClusterIP

Default type; accessible only within cluster:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-model
spec:
  type: ClusterIP
  selector:
    app: fraud-model
  ports:
    - port: 80
      targetPort: 8080
```

Use ClusterIP for:
- Internal model services
- Inter-service communication
- Feature stores
- Model registries

Access within cluster:
- `http://fraud-model` (same namespace)
- `http://fraud-model.ml-serving` (different namespace)
- `http://fraud-model.ml-serving.svc.cluster.local` (FQDN)

### NodePort

Exposes service on each node's IP:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-model
spec:
  type: NodePort
  selector:
    app: fraud-model
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080  # 30000-32767
```

Access via `<NodeIP>:30080`. Useful for development and testing but not ideal for production.

### LoadBalancer

Provisions cloud load balancer:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-model
  annotations:
    # AWS-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
spec:
  type: LoadBalancer
  selector:
    app: fraud-model
  ports:
    - port: 80
      targetPort: 8080
```

Use for production external access. Cloud-specific annotations configure load balancer behavior.

### ExternalName

Maps service to external DNS name:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-feature-store
spec:
  type: ExternalName
  externalName: feature-store.company.com
```

Useful for referencing external services with Kubernetes DNS.

## Ingress

### Ingress Resource

Routes HTTP traffic to services:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - models.company.com
      secretName: tls-secret
  rules:
    - host: models.company.com
      http:
        paths:
          - path: /fraud
            pathType: Prefix
            backend:
              service:
                name: fraud-model
                port:
                  number: 80
          - path: /recommendation
            pathType: Prefix
            backend:
              service:
                name: recommendation-model
                port:
                  number: 80
```

### Path-Based Routing

Route to different models:

```yaml
rules:
  - http:
      paths:
        - path: /v1/fraud
          pathType: Prefix
          backend:
            service:
              name: fraud-model-v1
              port:
                number: 80
        - path: /v2/fraud
          pathType: Prefix
          backend:
            service:
              name: fraud-model-v2
              port:
                number: 80
```

### Host-Based Routing

Different hosts for different environments:

```yaml
rules:
  - host: staging.models.company.com
    http:
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: models-staging
              port:
                number: 80
  - host: models.company.com
    http:
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: models-production
              port:
                number: 80
```

## Load Balancing

### Default Round-Robin

Services distribute traffic evenly across pods by default.

### Session Affinity

Stick clients to specific pods:

```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
```

Useful when model state is cached per-pod, but prefer stateless designs.

### External Traffic Policy

Control source IP preservation:

```yaml
spec:
  externalTrafficPolicy: Local  # Preserve client IP
```

`Local` preserves source IP but may cause uneven load distribution.

## Service Discovery

### DNS-Based Discovery

Kubernetes provides DNS for services:

```python
# Access service from within cluster
import requests

# Same namespace
response = requests.post("http://fraud-model/predict", json=data)

# Different namespace
response = requests.post("http://fraud-model.ml-serving/predict", json=data)
```

### Environment Variables

Kubernetes injects service info:

```bash
FRAUD_MODEL_SERVICE_HOST=10.0.0.42
FRAUD_MODEL_SERVICE_PORT=80
```

Avoid environment variables for dynamic service discovery; prefer DNS.

## gRPC Services

### gRPC Model Serving

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-model-grpc
spec:
  type: ClusterIP
  selector:
    app: fraud-model
  ports:
    - name: grpc
      port: 50051
      targetPort: 50051
      protocol: TCP
```

### gRPC Ingress

Requires gRPC-compatible ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grpc-ingress
  annotations:
    nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
spec:
  rules:
    - host: grpc.models.company.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: fraud-model-grpc
                port:
                  number: 50051
```

## Headless Services

For direct pod access without load balancing:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-pods
spec:
  clusterIP: None  # Headless
  selector:
    app: model-server
  ports:
    - port: 8080
```

DNS returns all pod IPs. Use for:
- StatefulSets
- Custom load balancing
- Peer discovery in distributed systems

## ML Service Patterns

### A/B Testing with Services

Traffic splitting between model versions:

```yaml
# Version A Service
apiVersion: v1
kind: Service
metadata:
  name: fraud-model-a
spec:
  selector:
    app: fraud-model
    version: a
  ports:
    - port: 80
---
# Version B Service
apiVersion: v1
kind: Service
metadata:
  name: fraud-model-b
spec:
  selector:
    app: fraud-model
    version: b
  ports:
    - port: 80
```

Use Ingress or service mesh for traffic splitting.

### Canary with Weighted Services

Using Istio for traffic splitting:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fraud-model
spec:
  hosts:
    - fraud-model
  http:
    - route:
        - destination:
            host: fraud-model
            subset: v1
          weight: 90
        - destination:
            host: fraud-model
            subset: v2
          weight: 10
```

### Multi-Model Endpoint

Single service routing to multiple models:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-router
spec:
  selector:
    app: model-router
  ports:
    - port: 80
```

The router pod handles model selection logic.

## Health Checks

### Service-Level Checks

Services don't check pod health directly; they rely on endpoints:

```bash
kubectl get endpoints fraud-model
```

Only pods passing readiness probes appear in endpoints.

### External Health Checks

Cloud load balancers perform health checks:

```yaml
metadata:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-path: /health
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-interval: "10"
```

## Best Practices

### Use Descriptive Names

```yaml
metadata:
  name: fraud-detection-model-v2
  labels:
    app: fraud-detection
    version: v2
    team: ml-platform
```

### Document Ports

```yaml
ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  - name: grpc
    port: 50051
    targetPort: 50051
```

### Separate Internal and External Services

```yaml
# Internal service
apiVersion: v1
kind: Service
metadata:
  name: fraud-model-internal
spec:
  type: ClusterIP
---
# External service (different configuration)
apiVersion: v1
kind: Service
metadata:
  name: fraud-model-external
spec:
  type: LoadBalancer
```

## Common Pitfalls

### Selector Mismatches

Service selector must match pod labels:

```yaml
# Service
selector:
  app: fraud-model  # Must match pod labels

# Pod (in Deployment)
labels:
  app: fraud-model  # Must match service selector
```

### Port Confusion

```yaml
ports:
  - port: 80         # Service port (what clients use)
    targetPort: 8080 # Container port (where app listens)
```

### Missing Readiness Probes

Without readiness probes, traffic goes to unready pods. Always configure readiness probes on deployments.

### Ignoring DNS Caching

Service discovery can cache DNS. Handle connection failures gracefully in client code.
