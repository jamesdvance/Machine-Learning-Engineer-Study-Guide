# ConfigMaps and Secrets

## Summary

ConfigMaps and Secrets decouple configuration from container images, enabling the same image to run with different settings across environments. For ML systems, they store model paths, feature store endpoints, hyperparameters, and credentials without rebuilding images or exposing sensitive data in version control.

Key points to remember:

- ConfigMaps store non-sensitive configuration as key-value pairs or files
- Secrets store sensitive data with base64 encoding and access controls
- Both can be mounted as files or exposed as environment variables
- Changes to ConfigMaps/Secrets don't automatically restart pods
- Never commit Secrets to version control; use external secret management

## ConfigMaps

### Basic ConfigMap

Store configuration values:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  # Simple key-value pairs
  MODEL_NAME: fraud-detector
  MODEL_VERSION: v1.2.3
  BATCH_SIZE: "32"
  LOG_LEVEL: INFO

  # Multi-line configuration file
  config.yaml: |
    model:
      path: /models/fraud
      threshold: 0.85
    features:
      store_url: http://feature-store:8080
      timeout_ms: 100
```

### Using ConfigMaps as Environment Variables

Inject individual values:

```yaml
spec:
  containers:
    - name: model-server
      env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: model-config
              key: MODEL_NAME
        - name: BATCH_SIZE
          valueFrom:
            configMapKeyRef:
              name: model-config
              key: BATCH_SIZE
```

Inject all values at once:

```yaml
spec:
  containers:
    - name: model-server
      envFrom:
        - configMapRef:
            name: model-config
```

### Using ConfigMaps as Files

Mount configuration files:

```yaml
spec:
  containers:
    - name: model-server
      volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
  volumes:
    - name: config-volume
      configMap:
        name: model-config
        items:
          - key: config.yaml
            path: config.yaml
```

The file appears at `/app/config/config.yaml`.

### Creating ConfigMaps

From literal values:

```bash
kubectl create configmap model-config \
    --from-literal=MODEL_NAME=fraud-detector \
    --from-literal=BATCH_SIZE=32
```

From files:

```bash
kubectl create configmap model-config \
    --from-file=config.yaml \
    --from-file=features.json
```

From directories:

```bash
kubectl create configmap model-config \
    --from-file=./config/
```

## Secrets

### Basic Secret

Store sensitive data:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
type: Opaque
data:
  # Values must be base64 encoded
  API_KEY: c2VjcmV0LWFwaS1rZXktMTIz
  DB_PASSWORD: cGFzc3dvcmQxMjM=
```

Use stringData for plain text (encoded automatically):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
type: Opaque
stringData:
  API_KEY: secret-api-key-123
  DB_PASSWORD: password123
```

### Secret Types

Opaque (default):

```yaml
type: Opaque
```

Docker registry credentials:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: regcred
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-encoded-docker-config>
```

TLS certificates:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### Using Secrets as Environment Variables

```yaml
spec:
  containers:
    - name: model-server
      env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: model-secrets
              key: API_KEY
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: model-secrets
              key: DB_PASSWORD
```

### Using Secrets as Files

```yaml
spec:
  containers:
    - name: model-server
      volumeMounts:
        - name: secrets-volume
          mountPath: /app/secrets
          readOnly: true
  volumes:
    - name: secrets-volume
      secret:
        secretName: model-secrets
        defaultMode: 0400  # Read-only for owner
```

### Creating Secrets

From literal values:

```bash
kubectl create secret generic model-secrets \
    --from-literal=API_KEY=secret-api-key-123 \
    --from-literal=DB_PASSWORD=password123
```

From files:

```bash
kubectl create secret generic tls-secret \
    --from-file=tls.crt=./cert.pem \
    --from-file=tls.key=./key.pem
```

Docker registry secret:

```bash
kubectl create secret docker-registry regcred \
    --docker-server=myregistry.io \
    --docker-username=user \
    --docker-password=password
```

## ML Configuration Patterns

### Model Configuration

Separate model settings from code:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fraud-model-config
data:
  model.yaml: |
    model:
      name: fraud-detector
      version: v1.2.3
      path: /models/fraud
    inference:
      batch_size: 32
      max_sequence_length: 512
      threshold: 0.85
    preprocessing:
      normalize: true
      fill_missing: median
```

### Feature Store Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: feature-store-config
data:
  FEATURE_STORE_URL: http://feast-server:6566
  FEATURE_STORE_PROJECT: fraud_detection
  ONLINE_STORE_TYPE: redis

  features.yaml: |
    feature_views:
      - name: user_features
        ttl: 3600
      - name: transaction_features
        ttl: 300
```

### Environment-Specific Configuration

Development:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-dev
data:
  LOG_LEVEL: DEBUG
  ENABLE_PROFILING: "true"
  MODEL_PATH: /models/dev
```

Production:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-prod
data:
  LOG_LEVEL: INFO
  ENABLE_PROFILING: "false"
  MODEL_PATH: /models/prod
```

### API Keys and Credentials

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-credentials
type: Opaque
stringData:
  OPENAI_API_KEY: sk-...
  WANDB_API_KEY: ...
  AWS_ACCESS_KEY_ID: AKIA...
  AWS_SECRET_ACCESS_KEY: ...
  MLFLOW_TRACKING_TOKEN: ...
```

## External Secret Management

### AWS Secrets Manager

Using External Secrets Operator:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: model-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: model-secrets
  data:
    - secretKey: API_KEY
      remoteRef:
        key: prod/ml/api-keys
        property: api_key
```

### HashiCorp Vault

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: model-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: model-secrets
  data:
    - secretKey: DB_PASSWORD
      remoteRef:
        key: secret/data/ml/database
        property: password
```

### Sealed Secrets

Encrypt secrets for Git storage:

```bash
# Create sealed secret
kubeseal --format yaml < secret.yaml > sealed-secret.yaml
```

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: model-secrets
spec:
  encryptedData:
    API_KEY: AgB2...encrypted...
```

Only the cluster can decrypt sealed secrets.

## Configuration Updates

### Automatic Reloading

ConfigMaps mounted as volumes update automatically (with delay).

Watch for changes in application:

```python
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloader(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('config.yaml'):
            reload_config()

observer = Observer()
observer.schedule(ConfigReloader(), '/app/config', recursive=False)
observer.start()
```

### Rolling Restart on Config Change

Add checksum annotation to trigger redeployment:

```yaml
spec:
  template:
    metadata:
      annotations:
        checksum/config: {{ sha256sum .Values.config | quote }}
```

Or manually restart:

```bash
kubectl rollout restart deployment/fraud-model
```

### Immutable ConfigMaps

Prevent accidental changes:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config-v1
immutable: true
data:
  MODEL_VERSION: v1.2.3
```

Use versioned names and update deployment references.

## Best Practices

### Separate Concerns

```yaml
# Application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: INFO

---
# Model-specific configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_PATH: /models/fraud

---
# Credentials (Secret)
apiVersion: v1
kind: Secret
metadata:
  name: credentials
stringData:
  API_KEY: secret
```

### Use Namespaces for Isolation

```yaml
# Development namespace
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-dev
---
# Production namespace
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-prod
```

### Version Configuration

```yaml
metadata:
  name: model-config
  labels:
    app: fraud-model
    version: v1.2.3
    config-version: "2024-01-15"
```

### Limit Secret Access

Use RBAC to restrict secret access:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: model-secret-reader
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["model-secrets"]
    verbs: ["get"]
```

## Common Pitfalls

### Forgetting Base64 Encoding

```yaml
# Wrong: plain text in data field
data:
  API_KEY: my-secret-key

# Correct: base64 encoded
data:
  API_KEY: bXktc2VjcmV0LWtleQ==

# Or use stringData for automatic encoding
stringData:
  API_KEY: my-secret-key
```

### Environment Variable Naming

ConfigMap keys become environment variable names:

```yaml
# This creates invalid env var (hyphens not allowed)
data:
  model-path: /models  # Invalid as env var

# Use underscores instead
data:
  MODEL_PATH: /models  # Valid as env var
```

### Large ConfigMaps

ConfigMaps have a 1MB limit. For large configurations:
- Split into multiple ConfigMaps
- Use external storage and download at startup
- Consider using a configuration management system

### Secrets in Logs

Avoid logging secret values:

```python
# Bad: may log secret
logger.info(f"Connecting with key: {api_key}")

# Good: mask or omit
logger.info("Connecting to API with configured credentials")
```

### Not Rotating Secrets

Implement secret rotation:
- Use external secret managers with rotation
- Version secrets and update references
- Monitor secret age and usage
