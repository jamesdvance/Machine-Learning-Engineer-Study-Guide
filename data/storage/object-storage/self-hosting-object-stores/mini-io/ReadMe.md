# MinIO

## Summary

MinIO is a high-performance, S3-compatible object storage system designed for AI and ML workloads. It is the most widely deployed self-hosted S3 alternative, known for its complete API compatibility and sub-10ms latency on NVMe configurations. MinIO is the default choice for production ML infrastructure requiring on-premise object storage.

Key characteristics:

- **Performance**: Designed for high throughput; benchmarks show 325 GiB/s reads and 165 GiB/s writes on commodity hardware
- **S3 Compatibility**: Most complete S3 API implementation outside AWS, including S3 Select, versioning, and IAM policies
- **Deployment**: Kubernetes-native with operator, also runs on bare metal and VMs
- **License**: AGPLv3 for community edition; commercial license required for enterprise features
- **Scale**: From single-node development to exabyte-scale distributed deployments

When to use MinIO:

- High-performance training data storage requiring S3 compatibility
- Model serving with strict latency requirements
- Drop-in replacement for S3 in hybrid cloud architectures
- Kubernetes-native ML platforms

When to consider alternatives:

- Billions of tiny files (SeaweedFS may be more efficient)
- Enterprise unified storage needs (Ceph provides block/file/object)
- Minimal resource environments (Garage is lighter weight)
- Commercial use requires careful license consideration (AGPLv3)

Performance expectations for ML workloads:

| Configuration | Read Throughput | Write Throughput | Latency |
|---------------|-----------------|------------------|---------|
| Single node NVMe | 10-20 GB/s | 5-10 GB/s | <5ms |
| 4-node cluster | 30-60 GB/s | 15-30 GB/s | <10ms |
| 16-node cluster | 100+ GB/s | 50+ GB/s | <10ms |

---

## Architecture

### Core Components

MinIO's architecture is designed for simplicity and performance:

**MinIO Server**: The core process that handles S3 API requests, data storage, and cluster coordination. Each server manages a set of local drives and participates in distributed operations.

**Erasure Coding**: MinIO uses erasure coding rather than replication for data protection. The default configuration (EC:4) splits data into 12 data shards and 4 parity shards, allowing any 4 shards to be lost without data loss. This provides better storage efficiency than 3x replication while maintaining high durability.

**Distributed Mode**: In distributed deployments, MinIO forms a cluster where data is striped across all nodes. The cluster operates as a single namespace with consistent hashing for data placement.

### Deployment Models

**Standalone Mode**: Single server, single drive. Suitable for development and testing only. No redundancy.

```bash
minio server /data
```

**Distributed Mode**: Multiple servers with multiple drives each. Production configuration.

```bash
# 4-node cluster, 4 drives per node
minio server http://minio{1...4}/data{1...4}
```

**Kubernetes Operator**: Declarative deployment on Kubernetes with automatic healing, expansion, and upgrades.

```yaml
apiVersion: minio.min.io/v2
kind: Tenant
metadata:
  name: ml-storage
spec:
  pools:
    - servers: 4
      volumesPerServer: 4
      volumeClaimTemplate:
        spec:
          storageClassName: local-nvme
          resources:
            requests:
              storage: 1Ti
```

### Data Layout

MinIO stores objects directly on the filesystem with minimal metadata overhead:

```
/data/
  bucket-name/
    object-key           # Object data (or erasure coded shards)
    object-key.meta      # Metadata (optional)
```

For erasure-coded objects, shards are distributed across drives. The object key maps to a specific set of drives via consistent hashing.

## S3 API Compatibility

MinIO implements the complete S3 API, not a subset. This is critical for ML tooling compatibility.

### Supported Operations

**Bucket Operations:**
- CreateBucket, DeleteBucket, ListBuckets
- GetBucketLocation, GetBucketVersioning, PutBucketVersioning
- GetBucketPolicy, PutBucketPolicy, DeleteBucketPolicy
- GetBucketLifecycle, PutBucketLifecycle
- GetBucketNotification, PutBucketNotification

**Object Operations:**
- GetObject, PutObject, DeleteObject, CopyObject
- HeadObject, ListObjects, ListObjectsV2
- GetObjectTagging, PutObjectTagging
- Multipart uploads (CreateMultipartUpload, UploadPart, CompleteMultipartUpload)

**Advanced Features:**
- S3 Select (query CSV, JSON, Parquet in-place)
- Object locking and legal hold
- Server-side encryption (SSE-S3, SSE-C, SSE-KMS)
- Bucket and object versioning
- Lifecycle policies and transitions
- Replication (same-site and cross-site)

### Authentication and Authorization

MinIO uses AWS Signature Version 4 for authentication and implements IAM-compatible policies:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::training-data",
        "arn:aws:s3:::training-data/*"
      ]
    }
  ]
}
```

For ML workflows, typical configurations include:
- Service accounts for training jobs with read-only access to data buckets
- Write access to checkpoint and artifact buckets
- Admin access restricted to operations team

## Performance Optimization

### Hardware Configuration

**Drives**: NVMe SSDs provide the best performance. MinIO recommends dedicated drives (not shared with OS). For cost-sensitive deployments, HDDs work but with higher latency.

**Network**: 25 Gbps minimum per node for production; 100 Gbps for high-performance workloads. Network often becomes the bottleneck before storage.

**Memory**: 32 GB minimum, 128 GB recommended for large deployments. Memory is used for caching and connection handling.

**CPU**: Modern multi-core processors. MinIO uses SIMD instructions for erasure coding and checksums.

### Tuning for ML Workloads

**Large Sequential Reads (Training Data)**:
```bash
# Increase read-ahead for streaming workloads
echo 2048 > /sys/block/nvme0n1/queue/read_ahead_kb

# Tune network buffers
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
```

**High Concurrency (Many Workers)**:
```bash
# Increase connection limits
ulimit -n 1000000

# Enable connection pooling in clients
```

**Small Objects (Embeddings, Features)**:
```bash
# Use SSD caching tier
# Consider consolidating small files into tar archives
```

### Benchmarking

MinIO provides a built-in benchmarking tool:

```bash
# Write benchmark
mc admin speedtest minio-cluster --duration 60s

# Custom benchmark with warp
warp mixed --host minio-cluster:9000 \
    --access-key minioadmin \
    --secret-key minioadmin \
    --autoterm
```

Expected results on well-configured hardware:
- Single NVMe node: 5-10 GB/s
- 4-node NVMe cluster: 20-40 GB/s
- Network-limited at 100 Gbps: ~12 GB/s theoretical maximum

## ML Integration

### PyTorch DataLoader

```python
import boto3
from torch.utils.data import Dataset, DataLoader
import io

class MinIODataset(Dataset):
    def __init__(self, endpoint, bucket, prefix, access_key, secret_key):
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket = bucket

        # List all objects with prefix
        paginator = self.s3.get_paginator('list_objects_v2')
        self.keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            self.keys.extend([obj['Key'] for obj in page.get('Contents', [])])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        response = self.s3.get_object(Bucket=self.bucket, Key=self.keys[idx])
        data = response['Body'].read()
        # Parse and return data
        return self.parse(data)

# Usage
dataset = MinIODataset(
    endpoint='http://minio.internal:9000',
    bucket='training-data',
    prefix='images/',
    access_key='access_key',
    secret_key='secret_key'
)
loader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2)
```

### WebDataset Integration

For large-scale training, WebDataset with MinIO provides efficient streaming:

```python
import webdataset as wds
import os

# Configure S3 credentials for webdataset
os.environ['AWS_ACCESS_KEY_ID'] = 'access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'secret_key'
os.environ['S3_ENDPOINT_URL'] = 'http://minio.internal:9000'

# Stream sharded data
dataset = wds.WebDataset(
    's3://training-data/shards/shard-{0000..0999}.tar',
    shardshuffle=True
).shuffle(1000).decode('pil').to_tuple('jpg', 'cls')

loader = wds.WebLoader(dataset, batch_size=64, num_workers=8)
```

### Checkpoint Storage

```python
import torch
import boto3
from io import BytesIO

class MinIOCheckpointer:
    def __init__(self, endpoint, bucket, access_key, secret_key):
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket = bucket

    def save(self, state_dict, path):
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, path)

    def load(self, path):
        buffer = BytesIO()
        self.s3.download_fileobj(self.bucket, path, buffer)
        buffer.seek(0)
        return torch.load(buffer)

# Usage
checkpointer = MinIOCheckpointer(
    'http://minio.internal:9000',
    'checkpoints',
    'access_key',
    'secret_key'
)
checkpointer.save(model.state_dict(), 'model_epoch_10.pt')
```

### fsspec and Pandas

```python
import pandas as pd
import pyarrow.parquet as pq

storage_options = {
    'endpoint_url': 'http://minio.internal:9000',
    'key': 'access_key',
    'secret': 'secret_key'
}

# Read parquet directly from MinIO
df = pd.read_parquet(
    's3://analytics/features.parquet',
    storage_options=storage_options
)

# Write results back
df.to_parquet(
    's3://analytics/processed.parquet',
    storage_options=storage_options
)
```

## Deployment

### Kubernetes Deployment

The MinIO Operator provides production-grade Kubernetes deployment:

```bash
# Install operator
kubectl apply -k github.com/minio/operator

# Create tenant
kubectl apply -f - <<EOF
apiVersion: minio.min.io/v2
kind: Tenant
metadata:
  name: ml-storage
  namespace: minio
spec:
  image: minio/minio:latest
  pools:
    - servers: 4
      volumesPerServer: 4
      volumeClaimTemplate:
        spec:
          storageClassName: local-nvme
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 1Ti
  requestAutoCert: true
  users:
    - name: ml-user-secret
  configuration:
    name: ml-storage-config
EOF
```

### Docker Compose (Development)

```yaml
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  minio-data:
```

### Bare Metal Production

```bash
# Create systemd service
cat > /etc/systemd/system/minio.service <<EOF
[Unit]
Description=MinIO
Documentation=https://docs.min.io
Wants=network-online.target
After=network-online.target

[Service]
User=minio-user
Group=minio-user
EnvironmentFile=/etc/default/minio
ExecStart=/usr/local/bin/minio server \$MINIO_VOLUMES \$MINIO_OPTS
Restart=always
LimitNOFILE=1000000

[Install]
WantedBy=multi-user.target
EOF

# Configure environment
cat > /etc/default/minio <<EOF
MINIO_VOLUMES="http://minio{1...4}.internal/data{1...4}"
MINIO_OPTS="--console-address :9001"
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=changeme
EOF

systemctl enable --now minio
```

## Data Protection

### Erasure Coding

MinIO uses erasure coding by default. For a 16-drive setup:
- 12 data shards + 4 parity shards
- Can tolerate loss of any 4 drives
- Storage efficiency: 75% (vs 33% for 3x replication)

Custom erasure coding configuration:

```bash
# Set erasure code parity (default is calculated automatically)
minio server /data{1...16} --erasure-set-drive-count 16
```

### Bitrot Protection

MinIO includes automatic bitrot detection and healing:
- HighwayHash checksums on all data
- Background scanner detects corruption
- Automatic healing from parity shards

### Replication

Site replication for disaster recovery:

```bash
# Configure replication between two clusters
mc admin replicate add minio1 minio2
```

Bucket replication for granular control:

```bash
# Replicate specific bucket
mc replicate add minio1/training-data \
    --remote-bucket training-data \
    --remote-target minio2
```

## Monitoring

### Prometheus Metrics

MinIO exposes Prometheus metrics on the `/minio/v2/metrics` endpoint:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: minio
    metrics_path: /minio/v2/metrics/cluster
    scheme: http
    static_configs:
      - targets: ['minio.internal:9000']
```

Key metrics to monitor:
- `minio_cluster_capacity_raw_total_bytes`: Total raw capacity
- `minio_cluster_capacity_usable_free_bytes`: Available space
- `minio_s3_requests_total`: Request counts by operation
- `minio_s3_requests_errors_total`: Error counts
- `minio_node_drive_health`: Drive health status

### Alerting

Essential alerts:

```yaml
groups:
  - name: minio
    rules:
      - alert: MinIODiskSpace
        expr: minio_cluster_capacity_usable_free_bytes / minio_cluster_capacity_raw_total_bytes < 0.2
        for: 5m
        annotations:
          summary: MinIO cluster below 20% free space

      - alert: MinIODriveOffline
        expr: minio_node_drive_health == 0
        for: 5m
        annotations:
          summary: MinIO drive offline

      - alert: MinIOHighErrorRate
        expr: rate(minio_s3_requests_errors_total[5m]) > 10
        for: 5m
        annotations:
          summary: High S3 error rate
```

## Comparison with Alternatives

### MinIO vs SeaweedFS

| Aspect | MinIO | SeaweedFS |
|--------|-------|-----------|
| S3 Compatibility | Excellent, most complete | Good, covers common operations |
| Small Files | Less efficient | Optimized (40 bytes overhead) |
| Performance | Higher throughput | Lower latency for small files |
| Complexity | Medium | Medium |
| Community | Larger | Smaller but active |

Choose MinIO for S3 compatibility; SeaweedFS for billions of small files.

### MinIO vs Ceph

| Aspect | MinIO | Ceph |
|--------|-------|------|
| Focus | Object storage only | Unified (block/file/object) |
| Complexity | Medium | High |
| Minimum Scale | 4 nodes | 3+ nodes, 10+ OSDs |
| Enterprise Support | Commercial license | Red Hat, IBM |
| Performance | Higher for objects | Variable by interface |

Choose MinIO for dedicated object storage; Ceph for unified storage needs.

### MinIO vs Cloud S3

| Aspect | MinIO | AWS S3 |
|--------|-------|--------|
| Cost at Scale | Lower (self-managed) | Higher (pay-per-GB) |
| Latency | Lower (co-located) | Higher (network) |
| Ops Overhead | Higher | None |
| Features | Complete | Complete + extras |
| Global Distribution | Manual | Built-in |

Choose MinIO for cost at scale; S3 for simplicity and global reach.

## Best Practices

### Production Configuration

1. **Use distributed mode**: Minimum 4 nodes with 4 drives each
2. **Enable TLS**: Encrypt all traffic in transit
3. **Configure IAM**: Use service accounts with minimal permissions
4. **Set up monitoring**: Prometheus metrics and alerting
5. **Plan capacity**: Maintain 20-30% free space buffer
6. **Test recovery**: Regularly verify backup and restore procedures

### ML-Specific Recommendations

1. **Consolidate small files**: Package into tar archives or TFRecords
2. **Use parallel reads**: Configure DataLoader with multiple workers
3. **Cache locally**: Use local SSD cache for repeated access
4. **Separate buckets**: Training data, checkpoints, and artifacts in separate buckets
5. **Lifecycle policies**: Automatically expire old checkpoints

### Common Pitfalls

1. **Single drive mode in production**: No redundancy, data loss on drive failure
2. **Insufficient network bandwidth**: Storage faster than network
3. **Ignoring bitrot**: Disable healing at your peril
4. **No monitoring**: Silent failures until data loss
5. **Mixed drive sizes**: Reduces effective capacity
