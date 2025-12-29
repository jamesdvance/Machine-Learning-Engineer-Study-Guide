# Garage

## Summary

Garage is a lightweight, geo-distributed S3-compatible object storage system designed for self-hosting on minimal hardware. Unlike enterprise solutions that assume datacenter environments, Garage is built to operate reliably across unreliable networks and heterogeneous hardware. It ships as a single binary with sensible defaults, making it accessible to operators without distributed systems expertise.

Key characteristics:

- **Deployment**: Single binary, no external dependencies
- **Hardware**: Runs on minimal resources (1 GB RAM, ARM or x86)
- **Network**: Designed for high-latency, unreliable connections
- **Replication**: 3-zone redundancy by default
- **License**: AGPLv3, fully open source
- **Scale**: Small to medium deployments (not exabyte-scale)

When to use Garage:

- Geo-distributed storage across multiple locations
- Edge deployments with limited resources
- Self-hosting on heterogeneous or older hardware
- Backup and archival across unreliable networks
- Organizations prioritizing operational simplicity

When to consider alternatives:

- High-performance training workloads (MinIO)
- Enterprise support requirements (Ceph, MinIO commercial)
- Exabyte-scale storage (Ceph)
- Billions of small files (SeaweedFS)

Resource requirements:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86_64 (10+ years), ARMv7/v8 | Any modern processor |
| Memory | 1 GB | 2-4 GB |
| Storage | 16 GB | Based on data volume |
| Network | 50 Mbps, <200ms latency | Higher bandwidth preferred |

---

## Architecture

### Design Philosophy

Garage was created by Deuxfleurs, a French non-profit focused on digital autonomy. The design prioritizes:

1. **Operational simplicity**: Single binary, configuration file, done
2. **Resilience over performance**: Handles network partitions gracefully
3. **Heterogeneous hardware**: No requirement for uniform nodes
4. **Geographic distribution**: First-class support for multi-site deployments

### Core Components

Garage runs as a single process handling all functionality:

```
+------------------------------------------+
|              Garage Node                  |
|                                          |
|  +------------+  +------------+          |
|  |   S3 API   |  |  Admin API |          |
|  +-----+------+  +-----+------+          |
|        |               |                 |
|  +-----+---------------+------+          |
|  |      Request Handler       |          |
|  +------------+---------------+          |
|               |                          |
|  +------------+---------------+          |
|  |     Replication Engine     |          |
|  +------------+---------------+          |
|               |                          |
|  +------------+---------------+          |
|  |       Data Storage         |          |
|  +----------------------------+          |
+------------------------------------------+
```

Each node is identical; there are no specialized roles. Nodes discover each other, share cluster state, and coordinate replication automatically.

### Zone-Based Replication

Garage uses zones (not replicas) as the unit of redundancy:

```
Zone A (US-East)     Zone B (EU-West)     Zone C (Asia)
+-------------+      +-------------+      +-------------+
|   Node 1    |      |   Node 3    |      |   Node 5    |
|   Node 2    |      |   Node 4    |      |   Node 6    |
+-------------+      +-------------+      +-------------+
       |                   |                   |
       +-------------------+-------------------+
                    Data replicated
                    across all 3 zones
```

Default configuration: 3 zones, each object stored in all 3 zones. Cluster remains available if any single zone fails.

### Consistency Model

Garage uses a Dynamo-style consistency model:

- **Writes**: Succeed when a quorum of zones acknowledge
- **Reads**: May return slightly stale data under partition
- **Eventual consistency**: All zones converge within seconds normally

For ML workloads, this means:
- Checkpoints may not be immediately visible across all zones
- Use consistent reads for critical operations
- Suitable for data that tolerates brief inconsistency

### Data Layout

Data is stored using content-addressed blocks:

```
/data/
  objects/
    <bucket>/<key>           # Object metadata
  blocks/
    <hash_prefix>/
      <block_hash>           # Deduplicated data blocks
```

Benefits:
- Automatic deduplication of identical blocks
- Efficient storage of similar objects
- Simplified garbage collection

## S3 Compatibility

Garage implements a subset of the S3 API sufficient for most self-hosting use cases.

### Supported Operations

| Operation | Support | Notes |
|-----------|---------|-------|
| GetObject | Yes | Full support |
| PutObject | Yes | Full support |
| DeleteObject | Yes | Full support |
| HeadObject | Yes | Full support |
| ListObjectsV2 | Yes | Full support |
| CopyObject | Yes | Full support |
| CreateBucket | Yes | Via admin API or S3 |
| DeleteBucket | Yes | Via admin API or S3 |
| Multipart Upload | Yes | For large objects |
| Presigned URLs | Yes | GET and PUT |
| Bucket Versioning | No | Not implemented |
| Object Lock | No | Not implemented |
| S3 Select | No | Not implemented |
| Lifecycle Policies | No | Not implemented |

### Authentication

Garage uses S3-compatible access keys:

```bash
# Create access key via admin API
garage key create my-application

# Returns:
# Key ID: GK...
# Secret: ...
```

Or via configuration:

```toml
# garage.toml
[[s3_api.keys]]
key_id = "GK..."
secret_key = "..."
bucket = "*"  # Access to all buckets
```

### Limitations vs Full S3

For ML workloads, notable missing features:

1. **No versioning**: Cannot maintain multiple versions of objects
2. **No lifecycle policies**: Manual cleanup required
3. **No S3 Select**: Cannot query within objects
4. **Limited bucket policies**: Basic ACLs only

Workarounds:
- Implement versioning in object keys (`model_v001.pt`, `model_v002.pt`)
- External scripts for lifecycle management
- Download and process locally instead of S3 Select

## Deployment

### Single-Node Development

```bash
# Download and run
curl -LO https://garagehq.deuxfleurs.fr/releases/garage-linux-amd64
chmod +x garage-linux-amd64

# Create config
cat > garage.toml <<EOF
metadata_dir = "/var/lib/garage/meta"
data_dir = "/var/lib/garage/data"

replication_mode = "none"  # Single node

[s3_api]
api_bind_addr = "[::]:3900"
s3_region = "garage"

[admin]
api_bind_addr = "[::]:3903"
EOF

# Run
./garage-linux-amd64 -c garage.toml server
```

### Multi-Zone Production

Minimum production: 3 nodes across 3 zones.

**Node 1 (Zone A):**
```toml
metadata_dir = "/var/lib/garage/meta"
data_dir = "/var/lib/garage/data"

replication_mode = "3"

[rpc_secret]
# Generate with: openssl rand -hex 32
secret = "abc123..."

[s3_api]
api_bind_addr = "[::]:3900"
s3_region = "garage"

[admin]
api_bind_addr = "[::]:3903"

[consul_discovery]
# Or use static bootstrap
```

**Bootstrap cluster:**
```bash
# On first node, get node ID
garage node id

# On other nodes, connect to first
garage node connect <node1_id>@<node1_ip>:3901

# Assign zones
garage layout assign <node1_id> --zone=zone-a --capacity=1T
garage layout assign <node2_id> --zone=zone-b --capacity=1T
garage layout assign <node3_id> --zone=zone-c --capacity=1T

# Apply layout
garage layout apply --version 1
```

### Docker Compose

```yaml
version: '3.8'

services:
  garage1:
    image: dxflrs/garage:v0.9.0
    volumes:
      - ./garage1.toml:/etc/garage.toml
      - garage1-meta:/var/lib/garage/meta
      - garage1-data:/var/lib/garage/data
    ports:
      - "3901:3901"  # RPC
      - "3900:3900"  # S3 API
      - "3903:3903"  # Admin API
    command: /garage server

  garage2:
    image: dxflrs/garage:v0.9.0
    volumes:
      - ./garage2.toml:/etc/garage.toml
      - garage2-meta:/var/lib/garage/meta
      - garage2-data:/var/lib/garage/data
    ports:
      - "3911:3901"
      - "3910:3900"
      - "3913:3903"
    command: /garage server

  garage3:
    image: dxflrs/garage:v0.9.0
    volumes:
      - ./garage3.toml:/etc/garage.toml
      - garage3-meta:/var/lib/garage/meta
      - garage3-data:/var/lib/garage/data
    ports:
      - "3921:3901"
      - "3920:3900"
      - "3923:3903"
    command: /garage server

volumes:
  garage1-meta:
  garage1-data:
  garage2-meta:
  garage2-data:
  garage3-meta:
  garage3-data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: garage
spec:
  serviceName: garage
  replicas: 3
  selector:
    matchLabels:
      app: garage
  template:
    metadata:
      labels:
        app: garage
    spec:
      containers:
        - name: garage
          image: dxflrs/garage:v0.9.0
          ports:
            - containerPort: 3900
              name: s3
            - containerPort: 3901
              name: rpc
            - containerPort: 3903
              name: admin
          volumeMounts:
            - name: config
              mountPath: /etc/garage.toml
              subPath: garage.toml
            - name: data
              mountPath: /var/lib/garage
      volumes:
        - name: config
          configMap:
            name: garage-config
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: garage-s3
spec:
  selector:
    app: garage
  ports:
    - port: 3900
      name: s3
  type: ClusterIP
```

## ML Integration

### Basic S3 Access

```python
import boto3

def get_garage_client(endpoint, access_key, secret_key):
    """Create Garage S3 client."""
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='garage'  # Required for signature
    )

client = get_garage_client(
    'http://garage.internal:3900',
    'GK...',
    'secret...'
)

# Standard operations
client.upload_file('model.pt', 'models', 'latest/model.pt')
client.download_file('models', 'latest/model.pt', 'local_model.pt')
```

### Training Data Access

```python
from torch.utils.data import Dataset, DataLoader
import boto3

class GarageDataset(Dataset):
    """Dataset backed by Garage storage."""

    def __init__(self, endpoint, bucket, prefix, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='garage'
        )
        self.bucket = bucket

        # List all objects
        paginator = self.client.get_paginator('list_objects_v2')
        self.keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            self.keys.extend([obj['Key'] for obj in page.get('Contents', [])])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        response = self.client.get_object(Bucket=self.bucket, Key=self.keys[idx])
        data = response['Body'].read()
        return self.parse(data)

    def parse(self, data):
        # Parse binary data
        pass

# Usage
dataset = GarageDataset(
    endpoint='http://garage:3900',
    bucket='training-data',
    prefix='images/',
    access_key='GK...',
    secret_key='...'
)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Checkpoint Storage with Manual Versioning

Since Garage lacks native versioning, implement it in keys:

```python
import torch
from datetime import datetime
import boto3

class GarageCheckpointer:
    """Checkpoint manager with manual versioning for Garage."""

    def __init__(self, endpoint, bucket, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='garage'
        )
        self.bucket = bucket

    def save(self, state_dict, run_id, step):
        """Save checkpoint with timestamp-based versioning."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        key = f'checkpoints/{run_id}/step_{step:08d}_{timestamp}.pt'

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        self.client.upload_fileobj(buffer, self.bucket, key)
        return key

    def load_latest(self, run_id):
        """Load most recent checkpoint for run."""
        prefix = f'checkpoints/{run_id}/'
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return None

        # Sort by key (timestamp ensures ordering)
        latest = sorted(
            response['Contents'],
            key=lambda x: x['Key'],
            reverse=True
        )[0]

        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, latest['Key'], buffer)
        buffer.seek(0)
        return torch.load(buffer)

    def cleanup(self, run_id, keep_last=5):
        """Remove old checkpoints."""
        prefix = f'checkpoints/{run_id}/'
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return

        checkpoints = sorted(
            response['Contents'],
            key=lambda x: x['Key'],
            reverse=True
        )

        for obj in checkpoints[keep_last:]:
            self.client.delete_object(Bucket=self.bucket, Key=obj['Key'])
```

### Backup and Sync

Garage integrates well with standard backup tools:

```bash
# Sync local directory to Garage
rclone sync /local/data garage:bucket/prefix \
    --s3-endpoint=http://garage:3900 \
    --s3-access-key-id=GK... \
    --s3-secret-access-key=...

# Backup to Garage with encryption
restic -r s3:http://garage:3900/backups init
restic -r s3:http://garage:3900/backups backup /data/models
```

## Operations

### Cluster Management

```bash
# View cluster status
garage status

# View node information
garage node id
garage node info

# View layout
garage layout show

# Add capacity to node
garage layout assign <node_id> --zone=zone-a --capacity=2T
garage layout apply --version 2
```

### Bucket Management

```bash
# Create bucket
garage bucket create ml-data

# Grant access
garage bucket allow ml-data --key=GK... --read --write

# List buckets
garage bucket list

# View bucket info
garage bucket info ml-data
```

### Key Management

```bash
# Create key
garage key create training-job

# List keys
garage key list

# View key info
garage key info GK...
```

### Monitoring

Garage exposes Prometheus metrics:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: garage
    static_configs:
      - targets: ['garage1:3903', 'garage2:3903', 'garage3:3903']
    metrics_path: /metrics
```

Key metrics:
- `garage_cluster_nodes`: Number of nodes
- `garage_data_bytes`: Data stored per node
- `garage_api_request_duration_seconds`: S3 API latency
- `garage_rpc_request_duration_seconds`: Inter-node RPC latency
- `garage_block_resync_queue_length`: Pending resyncs

### Health Checks

```bash
# Check cluster health
curl http://garage:3903/health

# Check S3 availability
aws s3 ls --endpoint-url=http://garage:3900 s3://

# Check specific bucket
aws s3 ls --endpoint-url=http://garage:3900 s3://ml-data/
```

## Geo-Distribution Patterns

### Multi-Region Deployment

```
US-East           EU-West           Asia-Pacific
(Zone A)          (Zone B)          (Zone C)
+--------+        +--------+        +--------+
| Garage |<------>| Garage |<------>| Garage |
| Node 1 |  WAN   | Node 2 |  WAN   | Node 3 |
+--------+        +--------+        +--------+
    |                 |                 |
+---v---+         +---v---+         +---v---+
| Local |         | Local |         | Local |
| Disks |         | Disks |         | Disks |
+-------+         +-------+         +-------+

Replication: All data in all 3 zones
Read: From nearest zone
Write: Quorum across zones
```

### Edge Deployment

```
        Central Site
        +----------+
        |  Garage  |
        |  (zone-a)|
        +----+-----+
             |
     +-------+-------+
     |               |
+----v----+     +----v----+
|  Edge 1 |     |  Edge 2 |
| (zone-b)|     | (zone-c)|
+---------+     +---------+

Edge sites:
- Minimal hardware (Raspberry Pi, old laptops)
- Local data access for edge training
- Eventual consistency with central
```

## Comparison with Alternatives

### Garage vs MinIO

| Aspect | Garage | MinIO |
|--------|--------|-------|
| Target | Geo-distributed, edge | Datacenter, performance |
| Resources | Minimal (1GB RAM) | Higher (32GB+ recommended) |
| Performance | Moderate | High |
| S3 Compatibility | Basic | Complete |
| Operations | Simple | Moderate |
| Scale | Small-medium | Medium-large |

Choose Garage for geo-distribution and simplicity; MinIO for performance.

### Garage vs SeaweedFS

| Aspect | Garage | SeaweedFS |
|--------|--------|-----------|
| Focus | Geo-distribution | Small files |
| Deployment | Single binary | Multiple components |
| Replication | Zone-based | Configurable |
| Performance | Moderate | Higher for small files |
| Cloud Tiering | No | Yes |

Choose Garage for multi-site simplicity; SeaweedFS for optimized small file storage.

### Garage vs Cloud Storage

| Aspect | Garage | Cloud S3 |
|--------|--------|----------|
| Cost | Hardware only | Per-GB + egress |
| Control | Full | Provider-managed |
| Latency | Local | Network-dependent |
| Reliability | Self-managed | Provider SLA |
| Features | Basic S3 | Full S3 + extras |

Choose Garage for cost control and data sovereignty; cloud for simplicity.

## Best Practices

### Deployment

1. **Always use 3+ zones**: Single or dual zone deployments risk data loss
2. **Distribute zones geographically**: Protect against site failures
3. **Uniform zone capacity**: Balance data distribution
4. **Use SSDs for metadata**: Improves performance significantly

### Operations

1. **Monitor regularly**: Prometheus metrics, health checks
2. **Test failovers**: Verify cluster survives zone loss
3. **Document recovery procedures**: Node replacement, data recovery
4. **Keep backups**: Garage replication is not backup

### ML-Specific Recommendations

1. **Local caching**: Cache frequently accessed data locally
2. **Batch uploads**: Reduce per-request overhead
3. **Manual versioning**: Implement in key names since no native versioning
4. **Lifecycle scripts**: Automate checkpoint cleanup
5. **Consider use case fit**: Garage suits archival and backup more than high-performance training

### Common Pitfalls

1. **Single zone deployment**: No redundancy
2. **Insufficient network**: High latency degrades performance
3. **No monitoring**: Silent failures
4. **Treating as high-performance storage**: Garage prioritizes reliability over speed
5. **Ignoring consistency model**: Reads may be stale under partition
