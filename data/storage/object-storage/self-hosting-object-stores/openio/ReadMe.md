# OpenIO

## Summary

OpenIO is a software-defined object storage system designed for large-scale deployments on heterogeneous hardware. Its grid-based architecture and Conscience scoring technology enable dynamic load balancing without manual rebalancing when adding or removing nodes. OpenIO was acquired by OVHcloud in 2020 and now powers their object storage offering.

Key characteristics:

- **Architecture**: Grid of nodes (not ring-based like Ceph)
- **Hardware**: Runs on mixed hardware generations without performance penalties
- **Scaling**: Dynamic load balancing via Conscience technology
- **Minimum Resources**: 1 CPU core, 512 MB RAM, 4 GB storage
- **API Support**: S3 and OpenStack Swift compatible
- **Status**: Acquired by OVHcloud (2020), limited community development

When to use OpenIO:

- Large archival workloads (cold storage, backups)
- Organizations with diverse, aging hardware
- Existing OVHcloud infrastructure
- Need to grow storage incrementally without rebalancing

When to consider alternatives:

- Active community and long-term roadmap needed (MinIO, Ceph)
- High-performance training workloads (MinIO)
- Simpler deployment (MinIO, Garage)
- Enterprise support requirements (Ceph, MinIO commercial)

Hardware flexibility:

| Component | Minimum | Notes |
|-----------|---------|-------|
| CPU | 1 core | Scale with workload type |
| Memory | 512 MB | More for metadata-heavy workloads |
| Storage | 4 GB | Any disk type supported |
| Network | 1 NIC | More for high throughput |

---

## Architecture

### Grid-Based Design

Unlike traditional ring-based architectures (Ceph, Cassandra) that require careful rebalancing, OpenIO uses a grid where nodes operate independently:

```
Traditional Ring:                 OpenIO Grid:
+---+---+---+---+                +---+---+---+---+
|   |   |   |   |                | A | B | C | D |
+---+---+---+---+                +---+---+---+---+
  Fixed positions                | E | F | G | H |
  Must rebalance on              +---+---+---+---+
  add/remove                     | I | J | K | L |
                                 +---+---+---+---+
                                   Independent nodes
                                   Dynamic selection
```

Benefits:
- Add nodes without rebalancing existing data
- Mix different hardware generations
- Nodes fail independently without cascade effects
- Capacity grows linearly with nodes

### Conscience Technology

Conscience is OpenIO's dynamic load balancing system. Each service reports a "quality score" based on:

- Current CPU utilization
- Memory usage
- Disk I/O and space
- Network throughput
- Recent error rates

The grid uses these scores for data placement:

```
Node Selection for Write:

Scores:  A=85  B=72  C=91  D=45  E=88  F=33

High score = healthy, available
Low score = busy, degraded

Selection: C (91), E (88), A (85)
           Best available nodes chosen
```

When new nodes join:
- They start with high scores (empty, fast)
- Receive more new writes
- Gradually balance with existing nodes
- No background rebalancing storms

### Core Components

**Proxy**: Entry point for client requests. Routes to appropriate services based on Conscience scores. Stateless, horizontally scalable.

```
Client Request
      |
      v
+----------+
|  Proxy   |  Route based on Conscience
+----+-----+
     |
     +-----> Meta services (namespace lookup)
     |
     +-----> Data services (object storage)
```

**Meta Services**: Three levels of metadata services:

- **Meta0**: Root directory, maps container prefixes to Meta1
- **Meta1**: Maps container names to Meta2 locations
- **Meta2**: Container metadata, object listings

**Data Services**: Store actual object data. Each service manages local storage and reports health to Conscience.

**Account Service**: Manages account-level quotas and statistics.

### Data Layout

Objects are split into chunks distributed across nodes:

```
Object (100 MB)
      |
      v
+-----------+-----------+-----------+
| Chunk 1   | Chunk 2   | Chunk 3   |  (32 MB each)
| (Node A)  | (Node B)  | (Node C)  |
+-----------+-----------+-----------+

Chunk locations stored in Meta2
Chunks include position, hash, size
```

Chunk placement:
- Conscience selects nodes with best scores
- Respects placement policies (different racks, zones)
- Handles failures by selecting alternates

## S3 API Support

OpenIO provides S3 compatibility through its gateway:

### Supported Operations

| Operation | Support | Notes |
|-----------|---------|-------|
| GET/PUT/DELETE Object | Yes | Full support |
| HEAD Object | Yes | Full support |
| ListObjectsV2 | Yes | Full support |
| Multipart Upload | Yes | For large objects |
| CopyObject | Yes | Full support |
| Bucket Operations | Yes | Create, delete, list |
| Versioning | Yes | Per-container |
| Server-Side Encryption | Yes | Customer-provided keys |
| Bucket ACLs | Yes | Basic ACLs |
| Object Tagging | Yes | Full support |
| S3 Select | No | Not implemented |
| Object Lock | No | Not implemented |

### Configuration

```ini
# /etc/oio/sds.conf.d/s3-gateway.conf
[s3_gateway]
bind_addr = 0.0.0.0
bind_port = 6007
workers = 32

[credentials]
# Access keys configured here or via account service
```

### Client Usage

```python
import boto3

def get_openio_client(endpoint, access_key, secret_key):
    """Create OpenIO S3 client."""
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

client = get_openio_client(
    'http://openio.internal:6007',
    'access_key',
    'secret_key'
)

# Standard S3 operations
client.upload_file('data.tar', 'ml-data', 'datasets/v1.tar')
```

## Storage Pools and Tiering

OpenIO supports multiple storage pools for different media types:

### Pool Configuration

```yaml
# Storage policies define how data is placed

# Fast pool for hot data
rawx_fast:
  service_type: rawx
  pool_type: ssd
  min_distance: 1  # Different nodes

# Capacity pool for cold data
rawx_capacity:
  service_type: rawx
  pool_type: hdd
  min_distance: 2  # Different racks
```

### Automatic Tiering

```yaml
# Lifecycle policy
policies:
  hot_data:
    storage_policy: rawx_fast
    condition:
      age_days: 0-30

  warm_data:
    storage_policy: rawx_capacity
    condition:
      age_days: 30-90

  cold_data:
    storage_policy: rawx_ec_cold
    condition:
      age_days: 90+
```

Objects automatically migrate based on access patterns and age.

### Erasure Coding

OpenIO supports erasure coding for space efficiency:

```yaml
# EC policy: 6 data + 3 parity
storage_policy:
  name: EC_6_3
  data: 6
  parity: 3
  # Can lose 3 chunks, 50% overhead
```

Erasure coding trade-offs:
- More space efficient than replication
- Higher CPU for encode/decode
- Better for large, infrequently accessed objects

## Deployment

### Minimum Deployment

Single-node development (not for production):

```bash
# Install OpenIO
curl -sL https://get.openio.io/setup | bash

# Initialize namespace
openio cluster init OPENIO

# Start services
openio cluster start
```

### Production Cluster

Minimum production: 3 nodes for metadata HA, more for storage capacity.

```
+-------------+     +-------------+     +-------------+
|   Node 1    |     |   Node 2    |     |   Node 3    |
| - Meta0/1/2 |     | - Meta0/1/2 |     | - Meta0/1/2 |
| - Proxy     |     | - Proxy     |     | - Proxy     |
| - Account   |     | - Account   |     | - Account   |
| - RawX (SSD)|     | - RawX (SSD)|     | - RawX (SSD)|
+-------------+     +-------------+     +-------------+

+-------------+     +-------------+     +-------------+
|   Node 4    |     |   Node 5    |     |   Node 6    |
| - RawX x12  |     | - RawX x12  |     | - RawX x12  |
| (HDD pool)  |     | (HDD pool)  |     | (HDD pool)  |
+-------------+     +-------------+     +-------------+
```

### Ansible Deployment

OpenIO provides Ansible playbooks:

```yaml
# inventory
[all:vars]
openio_namespace = OPENIO
openio_replication_factor = 3

[meta]
node1 ansible_host=10.0.0.1
node2 ansible_host=10.0.0.2
node3 ansible_host=10.0.0.3

[rawx]
node1 ansible_host=10.0.0.1
node2 ansible_host=10.0.0.2
node3 ansible_host=10.0.0.3
node4 ansible_host=10.0.0.4 rawx_disks=["/dev/sd[b-m]"]
node5 ansible_host=10.0.0.5 rawx_disks=["/dev/sd[b-m]"]
node6 ansible_host=10.0.0.6 rawx_disks=["/dev/sd[b-m]"]
```

```bash
ansible-playbook -i inventory openio.yml
```

### Docker Deployment

```yaml
version: '3.8'

services:
  openio:
    image: openio/sds:latest
    volumes:
      - openio-data:/var/lib/oio
      - openio-conf:/etc/oio
    ports:
      - "6007:6007"  # S3 API
      - "6006:6006"  # Swift API
    environment:
      - OPENIO_NAMESPACE=OPENIO
      - OPENIO_IPADDR=0.0.0.0

volumes:
  openio-data:
  openio-conf:
```

## ML Integration

### Basic Data Access

```python
import boto3
from botocore.config import Config

class OpenIOClient:
    """OpenIO S3 client for ML workloads."""

    def __init__(self, endpoint, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                max_pool_connections=50,
                retries={'max_attempts': 3}
            )
        )

    def upload_dataset(self, local_path, bucket, key):
        """Upload with multipart for large files."""
        config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100MB
            multipart_chunksize=50 * 1024 * 1024,   # 50MB
            max_concurrency=10
        )
        self.client.upload_file(local_path, bucket, key, Config=config)

    def download_dataset(self, bucket, key, local_path):
        """Download with parallel chunks."""
        config = boto3.s3.transfer.TransferConfig(
            max_concurrency=10
        )
        self.client.download_file(bucket, key, local_path, Config=config)

client = OpenIOClient(
    'http://openio:6007',
    'access_key',
    'secret_key'
)
```

### Training Data Pipeline

```python
from torch.utils.data import Dataset, DataLoader
import boto3
from concurrent.futures import ThreadPoolExecutor

class OpenIODataset(Dataset):
    """Dataset with parallel loading from OpenIO."""

    def __init__(self, endpoint, bucket, prefix, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket = bucket

        # List objects
        self.keys = []
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            self.keys.extend([obj['Key'] for obj in page.get('Contents', [])])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        response = self.client.get_object(Bucket=self.bucket, Key=self.keys[idx])
        data = response['Body'].read()
        return self.parse(data)

    def parse(self, data):
        # Parse binary data into training sample
        pass

# Usage
dataset = OpenIODataset(
    endpoint='http://openio:6007',
    bucket='training-data',
    prefix='images/',
    access_key='key',
    secret_key='secret'
)
loader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2)
```

### Archival Workflow

OpenIO excels at archival workloads:

```python
import boto3
from datetime import datetime

class ArchivalManager:
    """Manage ML artifacts in OpenIO with tiering."""

    def __init__(self, client, bucket):
        self.client = client
        self.bucket = bucket

    def archive_experiment(self, run_id, artifacts):
        """Archive experiment artifacts."""
        prefix = f'archive/{run_id}/'

        for name, path in artifacts.items():
            key = f'{prefix}{name}'
            self.client.upload_file(path, self.bucket, key)

            # Tag with metadata for lifecycle
            self.client.put_object_tagging(
                Bucket=self.bucket,
                Key=key,
                Tagging={
                    'TagSet': [
                        {'Key': 'archive_date', 'Value': datetime.utcnow().isoformat()},
                        {'Key': 'run_id', 'Value': run_id}
                    ]
                }
            )

    def restore_experiment(self, run_id, local_dir):
        """Restore experiment from archive."""
        prefix = f'archive/{run_id}/'

        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        for obj in response.get('Contents', []):
            key = obj['Key']
            local_path = os.path.join(local_dir, os.path.basename(key))
            self.client.download_file(self.bucket, key, local_path)
```

## Operations

### Cluster Management

```bash
# View cluster status
openio cluster show

# List services
openio cluster list

# Check service health
openio cluster local conf

# Add storage node
openio cluster unlock rawx <service_id>
```

### Container Management

```bash
# Create container
openio container create my-bucket

# List containers
openio container list

# Show container properties
openio container show my-bucket
```

### Object Operations

```bash
# Upload object
openio object create my-bucket local-file.tar --name remote-name.tar

# List objects
openio object list my-bucket

# Download object
openio object save my-bucket remote-name.tar --file local-file.tar
```

### Monitoring

OpenIO exposes metrics for monitoring:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: openio
    static_configs:
      - targets: ['openio-proxy:6000']
    metrics_path: /metrics
```

Key metrics:
- Service scores (Conscience)
- Request latency
- Storage utilization
- Replication status
- Error rates

## Comparison with Alternatives

### OpenIO vs MinIO

| Aspect | OpenIO | MinIO |
|--------|--------|-------|
| Architecture | Grid, Conscience | Distributed, erasure |
| Hardware Flexibility | Excellent | Good |
| Scaling | No rebalancing | Requires planning |
| Community | Limited (OVHcloud) | Large, active |
| Performance | Good | Higher |
| Enterprise Support | OVHcloud | Commercial license |

Choose OpenIO for heterogeneous hardware; MinIO for performance and community.

### OpenIO vs Ceph

| Aspect | OpenIO | Ceph |
|--------|--------|------|
| Complexity | Medium | High |
| Rebalancing | Automatic, gradual | Manual or slow |
| Unified Storage | Object only | Block + File + Object |
| Enterprise Support | OVHcloud | Red Hat, IBM |
| Scale | Large | Exabyte |

Choose OpenIO for simpler operations; Ceph for unified storage.

### OpenIO vs SeaweedFS

| Aspect | OpenIO | SeaweedFS |
|--------|--------|-----------|
| Focus | Large archives | Small files |
| Architecture | Grid | Master-volume |
| Cloud Tiering | Native | Native |
| Community | Smaller | Active |
| POSIX Support | OIO-FS | FUSE |

Choose OpenIO for archival; SeaweedFS for billions of small files.

## Considerations

### OVHcloud Acquisition Impact

Since the 2020 acquisition by OVHcloud:

- Development focused on OVHcloud internal needs
- Community releases less frequent
- Public roadmap unclear
- Enterprise support through OVHcloud only

This affects long-term planning:
- Evaluate vendor lock-in risk
- Consider migration path to alternatives
- Monitor community activity

### When OpenIO Makes Sense

1. **OVHcloud customers**: Native integration
2. **Large archival workloads**: Tiering and EC support
3. **Mixed hardware environments**: Conscience handles heterogeneity
4. **Incremental growth**: No rebalancing storms

### When to Consider Alternatives

1. **Active development needed**: MinIO, Ceph have larger communities
2. **High-performance training**: MinIO performs better
3. **Long-term roadmap concerns**: Ceph or MinIO more predictable
4. **Enterprise support outside OVH**: Ceph (Red Hat) or MinIO

## Best Practices

### Deployment

1. **Separate metadata nodes**: Dedicated nodes for Meta services
2. **Multiple proxies**: Load balance for HA
3. **Storage pools by media**: SSD and HDD pools
4. **Conscience tuning**: Adjust scoring weights for workload

### Operations

1. **Monitor Conscience scores**: Identify degraded nodes
2. **Regular health checks**: Verify chunk integrity
3. **Capacity planning**: Account for EC overhead
4. **Document procedures**: Node addition, failure recovery

### ML-Specific Recommendations

1. **Use for archives**: Best fit for cold data
2. **Batch operations**: Reduce per-request overhead
3. **Enable EC for old data**: Space efficiency for archives
4. **Consider alternatives for hot data**: MinIO for active training

### Common Pitfalls

1. **Ignoring acquisition implications**: Plan for potential migration
2. **Single proxy**: No HA for clients
3. **Mixed workloads on same pool**: Separate hot and cold
4. **No monitoring**: Conscience issues go unnoticed
5. **Insufficient metadata resources**: Meta services need SSDs
