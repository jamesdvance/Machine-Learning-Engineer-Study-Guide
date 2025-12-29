# Ceph

## Summary

Ceph is a unified distributed storage system providing object, block, and file storage on a single platform. Built on RADOS (Reliable Autonomic Distributed Object Store), Ceph is battle-tested at exabyte scale in production at organizations like CERN, Bloomberg, and major cloud providers. For ML workloads, Ceph's RADOS Gateway (RGW) provides S3-compatible object storage with enterprise-grade reliability.

Key characteristics:

- **Architecture**: Unified storage (block via RBD, file via CephFS, object via RGW)
- **Scale**: Proven at exabyte scale; handles billions of objects
- **Durability**: Configurable replication and erasure coding
- **S3 Compatibility**: Excellent via RADOS Gateway
- **Enterprise Support**: Red Hat, IBM, SUSE
- **Complexity**: High; requires dedicated expertise

When to use Ceph:

- Organizations with existing Ceph infrastructure
- Need for unified block, file, and object storage
- Enterprise requirements (support contracts, certifications)
- Exabyte-scale storage needs
- Kubernetes persistent volumes plus object storage

When to consider alternatives:

- Small deployments (MinIO is simpler)
- Object-only needs without block/file (MinIO, SeaweedFS)
- Limited operations expertise
- Edge or geo-distributed deployments (Garage)

Storage interface comparison:

| Interface | Protocol | Use Case | ML Application |
|-----------|----------|----------|----------------|
| RGW | S3/Swift | Object storage | Training data, models, artifacts |
| RBD | Block device | VM/container volumes | Scratch space, databases |
| CephFS | POSIX filesystem | Shared filesystems | Legacy applications, home directories |

---

## Architecture

### RADOS Foundation

All Ceph storage interfaces are built on RADOS, which provides the distributed object store foundation:

```
                     Client Applications
                            |
          +-----------------+------------------+
          |                 |                  |
       +-----+           +-----+           +------+
       | RGW |           | RBD |           |CephFS|
       |     |           |     |           |      |
       +--+--+           +--+--+           +--+---+
          |                 |                  |
          +-----------------+------------------+
                            |
                        librados
                            |
                         RADOS
          +-----------------+------------------+
          |                 |                  |
       +-----+           +-----+           +-----+
       | OSD |           | OSD |           | OSD |
       +-----+           +-----+           +-----+
```

**Object Storage Daemons (OSDs)**: Store data on physical disks. Each OSD manages one disk and handles replication, recovery, and rebalancing. Production clusters typically have tens to thousands of OSDs.

**Monitor (MON)**: Maintains cluster state including maps of OSDs, placement groups, and CRUSH rules. Requires odd number (3, 5) for quorum. Lightweight; metadata only.

**Manager (MGR)**: Provides monitoring, orchestration, and dashboard functionality. Runs alongside monitors.

**RADOS Gateway (RGW)**: S3 and Swift compatible HTTP gateway. Stateless; scales horizontally.

### CRUSH Algorithm

CRUSH (Controlled Replication Under Scalable Hashing) determines data placement without central lookup:

```
Object Key
    |
    v
Hash Function
    |
    v
Placement Group (PG)
    |
    v
CRUSH Map
    |
    v
OSD Set (primary + replicas)
```

CRUSH enables:
- Deterministic placement (any client can compute location)
- Failure domain awareness (spread replicas across racks, rooms, datacenters)
- Weighted distribution (larger disks get more data)
- No central metadata server for lookups

### Placement Groups

Objects map to placement groups (PGs), which map to OSDs:

```
1M objects -> 1000 PGs -> 100 OSDs

Each PG:
- Contains ~1000 objects
- Maps to 3 OSDs (with replication=3)
- Managed as a unit for replication/recovery
```

PG count affects:
- Recovery granularity (fewer PGs = larger recovery units)
- Memory usage (each PG has overhead)
- Distribution uniformity (more PGs = better balance)

Recommended: 100-200 PGs per OSD.

## RADOS Gateway (RGW)

RGW provides S3-compatible object storage for Ceph clusters.

### S3 API Support

RGW implements comprehensive S3 API:

| Feature | Support | Notes |
|---------|---------|-------|
| Basic Operations | Full | GET, PUT, DELETE, HEAD, LIST |
| Multipart Upload | Full | Required for large objects |
| Versioning | Full | Per-bucket configuration |
| Lifecycle Policies | Full | Expiration, transition |
| Bucket Policies | Full | IAM-style policies |
| Object Lock | Full | Compliance and governance modes |
| S3 Select | Partial | CSV and JSON support |
| Server-Side Encryption | Full | SSE-S3, SSE-KMS, SSE-C |
| Replication | Full | Cross-site replication |

### Multi-Site Replication

Ceph RGW supports active-active multi-site deployments:

```
       Site A (US-East)              Site B (EU-West)
    +------------------+          +------------------+
    |     RGW Zone     |  <---->  |     RGW Zone     |
    |  (active-active) |   sync   |  (active-active) |
    +--------+---------+          +--------+---------+
             |                             |
    +--------+---------+          +--------+---------+
    |   RADOS Cluster  |          |   RADOS Cluster  |
    +------------------+          +------------------+
```

Configuration:
- Realm: Global namespace across sites
- Zonegroup: Collection of zones (sites)
- Zone: Single site with its own RADOS cluster

Benefits:
- Geographic redundancy
- Local read/write performance
- Eventual consistency across sites
- Automatic conflict resolution

### Performance Tuning

RGW performance depends on:

**Thread Pool Configuration**:
```ini
[client.rgw.gateway]
rgw_thread_pool_size = 512
rgw_frontends = beast port=7480 num_threads=512
```

**Bucket Index Sharding**:
```bash
# Shard large buckets to avoid hotspots
radosgw-admin bucket reshard --bucket=large-bucket --num-shards=128
```

**Garbage Collection**:
```ini
rgw_gc_max_objs = 32
rgw_gc_obj_min_wait = 7200  # 2 hours
rgw_gc_processor_period = 3600
```

## ML Integration

### S3 Access via Boto3

```python
import boto3
from botocore.config import Config

def get_ceph_client(endpoint, access_key, secret_key):
    """Create Ceph RGW client."""
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}  # Required for some RGW configs
        )
    )

client = get_ceph_client(
    'http://rgw.internal:7480',
    'access_key',
    'secret_key'
)

# Standard S3 operations work
client.upload_file('model.pt', 'ml-models', 'model_v1.pt')
client.download_file('ml-models', 'model_v1.pt', 'local_model.pt')
```

### Training Data Pipeline

```python
import boto3
from torch.utils.data import DataLoader, IterableDataset
import io

class CephTrainingDataset(IterableDataset):
    """Stream training data from Ceph RGW."""

    def __init__(self, rgw_endpoint, bucket, prefix, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=rgw_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket = bucket
        self.prefix = prefix

    def __iter__(self):
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                yield self.load_sample(obj['Key'])

    def load_sample(self, key):
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        data = response['Body'].read()
        return self.parse(data)

    def parse(self, data):
        # Parse binary data
        pass

# Usage
dataset = CephTrainingDataset(
    rgw_endpoint='http://rgw.internal:7480',
    bucket='training-data',
    prefix='images/',
    access_key='access_key',
    secret_key='secret_key'
)
loader = DataLoader(dataset, batch_size=32, num_workers=8)
```

### Checkpoint Management

```python
import torch
import boto3
from datetime import datetime

class CephCheckpointManager:
    """Manage training checkpoints in Ceph."""

    def __init__(self, rgw_endpoint, bucket, access_key, secret_key):
        self.client = boto3.client(
            's3',
            endpoint_url=rgw_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket = bucket

    def save_checkpoint(self, state_dict, run_id, epoch):
        """Save checkpoint with versioning."""
        key = f'checkpoints/{run_id}/epoch_{epoch:04d}.pt'
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        self.client.upload_fileobj(buffer, self.bucket, key)

        # Tag with metadata
        self.client.put_object_tagging(
            Bucket=self.bucket,
            Key=key,
            Tagging={
                'TagSet': [
                    {'Key': 'epoch', 'Value': str(epoch)},
                    {'Key': 'timestamp', 'Value': datetime.utcnow().isoformat()}
                ]
            }
        )
        return key

    def load_latest(self, run_id):
        """Load most recent checkpoint."""
        prefix = f'checkpoints/{run_id}/'
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return None

        # Sort by key (assumes epoch_NNNN.pt naming)
        latest = sorted(response['Contents'], key=lambda x: x['Key'])[-1]

        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, latest['Key'], buffer)
        buffer.seek(0)
        return torch.load(buffer)

    def cleanup_old(self, run_id, keep_last=5):
        """Remove old checkpoints, keep last N."""
        prefix = f'checkpoints/{run_id}/'
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return

        checkpoints = sorted(response['Contents'], key=lambda x: x['Key'])
        to_delete = checkpoints[:-keep_last]

        for obj in to_delete:
            self.client.delete_object(Bucket=self.bucket, Key=obj['Key'])
```

### High-Throughput Parallel Access

```python
from concurrent.futures import ThreadPoolExecutor
import boto3
from functools import partial

class ParallelCephReader:
    """High-throughput parallel reads from Ceph."""

    def __init__(self, endpoint, bucket, access_key, secret_key, max_workers=64):
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.max_workers = max_workers

        # Connection pool
        self._local = threading.local()

    @property
    def client(self):
        """Thread-local S3 client."""
        if not hasattr(self._local, 'client'):
            self._local.client = boto3.client(
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
        return self._local.client

    def read_object(self, key):
        """Read single object."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def read_batch(self, keys):
        """Read multiple objects in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.read_object, keys))
        return results

# Usage
reader = ParallelCephReader(
    endpoint='http://rgw.internal:7480',
    bucket='training-data',
    access_key='key',
    secret_key='secret',
    max_workers=64
)

# Read 1000 objects in parallel
data = reader.read_batch(object_keys)
```

## Deployment

### Minimum Production Cluster

```
Monitors (3):     Managers (2):     OSDs (per node):
+-----+           +-----+           +-----+
| MON |           | MGR |           | OSD | x 4-12
+-----+           +-----+           +-----+
| MON |           | MGR |
+-----+           +-----+
| MON |
+-----+

RGW (2+):
+-----+
| RGW |  (behind load balancer)
+-----+
| RGW |
+-----+
```

Minimum requirements:
- 3 monitor nodes (can be co-located)
- 2 manager nodes (can be co-located with monitors)
- 3+ OSD nodes with 4+ disks each
- 2+ RGW nodes (for S3 access)

### cephadm Deployment

Modern Ceph deployments use cephadm:

```bash
# Bootstrap cluster
cephadm bootstrap --mon-ip 10.0.0.1

# Add hosts
ceph orch host add node2 10.0.0.2
ceph orch host add node3 10.0.0.3

# Add OSDs
ceph orch apply osd --all-available-devices

# Add RGW
ceph orch apply rgw myrgw --placement="count:2"
```

### RGW Configuration

```bash
# Create realm, zonegroup, zone
radosgw-admin realm create --rgw-realm=default --default
radosgw-admin zonegroup create --rgw-zonegroup=default --master --default
radosgw-admin zone create --rgw-zone=default --master --default

# Create user
radosgw-admin user create --uid=mluser --display-name="ML User" \
    --access-key=accesskey --secret-key=secretkey

# Apply bucket quota
radosgw-admin quota set --quota-scope=bucket --uid=mluser \
    --max-size=10T --max-objects=100000000

# Enable versioning
aws s3api put-bucket-versioning \
    --endpoint-url=http://rgw:7480 \
    --bucket=ml-data \
    --versioning-configuration Status=Enabled
```

### Kubernetes Integration

Rook operator for Kubernetes deployment:

```yaml
apiVersion: ceph.rook.io/v1
kind: CephCluster
metadata:
  name: ml-cluster
  namespace: rook-ceph
spec:
  cephVersion:
    image: quay.io/ceph/ceph:v18
  dataDirHostPath: /var/lib/rook
  mon:
    count: 3
    allowMultiplePerNode: false
  mgr:
    count: 2
  storage:
    useAllNodes: true
    useAllDevices: true

---
apiVersion: ceph.rook.io/v1
kind: CephObjectStore
metadata:
  name: ml-object-store
  namespace: rook-ceph
spec:
  metadataPool:
    replicated:
      size: 3
  dataPool:
    erasureCoded:
      dataChunks: 4
      codingChunks: 2
  gateway:
    port: 80
    instances: 2
```

## Data Protection

### Replication vs Erasure Coding

**Replication** (simple, higher overhead):
```bash
# 3x replication pool
ceph osd pool create ml-replicated 128 128 replicated
ceph osd pool set ml-replicated size 3
```
- Overhead: 200% (3 copies)
- Read performance: Good (any replica)
- Recovery: Fast (copy from replica)

**Erasure Coding** (efficient, complex):
```bash
# EC 4+2 pool
ceph osd erasure-code-profile set ml-ec-profile \
    k=4 m=2 crush-failure-domain=host
ceph osd pool create ml-ec 128 128 erasure ml-ec-profile
```
- Overhead: 50% (4 data + 2 parity = 6 units for 4 units of data)
- Read performance: Lower (must read k chunks)
- Recovery: Slower (compute from parity)

For ML workloads:
- Replication for hot data (training data in active use)
- Erasure coding for cold data (archived datasets, old checkpoints)

### Failure Domains

CRUSH rules control data placement:

```bash
# Spread replicas across racks
ceph osd crush rule create-replicated ml-rack-rule \
    default rack host

# Apply to pool
ceph osd pool set ml-data crush_rule ml-rack-rule
```

Failure domain options:
- `osd`: Different OSDs (same host possible)
- `host`: Different hosts (same rack possible)
- `rack`: Different racks (same datacenter)
- `datacenter`: Different datacenters

### Scrubbing and Repair

Ceph automatically verifies and repairs data:

```bash
# Check pool health
ceph health detail

# Force deep scrub
ceph pg deep-scrub 1.a5

# Check scrub status
ceph pg dump_stuck inactive
```

## Monitoring

### Built-in Dashboard

Ceph provides a web dashboard:

```bash
# Enable dashboard
ceph mgr module enable dashboard
ceph dashboard create-self-signed-cert
ceph dashboard ac-user-create admin -i password.txt administrator
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ceph'
    static_configs:
      - targets: ['ceph-mgr:9283']
```

Key metrics:
- `ceph_health_status`: Cluster health (0=OK, 1=WARN, 2=ERR)
- `ceph_osd_in`: OSDs in the cluster
- `ceph_osd_up`: OSDs that are up
- `ceph_pool_bytes_used`: Pool utilization
- `ceph_rgw_req`: RGW request counts
- `ceph_rgw_failed_req`: RGW failures

### Alerting

```yaml
groups:
  - name: ceph
    rules:
      - alert: CephHealthWarn
        expr: ceph_health_status == 1
        for: 5m
        annotations:
          summary: "Ceph cluster health warning"

      - alert: CephOSDDown
        expr: ceph_osd_up < ceph_osd_in
        for: 5m
        annotations:
          summary: "Ceph OSD down"

      - alert: CephPoolNearFull
        expr: ceph_pool_bytes_used / ceph_pool_max_avail > 0.8
        for: 5m
        annotations:
          summary: "Ceph pool near full"
```

## Comparison with Alternatives

### Ceph vs MinIO

| Aspect | Ceph | MinIO |
|--------|------|-------|
| Scope | Block + File + Object | Object only |
| Complexity | High | Medium |
| Minimum Deploy | 3+ nodes, 10+ OSDs | 4 nodes |
| Enterprise Support | Red Hat, IBM | Commercial license |
| S3 Compatibility | Excellent | Excellent |
| Performance | Good | Higher for objects |

Choose Ceph for unified storage; MinIO for dedicated object storage.

### Ceph vs SeaweedFS

| Aspect | Ceph | SeaweedFS |
|--------|------|-----------|
| Small Files | Good (with tuning) | Excellent |
| Complexity | High | Medium |
| Enterprise Support | Strong | Limited |
| POSIX Support | CephFS | FUSE mount |
| Cloud Tiering | Limited | Native |

Choose Ceph for enterprise requirements; SeaweedFS for cost-optimized small file storage.

## Best Practices

### Capacity Planning

- Maintain 70% utilization maximum (recovery headroom)
- Plan for 3-year growth
- Account for OSD overhead (~1% per OSD)
- Size RGW instances for expected request rate

### Performance Optimization

1. **Use SSDs for journals/WAL/DB**: Critical for write performance
2. **Separate networks**: Public (client) and cluster (replication)
3. **Tune PG count**: 100-200 per OSD
4. **Enable caching**: BlueStore caching, RGW front-end caching
5. **Shard large buckets**: Prevent bucket index hotspots

### Operational Excellence

1. **Monitor proactively**: Dashboard, Prometheus, alerting
2. **Regular upgrades**: Stay on supported versions
3. **Test recovery**: Practice OSD replacement, node failure
4. **Document procedures**: Runbooks for common operations
5. **Capacity alerts**: 70%, 80%, 90% thresholds

### ML-Specific Recommendations

1. **EC for archives**: Erasure coding for cold training data
2. **Replication for active data**: 3x for data in active training
3. **RGW scaling**: Multiple RGW instances behind load balancer
4. **Large objects**: Use multipart upload for files >100MB
5. **Lifecycle policies**: Auto-transition old checkpoints to EC pools

### Common Pitfalls

1. **Undersized monitors**: MONs need SSDs and adequate memory
2. **Insufficient PGs**: Causes uneven distribution
3. **Mixed OSD sizes**: Complicates capacity planning
4. **No network separation**: Replication traffic impacts clients
5. **Ignoring HEALTH_WARN**: Small issues compound
