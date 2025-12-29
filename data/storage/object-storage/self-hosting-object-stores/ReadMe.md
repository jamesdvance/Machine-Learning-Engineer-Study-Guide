# Self-Hosting Object Stores

## Summary

Self-hosted object storage provides S3-compatible storage infrastructure on your own hardware, eliminating cloud storage costs for large ML datasets while maintaining API compatibility with cloud-native tools. These solutions range from lightweight single-binary deployments to enterprise-grade distributed storage systems capable of exabyte-scale storage.

Key decision factors:

| Solution | Best For | Complexity | Minimum Hardware | S3 Compatibility |
|----------|----------|------------|------------------|------------------|
| MinIO | Production ML workloads, high performance | Medium | 4 nodes recommended | Excellent |
| SeaweedFS | Billions of small files, cost efficiency | Medium | 1 master + 1 volume | Good |
| Ceph | Enterprise, unified storage (block/file/object) | High | 3+ nodes, 10+ OSDs | Excellent |
| Garage | Geo-distributed, edge deployments | Low | 3 nodes, minimal resources | Good |
| OpenIO | Large-scale archival, heterogeneous hardware | Medium-High | Minimal (1 core, 512MB) | Good |

When to self-host:

- Storage costs exceed $10,000/month and are growing
- Data residency requirements prohibit cloud storage
- Network bandwidth to cloud is a bottleneck for training
- Predictable, fixed costs are preferred over pay-per-use
- Existing on-premise hardware can be repurposed

When to use cloud object storage:

- Teams lack infrastructure expertise
- Workloads are bursty or unpredictable
- Integration with cloud ML services is critical
- Global distribution with minimal ops overhead is needed
- Compliance certifications from cloud providers are required

Cost comparison for 1 PB storage:

| Approach | Monthly Cost | Notes |
|----------|--------------|-------|
| AWS S3 Standard | ~$23,000 | Plus egress fees |
| Self-hosted MinIO | ~$3,000-5,000 | Hardware amortized, power, ops |
| Self-hosted SeaweedFS | ~$2,000-4,000 | Lower hardware requirements |
| Self-hosted Ceph | ~$4,000-6,000 | Higher ops overhead |

---

## Why Self-Host Object Storage

### The Economics of Scale

Cloud object storage pricing follows a simple model: pay per gigabyte stored plus per-gigabyte transferred. At small scale, this is economical. At ML scale, costs compound rapidly.

Consider a typical ML training workflow:
- 500 TB training dataset
- Read 3x during hyperparameter sweeps
- Store 50 TB of checkpoints
- 10 TB model artifacts

**Monthly cloud costs:**
- Storage: 550 TB x $0.023/GB = ~$12,650
- Reads: 1,500 TB egress x $0.09/GB = ~$135,000 (if cross-region)
- Even same-region: significant request costs at billions of objects

**Self-hosted alternative:**
- Hardware: ~$500,000 one-time (amortized over 4 years)
- Power/cooling: ~$2,000/month
- Operations: 0.5-1 FTE
- Monthly effective cost: ~$15,000 including ops

Break-even typically occurs at 200-500 TB for organizations with existing infrastructure teams.

### Data Gravity and Locality

Training throughput depends on data locality. Reading from cloud storage introduces:
- 10-100ms latency per request (vs sub-ms for local NVMe)
- Network bandwidth caps (even 100 Gbps saturates with parallel training)
- Egress costs when compute and storage are separated

Self-hosted storage co-located with GPU clusters eliminates these constraints. A properly configured MinIO or SeaweedFS cluster can saturate 100+ Gbps networks with parallel reads.

### Operational Control

Self-hosting provides:
- Full control over hardware configuration and optimization
- No dependency on cloud provider availability
- Ability to tune for specific workloads (small files vs large sequential reads)
- No API rate limits or throttling
- Custom retention and lifecycle policies

The trade-off: operational responsibility for availability, durability, and scaling.

## Architecture Patterns for ML

### Pattern 1: High-Performance Training Storage

For distributed training with frequent data access:

```
                    +-------------------+
                    |   Training Cluster |
                    |   (GPU nodes)      |
                    +---------+---------+
                              |
                    100 Gbps InfiniBand/Ethernet
                              |
                    +---------+---------+
                    |   MinIO Cluster    |
                    |   (NVMe SSDs)      |
                    |   - 4-16 nodes     |
                    |   - Erasure coding |
                    +-------------------+
```

**Configuration:**
- NVMe SSDs for low latency
- High-bandwidth networking (25-100 Gbps per node)
- Erasure coding (EC:4) for efficiency over replication
- Co-located in same rack or adjacent racks

**Expected performance:**
- 10-50 GB/s aggregate read throughput
- Sub-10ms latency for cached data
- 99.99% durability with EC:4

### Pattern 2: Cost-Optimized Data Lake

For large archival datasets accessed periodically:

```
                    +-------------------+
                    |   Compute Cluster  |
                    +---------+---------+
                              |
                       10 Gbps Ethernet
                              |
                    +---------+---------+
                    |   SeaweedFS/Ceph   |
                    |   (HDDs + SSD cache)|
                    |   - Dense storage  |
                    |   - Tiered caching |
                    +-------------------+
```

**Configuration:**
- High-density HDD nodes (60-100 drives per server)
- SSD caching tier for hot data
- Erasure coding for space efficiency
- Slower network acceptable for batch workloads

**Expected performance:**
- 1-5 GB/s aggregate throughput
- 50-200ms latency (acceptable for batch)
- Cost: ~$10/TB/month fully loaded

### Pattern 3: Geo-Distributed Storage

For multi-site deployments or edge training:

```
     Site A                Site B                Site C
+-------------+      +-------------+      +-------------+
| Garage Node |<---->| Garage Node |<---->| Garage Node |
| (Training)  |      | (Training)  |      | (Inference) |
+-------------+      +-------------+      +-------------+
       \                   |                   /
        \                  |                  /
         +--------+--------+--------+--------+
                  |  WAN Links  |
                  | (Async Replication) |
```

**Configuration:**
- Garage or Ceph multi-site for geo-distribution
- Asynchronous replication between sites
- Local reads, cross-site writes propagate
- Tolerates site failures

## Choosing a Solution

### MinIO: Production ML Standard

MinIO is the default choice for production ML workloads requiring high performance and S3 compatibility.

**Strengths:**
- Industry-leading S3 API compatibility
- Sub-10ms latency on NVMe configurations
- Native Kubernetes deployment (Operator)
- Active development, large community
- Erasure coding with bitrot protection

**Weaknesses:**
- AGPLv3 license requires source disclosure for modifications
- Enterprise features require commercial license
- Higher resource requirements than lightweight alternatives

**Best for:** GPU training clusters, model serving, production ML pipelines

### SeaweedFS: Billions of Small Files

SeaweedFS excels at storing massive quantities of small files with minimal overhead.

**Strengths:**
- O(1) disk seek for any file (40 bytes metadata overhead)
- Native cloud tiering (hot local, warm cloud)
- POSIX FUSE mount option
- Flexible metadata backends (MySQL, PostgreSQL, Redis, etc.)
- Erasure coding support

**Weaknesses:**
- Less mature S3 compatibility than MinIO
- Smaller community
- Documentation gaps

**Best for:** Image datasets, embeddings storage, training data archives

### Ceph: Enterprise Unified Storage

Ceph provides block, file, and object storage on a unified platform.

**Strengths:**
- Battle-tested at exabyte scale
- Strong enterprise support (Red Hat, IBM)
- Unified storage (use same cluster for block, file, object)
- Advanced features (tiering, encryption, compression)
- Excellent RADOS Gateway S3 compatibility

**Weaknesses:**
- Significant operational complexity
- Higher minimum deployment size
- Requires dedicated expertise

**Best for:** Large enterprises with existing Ceph infrastructure, unified storage needs

### Garage: Lightweight Geo-Distribution

Garage is designed for geo-distributed deployments on minimal hardware.

**Strengths:**
- Single binary, minimal dependencies
- Runs on heterogeneous hardware (ARM, x86, old machines)
- Designed for unreliable networks
- Low resource requirements (1GB RAM minimum)
- AGPL license, fully open source

**Weaknesses:**
- Not designed for high-performance workloads
- Smaller feature set than MinIO/Ceph
- Limited enterprise support

**Best for:** Edge deployments, multi-site backup, resource-constrained environments

### OpenIO: Heterogeneous Hardware

OpenIO optimizes for mixed hardware environments and large-scale archival.

**Strengths:**
- Runs on heterogeneous hardware without performance penalties
- Conscience technology eliminates rebalancing
- Scales from TB to exabytes
- Low minimum requirements (1 core, 512MB RAM)

**Weaknesses:**
- Acquired by OVHcloud in 2020, uncertain roadmap
- Smaller community than MinIO/Ceph
- Limited recent development activity

**Best for:** Organizations with diverse hardware, large archival workloads

## Integration with ML Frameworks

All S3-compatible solutions work with standard ML tooling:

```python
import boto3
from torch.utils.data import DataLoader
import webdataset as wds

# Generic S3-compatible client
def get_s3_client(endpoint, access_key, secret_key):
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

# MinIO example
minio_client = get_s3_client(
    'http://minio.internal:9000',
    'minioaccess',
    'miniosecret'
)

# SeaweedFS example
seaweed_client = get_s3_client(
    'http://seaweedfs.internal:8333',
    'seaweedaccess',
    'seaweedsecret'
)

# WebDataset with self-hosted storage
dataset = wds.WebDataset(
    'pipe:aws s3 cp s3://training-data/shards/shard-{0000..0999}.tar -',
    handler=wds.warn_and_continue
).shuffle(1000).decode('pil').to_tuple('jpg', 'json')

loader = DataLoader(dataset, batch_size=32, num_workers=8)
```

### fsspec Integration

```python
import fsspec

# Register custom endpoint
fs = fsspec.filesystem(
    's3',
    endpoint_url='http://minio.internal:9000',
    key='access_key',
    secret='secret_key'
)

# Use with pandas, PyArrow, etc.
import pandas as pd
df = pd.read_parquet('s3://bucket/data.parquet', storage_options={
    'endpoint_url': 'http://minio.internal:9000',
    'key': 'access_key',
    'secret': 'secret_key'
})
```

## Operational Considerations

### Durability and Redundancy

| Solution | Replication | Erasure Coding | Recommended Durability |
|----------|-------------|----------------|------------------------|
| MinIO | Yes | Yes (default EC:4) | 99.999999999% |
| SeaweedFS | Yes (rack/DC aware) | Yes (RS 10,4 default) | 99.9999999% |
| Ceph | Yes (configurable) | Yes | 99.999999999% |
| Garage | Yes (3-zone) | No | 99.99999% |
| OpenIO | Yes | Yes | 99.9999999% |

### Monitoring

All solutions expose Prometheus metrics. Essential metrics to monitor:

- Disk utilization and health
- Network throughput and errors
- Request latency (p50, p95, p99)
- Error rates by operation type
- Cluster health and node status

### Backup Strategy

Self-hosted storage requires explicit backup planning:

1. **Cross-site replication**: Native feature in most solutions
2. **Cloud backup**: Async replication to cloud for disaster recovery
3. **Snapshot-based**: Periodic snapshots to separate storage

A hybrid approach is common: self-hosted for performance, cloud for DR.

## Migration from Cloud Storage

### Planning

1. Inventory current storage: buckets, sizes, access patterns
2. Estimate bandwidth for migration (1 PB at 10 Gbps = ~9 days)
3. Plan for dual-write period during transition
4. Update application configurations (endpoint URLs, credentials)

### Execution

```bash
# Using rclone for migration
rclone sync s3:cloud-bucket minio:local-bucket \
    --progress \
    --transfers 64 \
    --checkers 32 \
    --s3-chunk-size 64M

# Verify migration
rclone check s3:cloud-bucket minio:local-bucket
```

### Cutover Strategy

1. **Dual-write phase**: Write to both cloud and self-hosted
2. **Read migration**: Gradually shift reads to self-hosted
3. **Cloud read-through**: Fall back to cloud for missing data
4. **Final sync**: Ensure all data migrated
5. **Cloud decommission**: Delete cloud data after validation period

## Best Practices

### Hardware Selection

- **NVMe SSDs**: For hot data and high-performance workloads
- **HDDs**: For archival and cost-sensitive workloads (use 7200 RPM enterprise drives)
- **Network**: Minimum 25 Gbps per node for performance workloads
- **Memory**: 32-64 GB per node for caching and metadata

### Configuration

- Enable erasure coding for production deployments
- Configure separate pools for hot/warm/cold data
- Set appropriate lifecycle policies for automatic tiering
- Enable encryption at rest for sensitive data
- Configure monitoring and alerting before production use

### Operations

- Document runbooks for common operations (node replacement, expansion)
- Test failure scenarios in staging
- Maintain capacity buffer (20-30% free space)
- Plan for hardware refresh every 3-5 years
- Keep software updated for security patches

## Conclusion

Self-hosted object storage is a viable and often economical choice for ML workloads at scale. MinIO provides the best balance of performance, compatibility, and operational simplicity for most use cases. SeaweedFS excels for small-file workloads, Ceph for enterprise unified storage needs, and Garage for geo-distributed edge deployments.

The decision to self-host should be based on total cost analysis including hardware, operations, and opportunity cost. Organizations with 500+ TB of data and existing infrastructure teams typically see significant cost savings. Those without infrastructure expertise or with highly variable workloads may find cloud storage more appropriate despite higher per-GB costs.
