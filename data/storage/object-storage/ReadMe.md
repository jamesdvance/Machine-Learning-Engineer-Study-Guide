# Object Storage

## Summary

Object storage is the foundational data storage layer for modern machine learning infrastructure. Unlike file systems (hierarchical) or block storage (disk-level), object storage organizes data as discrete objects in flat namespaces, each containing data, metadata, and a unique identifier. This architecture enables virtually unlimited scalability, high durability (11+ nines), and cost-effective storage for the massive datasets that ML workloads demand.

**Why Object Storage for ML:**
- **Scale**: Store petabytes of training data without capacity planning
- **Durability**: 99.999999999% (11 nines) or higherdata loss is essentially impossible
- **Cost**: 10-100x cheaper than block storage or managed filesystems
- **Integration**: Native support in TensorFlow, PyTorch, Spark, and all major ML frameworks
- **Accessibility**: HTTP-based APIs enable access from anywhere

**The Three Major Providers:**

| Feature | Amazon S3 | Google Cloud Storage | Azure Blob Storage |
|---------|-----------|---------------------|-------------------|
| **Max Object Size** | 50 TB | 5 TiB | 190.7 TiB |
| **Storage Classes** | 7+ classes | 4 classes + Autoclass | 4 tiers (in-account) |
| **Archive Retrieval** | Hours (Glacier) | Milliseconds | Hours (requires rehydration) |
| **Low-Latency Option** | Express One Zone | None | Premium Block Blob |
| **Hierarchical Namespace** | No | HNS (20x faster checkpoints) | ADLS Gen2 |
| **Ecosystem Maturity** | Broadest | TensorFlow/JAX optimized | Azure ML/Synapse integrated |

**Key Decision Factors:**
- **Existing cloud platform**: Use the object storage native to your cloud provider
- **Checkpointing needs**: GCS HNS or Azure ADLS Gen2 for distributed training
- **Latency requirements**: S3 Express One Zone or Azure Premium for sub-10ms access
- **Archive access patterns**: GCS Archive has no retrieval delay (milliseconds vs hours)
- **Cost sensitivity**: All three are comparable; differences emerge in request pricing and egress

**Common ML Storage Patterns:**
1. **Training data lake**: Raw datasets, preprocessed features, data versions
2. **Model registry**: Trained models with versioning and metadata
3. **Checkpoint storage**: Distributed training state for fault tolerance
4. **Feature store backend**: Precomputed features for training and inference
5. **Artifact storage**: Logs, metrics, experiment outputs

---

## What is Object Storage?

Object storage is a data architecture that manages data as objects rather than as files in a hierarchy (file storage) or blocks on a disk (block storage). Each object contains three components:

1. **Data**: The actual content (image, model weights, parquet file, etc.)
2. **Metadata**: Descriptive information (content-type, creation date, custom tags)
3. **Unique identifier**: A key that identifies the object within its container

Objects are stored in containers (called "buckets" in S3/GCS, "containers" in Azure) within a flat namespace. While you can simulate folder hierarchies using "/" delimiters in object keys (e.g., `training-data/images/batch-001/img.jpg`), the underlying storage is fundamentally flatthere are no actual directories.

### Why Flat Namespace?

The flat architecture enables massive scalability:
- **No metadata bottlenecks**: No directory tree to traverse or lock
- **Unlimited fanout**: Billions of objects in a single bucket
- **Horizontal scaling**: Add capacity by adding storage nodes
- **Simple consistency**: Object-level operations are atomic

The trade-off: operations that would be fast on filesystems (directory listing, rename) require iterating over objects. This is why recent features like GCS Hierarchical Namespace and Azure ADLS Gen2 add filesystem-like capabilities on top of object storage.

### Object Storage vs. Other Storage Types

| Characteristic | Object Storage | File Storage | Block Storage |
|---------------|---------------|--------------|---------------|
| **Addressing** | Unique key per object | Path hierarchy | LBA (disk sectors) |
| **Scalability** | Exabytes | Terabytes-Petabytes | Terabytes |
| **Latency** | 10-100ms | 1-10ms | <1ms |
| **Consistency** | Strong (modern) | Strong | Strong |
| **Cost** | Lowest | Medium | Highest |
| **Use case** | Data lakes, backups | Home directories, shared files | Databases, OS disks |

For ML workloads: Object storage for datasets and models, block storage for compute node OS and scratch space, file storage for legacy applications requiring POSIX semantics.

## Core Concepts Across Providers

### Storage Classes and Tiers

All three providers offer tiered storage optimized for different access patterns. The fundamental trade-off: cheaper storage costs come with retrieval fees and/or access delays.

**Hot/Standard Tier:**
- Lowest access cost, highest storage cost
- Millisecond latency
- Use for: Active training data, frequently accessed models

**Cool/Nearline Tier:**
- ~50% cheaper storage, retrieval fees apply
- 30-day minimum storage duration
- Use for: Recent experiments, backup datasets

**Cold/Coldline Tier:**
- ~70% cheaper storage, higher retrieval fees
- 90-day minimum storage duration
- Use for: Archived training runs, compliance data

**Archive Tier:**
- 85-95% cheaper storage, highest retrieval fees
- 180-365 day minimum storage duration
- **Critical difference**: GCS Archive has millisecond access; S3 Glacier and Azure Archive require hours for retrieval

**Automatic Tiering:**
- S3 Intelligent-Tiering, GCS Autoclass, Azure lifecycle policies
- Automatically move objects based on access patterns
- Small management fee but eliminates manual optimization

### Consistency Models

All three providers now offer strong read-after-write consistency:
- Newly written objects are immediately visible
- Overwrites and deletes are immediately consistent
- No eventual consistency windows

This is critical for ML workflows: you can write a checkpoint and immediately read it from another worker without race conditions.

### Versioning

Object versioning maintains previous versions when objects are overwritten:
- Each write creates a new version (not an overwrite)
- Retrieve any previous version by version ID
- Delete operations create delete markers (reversible)

**ML Use Case**: Version model files automatically. Every training run writing to `models/resnet50.pt` creates a new version. Roll back to any previous model instantly.

**Cost Impact**: Each version is a separate billable object. Use lifecycle policies to delete old versions after N days.

### Lifecycle Policies

Automate storage class transitions and deletions:
- Transition objects to cheaper tiers after N days
- Delete old versions or expired data automatically
- Abort incomplete multipart uploads (prevents orphaned costs)

Every ML storage bucket should have lifecycle policies. Manual management doesn't scale.

## Provider Deep Dive: Key Differentiators

### Amazon S3

**Strengths:**
- **Ecosystem maturity**: Broadest third-party tool support
- **S3 Express One Zone**: Sub-10ms latency for performance-critical workloads
- **S3 Vectors**: Native vector embedding storage (2 trillion vectors per bucket)
- **S3 Tables**: Apache Iceberg integration for query-optimized data lakes
- **50 TB objects**: Largest single-object support (increased December 2025)

**Weaknesses:**
- **No hierarchical namespace**: Directory operations require object-by-object iteration
- **Glacier retrieval times**: Hours to access archived data
- **Request pricing complexity**: More granular (and confusing) than competitors

**Best For:**
- AWS-native infrastructure
- Workloads requiring sub-10ms latency (Express One Zone)
- Organizations with heavy third-party tool usage

### Google Cloud Storage

**Strengths:**
- **Hierarchical Namespace (HNS)**: Atomic directory operations, 20x faster checkpointing
- **Archive with millisecond access**: No retrieval delay unlike S3/Azure
- **Anywhere Cache**: Built-in global caching for inference serving
- **Cloud Storage FUSE**: Filesystem access with metadata/file caching
- **Simpler pricing**: No per-GET charges for Standard storage

**Weaknesses:**
- **5 TiB object limit**: Smaller than S3 (50 TB) and Azure (190.7 TiB)
- **No low-latency tier**: No equivalent to S3 Express One Zone
- **Smaller ecosystem**: Fewer third-party integrations than S3

**Best For:**
- Distributed training with frequent checkpointing (HNS)
- TensorFlow/JAX workloads (first-class integration)
- Unpredictable archive access patterns (instant retrieval)

### Azure Blob Storage

**Strengths:**
- **In-account tiering**: Change blob tiers without moving data
- **Premium Block Blob**: Multi-AZ low latency (unlike S3 Express single-AZ)
- **ADLS Gen2**: Mature hierarchical namespace with POSIX ACLs
- **190.7 TiB objects**: Largest single-object support
- **Granular redundancy**: LRS, ZRS, GRS, GZRS options

**Weaknesses:**
- **Archive rehydration**: Hours to access (like S3 Glacier)
- **Smaller ML ecosystem**: Fewer native framework integrations than GCS
- **Managed Lustre recommended**: Suggests Blob alone isn't optimal for training

**Best For:**
- Azure-native infrastructure
- Enterprise workloads requiring granular redundancy control
- Synapse/Databricks-based ML pipelines

## Performance Optimization for ML

### Parallelization is Mandatory

Object storage throughput scales linearly with parallel requests:

| Access Pattern | Throughput |
|---------------|-----------|
| Single-threaded | ~10 MB/s |
| 100 threads | ~1 GB/s |
| 1,000+ threads | 10+ GB/s |

For a 1 TB dataset:
- Single-threaded: ~27 hours
- 1,000 threads: ~17 minutes

**Implementation**: Use multi-threaded data loaders (PyTorch DataLoader with `num_workers`, TensorFlow `tf.data.Dataset.interleave`).

### Optimal Request Sizing

Small requests waste overhead. Large requests maximize throughput:

- **Minimum recommended**: 5-10 MB per request
- **Optimal**: 50-100 MB per request for sequential reads
- **Multipart uploads**: Required for objects >5 GB, recommended for >100 MB

### Prefix Partitioning (S3-specific)

S3 scales throughput by prefix. Each prefix supports:
- 3,500 PUT/POST/DELETE requests/second
- 5,500 GET/HEAD requests/second

For high-throughput training:
```
# Bad: Single prefix, limited to 5,500 reads/sec
s3://bucket/images/img000001.jpg
s3://bucket/images/img000002.jpg

# Good: 100 prefixes = 550,000 reads/sec capacity
s3://bucket/images/shard-0001/img000001.jpg
s3://bucket/images/shard-0002/img000002.jpg
```

GCS and Azure handle prefix partitioning automatically.

### Co-locate Storage and Compute

Cross-region data transfer adds latency and egress costs:

| Scenario | Additional Latency | Egress Cost |
|----------|-------------------|-------------|
| Same region | 0 | Free |
| Cross-region | 20-100ms | $0.01-0.02/GB |
| Cross-cloud | 50-200ms | $0.05-0.12/GB |

**Rule**: Always place storage buckets in the same region as training compute.

### File Consolidation

Millions of small files are expensive and slow:
- Each GET/PUT incurs request costs
- Metadata operations don't parallelize well
- Directory listings become expensive

**Solution**: Package small files into larger objects:
- TFRecord (TensorFlow)
- WebDataset (PyTorch)
- Parquet (tabular data)
- Tar archives (general purpose)

Target: 100-10,000 shards of 100 MB - 1 GB each.

## Checkpointing Strategies

Distributed training requires periodic checkpointing for fault tolerance. Object storage characteristics significantly impact checkpoint performance.

### The Problem

Checkpointing a large model (e.g., 70B parameters) to object storage:
1. Serialize model state (tens of GB)
2. Upload to object storage (network bound)
3. Ensure all ranks see consistent state
4. Resume on failure

Traditional object storage makes step 2 slow and step 3 complex.

### Solution: Hierarchical Namespace

GCS HNS and Azure ADLS Gen2 enable atomic directory operations:

```
# Without HNS: Checkpoint rename requires N operations
for file in checkpoint_temp/*:
    copy(file, checkpoint_final/)
    delete(file)  # Partial failure = inconsistent state

# With HNS: Single atomic rename
rename(checkpoint_temp/, checkpoint_final/)  # All or nothing
```

GCS reports 20x faster checkpointing with HNS enabled.

### Low-Latency Options

For models where checkpoint time significantly impacts training:
- **S3 Express One Zone**: Sub-10ms writes
- **Azure Premium Block Blob**: Single-digit ms with multi-AZ durability
- **GCS + FUSE caching**: Cache writes locally, async upload

### Hybrid Approach

Many organizations use tiered checkpointing:
1. **Frequent checkpoints**: Local SSD or high-performance storage
2. **Periodic sync**: Copy to object storage every N checkpoints
3. **Final model**: Permanent storage in object storage with versioning

## Cost Optimization

### Storage Class Selection Matrix

| Data Type | Recommended Tier | Rationale |
|-----------|-----------------|-----------|
| Active training data | Standard/Hot | Frequent access, latency matters |
| Recent experiments (<30 days) | Standard/Hot | May need quick access |
| Archived experiments (30-90 days) | Cool/Nearline | Infrequent access |
| Old models (>90 days) | Cold/Coldline | Rarely accessed |
| Compliance archives | Archive | Predictably cold |

### Egress Cost Reduction

Data transfer out of cloud regions is expensive ($0.05-0.12/GB):

1. **Co-locate compute and storage**: Free transfer within region
2. **Use caching**: GCS Anywhere Cache, CloudFront, Azure CDN
3. **Requester pays**: Shift costs to data consumers for public datasets
4. **Compress data**: Reduce bytes transferred

### Request Cost Reduction

Request pricing varies by operation and storage class:

| Operation | S3 Standard | GCS Standard | Azure Hot |
|-----------|------------|--------------|-----------|
| PUT (per 1,000) | $0.005 | $0.05 | $0.05 |
| GET (per 1,000) | $0.0004 | Free | Free |

S3's per-GET charges add up at scale. GCS and Azure include GETs in storage cost.

**Reduction strategies:**
- Batch small files into larger objects
- Use caching for frequently accessed data
- S3 Select / Azure query acceleration to reduce data retrieved

### Lifecycle Automation

Don't manually manage storage classes. Configure policies:

```json
{
  "rules": [
    {
      "action": "transition_to_nearline",
      "condition": {"age_days": 30}
    },
    {
      "action": "transition_to_coldline",
      "condition": {"age_days": 90}
    },
    {
      "action": "delete",
      "condition": {"age_days": 365, "is_noncurrent": true}
    }
  ]
}
```

Every ML storage bucket should have lifecycle policies configured.

## Security Best Practices

### Authentication

| Approach | Security | Recommendation |
|----------|----------|----------------|
| IAM roles / Managed identities | Highest | Use for all production workloads |
| Service account keys | Medium | Rotate frequently, avoid if possible |
| Access keys | Lowest | Never commit to code, use secrets manager |

**Best practice**: Attach IAM roles to compute instances. Training jobs automatically inherit credentials without hardcoded secrets.

### Access Control

- **Block public access**: Enable by default on all buckets
- **Least privilege**: Grant minimum required permissions
- **Bucket policies**: Restrict access by IP, VPC, or identity
- **Signed URLs**: Time-limited access for temporary sharing

### Encryption

All three providers encrypt data at rest by default with provider-managed keys. For compliance requirements:
- **Customer-managed keys (CMK/CMEK)**: You control key rotation and access
- **Customer-supplied keys (CSE/CSEK)**: You manage keys entirely

In-transit encryption (TLS 1.2+) is enforced by default.

### Data Protection

- **Versioning**: Recover from accidental overwrites
- **Soft delete**: Recover from accidental deletions
- **Object lock / Immutability**: Prevent any modification (compliance/WORM)
- **Cross-region replication**: Disaster recovery

## Integration with ML Frameworks

### PyTorch

```python
import s3fs
from torch.utils.data import DataLoader

# S3 filesystem access
fs = s3fs.S3FileSystem()
with fs.open('s3://bucket/data.parquet') as f:
    df = pd.read_parquet(f)

# WebDataset for sharded training data
import webdataset as wds
dataset = wds.WebDataset('s3://bucket/shards/shard-{0000..0099}.tar')
loader = DataLoader(dataset, num_workers=8)
```

### TensorFlow

```python
import tensorflow as tf

# Native S3/GCS support
dataset = tf.data.TFRecordDataset('gs://bucket/train.tfrecord')
dataset = tf.data.TFRecordDataset('s3://bucket/train.tfrecord')

# Parallel reading from multiple files
files = tf.io.gfile.glob('gs://bucket/shards/*.tfrecord')
dataset = tf.data.TFRecordDataset(files, num_parallel_reads=8)
```

### JAX

```python
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp

# GCS checkpoint loading across TPU pods
checkpointer = ocp.StandardCheckpointer()
state = checkpointer.restore('gs://bucket/checkpoint/')
```

### Spark

```python
# Read from S3/GCS/Azure
df = spark.read.parquet('s3://bucket/features/')
df = spark.read.parquet('gs://bucket/features/')
df = spark.read.parquet('abfss://container@account.dfs.core.windows.net/features/')
```

## Common Pitfalls

### 1. Not Parallelizing Access
Single-threaded access achieves <1% of potential throughput. Always use multi-threaded data loaders.

### 2. Too Many Small Files
Millions of tiny files are slow and expensive. Consolidate into larger objects.

### 3. Ignoring Minimum Storage Durations
Deleting data before minimum duration (30-365 days depending on tier) still incurs full-duration charges.

### 4. Wrong Storage Class for Access Pattern
Using Archive tier for data that might need immediate access. Use Cold/Coldline for uncertain patterns.

### 5. Cross-Region Data Transfer
Placing storage and compute in different regions wastes money and time.

### 6. No Lifecycle Policies
Manual storage class management doesn't scale. Automate transitions and deletions.

### 7. Hardcoded Credentials
Access keys in code lead to security breaches. Use IAM roles / managed identities.

### 8. Public Bucket Misconfiguration
Accidentally exposing training data is a common security incident. Block public access by default.

## Decision Framework: Choosing a Provider

### Use S3 When:
- Your infrastructure is on AWS
- You need the broadest ecosystem of third-party tools
- Sub-10ms latency is critical (Express One Zone)
- You're storing vectors at scale (S3 Vectors)

### Use GCS When:
- Your infrastructure is on Google Cloud
- Distributed training checkpointing is frequent (HNS)
- You need instant archive access (no retrieval delay)
- TensorFlow/JAX are your primary frameworks

### Use Azure Blob When:
- Your infrastructure is on Azure
- Enterprise redundancy controls are required (LRS/ZRS/GRS/GZRS)
- You're using Azure ML, Synapse, or Databricks
- Multi-AZ low latency is needed (Premium Block Blob)

### Multi-Cloud Considerations

For multi-cloud strategies:
- Use abstraction libraries (fsspec, smart_open) for portable code
- Expect egress costs when moving data between clouds
- Each cloud's ML services integrate best with their native storage
- Consider data gravity: move compute to data, not vice versa

## Best Practices Checklist

### Configuration
-  Enable versioning for critical data (models, curated datasets)
-  Configure lifecycle policies for automatic tier transitions
-  Block public access unless explicitly required
-  Enable encryption (provider-managed minimum, CMK for compliance)
-  Set up cross-region replication for disaster recovery (critical data)

### Performance
-  Co-locate storage and compute in the same region
-  Use parallel access (hundreds to thousands of threads)
-  Consolidate small files into larger objects (100 MB - 1 GB)
-  Use prefix partitioning for S3 high-throughput workloads
-  Enable hierarchical namespace for checkpoint-heavy workloads (GCS/Azure)

### Cost
-  Use appropriate storage classes for access patterns
-  Configure lifecycle policies for automatic optimization
-  Monitor egress costs and use caching where beneficial
-  Delete incomplete multipart uploads automatically
-  Set budget alerts to catch unexpected cost spikes

### Security
-  Use IAM roles / managed identities instead of access keys
-  Apply least-privilege access policies
-  Enable soft delete for accidental deletion recovery
-  Audit access with cloud-native logging tools
-  Use private endpoints for sensitive workloads

## Conclusion

Object storage is the backbone of ML data infrastructure. Its combination of unlimited scale, extreme durability, and cost efficiency makes it the default choice for training data, model artifacts, and feature storage.

The three major providersS3, GCS, and Azure Bloboffer similar core capabilities with meaningful differentiators:

- **S3**: Ecosystem maturity and Express One Zone for latency
- **GCS**: Hierarchical Namespace for checkpointing and instant archive access
- **Azure**: In-account tiering and Premium multi-AZ storage

Choose based on your existing cloud platform first, then optimize for your specific workload characteristics. Regardless of provider, the fundamental principles apply: parallelize access, consolidate small files, automate lifecycle management, and always co-locate storage with compute.
