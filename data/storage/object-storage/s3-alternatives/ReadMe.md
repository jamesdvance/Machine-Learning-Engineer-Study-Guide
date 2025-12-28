# S3-Compatible Alternatives

## Summary

S3-compatible object storage providers offer significant cost savings over major cloud providers while maintaining API compatibility with the S3 ecosystem. These alternatives are particularly valuable for ML workloads with large datasets, high egress requirements, or predictable storage needs. The three leading alternatives each optimize for different use cases.

Quick comparison:

| Feature | Cloudflare R2 | Backblaze B2 | Wasabi |
|---------|---------------|--------------|--------|
| Storage (per TB/mo) | $15 | $6 | $6.99 |
| Egress | Free | $0.01/GB (free to CDN) | Free |
| API requests | Per-operation | Minimal | Free |
| Min retention | None | None | 90 days |
| Edge compute | Workers | No | No |
| Best for | High egress, edge | Lowest cost | Predictable pricing |

Key decision factors:

- **Highest egress volume?** Cloudflare R2 or Wasabi (both free egress)
- **Lowest absolute cost?** Backblaze B2 (especially with CDN partners)
- **Most predictable pricing?** Wasabi (flat rate, no per-request charges)
- **Edge compute integration?** Cloudflare R2 (Workers)
- **Short-lived data?** Cloudflare R2 or Backblaze B2 (no minimum retention)

When to use S3 alternatives:

- Storage costs are a significant budget item
- Egress costs are unpredictable or high
- Deep cloud ML service integration is not required
- S3 API compatibility is sufficient
- Data residency can be satisfied by available regions

When to stick with major clouds:

- Tight integration with cloud ML services (SageMaker, Vertex AI, Azure ML)
- Need for advanced features (S3 Select, storage classes, Glacier)
- Regulatory requirements for specific providers
- Sub-10ms latency requirements
- Enterprise support contracts required

---

## Cost Analysis

### The Economics of Alternative Storage

Major cloud providers charge premium prices for object storage, particularly for egress. S3 alternatives disrupt this model:

**Monthly cost for 100 TB storage, 50 TB egress:**

| Provider | Storage | Egress | Total |
|----------|---------|--------|-------|
| AWS S3 | $2,300 | $4,500 | $6,800 |
| Google Cloud Storage | $2,000 | $6,000 | $8,000 |
| Azure Blob | $1,800 | $4,350 | $6,150 |
| Cloudflare R2 | $1,500 | $0 | $1,500 |
| Backblaze B2 | $600 | $500 | $1,100 |
| Wasabi | $699 | $0 | $699 |

**Annual savings at this scale:**
- R2 vs S3: ~$63,600
- B2 vs S3: ~$68,400
- Wasabi vs S3: ~$73,200

### When Each Alternative Wins

**Cloudflare R2 wins when:**
- Egress is the primary cost driver
- Edge compute (Workers) adds value
- Global distribution matters
- No minimum retention constraints

**Backblaze B2 wins when:**
- Absolute lowest cost is the goal
- CDN partners (Cloudflare, Fastly) handle distribution
- Egress is moderate or uses free partners
- Native B2 API features are valuable

**Wasabi wins when:**
- Cost predictability is paramount
- Data is stored 90+ days
- High API request volume
- Simple pricing is preferred

## S3 API Compatibility

All three providers implement the S3 API, enabling use of existing tools:

```python
import boto3

# Generic S3-compatible client factory
def create_s3_client(provider, credentials):
    endpoints = {
        'r2': f'https://{credentials["account_id"]}.r2.cloudflarestorage.com',
        'b2': f'https://s3.{credentials["region"]}.backblazeb2.com',
        'wasabi': f'https://s3.{credentials["region"]}.wasabisys.com'
    }

    return boto3.client('s3',
        endpoint_url=endpoints[provider],
        aws_access_key_id=credentials['access_key'],
        aws_secret_access_key=credentials['secret_key']
    )

# Use interchangeably
r2 = create_s3_client('r2', r2_creds)
b2 = create_s3_client('b2', b2_creds)
wasabi = create_s3_client('wasabi', wasabi_creds)
```

### Compatibility Matrix

| Feature | R2 | B2 | Wasabi |
|---------|----|----|--------|
| GET/PUT/DELETE | Yes | Yes | Yes |
| Multipart upload | Yes | Yes | Yes |
| Presigned URLs | Yes | Yes | Yes |
| Versioning | Yes | Yes | Yes |
| Lifecycle rules | Basic | Basic | Basic |
| Bucket policies | Limited | Limited | Yes |
| S3 Select | No | No | No |
| Object Lock | No | Yes (native) | Yes |
| Storage classes | No | No | No |

### Working with Multiple Providers

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class StorageConfig:
    provider: str
    bucket: str
    prefix: str

class MultiProviderStorage:
    """Abstraction over multiple S3-compatible providers."""

    def __init__(self, configs: dict):
        self.clients = {}
        for name, config in configs.items():
            self.clients[name] = create_s3_client(config.provider, config.credentials)
            self.configs[name] = config

    def upload(self, storage_name: str, local_path: str, key: str):
        config = self.configs[storage_name]
        full_key = f'{config.prefix}/{key}' if config.prefix else key
        self.clients[storage_name].upload_file(local_path, config.bucket, full_key)

    def download(self, storage_name: str, key: str, local_path: str):
        config = self.configs[storage_name]
        full_key = f'{config.prefix}/{key}' if config.prefix else key
        self.clients[storage_name].download_file(config.bucket, full_key, local_path)

# Usage
storage = MultiProviderStorage({
    'hot': StorageConfig('r2', 'active-data', 'training'),
    'cold': StorageConfig('b2', 'archive', 'backup'),
    'models': StorageConfig('wasabi', 'ml-models', 'production')
})

storage.upload('hot', 'dataset.tar', 'datasets/v1.tar')
storage.download('models', 'model.pt', 'local_model.pt')
```

## ML Workflow Patterns

### Tiered Storage Architecture

Use different providers for different data tiers:

```
Training Pipeline:
                                    +------------------+
                                    |  Active Training |
  +----------------+  preprocess    |   (Cloudflare R2)|
  | Raw Data       | ------------> |   - Fast access  |
  | (Backblaze B2) |                |   - Free egress  |
  | - Lowest cost  |                +--------+---------+
  +----------------+                         |
                                            | train
                                            v
                                    +------------------+
                                    |  Model Registry  |
                                    |    (Wasabi)      |
                                    |  - Predictable   |
                                    |  - Long-term     |
                                    +------------------+
```

### Data Pipeline Example

```python
class MLDataPipeline:
    def __init__(self):
        self.archive = create_s3_client('b2', b2_creds)      # Cheap archive
        self.training = create_s3_client('r2', r2_creds)     # Fast training
        self.models = create_s3_client('wasabi', wasabi_creds)  # Model storage

    def prepare_training_data(self, dataset_name, version):
        """Move data from archive to training storage."""
        archive_key = f'datasets/{dataset_name}/v{version}.tar'
        training_key = f'active/{dataset_name}.tar'

        # Download from B2 (low cost storage)
        local_path = f'/tmp/{dataset_name}.tar'
        self.archive.download_file('archive-bucket', archive_key, local_path)

        # Upload to R2 (fast, free egress for training)
        self.training.upload_file(local_path, 'training-bucket', training_key)

        return f's3://training-bucket/{training_key}'

    def save_model(self, model_path, name, version):
        """Save trained model to long-term storage."""
        key = f'models/{name}/v{version}/model.pt'
        self.models.upload_file(model_path, 'models-bucket', key)
        return key
```

### Public Dataset Hosting

All three providers enable cost-effective public dataset hosting:

```python
def host_public_dataset(provider_client, bucket, dataset_path, dataset_name):
    """Upload and configure public dataset access."""

    # Upload dataset
    key = f'public/{dataset_name}'
    provider_client.upload_file(dataset_path, bucket, key)

    # Generate long-lived presigned URL (or use public bucket)
    url = provider_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=365 * 24 * 3600  # 1 year
    )

    return url

# Cost comparison for 10 TB dataset, 1000 downloads:
# S3: $230 storage + $9000 egress = $9,230/month
# R2: $150 storage + $0 egress = $150/month
# B2: $60 storage + $100 egress = $160/month
# Wasabi: $70 storage + $0 egress = $70/month
```

## Migration Strategies

### From S3 to Alternatives

```bash
# Using rclone for migration
rclone sync s3:source-bucket wasabi:dest-bucket \
    --progress \
    --transfers 32 \
    --checkers 16

# Verify migration
rclone check s3:source-bucket wasabi:dest-bucket

# For large datasets, use parallel jobs
parallel -j 4 rclone copy s3:bucket/{} wasabi:bucket/{} ::: prefix1 prefix2 prefix3 prefix4
```

### Gradual Migration

```python
class MigrationProxy:
    """Proxy that reads from old storage, writes to both."""

    def __init__(self, old_client, new_client, old_bucket, new_bucket):
        self.old = old_client
        self.new = new_client
        self.old_bucket = old_bucket
        self.new_bucket = new_bucket

    def get_object(self, key):
        """Read from old storage (can be switched later)."""
        return self.old.get_object(Bucket=self.old_bucket, Key=key)

    def put_object(self, key, body):
        """Write to both storages during migration."""
        self.old.put_object(Bucket=self.old_bucket, Key=key, Body=body)
        self.new.put_object(Bucket=self.new_bucket, Key=key, Body=body)

    def migrate_object(self, key):
        """Copy single object from old to new."""
        obj = self.old.get_object(Bucket=self.old_bucket, Key=key)
        self.new.put_object(Bucket=self.new_bucket, Key=key, Body=obj['Body'].read())
```

## Performance Comparison

### Throughput Testing

```python
import time
from concurrent.futures import ThreadPoolExecutor

def benchmark_provider(client, bucket, object_size_mb=100, num_objects=10):
    """Benchmark upload/download performance."""

    # Generate test data
    test_data = b'x' * (object_size_mb * 1024 * 1024)

    # Upload benchmark
    start = time.time()
    for i in range(num_objects):
        client.put_object(Bucket=bucket, Key=f'bench/obj_{i}', Body=test_data)
    upload_time = time.time() - start
    upload_throughput = (object_size_mb * num_objects) / upload_time

    # Download benchmark
    start = time.time()
    for i in range(num_objects):
        client.get_object(Bucket=bucket, Key=f'bench/obj_{i}')['Body'].read()
    download_time = time.time() - start
    download_throughput = (object_size_mb * num_objects) / download_time

    return {
        'upload_mb_per_sec': upload_throughput,
        'download_mb_per_sec': download_throughput
    }
```

### Typical Performance Expectations

| Provider | Single-stream | Multi-stream | Latency |
|----------|---------------|--------------|---------|
| R2 | 50-100 MB/s | 200-500 MB/s | 50-200ms |
| B2 | 50-100 MB/s | 200-500 MB/s | 100-300ms |
| Wasabi | 100+ MB/s | 500+ MB/s | 50-150ms |

All providers scale well with parallel connections. For training pipelines, use 10-50 concurrent connections.

## Limitations and Workarounds

### No Storage Classes

None of the alternatives offer Glacier-like archival tiers.

**Workaround:** Use B2 for "archive" (already cheapest), or maintain S3 Glacier for truly cold data.

### No S3 Select

Cannot query within objects.

**Workaround:** Use appropriate file formats (Parquet with predicate pushdown) and process client-side.

### Limited Enterprise Features

Reduced compliance certifications, SLAs, and support compared to major clouds.

**Workaround:** Use for non-critical data, maintain cloud storage for regulated workloads.

### Regional Availability

Fewer regions than major clouds.

**Workaround:** Use CDN for global distribution, or multi-provider for regional coverage.

## Decision Framework

### Choose Cloudflare R2 when:
- Egress is your primary cost concern
- Edge compute (Workers) integration adds value
- Global distribution without replication management is valuable
- You need flexible data lifecycle (no minimum retention)

### Choose Backblaze B2 when:
- Absolute lowest storage cost is the priority
- You use CDN partners (Cloudflare, Fastly) for free egress
- Native B2 features (object lock, large file resume) are needed
- Budget is extremely constrained

### Choose Wasabi when:
- Cost predictability matters most
- Data is stored long-term (90+ days)
- High API request volume
- Simple, all-inclusive pricing is preferred

### Hybrid Approach

Many organizations use multiple providers:

| Data Type | Provider | Rationale |
|-----------|----------|-----------|
| Active training data | R2 | Free egress, fast access |
| Raw archives | B2 | Lowest storage cost |
| Model registry | Wasabi | Predictable, long-term |
| Public datasets | R2 + CDN | Free distribution |
| Compliance data | Cloud provider | Enterprise features |

## Best Practices

1. **Start with workload analysis**: Calculate current storage, egress, and request volumes before choosing.

2. **Test compatibility**: Verify your tools work with S3 compatibility layer before committing.

3. **Consider total cost**: Include egress, requests, and minimum retention in calculations.

4. **Plan for migration**: Use abstraction layers and tools like rclone for flexibility.

5. **Monitor after migration**: Track actual costs and performance versus projections.

6. **Maintain fallback options**: Keep critical data accessible through multiple paths.

7. **Evaluate regularly**: Pricing and features change; reassess annually.
