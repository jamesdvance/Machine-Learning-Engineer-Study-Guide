# Wasabi

## Summary

Wasabi is an S3-compatible object storage service that offers flat-rate pricing with no egress fees and no API request charges. Positioned as "hot cloud storage," Wasabi provides predictable costs and aims to match or exceed AWS S3 performance at a fraction of the price. For ML practitioners, Wasabi offers a compelling option for large datasets with unpredictable access patterns.

Key points to remember:

- Flat-rate storage pricing: $6.99/TB/month (no per-GB pricing at scale)
- No egress fees
- No API request charges
- S3-compatible API
- 90-day minimum storage duration (pay-as-you-delete)
- 11 nines durability, 99.99% availability
- Multiple global regions

Pricing comparison:

| Cost Component | Wasabi | S3 Standard |
|----------------|--------|-------------|
| Storage (per TB/month) | $6.99 | ~$23 |
| Egress (per GB) | $0.00 | $0.09 |
| API requests | $0.00 | $0.0004-$5.00 per 1000 |
| Minimum duration | 90 days | None |

When to use Wasabi:

- Large datasets with unpredictable read patterns
- High egress workloads where cost predictability matters
- S3 replacement for cost-sensitive workloads
- Multi-region storage needs with predictable pricing
- Backup and archive storage with potential for frequent access

When to consider alternatives:

- Short-lived data (90-day minimum)
- Need for storage classes/tiering
- Deep integration with cloud ML services
- Low-latency requirements (edge presence limited)

---

## Understanding Wasabi's Pricing Model

### Flat-Rate Economics

Wasabi's pricing model differs fundamentally from traditional cloud storage:

**Traditional (S3) Model:**
- Pay per GB stored
- Pay per API request (GET, PUT, LIST, etc.)
- Pay per GB egress
- Complex cost prediction

**Wasabi Model:**
- Pay per TB stored (flat rate)
- No API charges
- No egress charges
- Simple, predictable costs

### Cost Comparison Scenarios

**Scenario 1: Active Training Dataset (10 TB, heavy reads)**
- Reads: 100 TB egress/month, 100M GET requests

| Provider | Storage | Egress | Requests | Total |
|----------|---------|--------|----------|-------|
| S3 | $230 | $9,000 | $40 | $9,270 |
| Wasabi | $69.90 | $0 | $0 | $69.90 |

**Scenario 2: Archival Data (100 TB, minimal access)**
- Reads: 1 TB egress/month, 1M GET requests

| Provider | Storage | Egress | Requests | Total |
|----------|---------|--------|----------|-------|
| S3 Standard | $2,300 | $90 | $0.40 | $2,390 |
| Wasabi | $699 | $0 | $0 | $699 |

### The 90-Day Minimum

Wasabi charges for a minimum of 90 days per object:

- Upload on day 1, delete on day 30: charged for 90 days
- Upload on day 1, delete on day 100: charged for 100 days
- Objects overwritten count as new uploads

**ML Implications:**
- Checkpoints deleted frequently may incur higher effective costs
- Training data stored long-term sees full benefit
- Calculate effective cost for your data lifecycle

```python
def calculate_effective_wasabi_cost(storage_gb, retention_days):
    """Calculate effective Wasabi cost accounting for minimum duration."""
    min_days = 90
    effective_days = max(retention_days, min_days)
    monthly_cost_per_tb = 6.99
    daily_cost_per_gb = (monthly_cost_per_tb / 1000) / 30

    return storage_gb * daily_cost_per_gb * effective_days
```

## S3 API Compatibility

Wasabi provides comprehensive S3 API support:

```python
import boto3

# Configure boto3 for Wasabi
wasabi = boto3.client('s3',
    endpoint_url='https://s3.wasabisys.com',  # Or region-specific
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY
)

# Standard S3 operations
wasabi.upload_file('model.pt', 'ml-bucket', 'models/model.pt')
wasabi.download_file('ml-bucket', 'models/model.pt', 'local_model.pt')

# Multipart upload for large files
config = boto3.s3.transfer.TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    max_concurrency=10
)
wasabi.upload_file('large_model.pt', 'ml-bucket', 'models/large.pt', Config=config)
```

### Supported Operations

Wasabi supports most S3 operations:

- Object CRUD: GET, PUT, DELETE, HEAD, COPY
- Multipart uploads
- Presigned URLs
- Bucket operations and policies
- Versioning
- Object tagging
- Cross-origin resource sharing (CORS)
- Server-side encryption

### Notable Differences from S3

- No S3 Select
- No storage classes (all storage is "hot")
- No Glacier-like archival
- No S3 Inventory
- No S3 Batch Operations
- Limited bucket policy syntax

## Wasabi for ML Workflows

### Training Data Storage

Wasabi works well for large training datasets:

```python
import s3fs

# Configure s3fs for Wasabi
fs = s3fs.S3FileSystem(
    key=WASABI_ACCESS_KEY,
    secret=WASABI_SECRET_KEY,
    endpoint_url='https://s3.wasabisys.com',
    client_kwargs={'region_name': 'us-east-1'}
)

# Read training data
import pandas as pd
with fs.open('bucket/training-data/features.parquet') as f:
    df = pd.read_parquet(f)

# List dataset shards
shards = fs.glob('bucket/training-data/shards/*.tar')
print(f"Found {len(shards)} training shards")
```

### Model Registry

Store models with free distribution:

```python
from datetime import datetime

class WasabiModelRegistry:
    def __init__(self, bucket):
        self.bucket = bucket
        self.client = wasabi

    def save_model(self, model_path, name, version, metadata=None):
        """Save model with metadata."""
        key = f'models/{name}/v{version}/model.pt'

        extra_args = {'Metadata': metadata or {}}
        extra_args['Metadata']['upload_time'] = datetime.now().isoformat()

        self.client.upload_file(model_path, self.bucket, key, ExtraArgs=extra_args)
        return key

    def load_model(self, name, version, local_path):
        """Load model from registry."""
        key = f'models/{name}/v{version}/model.pt'
        self.client.download_file(self.bucket, key, local_path)

    def list_versions(self, name):
        """List all versions of a model."""
        prefix = f'models/{name}/'
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter='/'
        )
        return [p['Prefix'].split('/')[-2] for p in response.get('CommonPrefixes', [])]
```

### Checkpoint Backup

For training checkpoint backup (with 90-day minimum consideration):

```python
def should_backup_to_wasabi(checkpoint_importance):
    """
    Determine if checkpoint should go to Wasabi.
    Only backup important checkpoints due to 90-day minimum.
    """
    # Backup every Nth checkpoint or final models
    return checkpoint_importance in ['final', 'milestone', 'best']

def backup_checkpoint(local_path, run_id, epoch, importance):
    """Backup checkpoint to Wasabi with importance filtering."""
    if not should_backup_to_wasabi(importance):
        return None

    key = f'checkpoints/{run_id}/epoch_{epoch:04d}_{importance}.pt'
    wasabi.upload_file(local_path, 'ml-checkpoints', key)
    return key
```

### Dataset Distribution

With free egress, Wasabi is excellent for dataset distribution:

```python
def generate_dataset_download_url(dataset_name, expiry_hours=24):
    """Generate presigned URL for dataset download."""
    key = f'public-datasets/{dataset_name}'

    url = wasabi.generate_presigned_url(
        'get_object',
        Params={'Bucket': 'datasets', 'Key': key},
        ExpiresIn=expiry_hours * 3600
    )
    return url

# No egress costs regardless of download volume
```

## Performance Characteristics

### Throughput

Wasabi claims to match or exceed S3 performance:

| Metric | Typical Value |
|--------|---------------|
| Single stream throughput | 100+ MB/s |
| Multi-stream throughput | 500+ MB/s |
| First-byte latency | 50-150ms |
| Request rate | High (no throttling limits) |

### Parallel Access

Like all object storage, parallelize for best throughput:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_upload(local_files, bucket, prefix, max_workers=20):
    """Upload files in parallel."""
    def upload_one(local_file):
        key = f'{prefix}/{local_file.name}'
        wasabi.upload_file(str(local_file), bucket, key)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_one, f): f for f in local_files}
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    return results
```

### Geographic Regions

Wasabi has global presence:

| Region | Endpoint |
|--------|----------|
| US East 1 | s3.us-east-1.wasabisys.com |
| US East 2 | s3.us-east-2.wasabisys.com |
| US Central 1 | s3.us-central-1.wasabisys.com |
| US West 1 | s3.us-west-1.wasabisys.com |
| EU Central 1 | s3.eu-central-1.wasabisys.com |
| EU Central 2 | s3.eu-central-2.wasabisys.com |
| EU West 1 | s3.eu-west-1.wasabisys.com |
| EU West 2 | s3.eu-west-2.wasabisys.com |
| AP Northeast 1 | s3.ap-northeast-1.wasabisys.com |
| AP Northeast 2 | s3.ap-northeast-2.wasabisys.com |
| AP Southeast 1 | s3.ap-southeast-1.wasabisys.com |
| AP Southeast 2 | s3.ap-southeast-2.wasabisys.com |

Choose regions close to your compute for best performance.

## Versioning and Data Protection

### Object Versioning

```python
# Enable versioning
wasabi.put_bucket_versioning(
    Bucket='ml-bucket',
    VersioningConfiguration={'Status': 'Enabled'}
)

# Upload creates new version
wasabi.upload_file('model_v2.pt', 'ml-bucket', 'models/model.pt')

# List versions
versions = wasabi.list_object_versions(Bucket='ml-bucket', Prefix='models/model.pt')
for v in versions.get('Versions', []):
    print(f"Version: {v['VersionId']}, Modified: {v['LastModified']}")

# Download specific version
wasabi.download_file(
    'ml-bucket',
    'models/model.pt',
    'old_model.pt',
    ExtraArgs={'VersionId': 'specific-version-id'}
)
```

### Object Lock

Wasabi supports object lock for compliance:

```python
# Create bucket with object lock
wasabi.create_bucket(
    Bucket='compliance-bucket',
    ObjectLockEnabledForBucket=True
)

# Set retention on upload
wasabi.put_object(
    Bucket='compliance-bucket',
    Key='audit/log.json',
    Body=log_data,
    ObjectLockMode='GOVERNANCE',
    ObjectLockRetainUntilDate=datetime(2025, 12, 31)
)
```

## Integration Patterns

### With rclone

rclone works well with Wasabi:

```bash
# Configure rclone
rclone config
# Choose s3, provider wasabi

# Sync dataset
rclone sync /local/data wasabi:bucket/data --progress --transfers 16

# Mount for filesystem access
rclone mount wasabi:bucket /mnt/wasabi --vfs-cache-mode full
```

### With DVC

Use Wasabi as DVC remote:

```bash
# Add Wasabi remote
dvc remote add -d wasabi s3://bucket/dvc
dvc remote modify wasabi endpointurl https://s3.wasabisys.com

# Configure credentials
dvc remote modify wasabi access_key_id $WASABI_ACCESS_KEY
dvc remote modify wasabi secret_access_key $WASABI_SECRET_KEY

# Push/pull data
dvc push
dvc pull
```

### With PyTorch DataLoader

```python
import webdataset as wds

# WebDataset with Wasabi
def wasabi_url(shard_pattern):
    return f's3://bucket/shards/{shard_pattern}'

dataset = wds.WebDataset(
    wasabi_url('shard-{0000..0099}.tar'),
    handler=wds.handlers.warn_and_continue
).decode('pil').to_tuple('jpg', 'json')

loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## Security Features

### Access Control

Wasabi supports IAM-style access control:

```python
# Create bucket policy
policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "AllowMLTeam",
        "Effect": "Allow",
        "Principal": {"AWS": ["arn:aws:iam::ACCOUNT:user/ml-team"]},
        "Action": ["s3:GetObject", "s3:PutObject"],
        "Resource": ["arn:aws:s3:::ml-bucket/*"]
    }]
}

wasabi.put_bucket_policy(Bucket='ml-bucket', Policy=json.dumps(policy))
```

### Encryption

```python
# Server-side encryption with Wasabi-managed keys
wasabi.put_object(
    Bucket='secure-bucket',
    Key='sensitive-data.pt',
    Body=data,
    ServerSideEncryption='AES256'
)

# Default encryption for bucket
wasabi.put_bucket_encryption(
    Bucket='secure-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'AES256'
            }
        }]
    }
)
```

## Cost Optimization

### Account for 90-Day Minimum

```python
def estimate_wasabi_cost(data_inventory):
    """
    Estimate monthly Wasabi cost accounting for minimum duration.

    data_inventory: List of (size_gb, retention_days) tuples
    """
    monthly_rate_per_tb = 6.99
    min_retention = 90

    total_gb_months = 0
    for size_gb, retention_days in data_inventory:
        effective_months = max(retention_days, min_retention) / 30
        total_gb_months += size_gb * effective_months

    return (total_gb_months / 1000) * monthly_rate_per_tb
```

### Lifecycle Policies

While Wasabi doesn't have storage classes, use lifecycle for cleanup:

```python
# Delete old checkpoints (after accounting for 90-day minimum)
wasabi.put_bucket_lifecycle_configuration(
    Bucket='checkpoints',
    LifecycleConfiguration={
        'Rules': [{
            'ID': 'delete-old-checkpoints',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'temp-checkpoints/'},
            'Expiration': {'Days': 90}  # Match minimum to avoid waste
        }]
    }
)
```

### Comparison Decision Matrix

| Factor | Wasabi | Cloudflare R2 | Backblaze B2 |
|--------|--------|---------------|--------------|
| Storage cost | $6.99/TB | $15/TB | $6/TB |
| Egress cost | Free | Free | $0.01/GB (free to CDN) |
| API costs | Free | Per-operation | Minimal |
| Minimum duration | 90 days | None | None |
| Regions | 12+ | Global edge | 3 |
| Edge compute | No | Workers | No |

## Practical Recommendations

1. **Mind the 90-day minimum**: Plan data lifecycle around this constraint. Wasabi is best for data stored 90+ days.

2. **Use for stable datasets**: Training data and model archives benefit most; frequently-changing checkpoints less so.

3. **Leverage free egress**: High-read workloads see significant savings compared to traditional cloud storage.

4. **Choose appropriate regions**: Place buckets near compute for best latency.

5. **Consider hybrid approaches**: Use Wasabi for bulk storage, cloud-native storage for tight ML service integration.

6. **Monitor effective costs**: Calculate actual cost-per-GB considering retention patterns.

7. **Use versioning wisely**: Versions count toward minimum storage, so clean up aggressively if costs matter.
