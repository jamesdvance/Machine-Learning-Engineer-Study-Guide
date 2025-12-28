# Backblaze B2

## Summary

Backblaze B2 is a low-cost object storage service offering S3-compatible APIs at approximately one-fifth the price of major cloud providers. Known for transparency in pricing and reliability metrics, B2 is particularly attractive for large-scale data storage where cost is the primary concern. For ML practitioners, B2 offers compelling economics for training data archives, model backups, and datasets where access latency is not critical.

Key points to remember:

- Storage cost: $0.006/GB/month (vs $0.023 for S3 Standard)
- Free egress to Cloudflare, Fastly, and other CDN partners
- S3-compatible API alongside native B2 API
- Simple, transparent pricing with no hidden fees
- Single storage class (no tiering)
- 10 GB free storage, 1 GB free daily egress

Pricing comparison:

| Cost Component | B2 | S3 Standard | Savings |
|----------------|-----|-------------|---------|
| Storage (per GB/month) | $0.006 | $0.023 | 74% |
| Egress (per GB) | $0.01 | $0.09 | 89% |
| Class B ops (per 10k) | $0.004 | $0.004 | 0% |
| Class C ops (per 10k) | Free | $0.0004 | 100% |

When to use B2:

- Large archival datasets where cost matters most
- Training data that is written once and read occasionally
- Backup storage for models and checkpoints
- Public dataset hosting (with CDN integration)
- Budget-constrained projects with large storage needs

When to consider alternatives:

- Low-latency requirements (major clouds have edge presence)
- Deep integration with cloud ML services needed
- Geographic data residency requirements
- Need for storage classes (Glacier-like archival)

---

## Understanding B2's Cost Advantage

### Storage Economics

B2's primary value proposition is dramatically lower storage costs:

| Storage Volume | B2 Cost/Month | S3 Cost/Month | Annual Savings |
|----------------|---------------|---------------|----------------|
| 1 TB | $6 | $23 | $204 |
| 10 TB | $60 | $230 | $2,040 |
| 100 TB | $600 | $2,300 | $20,400 |
| 1 PB | $6,000 | $23,000 | $204,000 |

For ML organizations storing petabytes of training data, B2 can reduce storage costs by hundreds of thousands of dollars annually.

### Free Egress Partners

B2 offers free egress to select CDN and compute partners through the Bandwidth Alliance:

- Cloudflare
- Fastly
- Bunny CDN
- Vultr

This combination (B2 storage + Cloudflare CDN) provides extremely cost-effective data distribution.

### Transparent Pricing

Unlike major cloud providers with complex pricing pages, B2's pricing is straightforward:

- Storage: $0.006/GB/month
- Downloads: $0.01/GB (free to Alliance partners)
- Class B transactions (downloads, metadata): $0.004 per 10,000
- Class C transactions (uploads, deletes): Free

No minimum storage duration, no retrieval fees, no hidden charges.

## S3 Compatibility

B2 provides an S3-compatible API, enabling use of existing S3 tools and libraries:

```python
import boto3

# Configure boto3 for B2
b2_s3 = boto3.client('s3',
    endpoint_url='https://s3.us-west-004.backblazeb2.com',  # Region-specific
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)

# Standard S3 operations
b2_s3.upload_file('model.pt', 'my-bucket', 'models/model.pt')
b2_s3.download_file('my-bucket', 'models/model.pt', 'local_model.pt')
```

### Supported S3 Operations

- Object operations: GET, PUT, DELETE, HEAD, COPY
- Multipart uploads
- Presigned URLs
- Bucket operations
- Object metadata

### Limitations vs S3

- No S3 Select (query within objects)
- No storage classes
- No cross-region replication
- No object lock/WORM (available in native API)
- Limited bucket policies

### Native B2 API

B2 also has a native API with additional features:

```python
from b2sdk.v2 import B2Api, InMemoryAccountInfo

# Initialize B2 API
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account('production', B2_KEY_ID, B2_APP_KEY)

# Get bucket
bucket = b2_api.get_bucket_by_name('my-bucket')

# Upload with B2-specific features
bucket.upload_local_file(
    local_file='model.pt',
    file_name='models/model.pt',
    file_info={'model_version': '1.0', 'framework': 'pytorch'}
)
```

Native API advantages:
- Large file upload resumption
- File versioning with explicit control
- Application keys with fine-grained permissions
- Server-side encryption options

## B2 for ML Workflows

### Training Data Archives

B2 is ideal for storing large training datasets that are accessed infrequently:

```python
from b2sdk.v2 import B2Api, InMemoryAccountInfo
from b2sdk.v2 import ScanPoliciesManager, parse_sync_folder

def sync_dataset_to_b2(local_dir, bucket_name, remote_prefix):
    """Sync training dataset to B2 for archival."""
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', B2_KEY_ID, B2_APP_KEY)

    bucket = b2_api.get_bucket_by_name(bucket_name)

    # Sync with parallel uploads
    source = parse_sync_folder(local_dir, b2_api)
    dest = parse_sync_folder(f'b2://{bucket_name}/{remote_prefix}', b2_api)

    policies = ScanPoliciesManager()
    synchronizer = bucket.synchronize(
        source_folder=source,
        dest_folder=dest,
        now_millis=None,
        policies_manager=policies
    )

    for action in synchronizer:
        action.run()
```

### Model Checkpoint Storage

For periodic checkpoint backups where cost matters more than latency:

```python
import datetime

def backup_checkpoint_to_b2(checkpoint_path, model_name, epoch):
    """Backup training checkpoint to B2."""
    timestamp = datetime.datetime.now().isoformat()
    remote_key = f'checkpoints/{model_name}/{timestamp}_epoch{epoch}.pt'

    b2_s3.upload_file(
        checkpoint_path,
        'ml-backups',
        remote_key,
        ExtraArgs={
            'Metadata': {
                'model': model_name,
                'epoch': str(epoch),
                'timestamp': timestamp
            }
        }
    )

    return remote_key

def restore_checkpoint_from_b2(remote_key, local_path):
    """Restore checkpoint from B2."""
    b2_s3.download_file('ml-backups', remote_key, local_path)
```

### Dataset Distribution

Combine B2 with Cloudflare for cost-effective dataset distribution:

```python
# Store dataset in B2
def upload_public_dataset(local_path, dataset_name):
    """Upload dataset for public distribution."""
    remote_key = f'public-datasets/{dataset_name}'
    b2_s3.upload_file(local_path, 'public-data', remote_key)

    # B2 bucket configured with Cloudflare CDN
    # Free egress through Bandwidth Alliance
    return f'https://data.yourdomain.com/{remote_key}'
```

### Large File Handling

B2 supports files up to 10 TB with optimized large file uploads:

```python
from b2sdk.v2 import B2Api
from b2sdk.v2.transfer import LargeFileUploadState

def upload_large_model(local_path, bucket_name, remote_key):
    """Upload large model with resume capability."""
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # Automatic multipart upload for large files
    # Configurable part size (minimum 5 MB)
    bucket.upload_local_file(
        local_file=local_path,
        file_name=remote_key,
        min_part_size=100 * 1024 * 1024  # 100 MB parts
    )
```

## Performance Characteristics

### Throughput

B2 provides reasonable throughput for cost-optimized storage:

| Metric | Typical Value |
|--------|---------------|
| Single stream download | 50-100 MB/s |
| Multi-stream download | 200-500 MB/s |
| Upload speed | 50-100 MB/s |
| First-byte latency | 100-300ms |

For high-throughput needs, parallelize:

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_download(bucket, keys, local_dir, workers=10):
    """Download multiple files in parallel."""
    def download_one(key):
        local_path = f'{local_dir}/{key.split("/")[-1]}'
        b2_s3.download_file(bucket, key, local_path)
        return local_path

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(download_one, keys))

    return results
```

### Geographic Presence

B2 has fewer regions than major cloud providers:

| Region | Endpoint |
|--------|----------|
| US West | s3.us-west-004.backblazeb2.com |
| US East | s3.us-east-005.backblazeb2.com |
| EU Central | s3.eu-central-003.backblazeb2.com |

For global access with low latency, pair B2 with a CDN.

### Durability and Availability

- Durability: 99.999999999% (11 nines, same as major providers)
- Availability: 99.9% SLA

Data is replicated across multiple drives and availability zones within each region.

## Cost Optimization Strategies

### Lifecycle Rules

Configure automatic deletion for transient data:

```python
# Set lifecycle rules via B2 native API
bucket = b2_api.get_bucket_by_name('ml-checkpoints')

# Delete files older than 30 days
rules = [
    {
        'daysFromHidingToDeleting': 1,
        'daysFromUploadingToHiding': 30,
        'fileNamePrefix': 'checkpoints/'
    }
]
bucket.update(lifecycle_rules=rules)
```

### Version Management

Control file versions to manage costs:

```python
# Keep only latest version (hide/delete old versions)
def cleanup_old_versions(bucket_name, prefix, keep_versions=3):
    """Keep only recent versions of each file."""
    bucket = b2_api.get_bucket_by_name(bucket_name)

    for file_version, folder in bucket.ls(prefix, recursive=True):
        versions = list(bucket.list_file_versions(file_version.file_name))

        # Hide old versions (will be deleted by lifecycle rule)
        for old_version in versions[keep_versions:]:
            bucket.hide_file(old_version.file_name)
```

### Strategic Data Placement

Use B2 for appropriate workloads:

| Data Type | Recommendation |
|-----------|----------------|
| Active training data | Primary cloud storage |
| Training data archives | B2 |
| Model checkpoints (active) | Primary cloud storage |
| Model checkpoint backups | B2 |
| Public datasets | B2 + CDN |
| Experiment logs | B2 |

## Integration Patterns

### With Cloudflare R2

Use B2 for cold storage, R2 for hot data:

```python
def migrate_cold_to_b2(r2_bucket, b2_bucket, age_days=30):
    """Move old data from R2 to B2 for cost savings."""
    import datetime

    cutoff = datetime.datetime.now() - datetime.timedelta(days=age_days)

    for obj in r2_client.list_objects_v2(Bucket=r2_bucket)['Contents']:
        if obj['LastModified'] < cutoff:
            # Download from R2
            data = r2_client.get_object(Bucket=r2_bucket, Key=obj['Key'])

            # Upload to B2
            b2_client.put_object(
                Bucket=b2_bucket,
                Key=obj['Key'],
                Body=data['Body'].read()
            )

            # Delete from R2
            r2_client.delete_object(Bucket=r2_bucket, Key=obj['Key'])
```

### With rclone

rclone provides efficient sync between B2 and other storage:

```bash
# Sync from S3 to B2
rclone sync s3:my-bucket b2:my-bucket --progress --transfers 16

# Sync specific prefix
rclone sync s3:data/training b2:data/training --progress

# Mount B2 as filesystem (for read-heavy workloads)
rclone mount b2:my-bucket /mnt/b2 --vfs-cache-mode full
```

### With DVC

Use B2 as a DVC remote for dataset versioning:

```bash
# Configure DVC remote
dvc remote add -d b2storage s3://my-bucket/dvc
dvc remote modify b2storage endpointurl https://s3.us-west-004.backblazeb2.com

# Push datasets
dvc push

# Pull on training machines
dvc pull
```

## Security Features

### Access Control

B2 supports fine-grained access through application keys:

```python
# Create restricted application key
key = b2_api.create_key(
    capabilities=['listBuckets', 'readFiles', 'writeFiles'],
    key_name='training-pipeline',
    bucket_id=bucket.id_,
    name_prefix='training-data/'  # Restrict to prefix
)
```

### Encryption

- In-transit: TLS 1.2+ enforced
- At-rest: Server-side encryption (SSE-B2) available

```python
# Enable SSE for bucket
bucket.update(
    default_server_side_encryption=EncryptionSetting(
        mode=EncryptionMode.SSE_B2,
        algorithm=EncryptionAlgorithm.AES256
    )
)
```

### Object Lock

Available through native API for compliance:

```python
# Create bucket with object lock
bucket = b2_api.create_bucket(
    'compliance-bucket',
    'allPrivate',
    is_file_lock_enabled=True
)

# Set retention on upload
bucket.upload_local_file(
    local_file='audit.log',
    file_name='logs/audit.log',
    file_retention=FileRetentionSetting(
        mode=RetentionMode.GOVERNANCE,
        retain_until=datetime.datetime.now() + datetime.timedelta(days=365)
    )
)
```

## Practical Recommendations

1. **Use for archival workloads**: B2 shines for data that is written once and read occasionally.

2. **Pair with CDN for distribution**: Free egress to Cloudflare makes B2 + Cloudflare extremely cost-effective.

3. **Parallelize transfers**: Single-stream performance is modest; parallelize for large transfers.

4. **Consider hybrid architecture**: Use B2 for cold data, primary cloud storage for hot data.

5. **Leverage native API**: For large files and advanced features, the native B2 API offers more capability than S3 compatibility layer.

6. **Plan for latency**: B2 is not optimized for sub-100ms access. Use caching or CDN for latency-sensitive workloads.

7. **Monitor egress costs**: While cheaper than major clouds, non-Alliance egress still has costs ($0.01/GB).
