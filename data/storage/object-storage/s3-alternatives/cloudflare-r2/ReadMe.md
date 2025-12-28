# Cloudflare R2

## Summary

Cloudflare R2 is an S3-compatible object storage service that eliminates egress fees, making it particularly attractive for data-intensive ML workloads with significant read traffic. Built on Cloudflare's global edge network, R2 offers automatic geographic distribution and integration with Cloudflare Workers for serverless compute at the edge.

Key points to remember:

- Zero egress fees (the primary differentiator)
- Full S3 API compatibility
- Automatic global distribution via Cloudflare's network
- Strong integration with Cloudflare Workers for edge compute
- Simpler pricing: storage + Class A/B operations only
- No storage classes or tiers (single tier)

Pricing comparison (approximate):

| Cost Component | R2 | S3 Standard |
|----------------|-----|-------------|
| Storage (per GB/month) | $0.015 | $0.023 |
| Egress (per GB) | $0.00 | $0.09 |
| Class A ops (per million) | $4.50 | $5.00 (PUT) |
| Class B ops (per million) | $0.36 | $0.40 (GET) |

When to use R2:

- High egress workloads (model serving, data distribution)
- Public dataset hosting
- Multi-region inference where data must be fetched from storage
- Cost-sensitive workloads with predictable storage but variable reads
- Edge ML inference with Cloudflare Workers

When to consider alternatives:

- Need for storage classes/tiers (archival data)
- Deep integration with AWS/GCP/Azure ML services
- Advanced features like S3 Select, Glacier, or lifecycle transitions
- Regulatory requirements for specific cloud providers

---

## Understanding R2's Value Proposition

### The Egress Cost Problem

Traditional cloud object storage charges significant fees for data leaving their network:

| Provider | Egress Cost (per GB) |
|----------|---------------------|
| AWS S3 | $0.09 |
| Google Cloud Storage | $0.12 |
| Azure Blob | $0.087 |
| Cloudflare R2 | $0.00 |

For ML workloads, this matters significantly:

**Model Serving Example**: Serving a 5 GB model to 10,000 inference requests per day:
- Daily egress: 50 TB
- Monthly egress: 1.5 PB
- S3 cost: $135,000/month in egress alone
- R2 cost: $0 in egress

**Dataset Distribution Example**: Public ML dataset (100 GB) downloaded 1,000 times/month:
- S3 cost: ~$9,000/month
- R2 cost: Storage only (~$1.50/month)

### S3 API Compatibility

R2 implements the S3 API, enabling drop-in replacement for many workloads:

```python
import boto3

# S3 configuration
s3_client = boto3.client('s3',
    endpoint_url='https://<account_id>.r2.cloudflarestorage.com',
    aws_access_key_id='<R2_ACCESS_KEY>',
    aws_secret_access_key='<R2_SECRET_KEY>'
)

# Standard S3 operations work
s3_client.upload_file('model.pt', 'my-bucket', 'models/model.pt')
s3_client.download_file('my-bucket', 'models/model.pt', 'local_model.pt')

# List objects
response = s3_client.list_objects_v2(Bucket='my-bucket', Prefix='models/')
```

Supported S3 operations include:
- Object CRUD (GET, PUT, DELETE, HEAD)
- Multipart uploads
- Presigned URLs
- Bucket operations (create, delete, list)
- Object metadata and tagging

Notable S3 features not supported:
- S3 Select (query within objects)
- Storage classes (Glacier, Intelligent-Tiering)
- Object Lock (WORM compliance)
- Replication rules
- Bucket policies (use Cloudflare's access controls)

## R2 for ML Workflows

### Model Artifact Storage

R2 is well-suited for storing and distributing trained models:

```python
import boto3
from botocore.config import Config

# Configure R2 client
r2 = boto3.client('s3',
    endpoint_url=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

def upload_model(local_path, model_name, version):
    """Upload model to R2 with versioning in key."""
    key = f'models/{model_name}/v{version}/model.pt'
    r2.upload_file(local_path, 'ml-models', key)
    return f'https://ml-models.{ACCOUNT_ID}.r2.cloudflarestorage.com/{key}'

def download_model(model_name, version, local_path):
    """Download model from R2."""
    key = f'models/{model_name}/v{version}/model.pt'
    r2.download_file('ml-models', key, local_path)
```

### Training Data Storage

For training data, R2 works with standard data loading patterns:

```python
import s3fs

# Configure s3fs for R2
fs = s3fs.S3FileSystem(
    key=R2_ACCESS_KEY,
    secret=R2_SECRET_KEY,
    endpoint_url=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com'
)

# Read parquet files
import pandas as pd
with fs.open('my-bucket/training-data/features.parquet') as f:
    df = pd.read_parquet(f)

# Stream large files
with fs.open('my-bucket/training-data/large_dataset.csv') as f:
    for chunk in pd.read_csv(f, chunksize=10000):
        process(chunk)
```

### Public Dataset Hosting

R2's zero egress makes it ideal for hosting public ML datasets:

```python
# Enable public access for a bucket
# (Done via Cloudflare dashboard or API)

# Generate public URLs
def get_public_url(bucket, key):
    return f'https://{bucket}.{ACCOUNT_ID}.r2.cloudflarestorage.com/{key}'

# Or use custom domain with Cloudflare
def get_custom_domain_url(key):
    return f'https://data.yourdomain.com/{key}'
```

## Integration with Cloudflare Workers

R2 integrates natively with Cloudflare Workers for edge compute, enabling interesting ML serving patterns.

### Edge Model Serving

```javascript
// Cloudflare Worker for model serving
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Fetch model from R2
    const model = await env.MODELS_BUCKET.get('model.onnx');
    if (!model) {
      return new Response('Model not found', { status: 404 });
    }

    // Process with ONNX runtime (simplified example)
    const modelData = await model.arrayBuffer();
    // ... inference logic ...

    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
```

### Smart Caching for Inference

```javascript
// Cache model at edge, refresh periodically
export default {
  async fetch(request, env, ctx) {
    const cacheKey = new Request(request.url, request);
    const cache = caches.default;

    // Check cache first
    let response = await cache.match(cacheKey);
    if (response) {
      return response;
    }

    // Fetch from R2 if not cached
    const object = await env.BUCKET.get('predictions/latest.json');
    response = new Response(object.body, {
      headers: {
        'Cache-Control': 'max-age=3600',
        'Content-Type': 'application/json'
      }
    });

    // Cache at edge
    ctx.waitUntil(cache.put(cacheKey, response.clone()));
    return response;
  }
};
```

## Performance Considerations

### Throughput and Latency

R2 performance characteristics:

| Metric | Typical Value |
|--------|---------------|
| First-byte latency | 50-200ms |
| Throughput per connection | 10-50 MB/s |
| Parallel connection scaling | Linear |
| Global edge caching | <10ms when cached |

For high-throughput training data access, use parallel connections:

```python
from concurrent.futures import ThreadPoolExecutor
import boto3

def download_shard(args):
    bucket, key, local_path = args
    r2.download_file(bucket, key, local_path)

# Parallel download of training shards
shards = [(bucket, f'shards/shard-{i:04d}.tar', f'local/shard-{i:04d}.tar')
          for i in range(100)]

with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(download_shard, shards)
```

### Geographic Distribution

R2 automatically distributes data across Cloudflare's network. Unlike traditional object storage with explicit regions, R2 data is accessible globally with automatic edge caching.

For ML inference:
- Models cached at edge locations near users
- No need to replicate buckets across regions
- Consistent access patterns worldwide

### Multipart Uploads

For large model files, use multipart uploads:

```python
from boto3.s3.transfer import TransferConfig

# Configure multipart upload
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,  # 100 MB
    max_concurrency=10,
    multipart_chunksize=100 * 1024 * 1024
)

# Upload large model
r2.upload_file(
    'large_model.pt',
    'models-bucket',
    'models/large_model.pt',
    Config=config
)
```

## Limitations and Workarounds

### No Storage Classes

R2 has a single storage tier. For archival data:

**Workaround**: Use R2 for active data, archive to S3 Glacier or similar for long-term cold storage. The lack of egress fees means migrating data out of R2 is free.

### No S3 Select

Cannot query within objects:

**Workaround**: Use Parquet with row group filtering, or process data through Workers.

### No Object Lock

Cannot enforce WORM (Write Once Read Many) compliance:

**Workaround**: Use AWS S3 with Object Lock for compliance-critical data.

### Limited Lifecycle Policies

Basic expiration rules only, no storage class transitions:

```python
# Set object expiration (supported)
r2.put_bucket_lifecycle_configuration(
    Bucket='my-bucket',
    LifecycleConfiguration={
        'Rules': [{
            'ID': 'expire-old-checkpoints',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'checkpoints/'},
            'Expiration': {'Days': 30}
        }]
    }
)
```

## Cost Analysis for ML Workloads

### Scenario 1: Model Registry

Storing 100 model versions (average 2 GB each):
- Storage: 200 GB x $0.015 = $3/month
- 10,000 downloads/month (egress): $0
- Operations: Minimal

**Total**: ~$5/month vs ~$1,800/month on S3 (mostly egress)

### Scenario 2: Training Data Lake

10 TB training dataset, read 10x during training:
- Storage: 10 TB x $0.015 = $150/month
- Egress (100 TB): $0
- GET operations (millions): ~$50

**Total**: ~$200/month vs ~$9,150/month on S3

### Scenario 3: Inference Serving

Serving embeddings (1 KB average) at 1M requests/day:
- Storage: 1 GB x $0.015 = negligible
- Egress (30 GB/day, 900 GB/month): $0
- GET operations (30M/month): ~$11

**Total**: ~$15/month vs ~$100/month on S3

### Break-Even Analysis

R2 becomes more cost-effective as egress increases:

| Monthly Egress | S3 Cost | R2 Cost | Savings |
|----------------|---------|---------|---------|
| 100 GB | $9 | ~$2 | 78% |
| 1 TB | $90 | ~$5 | 94% |
| 10 TB | $900 | ~$20 | 98% |
| 100 TB | $9,000 | ~$100 | 99% |

## Migration from S3

### Compatibility Testing

Test S3 operations before migrating:

```python
def test_r2_compatibility():
    """Test basic S3 operations on R2."""
    test_key = 'compatibility-test/test.txt'

    # PUT
    r2.put_object(Bucket=BUCKET, Key=test_key, Body=b'test data')

    # GET
    response = r2.get_object(Bucket=BUCKET, Key=test_key)
    assert response['Body'].read() == b'test data'

    # HEAD
    r2.head_object(Bucket=BUCKET, Key=test_key)

    # LIST
    r2.list_objects_v2(Bucket=BUCKET, Prefix='compatibility-test/')

    # DELETE
    r2.delete_object(Bucket=BUCKET, Key=test_key)

    print("All compatibility tests passed")
```

### Data Migration

Use rclone for efficient migration:

```bash
# Configure rclone for both S3 and R2
rclone config

# Sync data from S3 to R2
rclone sync s3:source-bucket r2:dest-bucket --progress --transfers 32

# Verify migration
rclone check s3:source-bucket r2:dest-bucket
```

### Code Changes

Minimal changes required for S3 SDK usage:

```python
# Before (S3)
s3 = boto3.client('s3')

# After (R2)
r2 = boto3.client('s3',
    endpoint_url=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY
)

# All other code remains the same
```

## Best Practices

1. **Use for egress-heavy workloads**: R2's value is in zero egress fees. For storage-heavy, low-read workloads, the difference is smaller.

2. **Leverage edge caching**: Use Cloudflare's CDN and Workers for frequently accessed data.

3. **Parallel access**: Like all object storage, parallelize for throughput.

4. **Monitor operations costs**: While egress is free, Class A/B operations add up at scale.

5. **Consider hybrid approaches**: Use R2 for serving/distribution, traditional cloud storage for deep integration with ML services.

6. **Enable public access carefully**: While great for public datasets, ensure proper access controls for private data.
