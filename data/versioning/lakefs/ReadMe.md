# LakeFS

## Summary

LakeFS is an open-source platform that provides Git-like version control for data lakes. It layers branching, committing, and merging semantics directly on top of object storage (S3, GCS, Azure), enabling atomic operations across millions of files without data duplication. Unlike DVC which extends Git, LakeFS provides a native data versioning layer that works transparently with existing data tools.

Key points to remember:

- Git semantics (branch, commit, merge, revert) for object storage
- Zero-copy branching: branches are metadata-only, no data duplication
- Works transparently with S3-compatible tools (Spark, Trino, Pandas)
- Atomic commits ensure consistency across multi-file operations
- Pre-commit and pre-merge hooks enable data quality gates
- Supports all object storage: S3, GCS, Azure Blob, MinIO
- Self-hosted server or managed service available

## Core Concepts

### Architecture

```
Applications (Spark, Trino, Python)
              |
              v
+----------------------------+
|         LakeFS             |
|  - Branching/Merging       |
|  - Commit History          |
|  - Access Control          |
+----------------------------+
              |
              v
+----------------------------+
|    Object Storage          |
|  (S3, GCS, Azure, MinIO)   |
+----------------------------+
```

LakeFS provides an S3-compatible API that intercepts requests and adds versioning semantics. Applications use the LakeFS endpoint instead of S3 directly, gaining version control without code changes.

### Key Concepts

**Repository**: A versioned namespace for data, similar to a Git repository. Each repository has a default branch (usually `main`).

**Branch**: An isolated workspace for changes. Branches are metadata pointers, not data copies. Creating a branch is instant regardless of data size.

**Commit**: An immutable snapshot of all data at a point in time. Commits are content-addressed and include metadata (message, timestamp, author).

**Merge**: Combines changes from one branch into another. LakeFS supports both merge and rebase operations.

**Reference**: Points to a commit (branch, tag, or commit ID).

### Zero-Copy Branching

```
main branch:
  file1.parquet -> object-abc
  file2.parquet -> object-def

After: lakectl branch create dev --source main

dev branch:
  file1.parquet -> object-abc  (same object!)
  file2.parquet -> object-def  (same object!)

After writing new file on dev:
dev branch:
  file1.parquet -> object-abc
  file2.parquet -> object-def
  file3.parquet -> object-ghi  (new object)
```

Branches share underlying objects. Only changed files create new objects. This enables instant branching of petabyte-scale datasets.

## Basic Workflow

### Setup

```bash
# Install lakectl CLI
curl -LO https://github.com/treeverse/lakeFS/releases/download/vX.Y.Z/lakeFS_X.Y.Z_linux_amd64.tar.gz
tar -xzvf lakeFS_*.tar.gz

# Configure credentials
lakectl config
# Enter LakeFS endpoint, access key, secret key

# Or use environment variables
export LAKECTL_SERVER_ENDPOINT_URL=https://lakefs.example.com
export LAKECTL_CREDENTIALS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### Creating a Repository

```bash
# Create repository with S3 storage backend
lakectl repo create lakefs://my-repo s3://my-bucket/lakefs-data

# Create repository with default branch name
lakectl repo create lakefs://my-repo s3://my-bucket/lakefs-data --default-branch main
```

### Branching and Committing

```bash
# Create a branch
lakectl branch create lakefs://my-repo/feature-branch --source lakefs://my-repo/main

# List branches
lakectl branch list lakefs://my-repo

# Upload data (via lakectl)
lakectl fs upload lakefs://my-repo/feature-branch/data/file.parquet ./local-file.parquet

# Or use S3 API directly (any S3-compatible tool works)
aws s3 cp local-file.parquet s3://my-repo/feature-branch/data/file.parquet --endpoint-url https://lakefs.example.com

# Check uncommitted changes
lakectl diff lakefs://my-repo/feature-branch

# Commit changes
lakectl commit lakefs://my-repo/feature-branch -m "Add training data"

# View commit history
lakectl log lakefs://my-repo/feature-branch
```

### Merging

```bash
# Merge feature branch into main
lakectl merge lakefs://my-repo/feature-branch lakefs://my-repo/main

# Handle conflicts if any
lakectl diff lakefs://my-repo/main..lakefs://my-repo/feature-branch

# Revert a commit if needed
lakectl revert lakefs://my-repo/main <commit-id>
```

### Accessing Data at Specific Versions

```bash
# List files at specific commit
lakectl fs ls lakefs://my-repo/<commit-id>/data/

# Read file at specific commit
lakectl fs cat lakefs://my-repo/<commit-id>/data/file.csv

# Create tag for easy reference
lakectl tag create lakefs://my-repo/v1.0 lakefs://my-repo/main

# Access via tag
lakectl fs ls lakefs://my-repo/v1.0/data/
```

## Integration with Data Tools

### Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.hadoop.fs.s3a.endpoint", "https://lakefs.example.com") \
    .config("spark.hadoop.fs.s3a.access.key", "AKIAIOSFODNN7EXAMPLE") \
    .config("spark.hadoop.fs.s3a.secret.key", "wJalrXUtnFEMI/K7MDENG") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# Read from specific branch
df = spark.read.parquet("s3a://my-repo/main/data/training/")

# Read from specific commit
df = spark.read.parquet("s3a://my-repo/<commit-id>/data/training/")

# Write to feature branch
df.write.parquet("s3a://my-repo/feature-branch/data/processed/")
```

### Pandas/PyArrow

```python
import pandas as pd
import s3fs

# Configure S3 filesystem for LakeFS
fs = s3fs.S3FileSystem(
    endpoint_url="https://lakefs.example.com",
    key="AKIAIOSFODNN7EXAMPLE",
    secret="wJalrXUtnFEMI/K7MDENG"
)

# Read from main branch
df = pd.read_parquet("s3://my-repo/main/data/file.parquet", filesystem=fs)

# Read from specific commit
df = pd.read_parquet("s3://my-repo/<commit-id>/data/file.parquet", filesystem=fs)

# Write to feature branch
df.to_parquet("s3://my-repo/feature/data/output.parquet", filesystem=fs)
```

### Trino/Presto

```sql
-- Create schema pointing to LakeFS
CREATE SCHEMA lakefs.mydata WITH (location = 's3a://my-repo/main/data/');

-- Query from main branch
SELECT * FROM lakefs.mydata.training_data;

-- Query from specific commit (use catalog with commit reference)
SELECT * FROM lakefs.mydata."<commit-id>".training_data;
```

### Python SDK

```python
import lakefs

# Initialize client
client = lakefs.Client(
    host="https://lakefs.example.com",
    access_key_id="AKIAIOSFODNN7EXAMPLE",
    secret_access_key="wJalrXUtnFEMI/K7MDENG"
)

# Get repository
repo = client.repository("my-repo")

# Create branch
branch = repo.branch("feature").create(source_reference="main")

# Upload file
branch.object("data/file.parquet").upload(open("file.parquet", "rb"))

# Commit
branch.commit(message="Add training data")

# Merge to main
main_branch = repo.branch("main")
main_branch.merge(branch)

# Read file
content = branch.object("data/file.parquet").reader().read()
```

## Data Quality Hooks

### Pre-Commit Hooks

LakeFS supports hooks that run before commits are finalized:

```yaml
# .lakefs_actions.yaml (in repository root)
on:
  pre-commit:
    branches:
      - main
      - staging

hooks:
  - id: validate-schema
    type: webhook
    properties:
      url: https://my-service/validate
      timeout: 5m

  - id: run-tests
    type: lua
    properties:
      script: |
        function validate(action_info, added_files)
          for _, f in ipairs(added_files) do
            if f.path:match("%.parquet$") then
              -- Validate parquet files
            end
          end
          return true
        end
```

### Pre-Merge Hooks

```yaml
on:
  pre-merge:
    branches:
      - main

hooks:
  - id: quality-gate
    type: webhook
    properties:
      url: https://my-service/quality-check
```

### Example Validation Hook

```python
from flask import Flask, request, jsonify
import pyarrow.parquet as pq

app = Flask(__name__)

@app.route("/validate", methods=["POST"])
def validate_hook():
    action = request.json

    # Check each added/modified file
    for change in action["hooks"][0]["staged_objects"]:
        path = change["path"]

        # Validate parquet schema
        if path.endswith(".parquet"):
            try:
                table = pq.read_table(f"s3://repo/{action['branch']}/{path}")
                required_cols = ["id", "timestamp", "value"]
                if not all(col in table.column_names for col in required_cols):
                    return jsonify({"status": "fail", "message": f"Missing columns in {path}"}), 400
            except Exception as e:
                return jsonify({"status": "fail", "message": str(e)}), 400

    return jsonify({"status": "success"})
```

## Isolation Patterns

### Development Workflow

```
main (production)
  |
  +-- staging (pre-production testing)
  |     |
  |     +-- feature-1 (developer work)
  |     +-- feature-2 (developer work)
  |
  +-- experiments
        |
        +-- exp-learning-rate
        +-- exp-architecture
```

```bash
# Developer creates feature branch
lakectl branch create lakefs://repo/feature-1 --source lakefs://repo/staging

# Work on feature
# ... upload data, commit changes ...

# Merge to staging for testing
lakectl merge lakefs://repo/feature-1 lakefs://repo/staging

# After validation, merge to main
lakectl merge lakefs://repo/staging lakefs://repo/main
```

### ETL Isolation

```bash
# Create isolated branch for ETL run
lakectl branch create lakefs://repo/etl-2024-01-15 --source lakefs://repo/main

# Run ETL pipeline on isolated branch
spark-submit etl.py --output s3://repo/etl-2024-01-15/processed/

# Validate results
python validate.py s3://repo/etl-2024-01-15/processed/

# If valid, merge to main
lakectl merge lakefs://repo/etl-2024-01-15 lakefs://repo/main

# If failed, discard branch (no impact on main)
lakectl branch delete lakefs://repo/etl-2024-01-15
```

## Comparison with Alternatives

### LakeFS vs DVC

| Aspect | LakeFS | DVC |
|--------|--------|-----|
| Model | Git on object storage | Git extension |
| Branching | Native, zero-copy | Git branches + data sync |
| Granularity | Object-level | File/directory |
| Integration | S3-compatible (transparent) | Explicit DVC commands |
| Infrastructure | Requires server | Uses existing Git |
| Best For | Data lake versioning | ML project versioning |

### LakeFS vs Delta Lake Time Travel

| Aspect | LakeFS | Delta Lake |
|--------|--------|------------|
| Scope | All object storage | Table format (Parquet) |
| Branching | Full Git-like branches | No branching |
| Query | S3 API | SQL/DataFrame |
| Format | Any file format | Parquet only |
| Isolation | Branch-level | N/A |
| Best For | Data lake workflows | Analytical tables |

### LakeFS vs Iceberg

| Aspect | LakeFS | Iceberg |
|--------|--------|---------|
| Scope | Storage layer | Table format |
| Versioning | Full Git semantics | Time travel |
| Branching | Yes | Limited (tags/snapshots) |
| Format | Any | Parquet/ORC |
| Integration | S3 API layer | Query engine level |
| Complementary | Yes | Yes |

LakeFS and Iceberg can be used together: LakeFS versions the underlying storage while Iceberg manages table metadata.

## Best Practices

### Repository Organization

```
lakefs://data-lake/
  +-- main/
      +-- raw/                    # Immutable raw data
      |   +-- source-a/
      |   +-- source-b/
      +-- staging/                # Intermediate data
      |   +-- cleaned/
      |   +-- transformed/
      +-- curated/                # Production-ready data
      |   +-- features/
      |   +-- training-data/
      +-- models/                 # Trained models
          +-- production/
          +-- experiments/
```

### Branching Strategy

1. **Main branch**: Production data, always stable
2. **Staging branch**: Integration testing
3. **Feature branches**: Individual development work
4. **Experiment branches**: ML experiments with data variations

### Commit Practices

```bash
# Use meaningful commit messages
lakectl commit lakefs://repo/main -m "Add Q4 2024 training data: 500k samples, cleaned duplicates"

# Tag important milestones
lakectl tag create lakefs://repo/v1.0-training-data lakefs://repo/main

# Use tags for model training snapshots
lakectl tag create lakefs://repo/model-v2.0-data lakefs://repo/<commit-id>
```

### Garbage Collection

```bash
# Configure retention policy
lakectl gc set-rules lakefs://repo rules.json

# rules.json
{
  "default_retention_days": 30,
  "branches": [
    {"branch_id": "main", "retention_days": 90}
  ]
}

# Run garbage collection
lakectl gc run lakefs://repo
```

## Deployment Options

### Self-Hosted

```yaml
# docker-compose.yml
version: '3'
services:
  lakefs:
    image: treeverse/lakefs:latest
    ports:
      - "8000:8000"
    environment:
      - LAKEFS_BLOCKSTORE_TYPE=s3
      - LAKEFS_BLOCKSTORE_S3_REGION=us-east-1
      - LAKEFS_AUTH_ENCRYPT_SECRET_KEY=<secret>
      - LAKEFS_DATABASE_TYPE=postgres
      - LAKEFS_DATABASE_POSTGRES_CONNECTION_STRING=postgres://user:pass@postgres:5432/lakefs
    depends_on:
      - postgres

  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: lakefs
```

### Kubernetes

```yaml
# Using Helm
helm repo add lakefs https://charts.lakefs.io
helm install lakefs lakefs/lakefs \
  --set blockstore.type=s3 \
  --set blockstore.s3.region=us-east-1 \
  --set database.type=postgres
```

### LakeFS Cloud

Managed service available at https://lakefs.cloud with:
- Automatic scaling
- Managed backups
- SSO integration
- Enterprise support

## When to Use LakeFS

### Good Fit

- Data lake environments with many files
- Teams needing isolation for ETL development
- CI/CD for data pipelines
- Multi-tool environments (Spark, Trino, etc.)
- Compliance requiring audit trails

### Consider Alternatives

- Single ML project: Consider DVC (simpler)
- Table-only workflows: Consider Delta Lake/Iceberg
- Real-time streaming: May need different approach
- Very simple versioning: Object storage versioning may suffice
