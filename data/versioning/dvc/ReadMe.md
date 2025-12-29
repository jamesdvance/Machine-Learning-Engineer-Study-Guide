# DVC (Data Version Control)

## Summary

DVC is an open-source version control system for machine learning projects that extends Git to handle large files, datasets, and ML models. It provides Git-like semantics (add, commit, push, pull) for data while storing actual content in remote storage (S3, GCS, Azure, SSH). DVC enables reproducible ML experiments by tracking data, code, and pipelines together.

Key points to remember:

- Git for data: familiar commands (dvc add, dvc push, dvc pull) for data versioning
- Stores pointers in Git, actual data in remote storage (S3, GCS, Azure, etc.)
- Lightweight metadata files (.dvc) track data versions
- Pipeline definitions (dvc.yaml) enable reproducible experiments
- Experiment tracking for comparing runs and metrics
- No server required: uses existing Git and cloud storage infrastructure
- Works with any file type: datasets, models, images, videos

## Core Concepts

### How DVC Works

DVC separates data versioning into two layers:

```
Git Repository                    Remote Storage (S3/GCS/Azure)
+------------------+              +----------------------+
| - code           |              | - actual data files  |
| - dvc.yaml       |   points to  | - cached versions    |
| - data.dvc      +------------->| - model weights      |
| - params.yaml    |              +----------------------+
+------------------+
```

When you run `dvc add data/`:
1. DVC computes content hash of data/
2. Moves data to local cache (.dvc/cache)
3. Creates data.dvc file with hash pointer
4. Git tracks the small .dvc file

When you run `dvc push`:
1. DVC uploads cached files to remote storage
2. Files are content-addressed by hash
3. Deduplication: identical files stored once

### DVC Files

**.dvc files**: Pointers to versioned data

```yaml
# data.dvc
outs:
- md5: abc123...
  size: 1234567890
  path: data
```

**dvc.yaml**: Pipeline definition

```yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - raw_data/
      - prepare.py
    outs:
      - processed_data/

  train:
    cmd: python train.py
    deps:
      - processed_data/
      - train.py
    params:
      - learning_rate
      - epochs
    outs:
      - model.pkl
    metrics:
      - metrics.json:
          cache: false
```

**dvc.lock**: Exact state of pipeline execution (auto-generated)

### Remotes

DVC supports multiple storage backends:

| Backend | Configuration |
|---------|---------------|
| Amazon S3 | `dvc remote add -d myremote s3://bucket/path` |
| Google Cloud Storage | `dvc remote add -d myremote gs://bucket/path` |
| Azure Blob | `dvc remote add -d myremote azure://container/path` |
| SSH/SFTP | `dvc remote add -d myremote ssh://user@host/path` |
| Local | `dvc remote add -d myremote /path/to/storage` |
| HTTP | `dvc remote add -d myremote https://example.com/path` |

## Basic Workflow

### Initial Setup

```bash
# Initialize DVC in existing Git repository
git init
dvc init

# Configure remote storage
dvc remote add -d storage s3://my-bucket/dvc-storage

# Optional: configure credentials
dvc remote modify storage access_key_id mykey
dvc remote modify storage secret_access_key mysecret
# Or use AWS CLI profile, IAM roles, etc.
```

### Tracking Data

```bash
# Add data file or directory to DVC
dvc add data/training_images/

# Creates data/training_images.dvc pointer file
# Adds data/training_images/ to .gitignore

# Commit pointer to Git
git add data/training_images.dvc data/.gitignore
git commit -m "Add training images"

# Push data to remote storage
dvc push
```

### Retrieving Data

```bash
# Clone repository (data not included)
git clone https://github.com/user/ml-project.git
cd ml-project

# Download data from remote
dvc pull

# Or pull specific files
dvc pull data/training_images.dvc
```

### Updating Data

```bash
# Modify data (add new images, clean dataset, etc.)
# ...

# Update DVC tracking
dvc add data/training_images/

# Commit and push
git add data/training_images.dvc
git commit -m "Update training images: added 500 new samples"
dvc push
```

## Pipelines

### Defining Pipelines

```yaml
# dvc.yaml
stages:
  download:
    cmd: python scripts/download.py ${urls.source}
    deps:
      - scripts/download.py
    params:
      - urls.source
    outs:
      - data/raw/

  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - scripts/preprocess.py
      - data/raw/
    params:
      - preprocess.image_size
      - preprocess.augmentation
    outs:
      - data/processed/

  train:
    cmd: python scripts/train.py
    deps:
      - scripts/train.py
      - data/processed/
    params:
      - model.architecture
      - train.learning_rate
      - train.epochs
      - train.batch_size
    outs:
      - models/model.pt
    plots:
      - training_curves.csv:
          x: epoch
          y: loss
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - scripts/evaluate.py
      - models/model.pt
      - data/test/
    metrics:
      - evaluation.json:
          cache: false
```

### Parameters File

```yaml
# params.yaml
urls:
  source: https://example.com/dataset.zip

preprocess:
  image_size: 224
  augmentation: true

model:
  architecture: resnet50

train:
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
```

### Running Pipelines

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# Force re-run (ignore cache)
dvc repro -f

# Run downstream stages from a specific point
dvc repro --downstream preprocess
```

### Pipeline Visualization

```bash
# Show pipeline DAG
dvc dag

# Output:
# +------------+
# | download   |
# +------------+
#       |
#       v
# +------------+
# | preprocess |
# +------------+
#       |
#       v
# +------------+
# |   train    |
# +------------+
#       |
#       v
# +------------+
# |  evaluate  |
# +------------+
```

## Experiment Tracking

### Running Experiments

```bash
# Run experiment with modified parameters
dvc exp run --set-param train.learning_rate=0.01

# Run multiple experiments in parallel
dvc exp run --queue --set-param train.learning_rate=0.001
dvc exp run --queue --set-param train.learning_rate=0.01
dvc exp run --queue --set-param train.learning_rate=0.1
dvc queue start --jobs 3

# Run experiment with name
dvc exp run -n high-lr --set-param train.learning_rate=0.1
```

### Comparing Experiments

```bash
# List all experiments
dvc exp show

# Compare specific experiments
dvc exp diff exp-abc123 exp-def456

# Show metrics
dvc metrics show

# Show params
dvc params diff
```

### Managing Experiments

```bash
# Apply experiment to workspace
dvc exp apply exp-abc123

# Create branch from experiment
dvc exp branch exp-abc123 feature/high-lr

# Remove experiments
dvc exp remove exp-abc123
dvc exp gc --workspace  # Clean up unreferenced experiments
```

## Python API

### Data Access

```python
import dvc.api

# Get URL to data file (for streaming)
url = dvc.api.get_url("data/dataset.csv", repo="https://github.com/user/repo")

# Open file directly
with dvc.api.open("data/dataset.csv", repo="https://github.com/user/repo") as f:
    content = f.read()

# Read specific version
with dvc.api.open("data/dataset.csv", rev="v1.0") as f:
    content = f.read()

# Get file hash and metadata
info = dvc.api.info("data/dataset.csv")
```

### Parameters and Metrics

```python
import dvc.api

# Read parameters
params = dvc.api.params_show()
learning_rate = params["train"]["learning_rate"]

# Read metrics
metrics = dvc.api.metrics_show()
accuracy = metrics["metrics.json"]["accuracy"]

# Read from specific revision
params = dvc.api.params_show(rev="v1.0")
```

### Programmatic Pipeline Execution

```python
from dvc.repo import Repo

repo = Repo()

# Reproduce pipeline
repo.reproduce()

# Run specific stage
repo.reproduce(targets=["train"])

# Push data
repo.push()

# Pull data
repo.pull()
```

## Integration with ML Frameworks

### PyTorch

```python
import torch
import dvc.api

# Load model from versioned path
model_path = "models/model.pt"
dvc.api.read(model_path)  # Ensures file is pulled

model = torch.load(model_path)

# Save and version model
torch.save(model.state_dict(), "models/model.pt")
# Then: dvc add models/model.pt && git commit && dvc push
```

### Hugging Face

```python
from transformers import AutoModel
import dvc.api

# Track model weights with DVC
# After download:
# dvc add models/bert-base/

# Load model
model = AutoModel.from_pretrained("models/bert-base")
```

### MLflow Integration

```python
import mlflow
import dvc.api

# Log DVC data version to MLflow
with mlflow.start_run():
    data_version = dvc.api.info("data/train.csv")["md5"]
    mlflow.log_param("data_version", data_version)

    # Train model...
    mlflow.log_metric("accuracy", accuracy)
```

## Comparison with Alternatives

### DVC vs LakeFS

| Aspect | DVC | LakeFS |
|--------|-----|--------|
| Model | Git extension | Git-like on object storage |
| Granularity | File/directory | Object storage native |
| Branching | Git branches | Native branches on storage |
| Atomicity | Pipeline-level | Transaction-level |
| Infrastructure | Uses existing Git | Separate server |
| Best For | ML projects | Data lake versioning |

### DVC vs Delta Lake Time Travel

| Aspect | DVC | Delta Lake |
|--------|-----|------------|
| Scope | Files/directories | Tables |
| Format | Any file type | Parquet |
| Query | File retrieval | SQL queries |
| Integration | Git workflow | Spark ecosystem |
| Best For | ML artifacts | Analytical data |

## Best Practices

### Repository Structure

```
project/
+-- .dvc/               # DVC metadata
|   +-- config          # Remote configuration
|   +-- cache/          # Local cache
+-- data/
|   +-- raw/            # Raw data (DVC tracked)
|   +-- processed/      # Processed data (DVC tracked)
+-- models/             # Model files (DVC tracked)
+-- src/                # Source code (Git tracked)
+-- dvc.yaml            # Pipeline definition
+-- params.yaml         # Parameters
+-- dvc.lock            # Pipeline state
+-- .gitignore          # Include .dvc patterns
```

### Workflow Recommendations

1. **Track large files with DVC**: Anything over 10 MB
2. **Use pipelines**: Define reproducible workflows in dvc.yaml
3. **Parameterize experiments**: Use params.yaml for hyperparameters
4. **Version data with code**: Commit .dvc files alongside code changes
5. **Use meaningful commit messages**: Describe data changes
6. **Clean cache periodically**: `dvc gc` to remove unused data

### Remote Storage Configuration

```bash
# Use IAM roles (AWS)
dvc remote modify storage access_key_id ""
dvc remote modify storage secret_access_key ""
# Relies on instance role or AWS_PROFILE

# Use service account (GCS)
dvc remote modify storage credentialpath /path/to/keyfile.json

# Enable caching
dvc config cache.type hardlink,symlink,copy
```

### CI/CD Integration

```yaml
# .github/workflows/train.yml
name: Train Model
on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install dvc[s3]
          pip install -r requirements.txt

      - name: Configure DVC
        run: |
          dvc remote modify storage access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
          dvc remote modify storage secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull data
        run: dvc pull

      - name: Run pipeline
        run: dvc repro

      - name: Push results
        run: dvc push
```

## Common Issues and Solutions

### Large Cache Size

```bash
# Remove unused cache entries
dvc gc --workspace

# Remove all but current version
dvc gc --workspace --force

# Check cache size
du -sh .dvc/cache
```

### Slow Operations

```bash
# Use symlinks instead of copies
dvc config cache.type symlink

# Enable parallel operations
dvc config core.jobs 4
```

### Remote Authentication

```bash
# Debug remote connection
dvc remote list
dvc remote default

# Test push/pull
dvc push --verbose
```

### Pipeline Reproducibility

```bash
# Check what changed
dvc status

# See pipeline dependencies
dvc dag --full

# Force full re-run
dvc repro --force
```

## When to Use DVC

### Good Fit

- ML projects with large datasets
- Teams already using Git
- Need for reproducible experiments
- Mixed file types (images, models, tabular)
- No dedicated data infrastructure

### Consider Alternatives

- Real-time data applications: Consider LakeFS
- SQL-centric workflows: Consider Delta Lake
- Enterprise feature stores: Consider Feast/Tecton
- Very large datasets (petabytes): Consider data lake solutions
