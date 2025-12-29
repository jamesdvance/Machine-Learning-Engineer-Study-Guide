# Data Versioning

## Summary

Data versioning enables tracking, reproducing, and managing changes to datasets over time. For ML systems, versioning is essential for experiment reproducibility, debugging model behavior, and compliance requirements. Different tools address versioning at different layers: project-level (DVC), data lake-level (LakeFS), and table-level (Delta Lake time travel).

Key points to remember:

- DVC extends Git for data: Git-like commands for versioning large files and ML artifacts
- LakeFS provides Git semantics on object storage: branching, merging, and isolation
- Delta Lake time travel offers table-level versioning: query historical data via SQL
- Choose based on granularity needs: project, data lake, or table
- Data versioning enables reproducible ML: same data version = same training results
- All approaches support point-in-time access for debugging and auditing
- Consider retention costs: historical data uses storage

## Why Data Versioning Matters

### The Reproducibility Problem

ML experiments depend on both code and data. Git tracks code changes, but:

```
Experiment A (2024-01-15):
  - Code: commit abc123
  - Data: ???
  - Result: 92% accuracy

Experiment B (2024-02-01):
  - Code: commit abc123 (same!)
  - Data: ??? (changed?)
  - Result: 87% accuracy
```

Without data versioning, reproducing Experiment A is impossible.

### Core Use Cases

| Use Case | Requirement | Solution |
|----------|-------------|----------|
| Experiment reproducibility | Exact data snapshot | Version at training time |
| Model debugging | Historical data access | Time travel queries |
| A/B testing | Parallel data branches | Branch-based versioning |
| Compliance/audit | Data lineage | Immutable version history |
| Rollback | Previous state recovery | Restore operations |
| Incremental processing | Change detection | Delta/changelog tracking |

## Versioning Approaches

### Project-Level: DVC

DVC extends Git workflows to include data versioning:

```
Git Repository               Remote Storage (S3/GCS)
+------------------+         +------------------+
| - code           |         | - datasets       |
| - data.dvc      +--------> | - models         |
| - dvc.yaml       |         | - cached files   |
+------------------+         +------------------+
```

**How it works:**
1. `dvc add data/` computes hash and creates pointer file
2. Git tracks the small `.dvc` pointer file
3. `dvc push` uploads data to remote storage
4. Teammates run `dvc pull` to get data

**Best for:**
- ML projects with code + data
- Teams already using Git
- Mixed file types (images, models, tabular)
- Reproducible ML pipelines

### Data Lake-Level: LakeFS

LakeFS provides Git operations directly on object storage:

```
LakeFS Layer                  Object Storage
+-------------------+         +------------------+
| - branches        |         | - data objects   |
| - commits         |   S3    | - versioned      |
| - merge/revert   +------->  | - zero-copy      |
+-------------------+   API   +------------------+
```

**How it works:**
1. Create branch: instant, zero-copy operation
2. Write data to branch via S3-compatible API
3. Commit changes with message
4. Merge to main when validated

**Best for:**
- Data lake environments
- ETL development isolation
- Multi-tool ecosystems (Spark, Trino, etc.)
- Large-scale data operations

### Table-Level: Delta Lake Time Travel

Delta Lake tracks every change to tables via transaction log:

```
Delta Table
+-------------------+
| _delta_log/       |  <- Version history
|   00001.json      |
|   00002.json      |
| data/             |  <- Parquet files
|   part-001.parquet|
+-------------------+
```

**How it works:**
1. Every write creates new version automatically
2. Query any version: `SELECT * FROM table VERSION AS OF 5`
3. Restore to previous state: `RESTORE TABLE table TO VERSION AS OF 5`
4. Vacuum removes old versions after retention period

**Best for:**
- SQL/DataFrame workflows
- Spark ecosystem
- Table-level auditing
- Quick rollbacks

## Comparison

### Feature Matrix

| Feature | DVC | LakeFS | Delta Lake |
|---------|-----|--------|------------|
| Scope | Project/files | Data lake | Tables |
| Branching | Via Git | Native | No |
| Merging | Via Git | Native | No |
| Query interface | File system | S3 API | SQL |
| Granularity | File | Object | Row |
| Format agnostic | Yes | Yes | Parquet only |
| Infrastructure | Git + storage | Server | Spark |

### Operational Comparison

| Aspect | DVC | LakeFS | Delta Lake |
|--------|-----|--------|------------|
| Setup complexity | Low | Medium | Low |
| Learning curve | Low (Git-like) | Low (Git-like) | Low (SQL) |
| Maintenance | Low | Medium | Low |
| Scaling | Good | Excellent | Excellent |
| Cost | Storage only | Storage + server | Storage only |

### When to Use Each

| Scenario | Recommended Tool |
|----------|------------------|
| ML project with Git | DVC |
| Data lake with many tools | LakeFS |
| Spark-based analytics | Delta Lake |
| Need branching for ETL | LakeFS |
| Table-level debugging | Delta Lake |
| Model + data versioning | DVC |
| Compliance/audit trail | Any (all provide history) |

## Complementary Usage

These tools can work together in a data stack:

```
                    LakeFS
                  (lake-level versioning)
                        |
    +-------------------+-------------------+
    |                   |                   |
Raw Data          Delta Lake Tables        DVC
(any format)      (table versioning)   (ML artifacts)
```

Example architecture:

1. **LakeFS** versions the entire data lake with branching
2. **Delta Lake** provides table-level time travel within LakeFS
3. **DVC** versions ML models and experiment artifacts

```python
# Training pipeline using multiple versioning layers

# 1. LakeFS: Create isolated branch for experiment
lakectl branch create lakefs://data-lake/exp-001 --source main

# 2. Delta Lake: Query training data with time travel
training_data = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-15") \
    .load("s3a://data-lake/exp-001/features")

# 3. DVC: Track model output
model.save("models/model.pt")
# dvc add models/model.pt && git commit && dvc push
```

## ML Workflow Patterns

### Experiment Reproducibility

```python
# Record data version with experiment
import mlflow
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/data/features")
data_version = dt.history(1).select("version").collect()[0][0]

with mlflow.start_run():
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", "/data/features")

    # Train model...
    mlflow.log_metric("accuracy", accuracy)
```

### Point-in-Time Training Data

```python
# Get features as they existed at a specific time
def get_training_snapshot(feature_path, label_path, as_of):
    features = spark.read.format("delta") \
        .option("timestampAsOf", as_of) \
        .load(feature_path)

    labels = spark.read.format("delta") \
        .option("timestampAsOf", as_of) \
        .load(label_path)

    return features.join(labels, "entity_id")

# Reproduce exact training data from past experiment
training_data = get_training_snapshot(
    "/data/features",
    "/data/labels",
    "2024-01-15 00:00:00"
)
```

### Development Isolation

```bash
# LakeFS: Isolated development branch
lakectl branch create lakefs://data-lake/dev-alice --source main

# Work on dev branch (all tools use LakeFS endpoint)
spark-submit preprocess.py --output s3a://data-lake/dev-alice/processed/

# Validate results
python validate.py s3a://data-lake/dev-alice/processed/

# Merge to main only after validation
lakectl merge lakefs://data-lake/dev-alice lakefs://data-lake/main
```

### Model Rollback

```python
# Scenario: Model performance degraded after data update

# 1. Find when performance changed
history = dt.history()
# Review operations and timestamps

# 2. Query historical data for comparison
old_data = spark.read.format("delta") \
    .option("versionAsOf", 5) \
    .load("/data/features")

new_data = spark.read.format("delta").load("/data/features")

# 3. Compare distributions
old_data.describe().show()
new_data.describe().show()

# 4. Restore if needed (Delta Lake)
dt.restoreToVersion(5)

# Or retrain with historical data
training_data = get_training_snapshot("/data/features", "/data/labels", "2024-01-15")
retrain_model(training_data)
```

## Retention and Cleanup

### Retention Strategies

| Tool | Retention Mechanism | Default |
|------|---------------------|---------|
| DVC | Manual cache management | Indefinite |
| LakeFS | Garbage collection policies | Configurable |
| Delta Lake | VACUUM command | 7 days minimum |

### Cost Considerations

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Historical data storage | Linear with retention | Aggressive cleanup policies |
| Small file overhead | High for frequent updates | Compaction |
| Metadata storage | Low | N/A |

### Cleanup Commands

```bash
# DVC: Clean unused cache
dvc gc --workspace

# LakeFS: Run garbage collection
lakectl gc run lakefs://repo

# Delta Lake: Vacuum old files
VACUUM my_table RETAIN 168 HOURS;
```

## Best Practices

### Version Documentation

```python
# Always log data version with experiments
def log_data_version(data_path, experiment_tracker):
    if data_path.startswith("s3://") or data_path.startswith("gs://"):
        # DVC-tracked: get hash from .dvc file
        version = get_dvc_hash(data_path)
    else:
        # Delta Lake: get version number
        dt = DeltaTable.forPath(spark, data_path)
        version = dt.history(1).select("version").collect()[0][0]

    experiment_tracker.log_param("data_version", version)
    experiment_tracker.log_param("data_path", data_path)
```

### Retention Alignment

```
Experiment Lifecycle     Data Retention
+------------------+     +------------------+
| Active dev: days |---->| Full history     |
| Review: weeks    |---->| 30-day retention |
| Archive: months  |---->| Key versions only|
+------------------+     +------------------+
```

### Tagging Important Versions

```bash
# DVC: Use Git tags
git tag -a v1.0-training-data -m "Training data for model v1.0"

# LakeFS: Use tags
lakectl tag create lakefs://repo/v1.0-data lakefs://repo/main

# Delta Lake: Document in external system
# (No native tagging, track version numbers externally)
```

## Further Reading

For detailed information on each versioning tool, see:

- [DVC (Data Version Control)](dvc/ReadMe.md)
- [LakeFS](lakefs/ReadMe.md)
- [Delta Lake Time Travel](delta-lake-time-travel/ReadMe.md)
