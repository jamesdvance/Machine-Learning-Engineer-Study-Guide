# Delta Lake Time Travel

## Summary

Delta Lake time travel provides the ability to query previous versions of data using version numbers or timestamps. Built on Delta Lake's transaction log, time travel enables reproducible ML training, data auditing, and recovery from accidental changes. Unlike full version control systems (DVC, LakeFS), Delta Lake time travel is table-scoped and focused on point-in-time queries rather than branching and merging.

Key points to remember:

- Query historical data using VERSION AS OF or TIMESTAMP AS OF
- Each write operation creates a new version automatically
- Transaction log stores complete history of table changes
- Vacuum controls retention and cleanup of historical data
- Restore command reverts table to previous state
- Useful for ML reproducibility, debugging, and compliance
- Not a replacement for full version control (no branching)

## Core Concepts

### Transaction Log

Delta Lake maintains a transaction log (`_delta_log/`) that records every change to the table:

```
table_path/
  +-- _delta_log/
  |   +-- 00000000000000000000.json  # Version 0
  |   +-- 00000000000000000001.json  # Version 1
  |   +-- 00000000000000000002.json  # Version 2
  |   +-- 00000000000000000010.checkpoint.parquet
  +-- part-00000-...parquet
  +-- part-00001-...parquet
```

Each version file records:
- Files added in this version
- Files removed in this version
- Metadata changes (schema evolution)
- Transaction metadata (timestamp, operation type)

### Versioning Model

```
Version 0: Initial table creation
    |
    v
Version 1: INSERT 1000 rows
    |
    v
Version 2: UPDATE 50 rows (creates new files, marks old as removed)
    |
    v
Version 3: DELETE 10 rows
    |
    v
Version 4: MERGE operation
```

Each write operation creates a new version. Versions are monotonically increasing integers.

### Data Retention

Historical data files are retained until explicitly cleaned:

```
Current version: 4
Vacuum threshold: 7 days

Files needed for versions 0-3: Still present on disk
Files needed for version 4: Current data

After VACUUM RETAIN 7 DAYS:
- Files older than 7 days AND not in current version: Deleted
- Versions requiring deleted files: No longer accessible
```

## Time Travel Queries

### Query by Version

```sql
-- Spark SQL
SELECT * FROM my_table VERSION AS OF 3;

-- Or using @ syntax
SELECT * FROM my_table@v3;

-- PySpark
df = spark.read.format("delta").option("versionAsOf", 3).load("/path/to/table")

# Python delta-rs
from deltalake import DeltaTable
dt = DeltaTable("/path/to/table")
df = dt.to_pandas(version=3)
```

### Query by Timestamp

```sql
-- Spark SQL
SELECT * FROM my_table TIMESTAMP AS OF '2024-01-15 10:30:00';

-- PySpark
df = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-15 10:30:00") \
    .load("/path/to/table")

# Python delta-rs
from datetime import datetime
from deltalake import DeltaTable
dt = DeltaTable("/path/to/table")
df = dt.to_pandas(datetime=datetime(2024, 1, 15, 10, 30, 0))
```

### View Table History

```sql
-- See all versions and metadata
DESCRIBE HISTORY my_table;

-- Limit history
DESCRIBE HISTORY my_table LIMIT 10;
```

```python
# PySpark
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/path/to/table")
history_df = dt.history()
history_df.show()

# Columns: version, timestamp, userId, userName, operation, operationParameters, ...
```

Example output:
```
+-------+--------------------+---------+-----------+---------+
|version|           timestamp|operation|operationParameters|...
+-------+--------------------+---------+-----------+---------+
|      4|2024-01-15 10:30:00|    MERGE|{predicate: id = ..|
|      3|2024-01-15 09:00:00|   DELETE|{predicate: active..|
|      2|2024-01-14 15:00:00|   UPDATE|{predicate: catego..|
|      1|2024-01-14 10:00:00|    WRITE|{mode: Append, par..|
|      0|2024-01-14 09:00:00|    WRITE|{mode: Overwrite....|
+-------+--------------------+---------+-----------+---------+
```

## Restore Operations

### Restore to Version

```sql
-- Restore table to version 3
RESTORE TABLE my_table TO VERSION AS OF 3;

-- Restore to timestamp
RESTORE TABLE my_table TO TIMESTAMP AS OF '2024-01-15 10:30:00';
```

```python
# PySpark
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/path/to/table")
dt.restoreToVersion(3)

# Or by timestamp
dt.restoreToTimestamp("2024-01-15 10:30:00")
```

Restore creates a new version that has the same state as the target version:

```
Version 0: Initial
    |
    v
Version 1: Added data
    |
    v
Version 2: Modified data
    |
    v
Version 3: Deleted data
    |
    v
Version 4: RESTORE TO VERSION 1  <-- New version with Version 1's state
```

### Restore Best Practices

- Restore creates new version (non-destructive)
- Test restore on non-production first
- Consider downstream dependencies
- Document restore operations

## Data Retention and Vacuum

### Vacuum Command

Removes data files no longer referenced by the current version:

```sql
-- Remove files older than 7 days (default)
VACUUM my_table;

-- Specify retention period
VACUUM my_table RETAIN 168 HOURS;  -- 7 days

-- Dry run (show files to be deleted)
VACUUM my_table DRY RUN;
```

```python
# PySpark
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/path/to/table")
dt.vacuum(168)  # Hours

# Python delta-rs
from deltalake import DeltaTable
dt = DeltaTable("/path/to/table")
dt.vacuum(retention_hours=168, dry_run=False)
```

### Retention Period

Default minimum retention is 7 days. To reduce:

```sql
-- Enable shorter retention (use with caution)
SET spark.databricks.delta.retentionDurationCheck.enabled = false;
VACUUM my_table RETAIN 0 HOURS;  -- Dangerous: removes all history
```

### Vacuum Impact on Time Travel

```
Before VACUUM RETAIN 24 HOURS:
- Version 0 (5 days ago): Accessible
- Version 1 (3 days ago): Accessible
- Version 2 (1 day ago): Accessible
- Version 3 (current): Accessible

After VACUUM RETAIN 24 HOURS:
- Version 0: NOT accessible (files deleted)
- Version 1: NOT accessible (files deleted)
- Version 2: Accessible
- Version 3: Accessible
```

### Retention Strategy

| Use Case | Recommended Retention |
|----------|----------------------|
| Development | 24-48 hours |
| Production | 7-30 days |
| Compliance/Audit | 30-90+ days |
| ML Reproducibility | Match experiment lifecycle |

## ML-Specific Use Cases

### Reproducible Training Data

```python
from delta.tables import DeltaTable
from deltalake import DeltaTable as DeltaTablePy

# Record version used for training
dt = DeltaTable.forPath(spark, "/data/features")
current_version = dt.history(1).select("version").collect()[0][0]

# Log with experiment
mlflow.log_param("data_version", current_version)

# Later: reproduce exact training data
training_data = spark.read.format("delta") \
    .option("versionAsOf", current_version) \
    .load("/data/features")
```

### Point-in-Time Feature Retrieval

```python
# Get features as they existed at training time
training_timestamp = "2024-01-15 00:00:00"

features = spark.read.format("delta") \
    .option("timestampAsOf", training_timestamp) \
    .load("/data/features")

labels = spark.read.format("delta") \
    .option("timestampAsOf", training_timestamp) \
    .load("/data/labels")

# Join for training dataset
training_df = features.join(labels, "entity_id")
```

### Model Rollback Scenarios

```python
# Scenario: Bad data ingested, affecting model training

# 1. Identify when bad data was ingested
history = dt.history()
history.filter(history.operation == "WRITE").show()

# 2. Find last good version
good_version = 5  # Before bad data

# 3. Option A: Query historical data for retraining
good_data = spark.read.format("delta") \
    .option("versionAsOf", good_version) \
    .load("/data/training")

# 4. Option B: Restore table (affects all users)
dt.restoreToVersion(good_version)
```

### Experiment Comparison

```python
# Compare features across versions
def compare_versions(path, v1, v2):
    df1 = spark.read.format("delta").option("versionAsOf", v1).load(path)
    df2 = spark.read.format("delta").option("versionAsOf", v2).load(path)

    print(f"Version {v1}: {df1.count()} rows")
    print(f"Version {v2}: {df2.count()} rows")

    # Compare distributions
    df1.describe().show()
    df2.describe().show()

compare_versions("/data/features", 5, 10)
```

## Comparison with Alternatives

### Delta Lake Time Travel vs DVC

| Aspect | Delta Lake Time Travel | DVC |
|--------|----------------------|-----|
| Scope | Single table | Entire project |
| Granularity | Row-level versions | File/directory |
| Branching | No | Yes (via Git) |
| Query Interface | SQL/DataFrame | File system |
| Format | Parquet (Delta) | Any file format |
| Best For | Tabular data history | ML project versioning |

### Delta Lake Time Travel vs LakeFS

| Aspect | Delta Lake Time Travel | LakeFS |
|--------|----------------------|--------|
| Scope | Table | Entire data lake |
| Branching | No | Full Git-like branching |
| Isolation | No | Branch-level isolation |
| Merge | No | Yes |
| Integration | Spark ecosystem | S3-compatible |
| Best For | Table versioning | Data lake workflows |

### Complementary Usage

Delta Lake time travel and version control tools serve different purposes:

```
LakeFS/DVC: Version the entire data lake or project
    |
    +-- Branches for development isolation
    +-- Commits for project milestones
    |
    v
Delta Lake: Version individual tables within
    |
    +-- Row-level history for debugging
    +-- Point-in-time queries for ML
    +-- ACID transactions for consistency
```

## Configuration Options

### Spark Configuration

```python
# Log retention for time travel (default 30 days)
spark.conf.set("spark.databricks.delta.properties.defaults.logRetentionDuration", "interval 30 days")

# Deleted file retention (for vacuum safety)
spark.conf.set("spark.databricks.delta.properties.defaults.deletedFileRetentionDuration", "interval 7 days")

# Enable change data feed for CDC use cases
spark.conf.set("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true")
```

### Table Properties

```sql
-- Set retention at table level
ALTER TABLE my_table SET TBLPROPERTIES (
  'delta.logRetentionDuration' = 'interval 60 days',
  'delta.deletedFileRetentionDuration' = 'interval 14 days'
);

-- Enable change data feed
ALTER TABLE my_table SET TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);
```

## Best Practices

### Version Management

1. **Document important versions**: Use comments or external tracking
2. **Tag critical versions**: Record version numbers for ML experiments
3. **Automate version logging**: Integrate with MLflow, W&B, etc.

```python
# Example: Log version with every training run
@log_experiment
def train_model(data_path):
    dt = DeltaTable.forPath(spark, data_path)
    version = dt.history(1).select("version").collect()[0][0]
    mlflow.log_param("data_version", version)

    data = spark.read.format("delta").load(data_path)
    # ... training logic
```

### Retention Strategy

1. **Balance cost vs utility**: Historical data uses storage
2. **Align with compliance**: Meet regulatory requirements
3. **Coordinate with vacuum**: Don't vacuum too aggressively

```python
# Scheduled vacuum with safe retention
def scheduled_maintenance(table_path, retention_days=30):
    dt = DeltaTable.forPath(spark, table_path)

    # Vacuum with retention
    dt.vacuum(retention_days * 24)  # Hours

    # Optimize for query performance
    dt.optimize().executeCompaction()
```

### Query Patterns

```python
# Efficient historical queries
def get_training_snapshot(path, version=None, timestamp=None):
    reader = spark.read.format("delta")

    if version is not None:
        reader = reader.option("versionAsOf", version)
    elif timestamp is not None:
        reader = reader.option("timestampAsOf", timestamp)

    return reader.load(path)

# Cache frequently accessed historical versions
cached_v5 = get_training_snapshot("/data/features", version=5).cache()
```

## Limitations

### No Branching

Delta Lake time travel provides linear history only:

```
v0 -> v1 -> v2 -> v3 -> v4 (current)

Cannot create:
      +-> v2a (experiment branch)
v0 -> v1
      +-> v2b (production branch)
```

For branching, use LakeFS or copy tables.

### Vacuum Removes History

Once vacuumed, historical versions are permanently lost:

```python
# DANGEROUS: This removes ability to time travel
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
dt.vacuum(0)
```

### Storage Overhead

Each version retains data files until vacuum:
- Update-heavy tables grow quickly
- Consider vacuum schedule for cost management
- Monitor table size with `DESCRIBE DETAIL`

```sql
-- Check table storage details
DESCRIBE DETAIL my_table;
-- Shows: numFiles, sizeInBytes, version, ...
```

## When to Use Delta Lake Time Travel

### Good Fit

- Debugging data issues in production
- Reproducible ML training data
- Regulatory compliance and auditing
- Recovery from accidental modifications
- Comparing data across time periods

### Consider Alternatives

- Full version control needs: Use DVC or LakeFS
- Branch-based development: Use LakeFS
- Non-tabular data: Use DVC
- Very long retention: Evaluate storage costs
