# Apache Iceberg

## Summary

Apache Iceberg is an open table format for huge analytic datasets that brings ACID transactions, schema evolution, and time travel to data lakes. Unlike file formats (Parquet, ORC), Iceberg is a table format that sits on top of file formats, providing database-like capabilities while maintaining the flexibility of object storage. Iceberg has become the de facto standard for modern lakehouses, with native support from Snowflake, Databricks, AWS, and major query engines.

Key points to remember:

- Table format, not file format: manages collections of Parquet/ORC files as tables
- ACID transactions enable reliable concurrent reads and writes
- Schema evolution: add, rename, remove columns without rewriting data
- Partition evolution: change partitioning without rewriting existing data
- Hidden partitioning: partition transforms applied automatically, no partition columns in data
- Time travel: query historical snapshots for reproducibility and debugging
- Metadata layer: catalog ’ metadata files ’ manifest lists ’ manifests ’ data files
- Engine-agnostic: works with Spark, Flink, Trino, Presto, Athena, BigQuery
- PyIceberg provides native Python API for table operations
- Compared to Delta Lake, Iceberg offers better engine interoperability
- Compared to Hive tables, Iceberg provides ACID and schema/partition evolution

## Architecture

### Metadata Hierarchy

```
Catalog (tracks table locations)
    
    ¼
Metadata File (current table state)
    
    ¼
Manifest List (snapshot of all manifests)
    
    ¼
Manifests (file-level metadata)
    
    ¼
Data Files (Parquet/ORC)
```

**Catalog**
Tracks table locations and current metadata:
- REST catalog (recommended for production)
- Hive Metastore
- AWS Glue
- JDBC/SQL catalogs
- Nessie (Git-like versioning)

**Metadata File**
JSON file containing complete table state:
- Current and historical schemas
- Partition specs
- Snapshots (pointers to manifest lists)
- Table properties

**Manifest List**
Avro file listing all manifests in a snapshot:
- Manifest file locations
- Partition range summaries
- Added/deleted file counts

**Manifest**
Avro file with data file metadata:
- File paths and sizes
- Column statistics (min/max, null counts)
- Partition values

**Data Files**
Actual data stored in columnar format:
- Parquet (default, recommended)
- ORC (alternative)
- Avro (for streaming)

### Transaction Model

```
Write Transaction:
1. Write new data files
2. Create new manifest(s)
3. Create new manifest list
4. Write new metadata file
5. Atomic update of catalog pointer

Read Transaction:
1. Get current metadata from catalog
2. Read manifest list ’ manifests
3. Apply partition pruning
4. Read only relevant data files
```

Optimistic concurrency:
- Writers check if table changed during write
- Retry if conflict detected
- No locks required for reads

## Core Features

### Schema Evolution

```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog("default")
table = catalog.load_table("db.customers")

# Add column
with table.update_schema() as update:
    update.add_column("email", StringType())

# Rename column
with table.update_schema() as update:
    update.rename_column("old_name", "new_name")

# Drop column
with table.update_schema() as update:
    update.delete_column("unused_column")

# Change type (widening only)
with table.update_schema() as update:
    update.update_column("id", LongType())  # int ’ long
```

Supported evolutions:
- Add columns (at any position)
- Rename columns
- Drop columns
- Reorder columns
- Widen types (int ’ long, float ’ double)

### Partition Evolution

```python
from pyiceberg.transforms import DayTransform, MonthTransform

# Initial partitioning by day
table = catalog.create_table(
    "db.events",
    schema=schema,
    partition_spec=PartitionSpec(
        PartitionField(source_id=1, transform=DayTransform(), name="day")
    )
)

# Later: switch to month partitioning
with table.update_spec() as update:
    update.remove_field("day")
    update.add_field("month", MonthTransform(), source_column_name="timestamp")
```

Benefits:
- No data rewriting required
- Old and new partitions coexist
- Query planning handles both automatically
- Gradual migration as new data arrives

### Hidden Partitioning

Partition values derived from data, not stored as columns:

```python
from pyiceberg.transforms import (
    DayTransform,
    HourTransform,
    MonthTransform,
    YearTransform,
    BucketTransform,
    TruncateTransform,
)

# Date transforms
spec = PartitionSpec(
    PartitionField(source_id=1, transform=YearTransform(), name="year"),
    PartitionField(source_id=1, transform=MonthTransform(), name="month"),
    PartitionField(source_id=1, transform=DayTransform(), name="day"),
    PartitionField(source_id=1, transform=HourTransform(), name="hour"),
)

# Hash bucketing
spec = PartitionSpec(
    PartitionField(source_id=2, transform=BucketTransform(16), name="id_bucket"),
)

# String truncation
spec = PartitionSpec(
    PartitionField(source_id=3, transform=TruncateTransform(4), name="name_prefix"),
)
```

Queries automatically apply partition filters:
```sql
-- Iceberg translates to partition filter
SELECT * FROM events WHERE timestamp = '2024-01-15 10:30:00'
-- Uses: year=2024, month=01, day=15, hour=10
```

### Time Travel

```python
# Query by snapshot ID
df = spark.read.option("snapshot-id", 123456789).table("db.events")

# Query by timestamp
df = spark.read.option("as-of-timestamp", "2024-01-15T10:00:00").table("db.events")

# List snapshots
for snapshot in table.history():
    print(f"{snapshot.snapshot_id}: {snapshot.timestamp_ms}")

# Rollback to snapshot
table.manage_snapshots().rollback_to_snapshot(snapshot_id).commit()

# Rollback to timestamp
table.manage_snapshots().rollback_to_timestamp(timestamp_ms).commit()
```

Use cases:
- ML training reproducibility
- Debugging data issues
- Auditing and compliance
- A/B testing with consistent data

### ACID Transactions

```python
# Atomic multi-table writes (Spark)
spark.sql("""
    BEGIN TRANSACTION
    INSERT INTO table1 SELECT ...
    INSERT INTO table2 SELECT ...
    COMMIT
""")

# Row-level operations
spark.sql("DELETE FROM events WHERE event_type = 'test'")
spark.sql("UPDATE events SET status = 'processed' WHERE id = 123")
spark.sql("""
    MERGE INTO target t
    USING source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
```

Iceberg v3 features:
- Deletion vectors (mark deletes without rewriting files)
- Row-level lineage tracking
- Improved MERGE performance

## PyIceberg Usage

### Catalog Configuration

```python
from pyiceberg.catalog import load_catalog

# REST catalog (recommended)
catalog = load_catalog(
    "default",
    **{
        "type": "rest",
        "uri": "http://localhost:8181",
        "credential": "user:password",
    }
)

# AWS Glue catalog
catalog = load_catalog(
    "glue",
    **{
        "type": "glue",
        "region_name": "us-east-1",
    }
)

# SQL catalog (SQLite, PostgreSQL)
catalog = load_catalog(
    "sql",
    **{
        "type": "sql",
        "uri": "postgresql://user:pass@localhost/iceberg",
    }
)
```

### Table Operations

```python
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField,
    StringType,
    LongType,
    TimestampType,
    DoubleType,
)

# Define schema
schema = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "name", StringType()),
    NestedField(3, "timestamp", TimestampType()),
    NestedField(4, "value", DoubleType()),
)

# Create table
table = catalog.create_table(
    "db.my_table",
    schema=schema,
    location="s3://bucket/warehouse/db/my_table",
)

# Load existing table
table = catalog.load_table("db.my_table")

# List tables
tables = catalog.list_tables("db")

# Drop table
catalog.drop_table("db.my_table")
```

### Reading Data

```python
import pyarrow as pa

# Read as PyArrow table
arrow_table = table.scan().to_arrow()

# Read as Pandas DataFrame
df = table.scan().to_pandas()

# With column selection
df = table.scan(selected_fields=["id", "name"]).to_pandas()

# With row filter
from pyiceberg.expressions import GreaterThan, EqualTo, And

df = table.scan(
    row_filter=And(
        GreaterThan("timestamp", "2024-01-01"),
        EqualTo("status", "active")
    )
).to_pandas()

# With snapshot
df = table.scan(snapshot_id=123456789).to_pandas()
```

### Writing Data

```python
import pyarrow as pa

# Append data
arrow_table = pa.Table.from_pandas(df)
table.append(arrow_table)

# Overwrite data
table.overwrite(arrow_table)

# Overwrite with filter (dynamic overwrite)
table.overwrite(arrow_table, overwrite_filter=EqualTo("date", "2024-01-15"))
```

## Spark Integration

### Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IcebergApp") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "rest") \
    .config("spark.sql.catalog.iceberg.uri", "http://localhost:8181") \
    .getOrCreate()
```

### SQL Operations

```python
# Create table
spark.sql("""
    CREATE TABLE iceberg.db.events (
        id BIGINT,
        event_type STRING,
        timestamp TIMESTAMP,
        payload STRING
    )
    USING iceberg
    PARTITIONED BY (days(timestamp))
""")

# Insert data
spark.sql("""
    INSERT INTO iceberg.db.events
    VALUES (1, 'click', current_timestamp(), '{}')
""")

# Time travel
spark.sql("SELECT * FROM iceberg.db.events VERSION AS OF 123456789")
spark.sql("SELECT * FROM iceberg.db.events TIMESTAMP AS OF '2024-01-15 10:00:00'")

# Metadata queries
spark.sql("SELECT * FROM iceberg.db.events.snapshots")
spark.sql("SELECT * FROM iceberg.db.events.history")
spark.sql("SELECT * FROM iceberg.db.events.files")
spark.sql("SELECT * FROM iceberg.db.events.manifests")
```

### DataFrame Operations

```python
# Read
df = spark.table("iceberg.db.events")

# Write (append)
df.writeTo("iceberg.db.events").append()

# Write (overwrite)
df.writeTo("iceberg.db.events").overwritePartitions()

# Create or replace
df.writeTo("iceberg.db.events").createOrReplace()
```

## Table Maintenance

### Compaction

Merge small files for better query performance:

```python
# Spark SQL
spark.sql("""
    CALL iceberg.system.rewrite_data_files(
        table => 'db.events',
        options => map('target-file-size-bytes', '134217728')  -- 128MB
    )
""")

# With filter
spark.sql("""
    CALL iceberg.system.rewrite_data_files(
        table => 'db.events',
        where => 'date >= "2024-01-01"'
    )
""")
```

### Snapshot Expiration

Remove old snapshots and associated data:

```python
# Expire snapshots older than timestamp
spark.sql("""
    CALL iceberg.system.expire_snapshots(
        table => 'db.events',
        older_than => TIMESTAMP '2024-01-01 00:00:00',
        retain_last => 10
    )
""")
```

### Orphan File Removal

Clean up unreferenced files:

```python
spark.sql("""
    CALL iceberg.system.remove_orphan_files(
        table => 'db.events',
        older_than => TIMESTAMP '2024-01-01 00:00:00'
    )
""")
```

### Metadata Cleanup

Rewrite manifests for better performance:

```python
spark.sql("""
    CALL iceberg.system.rewrite_manifests(
        table => 'db.events'
    )
""")
```

## ML Applications

### Feature Store Integration

```python
# Time travel for training data reproducibility
training_snapshot_id = table.current_snapshot().snapshot_id

# Log snapshot ID with model
mlflow.log_param("training_data_snapshot", training_snapshot_id)

# Reproduce training data later
training_data = table.scan(snapshot_id=training_snapshot_id).to_pandas()
```

### Point-in-Time Feature Retrieval

```python
# Features as of specific timestamp
features = spark.sql(f"""
    SELECT *
    FROM iceberg.feature_store.customer_features
    TIMESTAMP AS OF '{event_timestamp}'
    WHERE customer_id = '{customer_id}'
""")
```

### Incremental Training

```python
# Get changes since last training
from pyiceberg.expressions import GreaterThan

new_data = table.scan(
    row_filter=GreaterThan("_commit_timestamp", last_training_timestamp)
).to_pandas()

# Or use incremental append scan
changes = spark.read \
    .format("iceberg") \
    .option("start-snapshot-id", last_processed_snapshot) \
    .load("iceberg.db.events")
```

### Data Versioning for Experiments

```python
# Branch for experiment
spark.sql("""
    ALTER TABLE iceberg.db.training_data
    CREATE BRANCH experiment_v2
""")

# Write experimental data to branch
df.writeTo("iceberg.db.training_data.branch_experiment_v2").append()

# Compare results
main_data = spark.table("iceberg.db.training_data")
exp_data = spark.table("iceberg.db.training_data.branch_experiment_v2")
```

## Performance Optimization

### Partition Strategy

```python
# High cardinality: use bucketing
PartitionSpec(
    PartitionField(source_id=1, transform=BucketTransform(256), name="user_bucket")
)

# Time series: use temporal transforms
PartitionSpec(
    PartitionField(source_id=2, transform=DayTransform(), name="day")
)

# Multi-level for complex queries
PartitionSpec(
    PartitionField(source_id=1, transform=MonthTransform(), name="month"),
    PartitionField(source_id=2, transform=BucketTransform(16), name="region_bucket"),
)
```

### File Sizing

```python
# Target file size (default 512MB for Parquet)
table.update_properties({
    "write.target-file-size-bytes": "134217728"  # 128MB
})

# Minimum file size before splitting
table.update_properties({
    "write.distribution-mode": "hash"  # Better file sizes
})
```

### Metadata Optimization

```python
# Commit metadata tuning
table.update_properties({
    "commit.manifest.min-count-to-merge": "10",
    "commit.manifest.target-size-bytes": "8388608",  # 8MB
})
```

## Comparison with Alternatives

### Iceberg vs Delta Lake

| Aspect | Iceberg | Delta Lake |
|--------|---------|------------|
| Engine support | Multi-engine | Spark-centric |
| Schema evolution | Full | Full |
| Partition evolution | Native | Limited |
| Hidden partitioning | Yes | No |
| Time travel | Yes | Yes |
| ACID | Yes | Yes |
| Governance | Open spec | Databricks |

### Iceberg vs Hive Tables

| Aspect | Iceberg | Hive |
|--------|---------|------|
| ACID | Yes | Limited |
| Schema evolution | Yes | Append only |
| Partition evolution | Yes | No |
| Concurrent writes | Yes | Limited |
| Time travel | Yes | No |
| Performance | File-level stats | Directory listing |

## When to Use Iceberg

Iceberg is well-suited for:
- Multi-engine analytics (Spark, Trino, Flink, etc.)
- Data lakes requiring ACID transactions
- Tables with evolving schemas or partitions
- ML feature stores needing time travel
- Incremental data processing
- Lakehouse architectures

Consider alternatives when:
- Single-engine Databricks environment (Delta Lake native)
- Simple append-only logs (plain Parquet)
- Real-time streaming focus (Delta Lake, Hudi)
- Small datasets (overhead not justified)
