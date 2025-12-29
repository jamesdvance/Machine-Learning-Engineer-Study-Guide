# Delta Lake

## Summary

Delta Lake is an open-source storage layer that brings ACID transactions, scalable metadata handling, and unified batch and streaming data processing to data lakes built on cloud object stores. Created by Databricks and open-sourced in 2019, it extends Apache Parquet files with a file-based transaction log to provide reliability guarantees traditionally associated with data warehouses.

Key points to remember:

- Delta Lake stores data as Parquet files plus a transaction log in a `_delta_log` directory
- Provides ACID transactions at the table level (not multi-table)
- Uses optimistic concurrency control with no read/write locks
- Supports time travel to query previous versions of data
- Schema enforcement prevents bad data from corrupting tables
- Tightly integrated with Apache Spark, making it ideal for Spark-based workloads
- Compared to Iceberg, Delta Lake has stronger Databricks integration but less engine portability
- Compared to Hudi, Delta Lake focuses more on batch workloads while Hudi excels at streaming upserts

## Architecture

### Table Structure

A Delta Lake table consists of two components stored in a directory:

1. Data files in Apache Parquet format
2. A `_delta_log` subdirectory containing the transaction log

The transaction log is the control center for all operations. It tracks every change, schema update, and transaction as an ordered, append-only sequence of commits.

### Transaction Log Internals

The `_delta_log` directory contains:

- JSON commit files numbered sequentially (e.g., `00000000000000000001.json`, `00000000000000000002.json`)
- Checkpoint files in Parquet format created every 10 commits
- A `_last_checkpoint` file pointing to the most recent checkpoint

Each JSON commit file records the actions that constitute a single atomic transaction. After every 10 commits, Delta Lake consolidates the JSON files into a checkpoint Parquet file for faster metadata reads on large tables.

### Transaction Actions

When you perform an operation like INSERT, UPDATE, or DELETE, Delta Lake breaks it into discrete actions:

- Add file: Records a new data file added to the table
- Remove file: Soft-deletes a data file (tombstoning for time travel support)
- Update metadata: Changes table schema, name, or partitioning
- Set transaction: Records structured streaming micro-batch commits
- Change protocol: Enables new Delta Lake features
- Commit info: Stores operation metadata including timestamp and source

Delta Lake never updates Parquet files in place. Instead, it rewrites entire files containing affected rows. Old files are tombstoned rather than immediately deleted, enabling time travel queries.

## ACID Guarantees

### Atomicity

Transactions either succeed completely or fail without partial changes. The transaction log ensures atomicity by writing data files first, then committing a transaction log entry only after all files are successfully written. Failed transactions leave orphaned files that the VACUUM operation later cleans up.

### Consistency

Multiple concurrent readers always see a consistent view of the table. The three-stage write process reads the current table version, writes new data files, then validates and commits changes. Conflicts trigger exceptions rather than corrupting data.

### Isolation

Delta Lake provides snapshot isolation for reads and write-serializable isolation for writes. Readers see a consistent snapshot without being blocked by writers. Multiple writers are serialized to prevent conflicts.

### Durability

Committed changes are permanent, inheriting durability guarantees from the underlying cloud object storage (S3, GCS, ADLS, or HDFS).

## Concurrency Control

Delta Lake uses optimistic concurrency control rather than locks:

1. Each transaction attempts to write the next sequential JSON commit file
2. If another transaction already wrote that version, the current transaction retries with the next available version number
3. Only one transaction can succeed at any given version

This approach enables high-throughput concurrent writes without deadlocks or complex locking mechanisms. However, it means high-contention workloads may experience retries.

Multi-cluster writes are supported but may produce conflicts. Delta Lake guarantees no data corruption even when conflicts occur.

## Key Features

### Time Travel

Query previous versions of your data using version numbers or timestamps:

```sql
-- Query by version
SELECT * FROM my_table VERSION AS OF 5

-- Query by timestamp
SELECT * FROM my_table TIMESTAMP AS OF '2024-01-15'
```

Time travel is enabled by tombstoning removed files rather than deleting them immediately. The VACUUM command eventually removes old files based on retention settings.

### Schema Enforcement and Evolution

Delta Lake enforces schema on write, preventing data with mismatched schemas from being written. This catches data quality issues early rather than at query time.

Schema evolution supports:

- Adding new columns
- Renaming columns (with appropriate settings)
- Reordering columns
- Changing data types in compatible ways

### Unified Batch and Streaming

Delta Lake tables can be both a streaming source and sink. Structured Streaming jobs can write to Delta tables while batch jobs query them, or vice versa. The transaction log tracks streaming micro-batch commits to ensure exactly-once processing.

### MERGE (Upsert) Operations

The MERGE command enables efficient upserts:

```sql
MERGE INTO target
USING updates
ON target.id = updates.id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
```

This is essential for CDC (change data capture) pipelines and GDPR compliance scenarios requiring record-level updates or deletes.

### Data Compaction

Small file problems are common in streaming scenarios. Delta Lake provides the OPTIMIZE command to compact small files into larger ones for better query performance:

```sql
OPTIMIZE my_table
```

Z-ordering colocates related data for even faster queries on specific columns:

```sql
OPTIMIZE my_table ZORDER BY (date, region)
```

## Performance Considerations

### Metadata Scalability

The checkpoint mechanism allows Delta Lake to scale to tables with billions of partitions or files. Reading a single checkpoint file is faster than parsing thousands of JSON commit files.

### Data Skipping

Delta Lake maintains file-level statistics (min/max values) that enable skipping irrelevant files during queries. This is automatic for the first 32 columns by default.

### Caching

The Delta Lake cache stores remote Parquet data on local SSDs for faster repeated access. This is particularly effective for interactive workloads.

### Limitations

- Transactions are single-table only; no multi-table transaction support
- Heavy reliance on Spark limits portability compared to Iceberg
- Write amplification from file rewrites can be costly for update-heavy workloads

## Comparison with Other Table Formats

### Delta Lake vs Apache Iceberg

Iceberg offers broader engine support (Spark, Flink, Trino, Presto, Hive, Snowflake, BigQuery) while Delta Lake is most mature on Spark and Databricks. Iceberg has more flexible partition evolution capabilities. Delta Lake has tighter Databricks integration and a larger existing user base in the Spark ecosystem.

For organizations prioritizing multi-engine portability, Iceberg may be preferable. For Databricks shops or Spark-heavy environments, Delta Lake is the natural choice.

### Delta Lake vs Apache Hudi

Hudi was designed for streaming-heavy workloads with frequent record-level updates, originally at Uber for ride-sharing data. Hudi provides indexing for efficient record lookups and optimized incremental processing.

Delta Lake is more general-purpose and excels at batch workloads with occasional updates. If your use case involves millions of events per second with continuous upserts (IoT, clickstreams), Hudi may offer better performance.

### Interoperability

Delta Lake UniForm enables reading Delta tables as Iceberg or Hudi tables without data duplication. Apache XTable provides similar cross-format interoperability. These tools reduce lock-in concerns when choosing a table format.

## Usage with Spark

### Creating a Delta Table

```python
# Write DataFrame as Delta
df.write.format("delta").save("/path/to/table")

# Create managed table
df.write.format("delta").saveAsTable("my_table")
```

### Reading Delta Tables

```python
# Read Delta table
df = spark.read.format("delta").load("/path/to/table")

# Query with SQL
spark.sql("SELECT * FROM delta.`/path/to/table`")
```

### Streaming

```python
# Stream write to Delta
stream = (df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/checkpoint")
    .start("/path/to/table"))

# Stream read from Delta
stream_df = (spark.readStream
    .format("delta")
    .load("/path/to/table"))
```

## When to Use Delta Lake

Delta Lake is well-suited for:

- Data lakehouse architectures on Databricks
- Spark-based ETL pipelines requiring transactional guarantees
- Scenarios requiring time travel for auditing or debugging
- Mixed batch and streaming workloads
- Teams already invested in the Databricks ecosystem

Consider alternatives when:

- Multi-engine portability is critical (evaluate Iceberg)
- High-frequency streaming upserts dominate your workload (evaluate Hudi)
- You need multi-table transactions (consider a traditional data warehouse)

## Maintenance Operations

### VACUUM

Remove files no longer referenced by the transaction log:

```sql
VACUUM my_table RETAIN 168 HOURS
```

The retention period must exceed the longest-running query or time travel requirement.

### DESCRIBE HISTORY

View the transaction history:

```sql
DESCRIBE HISTORY my_table
```

### RESTORE

Revert to a previous version:

```sql
RESTORE TABLE my_table TO VERSION AS OF 5
```

## Cloud Storage Considerations

Delta Lake works on any cloud object store but behavior varies:

- S3: Requires careful configuration for consistent listings; S3 Guard or Delta Lake's log store can help
- ADLS Gen2: Native support with strong consistency
- GCS: Good support with atomic rename operations

For production deployments, enable the appropriate consistency features for your cloud provider to avoid edge cases with concurrent operations.
