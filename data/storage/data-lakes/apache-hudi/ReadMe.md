# Apache Hudi

## Summary

Apache Hudi (Hadoop Upserts Deletes and Incrementals) is an open-source data management framework originally developed at Uber in 2016 to efficiently handle high-velocity streaming data with frequent record-level updates. It provides ACID transactions on data lakes while optimizing for upsert-heavy workloads that would be expensive with traditional append-only lake formats.

Key points to remember:

- Two table types: Copy-on-Write (COW) for read-heavy, Merge-on-Read (MOR) for write-heavy workloads
- Indexing is central to Hudi's performance, enabling fast record lookups for upserts
- Timeline architecture tracks all table operations for time travel and incremental queries
- Incremental query support enables efficient CDC pipelines
- Best suited for streaming ingestion, high-frequency updates, and real-time analytics
- Compared to Delta Lake, Hudi excels at streaming upserts but has more operational complexity
- Compared to Iceberg, Hudi offers better record-level update performance but less engine portability

## Architecture

### Core Components

Hudi's architecture consists of several interconnected components:

- Write Client: API for batch or streaming writes, enabling insertions, updates, and deletes
- Timeline: Metadata tracking all operations on the table with timestamps
- File Groups: Logical grouping of related data files
- Indexes: Data structures mapping record keys to file groups for fast lookups
- Compaction Service: Background process merging log files into base files (MOR tables)
- Cleaning Service: Removes old file versions based on retention policy

### Timeline

The timeline is Hudi's backbone for tracking table state. Every operation (commit, compaction, cleaning) is recorded as an instant on the timeline with:

- Instant time: Timestamp identifying when the action occurred
- Action type: commit, deltacommit, compaction, clean, rollback, etc.
- State: requested, inflight, or completed

This enables time travel queries and incremental processing by querying changes between timeline instants.

### File Organization

Hudi organizes data into file groups, each identified by a unique file ID. A file group contains:

- Base file: Parquet file with the bulk of the data
- Log files (MOR only): Avro files with incremental changes
- File slices: Versioned snapshots of a file group at specific instants

Partitioning is optional but commonly used with date-based schemes for time-series data.

## Table Types

Hudi provides two fundamental table types that represent different tradeoffs between read and write performance.

### Copy-on-Write (COW)

COW tables optimize for read performance at the cost of write amplification.

How it works:
1. Incoming records are matched to existing file groups using the index
2. For updates, the entire base file is read and merged with changes
3. A new version of the Parquet base file is written
4. No log files are involved

Characteristics:
- Read latency: Low (only Parquet files to scan)
- Write latency: High (full file rewrites on updates)
- Storage efficiency: High (no log files)
- Query complexity: Simple (standard Parquet reads)

Best for:
- Read-heavy workloads with infrequent updates
- Batch ETL pipelines
- Reference data tables that change slowly
- Workloads where query performance is paramount

### Merge-on-Read (MOR)

MOR tables balance read and write performance by deferring merge operations.

How it works:
1. Incoming records are matched to file groups using the index
2. Updates are written to log files (Avro format) rather than rewriting base files
3. Compaction periodically merges log files into new base files
4. Queries merge base files with log files on the fly (or skip logs for read-optimized queries)

Characteristics:
- Read latency: Variable (depends on log file size and compaction frequency)
- Write latency: Low (append-only log writes)
- Storage efficiency: Lower during active ingestion (multiple log files)
- Query complexity: Higher (merge logic required)

Best for:
- Write-heavy workloads with continuous updates
- Streaming data ingestion
- Change Data Capture (CDC) pipelines
- GDPR/CCPA compliance requiring frequent deletes

### Choosing Between COW and MOR

```
                     COW                    MOR
Write frequency:     Low-Medium             High
Update patterns:     Batch updates          Streaming/continuous
Query latency needs: Strict SLAs            Flexible
Read-to-write ratio: High reads             More balanced
Operational ease:    Simpler                Requires compaction tuning
```

For most streaming use cases at scale, MOR is preferred. For analytics on relatively static data, COW provides simpler operations.

## Query Types

### Snapshot Queries

Return the latest state of the table as of the most recent completed commit. This is the default query behavior and works identically to querying any other table.

### Time Travel Queries

Query the table as of a specific point in time:

```sql
SELECT * FROM hudi_table TIMESTAMP AS OF '2024-01-15 10:00:00'
```

Useful for debugging, auditing, and machine learning feature stores requiring point-in-time correctness.

### Read Optimized Queries (MOR only)

Query only the base Parquet files, ignoring uncommitted log files:

```sql
SELECT * FROM hudi_table_ro
```

Provides faster query performance at the cost of potentially stale data. Useful when slight staleness is acceptable for better latency.

### Incremental Queries

Return only records that changed since a specified instant:

```python
spark.read.format("hudi") \
    .option("hoodie.datasource.query.type", "incremental") \
    .option("hoodie.datasource.read.begin.instanttime", "20240115100000") \
    .load(path)
```

Enables efficient incremental ETL pipelines without reprocessing entire tables.

### CDC Queries

Provide before-and-after images of changed records, similar to database CDC:

```python
spark.read.format("hudi") \
    .option("hoodie.datasource.query.type", "incremental") \
    .option("hoodie.datasource.read.incr.operation", "read_changes") \
    .load(path)
```

Returns columns indicating operation type (insert, update, delete) and previous values.

## Indexing

Indexing is critical to Hudi's upsert performance. When new records arrive, the index determines which file groups contain existing records with matching keys.

### Index Types

**Simple Index**
- Lightweight in-memory join
- Suitable for smaller datasets
- Default on Flink and Java engines

**Bloom Index**
- Uses bloom filters stored in file footers
- Probabilistic: may have false positives but no false negatives
- Efficient for temporal data with key ranges
- Prunes files before reading
- Default on Spark engine

**Bucket Index**
- Hash-based partitioning of records to file groups
- Consistent hashing ensures records always map to same bucket
- Eliminates index lookups entirely
- Best for large-scale tables with uniform key distribution

**HBase Index**
- External index stored in Apache HBase
- Good for very large tables needing fast point lookups
- Adds operational overhead of managing HBase cluster

**Record Level Index (RLI)**
- Introduced in Hudi 0.14
- Maps record keys directly to file locations in metadata table
- Significantly faster than bloom or simple indexes for large tables
- Recommended for 100B+ record tables

### Global vs Non-Global Indexes

Non-global indexes only search within partitions matching incoming records. Global indexes search across all partitions.

Global indexes are required when:
- Records can move between partitions (e.g., status changes)
- Partition is not known at write time

The cost is significant: a non-global index on a 1000-partition table with 1000 files per partition only examines files in matching partitions. A global index must examine all 1M files.

### Index Selection Guidelines

| Scenario | Recommended Index |
|----------|------------------|
| Temporal data with ordered keys | Bloom |
| Random updates, small table | Simple |
| Very large table, uniform distribution | Bucket |
| External low-latency requirements | HBase |
| Very large table, needs global | Record Level Index |

## Compaction and Cleaning

### Compaction (MOR Tables)

Compaction merges log files with base files to create new base file versions:

```python
# Inline compaction during writes
.option("hoodie.compact.inline", "true")
.option("hoodie.compact.inline.max.delta.commits", "5")

# Async compaction
.option("hoodie.compact.inline", "false")
# Then run: spark.read.format("hudi").load(path).write.format("hudi")...
```

Compaction strategies:
- Bound IO: Limits number of file groups compacted per run
- Unbound IO: Compacts all pending file groups
- Time-based: Compacts based on log file age

Frequent compaction reduces query latency but increases write costs. Tune based on query SLA requirements.

### Cleaning

Cleaning removes old file versions beyond the retention window:

```python
.option("hoodie.cleaner.policy", "KEEP_LATEST_COMMITS")
.option("hoodie.cleaner.commits.retained", "10")
```

Policies:
- KEEP_LATEST_COMMITS: Retain N most recent commits
- KEEP_LATEST_FILE_VERSIONS: Retain N versions per file group
- KEEP_LATEST_BY_HOURS: Retain commits within time window

## Usage with Spark

### Writing Data

```python
# Insert/Upsert to Hudi table
df.write.format("hudi") \
    .option("hoodie.table.name", "my_table") \
    .option("hoodie.datasource.write.recordkey.field", "id") \
    .option("hoodie.datasource.write.partitionpath.field", "date") \
    .option("hoodie.datasource.write.precombine.field", "timestamp") \
    .option("hoodie.datasource.write.operation", "upsert") \
    .mode("append") \
    .save("/path/to/table")
```

Key options:
- recordkey.field: Primary key column(s) for record identification
- partitionpath.field: Column(s) for partitioning
- precombine.field: Tie-breaker when multiple records have same key
- operation: insert, upsert, bulk_insert, delete

### Reading Data

```python
# Snapshot query
df = spark.read.format("hudi").load("/path/to/table")

# Read optimized query (MOR)
df = spark.read.format("hudi").load("/path/to/table/*_ro")

# Time travel
df = spark.read.format("hudi") \
    .option("as.of.instant", "20240115100000") \
    .load("/path/to/table")
```

### Streaming Writes

```python
stream = df.writeStream \
    .format("hudi") \
    .option("hoodie.table.name", "streaming_table") \
    .option("hoodie.datasource.write.recordkey.field", "id") \
    .option("checkpointLocation", "/checkpoint") \
    .outputMode("append") \
    .start("/path/to/table")
```

## Comparison with Other Table Formats

### Hudi vs Delta Lake

Delta Lake provides simpler operations with strong Spark and Databricks integration. Hudi provides better performance for upsert-heavy workloads through its indexing system.

| Aspect | Hudi | Delta Lake |
|--------|------|------------|
| Upsert performance | Better (indexing) | Good |
| Operational complexity | Higher | Lower |
| Streaming support | Excellent | Good |
| Ecosystem integration | Broader | Databricks-centric |
| Incremental queries | Native | Via change data feed |
| Index options | Multiple | Limited |

Choose Hudi when: High-frequency streaming upserts dominate your workload.
Choose Delta Lake when: Simplicity and Databricks integration are priorities.

### Hudi vs Apache Iceberg

Iceberg provides the broadest engine compatibility and flexible schema evolution. Hudi provides better record-level update performance and incremental processing.

| Aspect | Hudi | Iceberg |
|--------|------|---------|
| Engine compatibility | Good | Excellent |
| Record-level updates | Excellent | Good |
| Schema evolution | Good | Excellent |
| Partition evolution | Good | Excellent |
| Incremental processing | Native | Limited |

Choose Hudi when: Record-level updates and CDC are primary use cases.
Choose Iceberg when: Multi-engine portability is critical.

## Performance Tuning

### Write Performance

1. Choose appropriate index type for workload
2. Enable bucket index for large tables with uniform distribution
3. Tune parallelism to match cluster resources
4. Use bulk_insert for initial loads (no index lookups)
5. Enable clustering to improve file sizes

### Read Performance

1. Run compaction frequently enough to meet query SLAs
2. Use read-optimized queries when slight staleness is acceptable
3. Leverage file pruning through partition and column statistics
4. Enable metadata table for faster file listings

### Compaction Tuning

```python
# Inline compaction every 5 commits
.option("hoodie.compact.inline", "true")
.option("hoodie.compact.inline.max.delta.commits", "5")

# Limit compaction scope
.option("hoodie.compaction.strategy", "org.apache.hudi.table.action.compact.strategy.BoundedIOCompactionStrategy")
.option("hoodie.compaction.target.io", "500000")
```

## When to Use Apache Hudi

Hudi is well-suited for:
- Near real-time analytics on streaming data
- CDC pipelines from operational databases
- High-frequency upsert workloads
- GDPR/CCPA compliance requiring efficient deletes
- Event-driven architectures with billions of events

Consider alternatives when:
- Workload is primarily batch analytics (Delta Lake may be simpler)
- Multi-engine portability is paramount (Iceberg may be better)
- Team lacks experience with streaming data systems
- Operational simplicity outweighs upsert performance needs
