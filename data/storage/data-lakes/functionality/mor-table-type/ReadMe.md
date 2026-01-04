# Merge-on-Read (MOR) Table Type

## Summary

Merge-on-Read is a storage strategy that optimizes for write performance by appending changes to log files rather than rewriting data files. Updates and deletes are recorded as delta logs. During reads, the system merges base files with their corresponding log files to present the current state of the data. This approach trades read complexity for dramatically lower write latency.

Key points to remember:

- Writes append to log files instead of rewriting base Parquet files
- Reads must merge base files with log files, adding computational overhead
- Compaction periodically merges logs into base files to bound read costs
- Ideal for streaming ingestion and high-frequency upsert workloads
- Hudi explicitly supports MOR; Iceberg and Delta Lake have partial MOR features
- Requires operational management of compaction scheduling
- Read-optimized queries can skip logs for lower latency at the cost of data freshness

## How Merge-on-Read Works

### The Write Process

When updating a record in a MOR table:

```
1. Identify file group containing the record (via index)
2. Write the updated record to a log file
3. Commit metadata referencing the new log entry
```

No base file is read or rewritten. The log file is a small, append-only write.

For example, updating one record:
- Identify file group for the record
- Append ~100 bytes to log file
- Commit completes

Compare this to COW, which rewrites the entire 128MB base file.

### Visual Representation

```
Initial State:
  base-001.parquet: [row1, row2, row3, row4]

Update: row2 -> row2'

After Update:
  base-001.parquet: [row1, row2, row3, row4]  (unchanged)
  log-001.avro: [update: row2 -> row2']

Another Update: row3 -> row3'

After Second Update:
  base-001.parquet: [row1, row2, row3, row4]
  log-001.avro: [update: row2 -> row2', update: row3 -> row3']

Read Query (snapshot):
  Merge base-001.parquet + log-001.avro
  Result: [row1, row2', row3', row4]
```

### The Read Process

Reading from a MOR table requires merge logic:

```
1. Query metadata to get list of base files and log files
2. For each file group:
   a. Read base Parquet file
   b. Read associated log files
   c. Apply log entries to base data (merge)
3. Return merged results
```

This merge operation adds CPU and I/O overhead compared to reading pure Parquet files.

## File Group Architecture

### File Group Concept

In MOR tables, files are organized into file groups. Each file group contains:

**Base file**: A Parquet file containing the bulk of the data at some point in time.

**Log files**: One or more Avro files containing updates since the base file was written.

**File slices**: A versioned snapshot of a file group at a specific commit.

```
file-group-001/
  base-001-1.parquet       # Base file from commit 1
  log-001-2.avro          # Updates from commit 2
  log-001-3.avro          # Updates from commit 3
  log-001-4.avro          # Updates from commit 4
```

### Record Routing

When a record arrives, the system determines its file group:

1. Apply the index to the record key
2. Index returns the file group ID
3. If update: append to log file for that file group
4. If insert to new partition: create new file group

This routing ensures that all versions of a record are co-located within a single file group.

## Compaction

### Why Compaction is Essential

Without compaction, log files grow unbounded. This causes:
- Ever-increasing read latency (more logs to merge)
- Growing storage costs
- Query timeouts for large log accumulation

Compaction merges log files into base files, resetting the merge overhead:

```
Before Compaction:
  base-001.parquet + log-001.avro + log-002.avro + log-003.avro

After Compaction:
  base-002.parquet  (merged result)
  (log files removed)
```

### Compaction Strategies

**Inline compaction**: Compaction runs as part of the write operation.
```python
.option("hoodie.compact.inline", "true")
.option("hoodie.compact.inline.max.delta.commits", "5")
```
Pros: Automatic, no separate job
Cons: Increases write latency

**Async compaction**: Compaction runs as a separate process.
```python
.option("hoodie.compact.inline", "false")
# Run compaction separately
HoodieSparkUtils.runCompaction(spark, path)
```
Pros: Write latency unaffected
Cons: Requires scheduling and monitoring

**Scheduled compaction**: Compaction runs on a schedule (e.g., hourly).
```python
# Trigger compaction at defined intervals
# Often managed by orchestrators like Airflow
```

### Compaction Tuning

Balance between read and write performance:

**Aggressive compaction** (every 3-5 commits):
- Lower read latency
- Higher write amplification
- More frequent I/O spikes

**Conservative compaction** (every 20-50 commits):
- Higher read latency
- Lower write overhead
- Accumulating storage for logs

Key parameters:
```python
# Max delta commits before compaction
.option("hoodie.compact.inline.max.delta.commits", "5")

# Target file size after compaction
.option("hoodie.compaction.target.io", "500000")

# Strategy for selecting file groups
.option("hoodie.compaction.strategy", "org.apache.hudi.table.action.compact.strategy.BoundedIOCompactionStrategy")
```

## Query Types

### Snapshot Queries

Return the latest state by merging all base files with their logs:

```python
df = spark.read.format("hudi").load(path)
```

This is the default query type. It reflects all committed updates.

### Read-Optimized Queries

Read only base Parquet files, ignoring log files:

```python
df = spark.read.format("hudi").load(path + "/*_ro")
# Or with explicit option
.option("hoodie.datasource.query.type", "read_optimized")
```

Characteristics:
- Faster than snapshot queries (no merge)
- May miss recent updates (before compaction)
- Data freshness depends on compaction frequency

Use when:
- Slight staleness is acceptable
- Query latency is critical
- Updates are eventually consistent

### Incremental Queries

Read only changes since a specified commit:

```python
spark.read.format("hudi") \
    .option("hoodie.datasource.query.type", "incremental") \
    .option("hoodie.datasource.read.begin.instanttime", "20240115100000") \
    .load(path)
```

This is a MOR superpower: efficiently extract just the delta without scanning the entire table.

## Implementation by Format

### Apache Hudi

Hudi is the primary format with full MOR support, configured at table creation:

```python
df.write.format("hudi") \
    .option("hoodie.datasource.write.table.type", "MERGE_ON_READ") \
    .option("hoodie.table.name", "my_mor_table") \
    .option("hoodie.datasource.write.recordkey.field", "id") \
    .option("hoodie.datasource.write.precombine.field", "timestamp") \
    .save(path)
```

Hudi's MOR includes:
- Avro log files for updates
- Multiple index types for record routing
- Built-in compaction service
- Read-optimized and snapshot query types

### Delta Lake

Delta Lake is primarily COW but has introduced MOR-like features:

**Deletion vectors**: Mark rows as deleted without file rewrite.
```sql
ALTER TABLE my_table SET TBLPROPERTIES (
  'delta.enableDeletionVectors' = true
)
```

Deletion vectors are stored separately and applied during reads, similar to MOR delete logs.

### Apache Iceberg

Iceberg supports MOR for deletes via position delete files:

```sql
DELETE FROM my_table WHERE status = 'cancelled'
-- Creates position delete file
```

Position delete files list row positions to skip. Updates still require file rewrites (COW), but deletes can be deferred.

Merge on read for updates is on the Iceberg roadmap but not yet production-ready.

## Performance Characteristics

### Write Performance

MOR dramatically improves write performance:

| Metric | COW | MOR |
|--------|-----|-----|
| Single-row update latency | High (file rewrite) | Low (log append) |
| Batch update throughput | Limited by rewrites | Near-append speed |
| Write amplification | High | Low |
| Streaming suitability | Poor | Excellent |

### Read Performance

MOR read performance depends on log accumulation:

| Scenario | Read Performance |
|----------|------------------|
| After compaction | Excellent (pure Parquet) |
| Few log files | Good |
| Many log files | Degraded |
| Read-optimized query | Excellent (skips logs) |

### Performance Profiles

```
COW: Consistent read performance, variable write cost
     ˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆ  Read
     ˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆ  Write (varies by data)

MOR: Variable read performance, consistent write cost
     ˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆ  Read (varies by logs)
     ˆˆˆˆˆˆˆˆ  Write
```

## Operational Considerations

### Monitoring Requirements

MOR tables require active monitoring:

**Log file accumulation**:
```python
# Check pending compaction
spark.read.format("hudi").load(path + "/.hoodie/compaction")
```

**Compaction lag**: Time since last compaction.

**Query latency trends**: Increasing latency indicates compaction is needed.

**File group health**: Log count per file group.

### Compaction Job Management

Production MOR deployments need:

1. Scheduled compaction jobs (Airflow, Dagster, Databricks Jobs)
2. Monitoring for compaction failures
3. Alerting for log accumulation thresholds
4. Capacity planning for compaction compute

Example Airflow pattern:
```python
@dag(schedule_interval="0 * * * *")  # Hourly
def hudi_compaction():
    compact_table = SparkSubmitOperator(
        task_id="compact_mor_table",
        application="hudi-compaction-job.py",
        ...
    )
```

### Failure Modes

MOR-specific failure scenarios:

**Compaction backlog**: If compaction falls behind, read latency degrades progressively.

**Log corruption**: Corrupted log files can affect reads until compaction removes them.

**Resource contention**: Compaction competing with query workloads.

Mitigation strategies:
- Separate compaction clusters from query clusters
- Set compaction SLAs with alerting
- Monitor log file age and count

## When to Choose MOR

### Ideal Workloads

**Streaming ingestion**: Continuous data streams with frequent updates.
```python
# Spark structured streaming to Hudi MOR
stream.writeStream.format("hudi") \
    .option("hoodie.datasource.write.table.type", "MERGE_ON_READ") \
    .start(path)
```

**CDC pipelines**: Change data capture with high update velocity.

**Event-driven architectures**: Real-time event processing with upserts.

**GDPR/CCPA compliance**: Frequent deletes for user data removal.

### Warning Signs Against MOR

Consider COW if:
- Read latency SLAs are strict and non-negotiable
- Team lacks operational capacity for compaction management
- Updates are infrequent (daily or less)
- Query patterns do not tolerate stale data

## MOR vs COW Comparison

| Aspect | Merge-on-Read | Copy-on-Write |
|--------|---------------|---------------|
| Write latency | Low | High |
| Read latency | Variable | Low |
| Write amplification | Low | High |
| Query complexity | Complex (merge) | Simple |
| Operational overhead | Higher (compaction) | Low |
| Storage during ingestion | Grows (logs) | Stable |
| Data freshness | Immediate | Immediate |
| Streaming suitability | Excellent | Poor |

### Decision Framework

```
Choose MOR when:
  - Writes >> Reads (or balanced)
  - Updates are frequent (sub-minute)
  - Write latency is critical
  - Team can manage compaction
  - Read-optimized queries acceptable for some use cases

Choose COW when:
  - Reads >> Writes
  - Query SLAs are strict
  - Operational simplicity is valued
  - Updates are batched
```

## Performance Tuning

### Index Selection

Fast record routing is critical for MOR performance:

**Bloom index**: Good for temporal data with ordered keys.
```python
.option("hoodie.index.type", "BLOOM")
```

**Bucket index**: Best for large tables with uniform key distribution.
```python
.option("hoodie.index.type", "BUCKET")
.option("hoodie.bucket.index.num.buckets", "256")
```

**Record-level index**: For very large tables (100B+ records).
```python
.option("hoodie.index.type", "RECORD_INDEX")
```

### Log File Sizing

Balance between write efficiency and read overhead:

**Smaller logs** (1-5MB):
- More files to merge
- Higher metadata overhead
- Lower latency per write

**Larger logs** (50-100MB):
- Fewer files
- Higher write buffering latency
- More efficient compaction

```python
.option("hoodie.logfile.max.size", "33554432")  # 32MB
```

### Parallelism Tuning

Compaction parallelism:
```python
.option("hoodie.compaction.lazy.block.read", "true")
.option("hoodie.compact.inline.max.delta.commits", "5")
```

Write parallelism:
```python
.option("hoodie.insert.shuffle.parallelism", "200")
.option("hoodie.upsert.shuffle.parallelism", "200")
```

## Best Practices

### Design for Compaction

Plan compaction from the start:
- Schedule compaction jobs before deploying
- Set up monitoring and alerting
- Reserve compute capacity

### Use Read-Optimized When Possible

For dashboards and reports that tolerate slight staleness:
```python
# Faster queries, slightly stale data
spark.read.format("hudi").load(path + "/*_ro")
```

### Monitor Log Accumulation

Alert on:
- Log files older than threshold (e.g., 4 hours)
- Log count per file group exceeding threshold
- Compaction job failures

### Test Merge Overhead

Before production:
- Measure query latency with varying log counts
- Determine acceptable log accumulation
- Set compaction frequency accordingly

### Document Operational Runbooks

For MOR tables, document:
- Compaction schedule and triggers
- Monitoring dashboards
- Recovery procedures for compaction failures
- Escalation path for degraded performance
