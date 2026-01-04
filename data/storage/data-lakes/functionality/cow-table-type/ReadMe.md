# Copy-on-Write (COW) Table Type

## Summary

Copy-on-Write is a storage strategy where every write operation produces a new version of any affected files. When a record is updated or deleted, the entire data file containing that record is read, modified, and rewritten as a new file. The old file is marked for deletion. This approach prioritizes read performance at the cost of write amplification.

Key points to remember:

- Every update rewrites the entire file containing the affected record, even for single-row changes
- Read performance is optimal because queries only read Parquet files with no merge logic
- Write latency is higher due to full file rewrites
- Storage efficiency is good because data is always in columnar Parquet format
- Best suited for read-heavy workloads with batch updates
- Delta Lake and Iceberg are inherently COW; Hudi offers COW as one of two table types
- COW is simpler operationally than Merge-on-Read because no background compaction is required

## How Copy-on-Write Works

### The Write Process

When updating a record in a COW table:

```
1. Identify file containing the record (via index or scan)
2. Read the entire file into memory
3. Apply the update to affected rows
4. Write a new file with the modified data
5. Commit metadata: mark old file as removed, new file as added
```

For example, updating one record in a 128MB Parquet file:
- Read 128MB from storage
- Modify 1 row
- Write 128MB to storage
- Old file is tombstoned (deleted later during vacuum)

### Visual Representation

```
Before Update:
  file-001.parquet: [row1, row2, row3, row4]

Update: row2 -> row2'

After Update:
  file-001.parquet: (tombstoned, awaiting vacuum)
  file-002.parquet: [row1, row2', row3, row4]
```

The entire file is rewritten even though only one row changed.

### Read Process

Reading from a COW table is straightforward:

```
1. Query metadata to get list of active files
2. Read files directly (standard Parquet reads)
3. Return results
```

No merge logic, no log file processing, no on-the-fly transformations. This simplicity is COW's primary advantage.

## Write Amplification

### Understanding the Cost

Write amplification measures how much data is written compared to the logical change:

```
Write Amplification = Physical Data Written / Logical Data Changed
```

For a single-row update in a 128MB file:
- Logical change: ~100 bytes
- Physical write: 128MB
- Write amplification: ~1,000,000x

For batch updates affecting 10% of rows across all files:
- Logical change: 10% of data
- Physical write: 100% of data (all files rewritten)
- Write amplification: 10x

### When Write Amplification Matters

High write amplification impacts:

**Storage costs**: More data written means more API calls and higher cloud storage costs.

**Write latency**: Large files take longer to rewrite, increasing commit latency.

**Throughput**: Storage bandwidth becomes the bottleneck for update-heavy workloads.

**Concurrent writes**: More data movement increases conflict probability.

### Mitigating Write Amplification

**Smaller files**: Reduce target file size to limit rewrite scope.
```python
# Target 32MB files instead of 128MB
spark.conf.set("spark.sql.files.maxRecordsPerFile", 100000)
```

**Partition alignment**: Partition by update key to limit affected partitions.
```python
# Updates by customer only rewrite customer's partition
df.write.partitionBy("customer_id")
```

**Batch updates**: Accumulate changes and apply in larger batches.
```python
# Instead of 1000 single-row updates, do one batch update
```

## Read Performance Characteristics

### Why COW Excels at Reads

COW tables deliver optimal read performance because:

**Pure Parquet**: All data is in optimized columnar format with compression and encoding.

**No merge overhead**: Readers never combine multiple file versions.

**Predictable planning**: Query planning only needs the active file list.

**Full statistics**: Parquet files include complete column statistics for pruning.

### Query Performance Profile

| Query Type | COW Performance | Notes |
|------------|-----------------|-------|
| Full table scan | Excellent | Standard Parquet scan |
| Point lookup | Good | Requires scanning files |
| Range query | Excellent | Partition pruning + statistics |
| Aggregation | Excellent | Columnar format optimal |
| Join | Good to Excellent | Standard Spark/Presto optimizations |

### Comparison to MOR Reads

Merge-on-Read tables must combine base files with log files during reads. COW avoids this entirely:

```
COW Read:
  file-001.parquet -> [results]

MOR Read:
  file-001.parquet + log-001.avro + log-002.avro -> merge -> [results]
```

For analytical queries touching many records, COW's lack of merge overhead provides consistent performance.

## Implementation by Format

### Delta Lake

Delta Lake uses COW semantics exclusively. Every update rewrites affected files:

```python
# This rewrites every file containing matching rows
spark.sql("UPDATE my_table SET status = 'active' WHERE region = 'US'")
```

Deletion vectors (a recent addition) enable lightweight deletes by marking rows as deleted without file rewrite:

```python
# Enable deletion vectors
spark.sql("""
  ALTER TABLE my_table
  SET TBLPROPERTIES ('delta.enableDeletionVectors' = true)
""")
```

With deletion vectors, deletes are fast, but reads must filter marked rows.

### Apache Iceberg

Iceberg is also inherently COW. Updates trigger file rewrites:

```sql
UPDATE my_table SET quantity = 10 WHERE order_id = 12345
```

Iceberg supports merge-on-read for deletes via position delete files:

```sql
DELETE FROM my_table WHERE status = 'cancelled'
-- Creates position delete file, no immediate rewrite
```

Position deletes defer the rewrite cost, with eventual compaction:

```sql
CALL catalog.system.rewrite_data_files('my_table')
```

### Apache Hudi

Hudi explicitly supports COW as a table type, configured at table creation:

```python
.option("hoodie.datasource.write.table.type", "COPY_ON_WRITE")
```

COW behavior in Hudi:
- Updates trigger full file rewrites
- Inserts append new files (no rewrite)
- Deletes rewrite files with affected records removed

Hudi's indexing accelerates finding which files contain records to update:

```python
.option("hoodie.index.type", "BLOOM")  # or BUCKET, SIMPLE
```

## Operational Characteristics

### Simplicity

COW is operationally simpler than MOR:

- No compaction jobs to schedule
- No log files to manage
- Consistent file format (always Parquet)
- Standard Parquet tooling works directly

### Storage Footprint

COW storage characteristics:
- Active data size equals current table size
- Historical versions consume additional space (for time travel)
- No intermediate log files
- Regular vacuum removes old file versions

### Failure Recovery

Recovery from failed writes is straightforward:
- Uncommitted files are orphaned
- Next transaction log read ignores uncommitted work
- Vacuum cleans up orphaned files

No special recovery process needed compared to MOR's potential log corruption scenarios.

## When to Choose COW

### Ideal Workloads

**Read-heavy analytics**: Dashboards, reports, ad-hoc queries.

**Batch ETL pipelines**: Daily or hourly batch updates, not continuous streaming.

**Dimension tables**: Slowly changing dimensions with infrequent updates.

**Aggregation-heavy queries**: OLAP workloads benefit from pure columnar format.

**Regulatory compliance queries**: Time travel and auditing with predictable performance.

### Warning Signs for COW

Consider MOR if you observe:
- Update latency becoming unacceptable
- Storage API costs dominated by writes
- Write throughput limits pipeline performance
- High-frequency streaming updates (sub-minute)

## COW vs MOR Comparison

| Aspect | Copy-on-Write | Merge-on-Read |
|--------|---------------|---------------|
| Read latency | Low | Variable |
| Write latency | Higher | Lower |
| Write amplification | High | Low |
| Query complexity | Simple | Complex (merge) |
| Operational overhead | Low | Higher (compaction) |
| Storage during writes | Stable | Grows (logs) |
| Best for | Read-heavy, batch | Write-heavy, streaming |

### Decision Framework

```
Choose COW when:
  - Reads >> Writes (10:1 or more)
  - Updates are batched (hourly, daily)
  - Query SLAs are strict
  - Operational simplicity is valued
  - Team is new to table formats

Choose MOR when:
  - Writes are frequent (sub-minute)
  - Update latency is critical
  - Write amplification costs are prohibitive
  - Streaming ingestion is primary use case
```

## Performance Tuning

### File Size Optimization

Balance between write amplification and read performance:

**Smaller files** (32-64MB):
- Lower write amplification
- More files to manage
- Higher metadata overhead

**Larger files** (128-256MB):
- Higher write amplification
- Fewer files
- Better compression, faster scans

Recommendation: Start with 128MB, adjust based on update patterns.

### Clustering and Sorting

Organize data to minimize files affected by updates:

```sql
-- Delta Lake: Z-order by update key
OPTIMIZE my_table ZORDER BY (update_key)

-- Iceberg: Sort by update key
ALTER TABLE my_table WRITE ORDERED BY update_key
```

If updates cluster by certain columns (e.g., recent dates), sorting reduces the number of files to rewrite.

### Partition Strategy

Align partitions with update patterns:

```python
# If updates target specific dates
.partitionBy("date")

# If updates target specific regions
.partitionBy("region", "date")
```

Updates confined to specific partitions rewrite fewer files.

## Best Practices

### Design for Batch Updates

Structure pipelines to batch changes:
- Accumulate updates in staging area
- Apply as single batch operation
- Schedule during low-activity windows

### Monitor Write Amplification

Track metrics:
- Bytes written vs. logical changes
- Files rewritten per operation
- Commit duration trends

### Regular Compaction (Even for COW)

While COW does not require MOR-style compaction, periodic optimization improves layout:

```sql
-- Consolidate small files
OPTIMIZE my_table

-- Remove old versions
VACUUM my_table RETAIN 168 HOURS
```

### Test Update Patterns

Before production:
- Benchmark update operations at expected scale
- Measure write amplification
- Validate that read SLAs are met despite updates

### Document Table Type Choice

Record the reasoning:
- Why COW was chosen
- Expected read/write ratio
- Conditions that would trigger reconsideration
- Performance baselines for comparison
