# ACID Transactions in Data Lake Table Formats

## Summary

ACID transactions bring database-level reliability guarantees to data lakes, transforming append-only object storage into mutable, consistent data platforms. Without ACID properties, concurrent reads and writes on data lakes can produce corrupted results, partial updates, or inconsistent views of data.

Key points to remember:

- ACID stands for Atomicity, Consistency, Isolation, and Durability
- Table formats implement ACID through transaction logs that track commits as atomic units
- Optimistic concurrency control is the standard approach, avoiding locks in favor of conflict detection at commit time
- All three major formats (Delta Lake, Hudi, Iceberg) provide single-table ACID guarantees, not multi-table transactions
- Transaction isolation levels vary: snapshot isolation for reads, serializable for writes
- Failed transactions leave orphaned files that maintenance operations clean up
- ACID overhead is minimal for read-heavy workloads but adds latency to high-frequency writes

## Why Data Lakes Need ACID

Traditional data lakes store files directly on object storage without coordination. This creates several failure modes:

**Partial Writes**: A multi-file write operation that fails midway leaves some files written and others missing. Readers see incomplete data.

**Dirty Reads**: A query starts reading while a write is in progress. The query reads some new files and some old files, producing inconsistent results.

**Lost Updates**: Two concurrent writers both read the current state, make changes, and write back. The second writer overwrites the first writer's changes.

**Phantom Reads**: A long-running query sees different data at different points because other transactions committed during execution.

Table formats solve these problems by introducing a transaction log that serves as a single source of truth for table state.

## The ACID Properties

### Atomicity

Atomicity ensures that transactions are all-or-nothing. Either every file in a multi-file write commits successfully, or none of them do. Readers never see partial results.

Table formats achieve atomicity through a two-phase approach:

1. Write Phase: Data files are written to storage but not yet visible to readers
2. Commit Phase: A single atomic metadata update makes all files visible simultaneously

The commit phase relies on atomic operations provided by the storage layer. On object stores like S3, this is typically a conditional PUT operation on a metadata file. If the PUT fails because another transaction committed first, the entire transaction is rolled back.

**Delta Lake**: Writes a new JSON file to `_delta_log` with a sequential version number. The version number acts as a compare-and-swap mechanism.

**Apache Hudi**: Creates a new instant on the timeline. The instant transitions from requested to inflight to completed atomically.

**Apache Iceberg**: Updates a metadata pointer file that references the new snapshot. The pointer update is atomic.

### Consistency

Consistency ensures that transactions move the database from one valid state to another. In table formats, this primarily means:

- Schema constraints are enforced (column types, nullability)
- Partition constraints are maintained
- Invariants like unique primary keys (in formats that support them) are preserved

Schema enforcement happens at write time. Attempting to write data that violates the table schema produces an error rather than corrupting the table.

### Isolation

Isolation determines what concurrent transactions can see of each other's work. Table formats provide:

**Snapshot Isolation for Readers**: Each query sees a consistent snapshot of the table as of a specific version. Writes occurring during query execution are invisible. This means readers are never blocked by writers.

**Serializable Isolation for Writers**: Concurrent writers are serialized to prevent conflicts. If two transactions modify overlapping data, one succeeds and one fails with a conflict error.

The isolation model differs from traditional databases. There are no read locks, no write locks that block readers, and no deadlocks. This enables high read throughput even during active writes.

### Durability

Durability guarantees that committed transactions survive system failures. Table formats inherit durability from the underlying storage:

- S3: 99.999999999% durability through automatic replication
- GCS: Similar durability with multi-region options
- ADLS: Configurable redundancy levels
- HDFS: Replication factor determines durability

Once a commit completes and the storage acknowledges the write, the data is durable. Recovery after failures simply involves reading the transaction log to determine the last committed state.

## Concurrency Control Mechanisms

### Optimistic Concurrency Control

All three major table formats use optimistic concurrency control (OCC) rather than locking:

1. Transaction reads current table state (version N)
2. Transaction computes changes based on that state
3. Transaction attempts to commit as version N+1
4. If another transaction already committed N+1, the current transaction fails
5. Failed transactions can retry with the new state

OCC works well for data lake workloads because:
- Read operations are never blocked
- Most writes touch different partitions and do not conflict
- The cost of occasional retries is lower than the cost of lock management at scale

### Conflict Detection

When a conflict occurs, the transaction must determine if it can be resolved:

**File-Level Conflicts**: Two transactions modify the same file. This always requires a retry because the original file state is no longer valid.

**Partition-Level Conflicts**: Two transactions write to the same partition. Depending on the operation, this may or may not conflict.

**Append Conflicts**: Two transactions append new files. These typically do not conflict because neither transaction modifies existing data.

Delta Lake, Hudi, and Iceberg each have rules for determining which conflicts require retries:

| Conflict Type | Delta Lake | Hudi | Iceberg |
|---------------|------------|------|---------|
| Concurrent appends | OK | OK | OK |
| Concurrent updates to same rows | Retry | Retry | Retry |
| Schema change during transaction | Retry | Retry | Retry |
| Compaction during transaction | OK | OK | OK |

### Write-Ahead Logging

Some formats use write-ahead logging (WAL) patterns where intentions are recorded before execution:

**Hudi MOR Tables**: Write operations first log to Avro log files, which act as a WAL. The log entries are later merged into base files during compaction.

**Iceberg**: Each commit records the intended changes in a manifest file before the changes become visible.

This pattern enables recovery: if a transaction fails after writing the log but before committing, the next operation can detect and clean up the incomplete work.

## Implementation Across Formats

### Delta Lake

Delta Lake stores its transaction log in the `_delta_log` directory as sequentially numbered JSON files:

```
_delta_log/
  00000000000000000000.json
  00000000000000000001.json
  00000000000000000002.json
  ...
```

Each JSON file represents one atomic commit containing actions like:
- `add`: New file added to table
- `remove`: File tombstoned (soft delete for time travel)
- `metaData`: Schema or configuration change
- `commitInfo`: Transaction metadata

Atomicity relies on the fact that only one writer can successfully create a file with a given version number. If two transactions race for version 42, exactly one succeeds and one fails.

Every 10 commits, Delta Lake writes a checkpoint in Parquet format that consolidates the current state. This accelerates recovery by avoiding the need to replay the entire log.

### Apache Hudi

Hudi uses a timeline architecture stored in the `.hoodie` directory:

```
.hoodie/
  hoodie.properties
  20240115100000.commit.requested
  20240115100000.commit.inflight
  20240115100000.commit
```

Each timeline instant has three states:
- **Requested**: Transaction has started
- **Inflight**: Transaction is executing
- **Completed**: Transaction has committed

The state transitions are atomic. A completed instant file contains metadata about which file groups were modified, enabling conflict detection.

Hudi's timeline also tracks non-commit operations like compaction, cleaning, and rollback, providing a complete audit trail of table operations.

### Apache Iceberg

Iceberg uses a two-level metadata hierarchy:

1. **Metadata Files**: JSON files containing table schema, partition spec, and pointers to manifest lists
2. **Manifest Lists**: Avro files listing all manifest files for a snapshot
3. **Manifests**: Avro files listing data files with statistics

Commits work by:
1. Writing new manifest files for added/removed data files
2. Writing a new manifest list referencing all current manifests
3. Writing a new metadata file pointing to the new manifest list
4. Atomically updating the catalog pointer to the new metadata file

The catalog pointer update is the commit point. This can be a file rename (on HDFS), a conditional write (on S3), or a catalog service update (Hive Metastore, AWS Glue).

## Transaction Scope and Limitations

### Single-Table Transactions

All three formats provide ACID guarantees only at the single-table level. There is no support for multi-table transactions that atomically update multiple tables.

For workflows requiring multi-table consistency:
- Design idempotent pipelines that can be safely re-run
- Use staging tables and atomic swaps
- Accept eventual consistency between related tables
- Consider a data warehouse for strict multi-table requirements

### Isolation Level Limitations

While table formats provide strong isolation, some edge cases exist:

**Long-Running Queries**: A query that runs for hours sees a consistent snapshot, but that snapshot may be very stale by completion time.

**External Modifications**: Direct file modifications bypassing the table format break consistency guarantees.

**Metadata Conflicts**: Rapid schema changes during active writes can cause transaction failures.

### Performance Implications

ACID transactions add overhead:

**Write Latency**: Each commit requires metadata operations in addition to data writes. For streaming workloads with per-record commits, this overhead is significant.

**Conflict Retries**: High-contention workloads may experience retry storms where many transactions repeatedly fail and retry.

**Cleanup Costs**: Orphaned files from failed transactions require maintenance operations to clean up.

Mitigation strategies:
- Batch writes into larger transactions (streaming micro-batches)
- Partition data to reduce write conflicts
- Tune retry policies with exponential backoff
- Schedule regular maintenance operations

## Practical Considerations

### Choosing Commit Frequency

For streaming ingestion, the tradeoff is between latency and overhead:

- **High-frequency commits** (every second): Lowest latency, highest metadata overhead, more small files
- **Micro-batch commits** (every 1-5 minutes): Balanced latency and efficiency
- **Batch commits** (hourly or daily): Lowest overhead, highest latency

Most production streaming pipelines use micro-batching with 1-5 minute intervals.

### Handling Conflicts

When a transaction fails due to conflict:

1. Determine if retry is appropriate (most cases yes)
2. Re-read current table state
3. Recompute changes based on new state
4. Retry the commit

Delta Lake and Iceberg provide automatic retry logic with configurable attempts. Hudi exposes conflict resolution through its write client.

For workloads with frequent conflicts, consider:
- Partitioning to isolate concurrent writers
- Reducing transaction scope
- Using append-only patterns where possible

### Monitoring Transaction Health

Key metrics to track:

- Commit latency (time from write start to commit completion)
- Conflict rate (percentage of transactions requiring retry)
- Retry count per transaction
- Transaction log growth rate
- Orphaned file count

High conflict rates indicate hot partitions or suboptimal write patterns. Growing orphaned file counts indicate insufficient maintenance.

## Comparison Across Formats

| Aspect | Delta Lake | Hudi | Iceberg |
|--------|------------|------|---------|
| Concurrency model | Optimistic | Optimistic | Optimistic |
| Conflict granularity | File-level | File group | File-level |
| Automatic retries | Yes | Configurable | Yes |
| Transaction log format | JSON + Parquet | Timeline instants | JSON + Avro |
| Multi-table transactions | No | No | No |
| Streaming commit support | Native | Native | Good |

All three formats provide robust ACID guarantees suitable for production data lake workloads. The choice between them depends more on ecosystem fit and other features than on transaction semantics.
