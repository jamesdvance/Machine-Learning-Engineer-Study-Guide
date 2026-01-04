# Partition Management in Data Lake Table Formats

## Summary

Partitioning organizes data into subdirectories based on column values, enabling queries to skip irrelevant data entirely. When a query filters on partition columns, the table format reads only the matching partitions rather than scanning the entire table. This transforms full table scans into targeted reads, often improving performance by orders of magnitude.

Key points to remember:

- Partitioning physically separates data into directories based on partition column values
- Partition pruning eliminates irrelevant partitions from query plans before any data is read
- Over-partitioning creates many small files; under-partitioning creates large scans
- Common partition strategies: date-based (year/month/day), categorical (region, status)
- Partition evolution allows changing partition schemes without rewriting existing data
- Iceberg offers hidden partitioning, eliminating the need to manage partition columns in queries
- Delta Lake and Hudi use explicit partition columns similar to Hive-style partitioning

## How Partitioning Works

### Physical Layout

Partitioned tables store data in directory hierarchies:

```
my_table/
  year=2024/
    month=01/
      part-001.parquet
      part-002.parquet
    month=02/
      part-003.parquet
  year=2023/
    month=12/
      part-004.parquet
```

Each partition directory contains only the data matching its partition values. The directory structure itself encodes the partition information.

### Partition Pruning

When a query filters on partition columns, the table format identifies relevant partitions from metadata without reading data files:

```sql
SELECT * FROM my_table
WHERE year = 2024 AND month = 01
```

Query execution:
1. Metadata lookup identifies partitions matching `year=2024/month=01`
2. Only files in those directories are included in the scan plan
3. Other partitions are never touched

For a table with 10 years of monthly data (120 partitions), this query reads 1/120th of the data.

### Partition Column Handling

In Hive-style partitioning (Delta Lake, Hudi), partition columns are not stored in data files. They are derived from directory paths:

```python
# Writing partitioned data
df.write.format("delta") \
    .partitionBy("year", "month") \
    .save(path)
```

The partition columns are physically removed from Parquet files and reconstructed from paths during reads.

Iceberg handles this differently with hidden partitioning, discussed below.

## Partition Strategy Design

### Choosing Partition Columns

Select partition columns based on:

**Query patterns**: Partition by columns that appear in WHERE clauses. If most queries filter by date, partition by date.

**Cardinality**: Partition columns should have moderate cardinality (hundreds to thousands of values, not millions).

**Data distribution**: Partitions should be roughly equal in size for balanced processing.

### Common Partition Schemes

**Date-based partitioning**: The most common approach for time-series data.

```python
# Fine-grained: year/month/day
.partitionBy("year", "month", "day")

# Coarser: year/month
.partitionBy("year", "month")

# Derived: Single date partition
df.withColumn("date", to_date("timestamp")) \
  .write.partitionBy("date")
```

**Categorical partitioning**: For data with natural categories.

```python
# Geographic partitioning
.partitionBy("region", "country")

# Status-based
.partitionBy("status")  # active, archived, deleted
```

**Composite partitioning**: Combining time and category.

```python
.partitionBy("region", "year", "month")
```

### Partition Granularity Tradeoffs

**Too many partitions (over-partitioning)**:
- Creates many small files
- Increases metadata overhead
- Slows down file listing operations
- May hurt query performance despite pruning

**Too few partitions (under-partitioning)**:
- Each partition contains too much data
- Limited pruning benefit
- Queries must scan more data than necessary

**Sizing guidelines**:
- Target 100MB to 1GB of data per partition
- Avoid partitions with fewer than 10 files
- Limit total partition count to tens of thousands

### Partition Column Data Types

Use simple types for partition columns:
- Strings: Most flexible, work everywhere
- Integers: Efficient for year, month, day
- Dates: Clean but may require conversion

Avoid:
- Timestamps with milliseconds (creates too many partitions)
- Floats (rounding issues)
- Complex types (not supported)

## Partition Evolution

### The Problem

Business requirements change. A table initially partitioned by day may need hour-level granularity for recent data. Traditionally, this required:
1. Create new table with new partitioning
2. Copy all data
3. Swap references
4. Delete old table

This is expensive and disruptive.

### Evolution in Iceberg

Iceberg supports partition evolution as a metadata-only operation:

```sql
-- Original partitioning
CREATE TABLE events (
  event_id BIGINT,
  event_type STRING,
  timestamp TIMESTAMP
) PARTITIONED BY (days(timestamp))

-- Add hour-level partitioning for new data
ALTER TABLE events ADD PARTITION FIELD hours(timestamp)
```

After evolution:
- Existing files retain their original partitioning
- New writes use the new partition scheme
- Queries transparently handle both layouts

### Evolution in Delta Lake and Hudi

Delta Lake and Hudi do not support partition evolution in the same way. Changing partition columns requires rewriting data:

```python
# Delta Lake: Rewrite with new partitioning
spark.read.format("delta").load(old_path) \
    .write.format("delta") \
    .partitionBy("new_column") \
    .save(new_path)
```

Some workarounds:
- Add new partition column as a regular column
- Use Z-ordering for multi-dimensional clustering
- Design partitioning carefully upfront

## Hidden Partitioning (Iceberg)

### The Problem with Explicit Partitioning

Hive-style partitioning requires users to know the partition structure:

```sql
-- User must know table is partitioned by year and month
SELECT * FROM events
WHERE year = 2024 AND month = 01
  AND timestamp >= '2024-01-15' AND timestamp < '2024-01-16'
```

If users filter only on timestamp without including year and month, partition pruning does not occur. This leads to accidental full table scans.

### Iceberg's Solution

Iceberg decouples partition values from partition columns. You define partition transforms on existing columns:

```sql
CREATE TABLE events (
  event_id BIGINT,
  event_type STRING,
  timestamp TIMESTAMP
) PARTITIONED BY (days(timestamp))
```

Users query naturally:

```sql
SELECT * FROM events
WHERE timestamp >= '2024-01-15' AND timestamp < '2024-01-16'
```

Iceberg automatically:
1. Translates the timestamp filter to partition values
2. Prunes partitions based on the transform
3. Returns correct results

### Partition Transforms

Iceberg supports several transforms:

| Transform | Example | Use Case |
|-----------|---------|----------|
| `identity(col)` | `identity(region)` | Categorical partitioning |
| `years(col)` | `years(timestamp)` | Coarse time partitioning |
| `months(col)` | `months(timestamp)` | Monthly partitioning |
| `days(col)` | `days(timestamp)` | Daily partitioning |
| `hours(col)` | `hours(timestamp)` | Hourly partitioning |
| `bucket(n, col)` | `bucket(16, user_id)` | Hash-based distribution |
| `truncate(n, col)` | `truncate(10, zipcode)` | Prefix-based grouping |

Multiple transforms can be combined:

```sql
PARTITIONED BY (days(timestamp), bucket(16, user_id))
```

## Dynamic Partition Management

### Adding Partitions

In most formats, partitions are created automatically when data is written:

```python
# Writing to a new partition creates it automatically
df.write.format("delta").mode("append").save(path)
# If data has year=2024, month=02, that partition is created
```

Some systems require explicit partition creation:
```sql
ALTER TABLE my_table ADD PARTITION (year=2024, month=02)
```

### Dropping Partitions

Remove partitions (and their data) when no longer needed:

**Delta Lake**:
```sql
DELETE FROM my_table WHERE year = 2020
-- Then vacuum to remove files
VACUUM my_table
```

**Hudi**:
```python
.option("hoodie.datasource.write.operation", "delete")
.option("hoodie.datasource.write.partitionpath.field", "year")
```

**Iceberg**:
```sql
DELETE FROM my_table WHERE year = 2020
-- Or use partition expiration
CALL system.expire_partitions('my_table', 'year < 2021')
```

### Partition Repair

When files are added or removed outside the table format (not recommended), partitions may need repair:

**Hive-compatible repair**:
```sql
MSCK REPAIR TABLE my_table
```

This scans storage and updates the metastore with discovered partitions.

## Partition Optimization

### Small File Compaction

Partitions with many small files hurt performance. Compact files within partitions:

**Delta Lake**:
```sql
OPTIMIZE my_table WHERE year = 2024
```

**Hudi**:
```python
# Clustering operation
.option("hoodie.clustering.inline", "true")
```

**Iceberg**:
```sql
CALL system.rewrite_data_files('my_table', strategy => 'binpack')
```

### Partition-Level Statistics

Table formats maintain statistics per file and partition. These enable:
- Predicate pushdown beyond partition pruning
- Min/max filtering on non-partition columns
- Accurate query planning

```sql
-- Delta Lake: Analyze table for statistics
ANALYZE TABLE my_table COMPUTE STATISTICS FOR ALL COLUMNS
```

### Z-Ordering (Multi-Dimensional Clustering)

When queries filter on multiple non-partition columns, Z-ordering colocates related data:

**Delta Lake**:
```sql
OPTIMIZE my_table ZORDER BY (user_id, product_id)
```

**Iceberg** (sorting):
```sql
ALTER TABLE my_table WRITE ORDERED BY user_id, product_id
```

Z-ordering is not a replacement for partitioning but complements it for secondary access patterns.

## Implementation Details by Format

### Delta Lake

Delta Lake uses Hive-style explicit partitioning:

```python
df.write.format("delta") \
    .partitionBy("year", "month") \
    .save(path)
```

Configuration options:
```python
# Auto-optimize writes to reduce small files
.option("spark.databricks.delta.autoOptimize.enabled", "true")

# Auto-compact small files
.option("spark.databricks.delta.autoCompact.enabled", "true")
```

### Apache Hudi

Hudi partitioning:

```python
.option("hoodie.datasource.write.partitionpath.field", "date")
```

Hudi supports custom partition formats:
```python
# URL-encode partition values (default)
.option("hoodie.datasource.write.hive_style_partitioning", "false")

# Hive-style key=value format
.option("hoodie.datasource.write.hive_style_partitioning", "true")
```

### Apache Iceberg

Iceberg with hidden partitioning:

```sql
CREATE TABLE events (
  event_id BIGINT,
  timestamp TIMESTAMP,
  region STRING
) PARTITIONED BY (days(timestamp), region)
```

Partition evolution:
```sql
-- Add new partition field
ALTER TABLE events ADD PARTITION FIELD hours(timestamp)

-- Remove partition field (new data only)
ALTER TABLE events DROP PARTITION FIELD days(timestamp)
```

## Best Practices

### Design for Query Patterns

Analyze your most common queries before choosing partition columns:

1. Identify frequent filter conditions
2. Estimate cardinality of candidate columns
3. Model expected partition sizes
4. Validate with sample data

### Avoid Over-Partitioning

Signs of over-partitioning:
- Thousands of partitions with only a few files each
- Slow metadata operations
- File listing timeouts

Solutions:
- Use coarser time granularity (month instead of day)
- Remove low-cardinality partition levels
- Use bucketing instead of partitioning for high-cardinality columns

### Monitor Partition Health

Track partition metrics:
- Number of files per partition
- Average file size per partition
- Partition count growth rate
- Query planning time

Set alerts for:
- Partitions with too many small files
- Unexpectedly empty partitions
- Partition count exceeding thresholds

### Handle Late-Arriving Data

Data that arrives after its logical partition time creates challenges:

1. **Accept small files**: Late data creates small files in historical partitions
2. **Periodic compaction**: Schedule regular compaction for affected partitions
3. **Partition by arrival time**: Add a secondary partition for arrival date
4. **Lambda architecture**: Process late data in a separate late-data table

### Consider Future Needs

Partition schemes are difficult to change:
- Allow for granularity increases
- Consider composite schemes from the start
- Prefer Iceberg for evolving requirements
- Document partition strategy and rationale
