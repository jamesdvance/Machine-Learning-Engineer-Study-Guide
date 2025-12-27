# Apache Iceberg

## Core Concept

Iceberg sits between your compute engine (Spark, Trino, Flink, etc.) and your underlying file storage (S3, HDFS, etc.). It provides a table abstraction over collections of data files, tracking which files belong to each version of a table.

## Key Features

### Schema Evolution

Lets you add, drop, rename, or reorder columns without rewriting data. Iceberg tracks schema changes in metadata, so queries automatically use the correct schema for each data file.

### Hidden Partitioning

Separates the physical layout from the logical queries. You define partition transforms (like extracting month from a timestamp), and Iceberg handles the rest—users query the timestamp column directly without needing to know the partition structure.

### Time Travel and Versioning

Maintains snapshots of the table at different points in time. You can query historical versions, roll back bad writes, or audit changes. Each commit creates an atomic snapshot.

### ACID Transactions

Ensures that readers see consistent data even during concurrent writes. Writers create new snapshots atomically, and readers continue using their snapshot until they explicitly refresh.

### File-Level Tracking

Maintains metadata about every data file: row counts, column min/max values, null counts. This enables aggressive query pruning—the engine can skip files that definitely don't contain matching rows without opening them.

### Format Independence

The underlying data can be Parquet, ORC, or Avro. Iceberg manages the metadata layer regardless of file format.

## Practical Benefits

Iceberg largely solves the "small files problem" through compaction and the performance issues of Hive-style partitioning. It also makes schema changes and partition evolution much safer operations that don't require data rewrites or downtime.