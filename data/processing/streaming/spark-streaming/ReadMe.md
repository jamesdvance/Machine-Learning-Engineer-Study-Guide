# Spark Structured Streaming

## Summary

Spark Structured Streaming is a stream processing engine built on the Spark SQL engine. It uses a micro-batch processing model where incoming data is treated as a continuously appended table, enabling the same DataFrame/SQL API for both batch and streaming workloads. For ML pipelines, Structured Streaming provides familiar APIs, excellent integration with MLlib, and unified batch-streaming feature engineering.

Key points to remember:

- Micro-batch processing model (not true streaming)
- Same DataFrame API for batch and streaming
- Exactly-once processing with checkpoint-based recovery
- Native integration with Kafka, Delta Lake, and cloud storage
- Watermarks for handling late data
- Continuous processing mode for lower latency (experimental)
- Best for teams with existing Spark infrastructure

## Core Concepts

### Streaming as Tables

```
                    Input Stream
                         |
                         v
               +------------------+
               | Unbounded Table  |
               |   (DataStream)   |
               +------------------+
                         |
            DataFrame Operations (same API!)
                         |
                         v
               +------------------+
               |  Output Table    |
               | (Streaming Sink) |
               +------------------+
```

### Basic Streaming Query

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("StructuredStreaming") \
    .getOrCreate()

# Read streaming data from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "user-events") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON payload
parsed = df.select(
    col("key").cast("string").alias("user_id"),
    from_json(col("value").cast("string"), schema).alias("data")
).select("user_id", "data.*")

# Process
result = parsed \
    .filter(col("event_type") == "click") \
    .groupBy("user_id") \
    .count()

# Write to console (for debugging)
query = result.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

### Output Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Append | Only new rows to sink | Log processing, non-aggregation |
| Complete | Entire result table | Aggregations with small result |
| Update | Only changed rows | Aggregations with keyed updates |

```python
# Append mode (default for non-aggregations)
query = df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .start("output/")

# Complete mode (for aggregations)
query = df.groupBy("category").count() \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Update mode
query = df.groupBy("user_id").agg(sum("amount")) \
    .writeStream \
    .outputMode("update") \
    .format("console") \
    .start()
```

## Windowed Aggregations

### Tumbling Windows

```python
from pyspark.sql.functions import window

# 10-minute tumbling windows
windowed = df \
    .withWatermark("timestamp", "5 minutes") \
    .groupBy(
        col("user_id"),
        window(col("timestamp"), "10 minutes")
    ) \
    .agg(
        count("*").alias("event_count"),
        sum("amount").alias("total_amount")
    )
```

### Sliding Windows

```python
# 10-minute windows, sliding every 5 minutes
windowed = df \
    .withWatermark("timestamp", "5 minutes") \
    .groupBy(
        col("user_id"),
        window(col("timestamp"), "10 minutes", "5 minutes")
    ) \
    .count()
```

### Session Windows

```python
from pyspark.sql.functions import session_window

# Session windows with 30-minute gap
sessioned = df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        col("user_id"),
        session_window(col("timestamp"), "30 minutes")
    ) \
    .agg(
        count("*").alias("session_events"),
        min("timestamp").alias("session_start"),
        max("timestamp").alias("session_end")
    )
```

## Watermarks and Late Data

### Watermark Configuration

```python
# Allow data up to 10 minutes late
df_with_watermark = df \
    .withWatermark("event_time", "10 minutes")

# Aggregation with watermark
result = df_with_watermark \
    .groupBy(
        window(col("event_time"), "5 minutes"),
        col("user_id")
    ) \
    .count()
```

### Understanding Watermarks

```
Event Time: |---10:00---|---10:05---|---10:10---|---10:15---|
                |           |           |           |
Events:      e1(10:02)   e2(10:07)   e3(10:03)   e4(10:12)
                                        ^ late!
Watermark (10 min threshold):
  At 10:05: watermark = 09:55 (accept all)
  At 10:10: watermark = 10:00 (e3 at 10:03 accepted)
  At 10:15: watermark = 10:05 (events before 10:05 may be dropped)
```

## Kafka Integration

### Reading from Kafka

```python
# Subscribe to topics
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker1:9092,broker2:9092") \
    .option("subscribe", "topic1,topic2") \
    .option("startingOffsets", "earliest") \
    .load()

# Or use pattern subscription
df = spark.readStream \
    .format("kafka") \
    .option("subscribePattern", "events-.*") \
    .load()

# Kafka message schema
# key: binary
# value: binary
# topic: string
# partition: int
# offset: long
# timestamp: timestamp
# timestampType: int
```

### Writing to Kafka

```python
# Prepare output (key and value as strings/binary)
output = df.select(
    col("user_id").alias("key"),
    to_json(struct("*")).alias("value")
)

# Write to Kafka
query = output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output-topic") \
    .option("checkpointLocation", "/checkpoints/kafka-sink") \
    .start()
```

## Delta Lake Integration

### Streaming to Delta

```python
# Write stream to Delta Lake
query = df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoints/delta") \
    .start("/delta/events")

# With partitioning
query = df.writeStream \
    .format("delta") \
    .partitionBy("date", "region") \
    .option("checkpointLocation", "/checkpoints/delta") \
    .start("/delta/events")
```

### Reading Delta as Stream

```python
# Stream from Delta Lake (detects new files)
df = spark.readStream \
    .format("delta") \
    .option("maxFilesPerTrigger", 1000) \
    .load("/delta/events")

# Read changes (CDC)
df = spark.readStream \
    .format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 0) \
    .load("/delta/events")
```

## ML Feature Pipelines

### Real-Time Feature Engineering

```python
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler

# Read events stream
events = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "user-events") \
    .load()

# Parse and add features
features = events \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("hour_of_day", hour("timestamp")) \
    .withColumn("day_of_week", dayofweek("timestamp")) \
    .withColumn("is_weekend", dayofweek("timestamp").isin([1, 7]).cast("int"))

# Windowed aggregations
user_features = features \
    .withWatermark("timestamp", "1 hour") \
    .groupBy(
        col("user_id"),
        window(col("timestamp"), "1 hour")
    ) \
    .agg(
        count("*").alias("events_per_hour"),
        sum(when(col("event_type") == "click", 1).otherwise(0)).alias("clicks"),
        sum(when(col("event_type") == "view", 1).otherwise(0)).alias("views"),
        avg("amount").alias("avg_amount")
    ) \
    .withColumn("click_rate", col("clicks") / col("views"))

# Write to feature store
query = user_features.writeStream \
    .outputMode("update") \
    .foreachBatch(write_to_feature_store) \
    .start()
```

### Batch Model Inference in Stream

```python
from pyspark.ml import PipelineModel

# Load pre-trained model
model = PipelineModel.load("/models/fraud-detector")

# Apply model to stream
def predict_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Apply model
    predictions = model.transform(batch_df)

    # Write predictions
    predictions.select("user_id", "prediction", "probability") \
        .write \
        .format("delta") \
        .mode("append") \
        .save("/delta/predictions")

query = features.writeStream \
    .foreachBatch(predict_batch) \
    .option("checkpointLocation", "/checkpoints/inference") \
    .start()
```

### Stream-Static Joins

```python
# Static dimension table
user_profiles = spark.read.parquet("/data/user_profiles")

# Join streaming events with static profiles
enriched = events.join(
    user_profiles,
    events.user_id == user_profiles.user_id,
    "left"
)

# Write enriched stream
query = enriched.writeStream \
    .format("delta") \
    .start("/delta/enriched-events")
```

### Stream-Stream Joins

```python
# Two event streams
clicks = spark.readStream.format("kafka") \
    .option("subscribe", "clicks").load()

impressions = spark.readStream.format("kafka") \
    .option("subscribe", "impressions").load()

# Add watermarks for both streams
clicks_wm = clicks.withWatermark("click_time", "10 minutes")
impressions_wm = impressions.withWatermark("impression_time", "10 minutes")

# Join on user_id within time window
joined = clicks_wm.join(
    impressions_wm,
    expr("""
        clicks.user_id = impressions.user_id AND
        click_time BETWEEN impression_time AND impression_time + INTERVAL 1 HOUR
    """),
    "inner"
)
```

## Checkpointing and Recovery

### Checkpoint Configuration

```python
query = df.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "s3://bucket/checkpoints/my-query") \
    .option("path", "s3://bucket/output/") \
    .start()
```

### Checkpoint Contents

```
checkpoints/
  +-- commits/        # Completed micro-batches
  +-- offsets/        # Source offsets per batch
  +-- sources/        # Source-specific metadata
  +-- state/          # Aggregation state
  +-- metadata        # Query metadata
```

### Recovery Behavior

```python
# Query automatically resumes from checkpoint
# If checkpoint exists, continues from last committed batch
# If checkpoint doesn't exist, starts fresh based on startingOffsets

# Change checkpoint to reset (lose state)
query = df.writeStream \
    .option("checkpointLocation", "new-checkpoint-path") \
    .start()
```

## Performance Tuning

### Trigger Intervals

```python
from pyspark.sql.streaming import Trigger

# Default: process as fast as possible
query = df.writeStream.start()

# Fixed interval micro-batch
query = df.writeStream \
    .trigger(Trigger.ProcessingTime("10 seconds")) \
    .start()

# One-time micro-batch (for backfill)
query = df.writeStream \
    .trigger(Trigger.Once()) \
    .start()

# Available triggers (low latency, experimental)
query = df.writeStream \
    .trigger(Trigger.AvailableNow()) \
    .start()
```

### Parallelism Configuration

```python
# Kafka partition parallelism
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic") \
    .option("minPartitions", 100)  # Override Kafka partitions \
    .load()

# Shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", 200)
```

### State Store Configuration

```python
# State store cleanup
spark.conf.set("spark.sql.streaming.stateStore.providerClass",
    "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider")

# State timeout (for mapGroupsWithState)
spark.conf.set("spark.sql.streaming.stateStore.maintenanceInterval", "10min")
```

## Comparison with Flink

| Feature | Spark Streaming | Flink |
|---------|-----------------|-------|
| Model | Micro-batch | True streaming |
| Latency | 100ms+ | Sub-second |
| API | DataFrame/SQL | DataStream/SQL |
| State | External or checkpoint | First-class |
| Recovery | Checkpoint-based | Checkpoint-based |
| Ecosystem | Larger | Growing |
| Unified Batch/Stream | Excellent | Good |

### When to Choose Spark

- Existing Spark infrastructure
- Same API for batch and streaming
- Team expertise in Spark
- Moderate latency requirements
- Need MLlib integration

### When to Choose Flink

- True streaming latency
- Complex event processing
- Large stateful operations
- Event time processing critical

## Monitoring

### Query Progress

```python
# Get query progress
query = df.writeStream.start()

# Check status
print(query.status)
print(query.lastProgress)

# Recent progress
for p in query.recentProgress:
    print(f"Batch {p['batchId']}: {p['numInputRows']} rows, "
          f"{p['processedRowsPerSecond']} rows/sec")
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| inputRowsPerSecond | Input rate | Sudden drop |
| processedRowsPerSecond | Processing rate | < input rate |
| numInputRows | Rows per batch | Varies |
| batchDuration | Batch processing time | > trigger interval |
| stateOperators | State size | Growing unbounded |

### Streaming Query Listener

```python
from pyspark.sql.streaming import StreamingQueryListener

class MyListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        print(f"Query started: {event.id}")

    def onQueryProgress(self, event):
        print(f"Progress: {event.progress.numInputRows} rows")

    def onQueryTerminated(self, event):
        print(f"Query terminated: {event.id}")

spark.streams.addListener(MyListener())
```

## Best Practices

### Idempotent Writes

```python
def write_to_database(batch_df, batch_id):
    # Use batch_id for idempotency
    batch_df \
        .withColumn("batch_id", lit(batch_id)) \
        .write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost/db") \
        .option("dbtable", "events") \
        .mode("append") \
        .save()

query = df.writeStream \
    .foreachBatch(write_to_database) \
    .start()
```

### Handling Schema Evolution

```python
# For Kafka JSON, handle schema evolution
schema = spark.read.json("sample_events.json").schema

# Or use schema registry
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .load() \
    .select(
        from_avro(col("value"), schema_registry_config).alias("data")
    )
```

### Graceful Shutdown

```python
import signal

query = df.writeStream.start()

def shutdown_handler(signum, frame):
    print("Stopping query...")
    query.stop()

signal.signal(signal.SIGTERM, shutdown_handler)
query.awaitTermination()
```
