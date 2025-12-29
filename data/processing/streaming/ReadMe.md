# Stream Processing

## Summary

Stream processing enables real-time data transformation and analysis as events occur, rather than waiting for batch collection. For ML systems, streaming is essential for real-time feature computation, online inference, and continuous model monitoring. The three dominant technologies in this space are Apache Kafka (messaging backbone), Apache Flink (stateful stream processing), and Spark Structured Streaming (micro-batch processing with familiar Spark APIs).

Key points to remember:

- Kafka: Distributed messaging platform for high-throughput event streaming
- Flink: True streaming with stateful processing and exactly-once semantics
- Spark Streaming: Micro-batch model with unified batch-streaming API
- Choose based on latency requirements, state complexity, and existing infrastructure
- Event time processing handles out-of-order data via watermarks
- Exactly-once processing available across all platforms with proper configuration
- Real-time ML requires balancing latency, throughput, and accuracy

## Processing Models

### True Streaming vs Micro-Batch

```
True Streaming (Flink):
  Events: e1 -> e2 -> e3 -> e4 -> e5 -> ...
            |     |     |     |     |
  Process:  p1    p2    p3    p4    p5
            |     |     |     |     |
  Output:   o1    o2    o3    o4    o5

  Latency: per-event (sub-second)

Micro-Batch (Spark Streaming):
  Events: [e1, e2, e3] -> [e4, e5, e6] -> ...
               |               |
  Process:    batch1         batch2
               |               |
  Output:      o1              o2

  Latency: per-batch (100ms+)
```

### Trade-offs

| Aspect | True Streaming | Micro-Batch |
|--------|----------------|-------------|
| Latency | Sub-second | 100ms - seconds |
| Throughput | Lower per-event overhead | Higher batch efficiency |
| Complexity | Higher | Lower |
| Recovery | Checkpoint + event replay | Checkpoint + batch replay |
| API | Stream-specific | Batch-like |

## Platform Comparison

### Feature Matrix

| Feature | Kafka | Flink | Spark Streaming |
|---------|-------|-------|-----------------|
| Role | Messaging | Processing | Processing |
| Model | Log-based | True streaming | Micro-batch |
| State | None (external) | First-class | Checkpoint-based |
| SQL | KSQL/ksqlDB | Streaming SQL | Structured SQL |
| Exactly-once | Producer level | Native | Checkpoint-based |
| Latency | 1-10ms delivery | Sub-second | 100ms+ |

### Architecture Patterns

**Pattern 1: Kafka + Flink**
```
Sources --> Kafka --> Flink --> Kafka --> Sinks
                        |
                    State Store
```
Best for: Low-latency stateful processing

**Pattern 2: Kafka + Spark Streaming**
```
Sources --> Kafka --> Spark --> Delta Lake --> Downstream
                        |
                    Checkpoint
```
Best for: Unified batch-streaming pipelines

**Pattern 3: Kafka + KSQL**
```
Sources --> Kafka --> KSQL --> Kafka --> Sinks
```
Best for: Simple SQL-based transformations

## Decision Framework

### When to Choose Kafka

- High-throughput event ingestion
- Decoupling producers and consumers
- Event replay and retention needed
- Multiple consumers per event stream
- Building streaming architecture backbone

### When to Choose Flink

- Sub-second latency requirements
- Complex stateful processing
- Event time with sophisticated watermarks
- Complex event processing (CEP)
- Large state (TB+) with exactly-once

### When to Choose Spark Streaming

- Existing Spark infrastructure
- Same API for batch and streaming
- Moderate latency acceptable (100ms+)
- Integration with Delta Lake/MLlib
- Team expertise in Spark

## ML Streaming Patterns

### Real-Time Feature Computation

```
Events (Kafka)
     |
     v
+------------------+
| Feature Pipeline |  (Flink/Spark)
| - Aggregations   |
| - Joins          |
| - Transforms     |
+------------------+
     |
     +---> Online Store (Redis)
     |
     +---> Offline Store (Delta Lake)
```

**Flink Implementation:**
```python
# Stateful aggregation per user
class UserFeatures(KeyedProcessFunction):
    def process_element(self, event, ctx):
        # Update running statistics
        state = self.get_state()
        state['event_count'] += 1
        state['last_seen'] = event['timestamp']

        # Emit features
        yield {
            'user_id': ctx.get_current_key(),
            'events_24h': state['event_count'],
            'recency': time.now() - state['last_seen']
        }
```

**Spark Implementation:**
```python
# Windowed aggregation
features = events \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        col("user_id"),
        window(col("timestamp"), "1 hour")
    ) \
    .agg(
        count("*").alias("events_per_hour"),
        avg("amount").alias("avg_amount")
    )
```

### Online Model Inference

```
Requests (Kafka)
     |
     v
+------------------+
| Feature Lookup   |  (Online Store)
+------------------+
     |
     v
+------------------+
| Model Inference  |  (Flink/Spark)
+------------------+
     |
     v
Predictions (Kafka)
```

**Implementation:**
```python
# Flink batch inference
class ModelInference(KeyedProcessFunction):
    def open(self, ctx):
        self.model = load_model()
        self.feature_client = FeatureStoreClient()

    def process_element(self, request, ctx):
        # Fetch features
        features = self.feature_client.get_online_features(
            entity_id=request['user_id']
        )

        # Predict
        prediction = self.model.predict(features)

        yield {
            'request_id': request['request_id'],
            'prediction': prediction,
            'timestamp': ctx.timestamp()
        }
```

### Training Data Collection

```
Predictions + Outcomes (Kafka)
     |
     v
+------------------+
| Join & Aggregate |  (Flink/Spark)
+------------------+
     |
     v
Training Data (Delta Lake)
```

**Implementation:**
```python
# Stream-stream join for label collection
predictions = spark.readStream.format("kafka") \
    .option("subscribe", "predictions").load()

outcomes = spark.readStream.format("kafka") \
    .option("subscribe", "outcomes").load()

# Join on request_id within time window
labeled = predictions.join(
    outcomes,
    expr("""
        predictions.request_id = outcomes.request_id AND
        outcome_time BETWEEN prediction_time
            AND prediction_time + INTERVAL 24 HOURS
    """)
)

# Write to training data store
labeled.writeStream \
    .format("delta") \
    .start("/data/training")
```

## Event Time Processing

### Watermarks

Watermarks track progress in event time, handling late data:

```
Event Time:  10:00   10:05   10:10   10:15   10:20
              |       |       |       |       |
Events:      e1      e2      e3*     e4      e5
                              ^ arrives late at 10:12

Watermark (5 min):
  At 10:10: watermark = 10:05
  At 10:15: watermark = 10:10 (e3 at 10:03 is late but within threshold)
  At 10:20: watermark = 10:15 (events before 10:15 may be dropped)
```

**Flink:**
```python
WatermarkStrategy \
    .for_bounded_out_of_orderness(Duration.of_seconds(30)) \
    .with_timestamp_assigner(lambda e, _: e['timestamp'])
```

**Spark:**
```python
df.withWatermark("event_time", "30 seconds")
```

### Window Types

| Window Type | Description | Use Case |
|-------------|-------------|----------|
| Tumbling | Fixed, non-overlapping | Hourly reports |
| Sliding | Fixed, overlapping | Moving averages |
| Session | Gap-based | User sessions |

## Exactly-Once Semantics

### Kafka Producer

```python
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 2147483647
})
```

### Flink Checkpointing

```python
env.enable_checkpointing(60000)
env.get_checkpoint_config().set_checkpointing_mode(
    CheckpointingMode.EXACTLY_ONCE
)
```

### Spark Checkpointing

```python
query = df.writeStream \
    .option("checkpointLocation", "/checkpoints/query") \
    .start()
```

### End-to-End Exactly-Once

For true exactly-once from source to sink:

1. Idempotent producer (Kafka)
2. Exactly-once processing (Flink/Spark)
3. Transactional or idempotent sink

```python
# Idempotent sink pattern
def write_with_dedup(batch_df, batch_id):
    # Use natural key for upsert
    batch_df.write \
        .format("delta") \
        .option("mergeSchema", "true") \
        .mode("overwrite") \
        .option("replaceWhere", f"batch_id = {batch_id}") \
        .save("/output")
```

## Operational Considerations

### Monitoring Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Consumer lag | Messages behind head | Growing trend |
| Processing latency | End-to-end time | > SLA |
| Checkpoint duration | State save time | > interval |
| Backpressure | Processing slower than input | Any |
| Error rate | Failed records | > 0.1% |

### Scaling Strategies

**Kafka:**
- Add partitions for parallelism
- Add brokers for throughput
- Increase replication for durability

**Flink:**
- Increase parallelism
- Add task managers
- Tune checkpointing

**Spark:**
- Increase executors
- Tune micro-batch interval
- Partition input data

### Failure Recovery

| Platform | Recovery Mechanism | Data Loss |
|----------|-------------------|-----------|
| Kafka | Replication + consumer offsets | None |
| Flink | Checkpoint + barrier | None (exactly-once) |
| Spark | Checkpoint + WAL | None (checkpoint enabled) |

## Best Practices

### Schema Management

```python
# Use Schema Registry for Kafka
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

schema_registry = SchemaRegistryClient({'url': 'http://localhost:8081'})
serializer = AvroSerializer(schema_registry, schema_str)
```

### Backpressure Handling

```python
# Flink: automatic backpressure
# Spark: rate limiting
spark.readStream \
    .format("kafka") \
    .option("maxOffsetsPerTrigger", 100000) \
    .load()
```

### Dead Letter Queues

```python
# Route failed records to DLQ
def process_with_dlq(event):
    try:
        return transform(event)
    except Exception as e:
        producer.produce('dead-letter-queue', event)
        return None
```

### Graceful Shutdown

```python
# Spark: stop query gracefully
import signal

def shutdown_handler(signum, frame):
    for query in spark.streams.active:
        query.stop()

signal.signal(signal.SIGTERM, shutdown_handler)
```

## Common Pitfalls

1. **Unbounded state**: Aggregations without TTL grow forever
2. **Clock skew**: Event time misaligned across sources
3. **Small files**: High-frequency writes create many small files
4. **Checkpoint bloat**: Large state increases recovery time
5. **Join explosion**: Unbounded stream-stream joins

### Solutions

```python
# State TTL in Flink
ttl_config = StateTtlConfig.builder(Time.days(7)) \
    .set_update_type(UpdateType.OnCreateAndWrite) \
    .build()

# Compaction in Spark/Delta
df.writeStream \
    .trigger(Trigger.ProcessingTime("10 minutes")) \
    .option("optimizeWrite", "true") \
    .start()
```

## Further Reading

For detailed information on each platform, see:

- [Apache Kafka](apache-kafka/ReadMe.md) - Distributed messaging
- [Apache Flink](apache-flink/ReadMe.md) - Stateful stream processing
- [Spark Structured Streaming](spark-streaming/ReadMe.md) - Micro-batch streaming
