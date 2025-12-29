# Apache Kafka

## Summary

Apache Kafka is a distributed event streaming platform used for building real-time data pipelines and streaming applications. For ML systems, Kafka serves as the backbone for feature computation, model serving events, and training data collection. It handles high-throughput, fault-tolerant messaging between producers and consumers while maintaining message ordering within partitions.

Key points to remember:

- Distributed, partitioned log with configurable retention
- Producers write to topics; consumers read from topics
- Messages ordered within partitions, not across partitions
- Consumer groups enable parallel processing and fault tolerance
- Kafka Connect for integrating with external systems
- Schema Registry for message schema management (Avro, Protobuf)
- Exactly-once semantics available for critical pipelines
- Typical latency: 1-10ms end-to-end

## Core Concepts

### Architecture

```
Producers         Kafka Cluster                 Consumers
+-------+        +------------------+           +--------+
| App A +------->| Topic: events    +---------->| App X  |
+-------+        |  Partition 0     |           +--------+
                 |  Partition 1     |
+-------+        |  Partition 2     |           +--------+
| App B +------->|                  +---------->| App Y  |
+-------+        +------------------+           +--------+
                        |
                 +------v------+
                 |  ZooKeeper  |  (or KRaft)
                 +-------------+
```

### Topics and Partitions

**Topic**: Logical category for messages (e.g., `user-events`, `predictions`)

**Partition**: Physical unit of parallelism within a topic

```
Topic: user-events (3 partitions)

Partition 0: [msg0, msg3, msg6, msg9,  ...]  -> Leader: Broker 1
Partition 1: [msg1, msg4, msg7, msg10, ...]  -> Leader: Broker 2
Partition 2: [msg2, msg5, msg8, msg11, ...]  -> Leader: Broker 3
```

Messages are distributed to partitions by key hash (or round-robin if no key).

### Producers

```python
from confluent_kafka import Producer

producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'acks': 'all',  # Wait for all replicas
    'retries': 3
})

# Send message with key (ensures ordering for same key)
producer.produce(
    topic='user-events',
    key='user123',
    value='{"event": "click", "timestamp": 1234567890}',
    callback=delivery_callback
)

producer.flush()  # Wait for all messages to be delivered
```

### Consumers

```python
from confluent_kafka import Consumer

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'feature-pipeline',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False  # Manual commit for exactly-once
})

consumer.subscribe(['user-events'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    if msg.error():
        handle_error(msg.error())
        continue

    process_message(msg.value())
    consumer.commit(msg)  # Commit after processing
```

### Consumer Groups

Consumer groups enable parallel processing with automatic partition assignment:

```
Consumer Group: feature-pipeline
+-------------+     Partition 0
| Consumer 1  | <-- Partition 1
+-------------+

+-------------+     Partition 2
| Consumer 2  | <-- Partition 3
+-------------+

When Consumer 2 fails:
+-------------+     Partition 0
| Consumer 1  | <-- Partition 1
|             | <-- Partition 2  (rebalanced)
|             | <-- Partition 3  (rebalanced)
+-------------+
```

## ML Use Cases

### Real-Time Feature Pipeline

```python
from confluent_kafka import Consumer, Producer
import json

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'feature-computation'
})
consumer.subscribe(['raw-events'])

producer = Producer({'bootstrap.servers': 'localhost:9092'})

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue

    event = json.loads(msg.value())

    # Compute features
    features = {
        'user_id': event['user_id'],
        'session_length': compute_session_length(event),
        'click_rate': compute_click_rate(event),
        'timestamp': event['timestamp']
    }

    # Publish to features topic
    producer.produce(
        'user-features',
        key=event['user_id'],
        value=json.dumps(features)
    )
    consumer.commit(msg)
```

### Training Data Collection

```python
# Collect prediction logs for model retraining

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'training-data-collector'
})
consumer.subscribe(['predictions', 'outcomes'])

# Buffer for batch writes
buffer = []
BATCH_SIZE = 10000

while True:
    msg = consumer.poll(1.0)
    if msg:
        buffer.append(json.loads(msg.value()))

    if len(buffer) >= BATCH_SIZE:
        # Write to data lake
        df = pd.DataFrame(buffer)
        df.to_parquet(f's3://training-data/{timestamp}.parquet')
        buffer = []
        consumer.commit()
```

### Model Serving Events

```python
# Async prediction service

from confluent_kafka import Consumer, Producer
import torch

model = torch.load('model.pt')

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'prediction-service'
})
consumer.subscribe(['prediction-requests'])

producer = Producer({'bootstrap.servers': 'localhost:9092'})

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue

    request = json.loads(msg.value())

    # Make prediction
    features = torch.tensor(request['features'])
    prediction = model(features).item()

    # Publish result
    result = {
        'request_id': request['request_id'],
        'prediction': prediction,
        'model_version': 'v1.2.3'
    }
    producer.produce(
        'prediction-results',
        key=request['request_id'],
        value=json.dumps(result)
    )
    consumer.commit(msg)
```

## Schema Registry

Schema Registry provides schema management for Kafka messages:

```python
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer

# Define schema
user_schema = """
{
    "type": "record",
    "name": "UserEvent",
    "fields": [
        {"name": "user_id", "type": "string"},
        {"name": "event_type", "type": "string"},
        {"name": "timestamp", "type": "long"},
        {"name": "properties", "type": {"type": "map", "values": "string"}}
    ]
}
"""

schema_registry = SchemaRegistryClient({'url': 'http://localhost:8081'})
serializer = AvroSerializer(schema_registry, user_schema)
deserializer = AvroDeserializer(schema_registry)

# Produce with schema
producer.produce(
    'user-events',
    value=serializer({'user_id': '123', 'event_type': 'click', ...})
)

# Consume with schema
msg = consumer.poll()
event = deserializer(msg.value())
```

Benefits:
- Schema evolution with compatibility checking
- Compact binary serialization (Avro)
- Centralized schema management
- Consumer doesn't need hardcoded schema

## Kafka Connect

Kafka Connect integrates Kafka with external systems:

```json
// Source connector: Database -> Kafka
{
    "name": "postgres-source",
    "config": {
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
        "database.hostname": "postgres",
        "database.port": "5432",
        "database.user": "kafka",
        "database.password": "secret",
        "database.dbname": "app",
        "table.include.list": "public.users,public.events"
    }
}

// Sink connector: Kafka -> S3
{
    "name": "s3-sink",
    "config": {
        "connector.class": "io.confluent.connect.s3.S3SinkConnector",
        "s3.bucket.name": "data-lake",
        "s3.region": "us-east-1",
        "topics": "user-events",
        "format.class": "io.confluent.connect.s3.format.parquet.ParquetFormat"
    }
}
```

Common connectors for ML:
- **Debezium**: CDC from databases
- **S3 Sink**: Archive to data lake
- **Elasticsearch Sink**: Search/analytics
- **JDBC Source/Sink**: Database integration

## Kafka Streams

Kafka Streams enables stream processing in Java/Scala:

```java
StreamsBuilder builder = new StreamsBuilder();

// Read from topic
KStream<String, UserEvent> events = builder.stream("user-events");

// Aggregate session metrics
KTable<String, SessionMetrics> sessions = events
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(30)))
    .aggregate(
        SessionMetrics::new,
        (key, event, metrics) -> metrics.update(event)
    );

// Output to topic
sessions.toStream().to("session-metrics");
```

For Python, use Faust (Kafka Streams-like) or Flink.

## Configuration Best Practices

### Producer Configuration

```python
producer_config = {
    'bootstrap.servers': 'kafka1:9092,kafka2:9092,kafka3:9092',

    # Reliability
    'acks': 'all',              # Wait for all replicas
    'retries': 2147483647,      # Infinite retries
    'enable.idempotence': True, # Exactly-once within partition

    # Performance
    'batch.size': 16384,        # 16KB batch
    'linger.ms': 5,             # Wait for batch
    'compression.type': 'lz4',  # Compress messages

    # Monitoring
    'statistics.interval.ms': 60000
}
```

### Consumer Configuration

```python
consumer_config = {
    'bootstrap.servers': 'kafka1:9092,kafka2:9092,kafka3:9092',
    'group.id': 'my-consumer-group',

    # Offset management
    'auto.offset.reset': 'earliest',  # Start from beginning
    'enable.auto.commit': False,      # Manual commit

    # Performance
    'fetch.min.bytes': 1,
    'fetch.max.wait.ms': 500,
    'max.poll.records': 500,

    # Session management
    'session.timeout.ms': 30000,
    'heartbeat.interval.ms': 10000
}
```

### Topic Configuration

```bash
# Create topic with configuration
kafka-topics.sh --create --topic user-events \
    --partitions 12 \
    --replication-factor 3 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete \
    --config min.insync.replicas=2
```

Key settings:
- **partitions**: Parallelism level (cannot decrease)
- **replication-factor**: Copies for fault tolerance
- **retention.ms**: How long to keep messages
- **cleanup.policy**: delete or compact
- **min.insync.replicas**: Required acks for write

## Monitoring

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Consumer lag | Messages behind head | > 10K or increasing |
| Request latency | Producer/consumer latency | > 100ms |
| Under-replicated partitions | Replicas behind leader | > 0 |
| Active controllers | Cluster leadership | != 1 |
| ISR shrinks | Replica failures | Any |

### Monitoring with Python

```python
from confluent_kafka.admin import AdminClient

admin = AdminClient({'bootstrap.servers': 'localhost:9092'})

# Get consumer group lag
def get_consumer_lag(group_id):
    groups = admin.list_consumer_groups().result()
    offsets = admin.list_consumer_group_offsets([group_id]).result()
    # Calculate lag from topic high watermarks
    return calculate_lag(offsets)
```

## Comparison with Alternatives

| Feature | Kafka | AWS Kinesis | Google Pub/Sub |
|---------|-------|-------------|----------------|
| Deployment | Self-hosted or managed | Managed | Managed |
| Ordering | Partition-level | Shard-level | None guaranteed |
| Retention | Configurable (unlimited) | 7 days max | 31 days max |
| Replay | Full replay | Limited | Limited |
| Throughput | Very high | High | High |
| Latency | 1-10ms | 200ms | 100ms |

## Best Practices

### Partitioning Strategy

```python
# Use meaningful keys for ordering and locality
producer.produce(
    'user-events',
    key=user_id,  # Same user -> same partition -> ordered
    value=event_data
)

# Calculate partitions based on expected throughput
# ~1MB/s per partition is sustainable
num_partitions = max_throughput_mb / 1
```

### Error Handling

```python
def delivery_callback(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")
        # Retry, dead-letter, or alert
    else:
        logger.debug(f"Delivered to {msg.topic()}[{msg.partition()}]")

# Consumer error handling
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue  # Normal
        else:
            logger.error(f"Consumer error: {msg.error()}")
            continue

    try:
        process(msg)
        consumer.commit(msg)
    except ProcessingError as e:
        # Send to dead-letter topic
        producer.produce('dead-letter', value=msg.value())
        consumer.commit(msg)
```

### Exactly-Once Processing

```python
# Enable idempotent producer
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 2147483647
})

# Transactional producer (for cross-topic exactly-once)
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'transactional.id': 'my-producer-1'
})
producer.init_transactions()

producer.begin_transaction()
try:
    producer.produce('topic1', value='msg1')
    producer.produce('topic2', value='msg2')
    producer.commit_transaction()
except Exception:
    producer.abort_transaction()
```

## When to Use Kafka

### Good Fit

- High-throughput event streaming
- Real-time feature pipelines
- Decoupled microservices
- Event sourcing architectures
- Log aggregation
- Long-term message retention

### Consider Alternatives

- Simple pub/sub: Consider cloud-native (Pub/Sub, SNS)
- RPC-style messaging: Consider gRPC
- Very low latency (< 1ms): Consider in-memory solutions
- Small teams, managed preference: Consider managed Kafka or alternatives
