# Apache Flink

## Summary

Apache Flink is a distributed stream processing framework designed for stateful computations over unbounded and bounded data streams. Unlike batch-oriented systems that process data in chunks, Flink processes records one-at-a-time with exactly-once semantics, making it ideal for real-time ML feature computation, event-driven applications, and complex event processing.

Key points to remember:

- True stream processing: processes records as they arrive, not micro-batches
- Stateful processing with exactly-once guarantees via checkpointing
- Event time processing with watermarks for out-of-order events
- Unified batch and streaming API (DataStream API)
- Native Kafka integration for streaming ML pipelines
- Table API and SQL for declarative stream processing
- Lower latency than Spark Streaming for stateful operations
- PyFlink enables Python-based stream processing

## Core Concepts

### Architecture

```
+------------------+     +------------------+     +------------------+
|   Job Manager    |     |   Task Manager   |     |   Task Manager   |
|  (Coordinator)   |<--->|   (Worker 1)     |     |   (Worker 2)     |
|                  |     |   +----------+   |     |   +----------+   |
|  - Scheduling    |     |   | Task 1   |   |     |   | Task 3   |   |
|  - Checkpointing |     |   +----------+   |     |   +----------+   |
|  - Recovery      |     |   | Task 2   |   |     |   | Task 4   |   |
+------------------+     |   +----------+   |     |   +----------+   |
                         +------------------+     +------------------+
```

### DataStream API

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer

env = StreamExecutionEnvironment.get_execution_environment()

# Read from Kafka
kafka_consumer = FlinkKafkaConsumer(
    topics='user-events',
    deserialization_schema=JsonRowDeserializationSchema.builder()
        .type_info(Types.ROW([
            Types.STRING(),  # user_id
            Types.STRING(),  # event_type
            Types.LONG()     # timestamp
        ]))
        .build(),
    properties={'bootstrap.servers': 'localhost:9092'}
)

stream = env.add_source(kafka_consumer)

# Process stream
result = stream \
    .filter(lambda x: x['event_type'] == 'click') \
    .key_by(lambda x: x['user_id']) \
    .process(ClickAggregator())

result.add_sink(kafka_sink)
env.execute('Click Aggregation')
```

### Time and Watermarks

Flink distinguishes three time concepts:

```
Event Time:      When event actually occurred (embedded in data)
Ingestion Time:  When event entered Flink
Processing Time: When event is processed

Event time with watermarks:
  Events:     e1(t=10) e3(t=30) e2(t=20) e4(t=40)
                |         |         |         |
  Watermarks: ----W(15)-----W(25)-----W(35)-----W(45)-->
                |         |         |         |
  Windows:    [0-20)    [20-40)   [40-60)
              complete  complete  active
```

```python
from pyflink.datastream import WatermarkStrategy
from pyflink.common import Duration

# Define watermark strategy
watermark_strategy = WatermarkStrategy \
    .for_bounded_out_of_orderness(Duration.of_seconds(5)) \
    .with_timestamp_assigner(lambda event, _: event['timestamp'])

stream = env.add_source(source) \
    .assign_timestamps_and_watermarks(watermark_strategy)
```

### State Management

Flink maintains state across records:

```python
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.datastream.functions import KeyedProcessFunction

class SessionTracker(KeyedProcessFunction):
    def __init__(self):
        self.session_state = None

    def open(self, runtime_context):
        self.session_state = runtime_context.get_state(
            ValueStateDescriptor('session', Types.PICKLED_OBJECT())
        )

    def process_element(self, value, ctx):
        # Get current state
        session = self.session_state.value() or {'count': 0, 'start': None}

        # Update state
        session['count'] += 1
        if session['start'] is None:
            session['start'] = value['timestamp']

        # Save state
        self.session_state.update(session)

        # Emit aggregation
        yield {
            'user_id': ctx.get_current_key(),
            'session_events': session['count'],
            'session_duration': value['timestamp'] - session['start']
        }
```

### Checkpointing

Checkpoints enable exactly-once processing and recovery:

```python
from pyflink.datastream import CheckpointingMode

env = StreamExecutionEnvironment.get_execution_environment()

# Configure checkpointing
env.enable_checkpointing(60000)  # Every 60 seconds
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
env.get_checkpoint_config().set_min_pause_between_checkpoints(30000)
env.get_checkpoint_config().set_checkpoint_timeout(120000)

# Configure state backend
from pyflink.datastream.state_backend import RocksDBStateBackend
env.set_state_backend(RocksDBStateBackend('s3://checkpoints/'))
```

## Windowing

### Window Types

```python
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows, SessionWindowTimeGap
from pyflink.common import Time

# Tumbling window: fixed-size, non-overlapping
stream.key_by(lambda x: x['user_id']) \
    .window(TumblingEventTimeWindows.of(Time.minutes(5))) \
    .aggregate(aggregate_func)

# Sliding window: fixed-size, overlapping
stream.key_by(lambda x: x['user_id']) \
    .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(5))) \
    .aggregate(aggregate_func)

# Session window: gap-based
stream.key_by(lambda x: x['user_id']) \
    .window(SessionWindowTimeGap.of(Time.minutes(30))) \
    .aggregate(aggregate_func)
```

### Window Aggregations

```python
from pyflink.datastream.functions import AggregateFunction

class ClickRateAggregator(AggregateFunction):
    def create_accumulator(self):
        return {'clicks': 0, 'views': 0}

    def add(self, value, accumulator):
        if value['event_type'] == 'click':
            accumulator['clicks'] += 1
        elif value['event_type'] == 'view':
            accumulator['views'] += 1
        return accumulator

    def get_result(self, accumulator):
        views = accumulator['views']
        if views == 0:
            return 0.0
        return accumulator['clicks'] / views

    def merge(self, a, b):
        return {
            'clicks': a['clicks'] + b['clicks'],
            'views': a['views'] + b['views']
        }
```

## Table API and SQL

### SQL for Streaming

```python
from pyflink.table import StreamTableEnvironment

t_env = StreamTableEnvironment.create(env)

# Create table from Kafka
t_env.execute_sql("""
    CREATE TABLE user_events (
        user_id STRING,
        event_type STRING,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'user-events',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# Streaming SQL query
t_env.execute_sql("""
    SELECT
        user_id,
        TUMBLE_START(event_time, INTERVAL '1' HOUR) as window_start,
        COUNT(*) as event_count,
        COUNT(CASE WHEN event_type = 'click' THEN 1 END) as clicks
    FROM user_events
    GROUP BY
        user_id,
        TUMBLE(event_time, INTERVAL '1' HOUR)
""").print()
```

### Table API

```python
from pyflink.table.expressions import col, lit

# Read table
events = t_env.from_path("user_events")

# Transform
result = events \
    .filter(col("event_type") == "click") \
    .group_by(col("user_id")) \
    .select(
        col("user_id"),
        col("event_time").count.alias("click_count")
    )

# Write to sink
result.execute_insert("output_table")
```

## ML Feature Pipelines

### Real-Time Feature Computation

```python
from pyflink.datastream.functions import KeyedProcessFunction
from pyflink.datastream.state import MapStateDescriptor, ValueStateDescriptor

class FeatureComputer(KeyedProcessFunction):
    """Compute real-time features per user."""

    def open(self, runtime_context):
        # State for running statistics
        self.event_count = runtime_context.get_state(
            ValueStateDescriptor('count', Types.LONG())
        )
        self.last_events = runtime_context.get_list_state(
            ListStateDescriptor('recent', Types.PICKLED_OBJECT())
        )
        self.category_counts = runtime_context.get_map_state(
            MapStateDescriptor('categories', Types.STRING(), Types.LONG())
        )

    def process_element(self, event, ctx):
        user_id = ctx.get_current_key()

        # Update count
        count = (self.event_count.value() or 0) + 1
        self.event_count.update(count)

        # Update recent events (keep last 100)
        recent = list(self.last_events.get())
        recent.append(event)
        if len(recent) > 100:
            recent = recent[-100:]
        self.last_events.update(recent)

        # Update category counts
        category = event.get('category', 'unknown')
        cat_count = self.category_counts.get(category) or 0
        self.category_counts.put(category, cat_count + 1)

        # Compute features
        features = {
            'user_id': user_id,
            'total_events': count,
            'events_per_hour': self._compute_rate(recent),
            'top_category': self._get_top_category(),
            'recency_seconds': ctx.timestamp() - recent[-1]['timestamp'],
            'timestamp': ctx.timestamp()
        }

        yield features

    def _compute_rate(self, events):
        if len(events) < 2:
            return 0
        time_span = events[-1]['timestamp'] - events[0]['timestamp']
        if time_span == 0:
            return 0
        return len(events) / (time_span / 3600000)  # per hour

    def _get_top_category(self):
        max_cat, max_count = None, 0
        for cat, count in self.category_counts.items():
            if count > max_count:
                max_cat, max_count = cat, count
        return max_cat
```

### Feature Store Integration

```python
# Write features to Redis (online store)
from pyflink.datastream.connectors.redis import RedisSink

redis_sink = RedisSink.builder() \
    .set_host('localhost') \
    .set_port(6379) \
    .set_mapper(FeatureMapper()) \
    .build()

feature_stream.add_sink(redis_sink)

# Write to Kafka for batch materialization
kafka_sink = FlinkKafkaProducer(
    topic='features-changelog',
    serialization_schema=JsonRowSerializationSchema.builder()
        .with_type_info(feature_type_info)
        .build(),
    producer_config={'bootstrap.servers': 'localhost:9092'}
)

feature_stream.add_sink(kafka_sink)
```

## Comparison with Spark Streaming

| Feature | Flink | Spark Streaming |
|---------|-------|-----------------|
| Processing Model | True streaming | Micro-batch |
| Latency | Sub-second | 100ms+ (micro-batch) |
| State Management | First-class, queryable | Requires external |
| Event Time | Native watermarks | Requires configuration |
| Exactly-Once | Native checkpointing | Requires careful setup |
| SQL | Streaming SQL | Structured Streaming SQL |
| Python API | PyFlink | PySpark |
| Maturity | Growing | Very mature |

### When to Choose Flink

- True streaming latency requirements
- Complex event time processing
- Large stateful operations
- CEP (Complex Event Processing)
- Streaming SQL use cases

### When to Choose Spark

- Existing Spark infrastructure
- Batch + streaming unified
- Simple streaming requirements
- Team expertise in Spark
- Micro-batch latency acceptable

## Configuration and Tuning

### Parallelism

```python
# Job-level parallelism
env.set_parallelism(16)

# Operator-level parallelism
stream.key_by(...) \
    .process(MyProcessor()).set_parallelism(32) \
    .sink(...).set_parallelism(8)
```

### Memory Configuration

```yaml
# flink-conf.yaml
taskmanager.memory.process.size: 4096m
taskmanager.memory.managed.fraction: 0.4
taskmanager.memory.network.fraction: 0.1

# For stateful jobs
state.backend: rocksdb
state.backend.rocksdb.memory.managed: true
```

### Checkpoint Tuning

```python
config = env.get_checkpoint_config()

# Balance between recovery time and overhead
config.set_checkpoint_interval(60000)  # 60 seconds
config.set_min_pause_between_checkpoints(30000)

# For large state
config.set_checkpoint_timeout(600000)  # 10 minutes

# Incremental checkpoints for RocksDB
env.set_state_backend(
    RocksDBStateBackend('s3://checkpoints/', True)  # True = incremental
)
```

## Deployment

### Kubernetes

```yaml
# FlinkDeployment (Flink Kubernetes Operator)
apiVersion: flink.apache.org/v1beta1
kind: FlinkDeployment
metadata:
  name: feature-pipeline
spec:
  image: flink:1.17
  flinkVersion: v1_17
  flinkConfiguration:
    taskmanager.numberOfTaskSlots: "4"
    state.backend: rocksdb
    state.checkpoints.dir: s3://checkpoints/
  serviceAccount: flink
  jobManager:
    resource:
      memory: "2048m"
      cpu: 1
  taskManager:
    resource:
      memory: "4096m"
      cpu: 2
    replicas: 4
  job:
    jarURI: local:///opt/flink/jobs/feature-pipeline.jar
    parallelism: 16
    upgradeMode: savepoint
```

### Savepoints for Upgrades

```bash
# Trigger savepoint
flink savepoint <job-id> s3://savepoints/

# Cancel with savepoint
flink cancel -s s3://savepoints/ <job-id>

# Resume from savepoint
flink run -s s3://savepoints/savepoint-xxx -p 16 job.jar
```

## Monitoring

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| numRecordsInPerSecond | Input throughput | Sudden drop |
| numRecordsOutPerSecond | Output throughput | Sudden drop |
| currentInputWatermark | Progress of event time | Stalled |
| lastCheckpointDuration | Checkpoint time | > 60s |
| lastCheckpointSize | State size | Growing rapidly |
| fullRestarts | Job restarts | > 0 |

### Metrics in Python

```python
from pyflink.datastream.functions import RichMapFunction

class MeteredProcessor(RichMapFunction):
    def open(self, runtime_context):
        self.counter = runtime_context.get_metrics_group() \
            .counter('processed_records')
        self.latency = runtime_context.get_metrics_group() \
            .histogram('processing_latency', ...)

    def map(self, value):
        start = time.time()
        result = process(value)
        self.counter.inc()
        self.latency.update(int((time.time() - start) * 1000))
        return result
```

## Best Practices

### Idempotent Sinks

```python
class IdempotentSink(SinkFunction):
    def invoke(self, value, context):
        # Use deterministic ID for deduplication
        record_id = f"{value['user_id']}_{value['timestamp']}"

        # Upsert (not insert) to handle reprocessing
        self.redis.set(record_id, json.dumps(value))
```

### Graceful Failure Handling

```python
from pyflink.datastream.functions import ProcessFunction

class SafeProcessor(ProcessFunction):
    def process_element(self, value, ctx):
        try:
            result = risky_operation(value)
            yield result
        except Exception as e:
            # Emit to side output for dead-letter processing
            ctx.output(self.dead_letter_tag, {
                'original': value,
                'error': str(e)
            })
```

### State TTL

```python
from pyflink.datastream.state import StateTtlConfig

ttl_config = StateTtlConfig.builder(Time.days(7)) \
    .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
    .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired) \
    .build()

state_descriptor = ValueStateDescriptor('state', Types.PICKLED_OBJECT())
state_descriptor.enable_time_to_live(ttl_config)
```
