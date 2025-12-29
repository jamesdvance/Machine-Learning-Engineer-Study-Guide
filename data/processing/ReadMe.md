# Data Processing

## Summary

Data processing encompasses the transformation, aggregation, and movement of data through ML systems. Processing can be categorized by timing (batch vs streaming), by paradigm (ETL vs ELT), and by purpose (feature engineering, training data preparation, inference). Modern ML platforms typically combine multiple processing approaches: batch for historical feature computation, streaming for real-time features, and ELT pipelines for data integration.

Key points to remember:

- Batch processing: Large-scale transformations on complete datasets (Spark, Dask, Ray)
- Stream processing: Real-time transformations as data arrives (Kafka, Flink, Spark Streaming)
- ETL/ELT: Data integration patterns for moving data between systems
- Choose batch for throughput, streaming for latency
- Modern stack: ELT with dbt for transformations, streaming for real-time features
- Processing choice impacts feature freshness, infrastructure cost, and complexity

## Processing Paradigms

### Batch vs Streaming

| Aspect | Batch | Streaming |
|--------|-------|-----------|
| Latency | Minutes to hours | Sub-second to seconds |
| Data scope | Complete dataset | Individual events |
| Compute model | Job-based | Continuous |
| State | Stateless or checkpointed | First-class state |
| Cost | Compute when running | Always running |
| Use case | Historical features | Real-time features |

### When to Use Each

**Batch Processing:**
- Historical feature computation
- Training data preparation
- Model training and evaluation
- Daily/hourly aggregations
- Large-scale data transformation

**Stream Processing:**
- Real-time feature updates
- Online inference pipelines
- Event-driven architectures
- Fraud detection and alerting
- Session-based aggregations

### Hybrid Approaches

Most ML systems combine batch and streaming:

```
Batch Layer (daily/hourly)
+------------------------+
| - Historical features  |
| - Model retraining     |
| - Backfill operations  |
+------------------------+
          |
          v
+------------------------+
| Feature Store          |
| - Offline store (batch)|
| - Online store (stream)|
+------------------------+
          ^
          |
Stream Layer (real-time)
+------------------------+
| - Real-time features   |
| - Event aggregations   |
| - Low-latency serving  |
+------------------------+
```

## Batch Processing

### Framework Selection

| Framework | Best For | Scale | API |
|-----------|----------|-------|-----|
| Apache Spark | Large-scale ETL, SQL analytics | TB-PB | SQL/DataFrame |
| Dask | Pandas at scale, Python-native | GB-TB | pandas-like |
| Ray Data | ML preprocessing, GPU workloads | GB-TB | Python |

### Decision Guide

```
Data Size < 10 GB?
  Yes -> pandas (single machine)
  No -> Continue

Need SQL or existing Spark infrastructure?
  Yes -> Apache Spark
  No -> Continue

ML pipeline with GPU preprocessing?
  Yes -> Ray Data
  No -> Continue

Scaling pandas/NumPy code?
  Yes -> Dask
  No -> Spark (default large-scale)
```

### Common Patterns

**Feature Engineering:**
```python
# Spark
df = spark.read.parquet("events/")
features = df.groupBy("user_id").agg(
    count("*").alias("event_count"),
    avg("amount").alias("avg_amount")
)
features.write.parquet("features/")

# Ray Data
ds = ray.data.read_parquet("events/")
ds = ds.map_batches(compute_features)
ds.write_parquet("features/")
```

**Training Data Preparation:**
```python
# Load and join data
events = spark.read.parquet("events/")
labels = spark.read.parquet("labels/")
training = events.join(labels, "entity_id")

# Split and save
train, test = training.randomSplit([0.8, 0.2])
train.write.parquet("training/train/")
test.write.parquet("training/test/")
```

## Stream Processing

### Platform Selection

| Platform | Model | Latency | Best For |
|----------|-------|---------|----------|
| Kafka | Messaging | 1-10ms | Event backbone |
| Flink | True streaming | Sub-second | Stateful processing |
| Spark Streaming | Micro-batch | 100ms+ | Unified batch/stream |

### Decision Guide

```
Need message queue/event backbone?
  Yes -> Apache Kafka
  No -> Continue

Sub-second latency with complex state?
  Yes -> Apache Flink
  No -> Continue

Existing Spark infrastructure?
  Yes -> Spark Structured Streaming
  No -> Flink (pure streaming) or Spark (unified)
```

### Common Patterns

**Real-Time Feature Aggregation:**
```python
# Flink
events.key_by(lambda x: x['user_id']) \
    .window(TumblingEventTimeWindows.of(Time.minutes(5))) \
    .aggregate(FeatureAggregator())

# Spark Streaming
events.withWatermark("timestamp", "5 minutes") \
    .groupBy(window("timestamp", "5 minutes"), "user_id") \
    .agg(count("*"), avg("amount"))
```

**Feature Store Integration:**
```python
# Compute features and write to online/offline stores
features = compute_streaming_features(events)

# Online store (low-latency serving)
features.write_to_redis(redis_config)

# Offline store (training)
features.write_to_delta(delta_path)
```

## ETL and ELT

### Pattern Comparison

| Aspect | ETL | ELT |
|--------|-----|-----|
| Transform location | External compute | Warehouse compute |
| Raw data access | No | Yes |
| Flexibility | Lower | Higher |
| Modern stack | Legacy | Preferred |

### Modern Data Stack

```
Sources --> Fivetran/Airbyte --> Warehouse --> dbt --> Analytics
            (Extract, Load)                   (Transform)
```

### Key Components

| Tool | Role | Type |
|------|------|------|
| Fivetran | Managed connectors | Commercial |
| Airbyte | Open-source connectors | Open-source |
| dbt | SQL transformations | Open-source |

### ML Integration

```sql
-- dbt model for ML features
SELECT
    user_id,
    COUNT(*) as total_events,
    SUM(CASE WHEN event = 'purchase' THEN 1 END) as purchases,
    AVG(amount) as avg_amount,
    CURRENT_TIMESTAMP() as computed_at
FROM {{ ref('stg_events') }}
GROUP BY user_id
```

## ML-Specific Considerations

### Feature Freshness Spectrum

| Freshness | Processing | Use Case |
|-----------|------------|----------|
| Daily | Batch | User profiles, historical aggregates |
| Hourly | Batch | Session summaries, recent activity |
| Minutes | Micro-batch | Near-real-time recommendations |
| Seconds | Streaming | Fraud detection, personalization |

### Training Data Pipeline

```
Raw Events
    |
    v
+------------------+
| Batch Processing |  (historical)
| - Feature eng    |
| - Label joining  |
+------------------+
    |
    v
+------------------+
| Training Store   |  (versioned)
| - Delta Lake     |
| - Parquet        |
+------------------+
    |
    v
+------------------+
| Model Training   |
| - PyTorch/TF     |
| - Distributed    |
+------------------+
```

### Serving Pipeline

```
Request
    |
    v
+------------------+
| Feature Lookup   |  (online store)
| - Redis          |
| - DynamoDB       |
+------------------+
    |
    +---- Missing? -----> Streaming Compute
    |                            |
    v                            v
+------------------+    +------------------+
| Model Inference  |    | Backfill to      |
| - Real-time      |    | Online Store     |
+------------------+    +------------------+
    |
    v
Response
```

## Orchestration

### Pipeline Scheduling

| Tool | Type | Best For |
|------|------|----------|
| Airflow | DAG-based | Complex dependencies |
| Dagster | Software-defined | ML pipelines |
| Prefect | Workflow orchestration | Python-native |

### Example DAG

```python
# Airflow DAG for ML pipeline
with DAG('ml_pipeline', schedule='@daily') as dag:

    # Extract and load
    sync_data = FivetranOperator(task_id='sync_data')

    # Transform with dbt
    dbt_run = BashOperator(
        task_id='dbt_run',
        bash_command='dbt run'
    )

    # Compute features
    compute_features = SparkSubmitOperator(
        task_id='compute_features',
        application='feature_pipeline.py'
    )

    # Train model
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_fn
    )

    sync_data >> dbt_run >> compute_features >> train_model
```

## Cost Optimization

### Batch Processing

| Strategy | Impact |
|----------|--------|
| Right-size clusters | Match cluster to workload |
| Spot/preemptible instances | 60-90% cost savings |
| Auto-scaling | Scale with demand |
| Efficient formats | Parquet reduces I/O |

### Stream Processing

| Strategy | Impact |
|----------|--------|
| Right-size parallelism | Match partitions to load |
| Checkpointing tuning | Balance recovery vs overhead |
| State TTL | Limit state growth |
| Windowing | Bound computation scope |

### ELT

| Strategy | Impact |
|----------|--------|
| Incremental models | Process only new data |
| Materialization choice | Views for light transforms |
| Scheduled runs | Run during off-peak |
| Warehouse credits | Match compute to need |

## Best Practices

### Data Quality

```python
# Validate before processing
def validate_batch(df):
    assert df.filter(col("id").isNull()).count() == 0, "Null IDs found"
    assert df.count() > 0, "Empty batch"
    return df

# Validate in streaming
def validate_record(record):
    if record.get('amount') < 0:
        send_to_dlq(record)
        return None
    return record
```

### Idempotency

```python
# Batch: use overwrite or merge
df.write.mode("overwrite").parquet(output_path)

# Streaming: use deterministic IDs
record_id = f"{event['user_id']}_{event['timestamp']}"
upsert(record_id, record)
```

### Monitoring

| Metric | Batch | Streaming |
|--------|-------|-----------|
| Throughput | Rows/second | Events/second |
| Latency | Job duration | End-to-end latency |
| Errors | Failed jobs | Error rate |
| Resources | CPU/memory usage | Consumer lag |

## Further Reading

For detailed information on each processing area, see:

- [Batch Processing](batch/ReadMe.md) - Spark, Dask, Ray Data
- [Stream Processing](streaming/ReadMe.md) - Kafka, Flink, Spark Streaming
- [ETL/ELT](etl-elt/ReadMe.md) - Fivetran, Airbyte, dbt
