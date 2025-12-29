# Data Warehouses

## Summary

Cloud data warehouses provide managed, scalable infrastructure for analytical workloads that traditional OLTP databases handle poorly. Unlike transactional databases optimized for single-row operations, data warehouses use columnar storage and massively parallel processing to efficiently scan and aggregate large datasets.

The three dominant cloud data warehouses are Snowflake, Google BigQuery, and Amazon Redshift. Each separates storage from compute to varying degrees, offers SQL-based interfaces, and provides integrated machine learning capabilities. The choice between them often comes down to existing cloud investments, operational preferences, and specific feature requirements.

Key points to remember:

- Columnar storage enables efficient analytical queries by reading only relevant columns
- Separation of storage and compute allows independent scaling of each resource
- Snowflake: Multi-cloud, minimal administration, virtual warehouse model
- BigQuery: Fully serverless, GCP-native, pay-per-query or reserved slots
- Redshift: AWS-native, fine-grained tuning, RA3 managed storage or serverless
- All three support SQL-based ML: Snowpark ML, BigQuery ML, Redshift ML
- Semi-structured data (JSON) is natively supported across all platforms

## Core Concepts

### Columnar vs Row Storage

Traditional row-oriented databases store each row together:
```
Row 1: [id=1, name="Alice", amount=100, date="2024-01-15"]
Row 2: [id=2, name="Bob", amount=200, date="2024-01-16"]
```

Columnar databases store each column together:
```
id:     [1, 2, 3, ...]
name:   ["Alice", "Bob", "Carol", ...]
amount: [100, 200, 150, ...]
date:   ["2024-01-15", "2024-01-16", ...]
```

Analytical benefits:
- Queries reading few columns scan less data
- Similar values compress better together
- Vectorized processing improves CPU efficiency
- Aggregations operate on contiguous memory

### Massively Parallel Processing (MPP)

Data warehouses distribute data and queries across multiple nodes:

1. Leader/coordinator receives query
2. Query compiled and optimized
3. Work distributed to compute nodes
4. Each node processes its data partition
5. Results aggregated and returned

Parallelism enables processing petabytes in seconds to minutes rather than hours.

### Separation of Storage and Compute

Modern cloud warehouses decouple these layers:

**Storage Layer**
- Data persisted in cloud object storage (S3, GCS, Azure Blob)
- Pay for data volume regardless of compute
- High durability and availability
- Shared across compute resources

**Compute Layer**
- Processing resources that execute queries
- Scale independently of data volume
- Pay only when running queries (varies by platform)
- Multiple compute pools can access same data

This separation enables:
- Scaling compute without moving data
- Multiple workloads sharing data without duplication
- Cost optimization by right-sizing each layer

## Platform Comparison

### Architecture Approaches

| Feature | Snowflake | BigQuery | Redshift |
|---------|-----------|----------|----------|
| Compute Model | Virtual Warehouses | Serverless slots | Nodes or Serverless |
| Storage | Managed (cloud storage) | Managed (Colossus) | RA3 managed or DC2 local |
| Scaling | Instant resize | Automatic | Resize or elastic |
| Multi-cloud | AWS, Azure, GCP | GCP only | AWS only |
| Admin Overhead | Minimal | Zero | Moderate (unless Serverless) |

### Pricing Models

**Snowflake**
- Credits consumed per-second of warehouse runtime
- Storage charged per TB/month
- Suspend warehouses when idle for cost control
- Predictable costs with explicit warehouse sizing

**BigQuery**
- On-demand: $6.25 per TB scanned
- Capacity: Reserved slots (1 slot approximately 1 vCPU)
- Free first 1 TB/month on-demand
- Storage: Active ($0.02/GB) vs Long-term ($0.01/GB)

**Redshift**
- RA3: Per node-hour plus managed storage
- DC2: Per node-hour (storage included)
- Serverless: Per RPU-second
- Reserved instances for 1-3 year commitments

### When to Choose Each

**Choose Snowflake when:**
- Multi-cloud deployment is required or planned
- Operational simplicity is a priority
- Data sharing across organizations is important
- Team prefers explicit compute control without cluster management

**Choose BigQuery when:**
- GCP is your primary cloud
- Variable workloads make pay-per-query attractive
- Zero infrastructure management is desired
- SQL-based ML with BQML fits your workflow

**Choose Redshift when:**
- AWS is deeply embedded in your architecture
- Fine-grained performance tuning is needed
- Kinesis streaming integration is required
- Reserved capacity provides cost advantages

## Machine Learning Capabilities

All three platforms enable ML directly within the data warehouse:

### Snowflake (Snowpark ML)

```python
from snowflake.ml.modeling.xgboost import XGBClassifier

model = XGBClassifier(input_cols=features, label_cols=["target"])
model.fit(train_df)
predictions = model.predict(test_df)
```

Characteristics:
- Python-based API with scikit-learn compatibility
- Distributed execution via Snowpark
- Model registry for versioning
- Snowpark-optimized warehouses for training

### BigQuery (BQML)

```sql
CREATE MODEL mydataset.churn_model
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['churned'])
AS SELECT * FROM training_data;

SELECT * FROM ML.PREDICT(MODEL mydataset.churn_model,
  (SELECT * FROM new_data));
```

Characteristics:
- Pure SQL interface
- Automatic preprocessing
- Vertex AI integration for advanced models
- Import TensorFlow/ONNX models

### Redshift (Redshift ML)

```sql
CREATE MODEL churn_predictor
FROM training_data
TARGET churned
FUNCTION predict_churn
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftML';

SELECT predict_churn(tenure, charges) FROM customers;
```

Characteristics:
- SageMaker Autopilot for model training
- Models compiled for local inference
- Bring-your-own-model support
- Remote inference to SageMaker endpoints

### ML Comparison

| Capability | Snowflake | BigQuery | Redshift |
|------------|-----------|----------|----------|
| Interface | Python API | SQL | SQL |
| Training Location | Snowflake compute | BigQuery | SageMaker |
| Inference | Local | Local | Local or remote |
| AutoML | Limited | Yes | Yes (Autopilot) |
| Deep Learning | Via Snowpark | Import models | SageMaker |
| Feature Store | Snowflake FS | Vertex FS | SageMaker FS |

## Semi-Structured Data

All platforms handle JSON natively, with different approaches:

### Snowflake (VARIANT)

```sql
CREATE TABLE events (data VARIANT);

SELECT
  data:user.name::STRING AS user_name,
  data:event.type::STRING AS event_type
FROM events;
```

### BigQuery (STRUCT/ARRAY)

```sql
CREATE TABLE events (
  user STRUCT<id INT64, name STRING>,
  items ARRAY<STRUCT<product STRING, quantity INT64>>
);

SELECT user.name, item.product
FROM events, UNNEST(items) AS item;
```

### Redshift (SUPER)

```sql
CREATE TABLE events (data SUPER);

SELECT
  data.user.name AS user_name,
  data.event.type AS event_type
FROM events;
```

Snowflake's VARIANT is most flexible for schema-on-read. BigQuery's STRUCT/ARRAY requires schema definition but provides stronger typing. Redshift's SUPER type offers similar flexibility to VARIANT.

## Time Travel and Data Recovery

All platforms support querying historical data:

| Platform | Default Retention | Maximum |
|----------|-------------------|---------|
| Snowflake | 1 day (Standard), 90 days (Enterprise) | 90 days |
| BigQuery | 7 days | 7 days |
| Redshift | Based on snapshot schedule | Via snapshots |

```sql
-- Snowflake
SELECT * FROM my_table AT(TIMESTAMP => '2024-01-15 10:00:00');

-- BigQuery
SELECT * FROM my_table FOR SYSTEM_TIME AS OF '2024-01-15 10:00:00';

-- Redshift
-- Uses snapshot restore rather than point-in-time queries
```

## Data Sharing

Sharing live data without copying:

| Platform | Mechanism | Cross-Cloud | Cross-Account |
|----------|-----------|-------------|---------------|
| Snowflake | Secure Data Sharing | Via replication | Yes |
| BigQuery | Analytics Hub | No | Yes |
| Redshift | Datashares | No | Yes (within AWS) |

Snowflake's data sharing is most mature, enabling both internal sharing and marketplace data products. BigQuery Analytics Hub provides similar capabilities within GCP. Redshift datashares work across AWS accounts but not across clouds.

## Streaming Data

Real-time data ingestion approaches:

**Snowflake**
- Snowpipe for continuous loading from S3/Azure/GCS
- Streams for CDC on tables
- Kafka connector available

**BigQuery**
- Streaming inserts API
- Pub/Sub integration
- Dataflow for complex streaming ETL

**Redshift**
- Streaming ingestion from Kinesis
- Zero-ETL for Aurora/DynamoDB
- Materialized views over streams

For high-velocity streaming, Redshift's native Kinesis integration provides the lowest latency. BigQuery and Snowflake typically introduce slightly more delay through their ingestion mechanisms.

## Performance Optimization

### Partitioning/Clustering

| Platform | Partitioning | Clustering |
|----------|--------------|------------|
| Snowflake | Automatic micro-partitions | Manual clustering keys |
| BigQuery | Time or integer range | Up to 4 columns |
| Redshift | Manual (distkey/sortkey) | Automatic or manual |

Snowflake requires the least manual tuning with automatic micro-partitioning. BigQuery's partitioning directly impacts query costs. Redshift offers the most control but requires more expertise.

### Caching

All platforms cache query results:
- Snowflake: Result cache (24 hours) + data cache
- BigQuery: Cached results (free, 24 hours)
- Redshift: Result cache + AQUA accelerator

### Materialized Views

```sql
-- All three support materialized views
CREATE MATERIALIZED VIEW daily_metrics AS
SELECT date, SUM(amount) FROM transactions GROUP BY date;
```

Automatic refresh varies:
- Snowflake: Manual or scheduled refresh
- BigQuery: Automatic on base table change
- Redshift: Automatic or manual refresh

## Governance and Security

Common security features:
- Role-based access control (RBAC)
- Column-level security
- Row-level security (policies)
- Encryption at rest and in transit
- Audit logging

Platform-specific strengths:
- Snowflake: Dynamic data masking, secure views for sharing
- BigQuery: Integration with GCP IAM and VPC Service Controls
- Redshift: Deep AWS IAM integration, VPC isolation

## Migration Considerations

### Data Migration

Moving data between platforms:
- Export to cloud storage (Parquet recommended)
- Use platform-native loading (COPY, bq load)
- Consider third-party tools (Fivetran, Airbyte) for ongoing sync

### Schema Translation

Most SQL is portable, but watch for:
- Data type differences (especially timestamps, numerics)
- Function name variations
- Semi-structured data syntax
- Platform-specific extensions

### Workload Assessment

Before migrating:
1. Profile existing query patterns
2. Identify performance-critical queries
3. Test representative workloads on target platform
4. Compare costs with realistic usage patterns

## Cost Management

### Common Strategies

1. Right-size compute resources
2. Use auto-suspend/auto-pause features
3. Optimize queries to reduce scanned data
4. Monitor and alert on spending
5. Consider reserved capacity for predictable workloads

### Platform-Specific Tips

**Snowflake:**
- Aggressive auto-suspend (60-300 seconds)
- Resource monitors with credit limits
- Separate warehouses by workload priority

**BigQuery:**
- Partition tables to reduce scanned data
- Use LIMIT for exploration queries
- Consider flat-rate for predictable usage

**Redshift:**
- Reserved instances for stable workloads
- Concurrency scaling limits
- Pause clusters during off-hours

## Further Reading

For detailed information on each platform, see:
- [Snowflake](snowflake/ReadMe.md)
- [BigQuery](bigquery/ReadMe.md)
- [Redshift](redshift/ReadMe.md)
