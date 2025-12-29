# Google BigQuery

## Summary

BigQuery is Google Cloud's fully managed, serverless enterprise data warehouse that can analyze petabytes of data using SQL. Unlike traditional data warehouses, BigQuery requires no infrastructure management, automatically scales compute resources, and charges based on data scanned rather than provisioned capacity.

Key points to remember:

- Fully serverless: no clusters to manage, no capacity planning required
- Separation of storage (Colossus) and compute (Dremel) enables independent scaling
- Pricing models: on-demand (per TB scanned) or capacity-based (slots)
- Columnar storage format (Capacitor) optimized for analytical queries
- Native support for nested and repeated fields via STRUCT and ARRAY types
- BigQuery ML enables machine learning directly in SQL without data movement
- Tight integration with GCP services (Vertex AI, Dataflow, Pub/Sub)
- Compared to Snowflake, BigQuery is more serverless but offers less compute control
- Compared to Redshift, BigQuery eliminates cluster management entirely

## Architecture

### Core Infrastructure Components

BigQuery is built on five Google infrastructure technologies:

**Dremel (Query Engine)**
The execution engine that processes SQL queries. Dremel parses queries into an execution tree with:
- Root node: Receives query, breaks into tasks
- Intermediate nodes (Mixers): Aggregate partial results
- Leaf nodes (Slots): Read data and perform computation

Each query can use thousands of slots in parallel, with results flowing up the tree for aggregation.

**Colossus (Distributed Storage)**
Google's distributed file system storing all BigQuery data. Features:
- Automatic replication across multiple locations
- Handles reliability and fault tolerance
- Petabyte-scale capacity
- Geo-replication for disaster recovery

**Capacitor (Columnar Format)**
Proprietary columnar storage format optimized for analytics:
- Each column stored separately
- Aggressive compression per column
- Encoding optimized for data type
- Supports nested and repeated structures

**Jupiter (Network)**
High-bandwidth network connecting storage and compute:
- Petabit-scale bisection bandwidth
- Enables separation of storage and compute
- Data can be read at memory-like speeds despite separation

**Borg (Cluster Management)**
Google's cluster orchestration system:
- Dynamically allocates compute resources (slots)
- Scales to thousands of machines per query
- Manages multi-tenancy across customers

### Serverless Model

BigQuery's serverless architecture means:
- No clusters to provision or manage
- No idle compute costs with on-demand pricing
- Automatic scaling based on query complexity
- Resources allocated per-query, released immediately after

This differs fundamentally from Snowflake and Redshift, which require explicit compute resource management.

### Slots

Slots are BigQuery's unit of compute capacity:
- One slot approximately equals one vCPU
- On-demand queries dynamically allocate slots
- Capacity pricing reserves dedicated slots
- Slot estimates visible in query plan

## Pricing Models

### On-Demand Pricing

Pay per query based on bytes processed:
- $6.25 per TB scanned (as of 2024)
- First 1 TB per month free
- Charged only for columns referenced in query
- Partitioning and clustering reduce scanned data

Best for:
- Unpredictable or sporadic workloads
- Development and experimentation
- Organizations new to BigQuery

### Capacity Pricing (Editions)

Reserve dedicated slots with monthly or annual commitments:

| Edition | Slots | Use Case |
|---------|-------|----------|
| Standard | 100+ | General analytics |
| Enterprise | 100+ | Advanced features, multi-region |
| Enterprise Plus | 100+ | Critical workloads, highest SLA |

Benefits:
- Predictable costs for high-volume workloads
- No per-query charges
- Autoscaling within reserved capacity

### Cost Optimization

```sql
-- Check query cost before running
SELECT
  total_bytes_processed / (1024*1024*1024*1024) AS tb_processed,
  total_bytes_processed / (1024*1024*1024*1024) * 6.25 AS estimated_cost
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE job_id = 'your-job-id';
```

Strategies:
- Use partitioning to limit data scanned
- Cluster tables on frequently filtered columns
- Avoid SELECT *
- Use materialized views for repeated queries
- Preview query cost in console before execution

## Key Features

### Partitioning

Divide tables into segments for efficient querying:

```sql
-- Time-based partitioning
CREATE TABLE events
PARTITION BY DATE(event_time)
AS SELECT * FROM raw_events;

-- Integer range partitioning
CREATE TABLE transactions
PARTITION BY RANGE_BUCKET(customer_id, GENERATE_ARRAY(0, 1000000, 10000))
AS SELECT * FROM raw_transactions;
```

Benefits:
- Queries filter partitions before scanning
- Reduced bytes processed = lower costs
- Automatic partition expiration available

### Clustering

Organize data within partitions by specified columns:

```sql
CREATE TABLE sales
PARTITION BY DATE(sale_date)
CLUSTER BY region, product_category
AS SELECT * FROM raw_sales;
```

Clustering benefits:
- Further reduces data scanned within partitions
- Most effective for high-cardinality columns
- Automatically maintained during writes
- Up to 4 clustering columns

### Nested and Repeated Fields

Native support for denormalized schemas:

```sql
CREATE TABLE orders (
  order_id INT64,
  customer STRUCT<
    id INT64,
    name STRING,
    email STRING
  >,
  items ARRAY<STRUCT<
    product_id INT64,
    quantity INT64,
    price FLOAT64
  >>
);

-- Query nested data
SELECT
  order_id,
  customer.name,
  item.product_id,
  item.quantity
FROM orders, UNNEST(items) AS item
WHERE customer.id = 12345;
```

Advantages:
- Avoid expensive joins
- Preserve data locality
- Natural fit for JSON-like data

### Streaming Inserts

Real-time data ingestion:

```python
from google.cloud import bigquery

client = bigquery.Client()
table_id = "project.dataset.table"

rows = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

errors = client.insert_rows_json(table_id, rows)
```

Characteristics:
- Data available for query within seconds
- Higher cost than batch loading
- Best-effort deduplication with insert IDs

### Materialized Views

Precomputed query results automatically refreshed:

```sql
CREATE MATERIALIZED VIEW daily_metrics AS
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS event_count,
  SUM(value) AS total_value
FROM events
GROUP BY date;
```

BigQuery automatically:
- Refreshes views when base tables change
- Uses views transparently in query optimization
- Charges for storage and refresh processing

### External Tables

Query data in Cloud Storage without loading:

```sql
CREATE EXTERNAL TABLE logs
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/logs/*.parquet']
);
```

Useful for:
- Querying data lakes
- One-time analysis of external data
- Federated queries across sources

## BigQuery ML (BQML)

### Overview

Train and deploy ML models using SQL:

```sql
-- Create a logistic regression model
CREATE OR REPLACE MODEL mydataset.churn_model
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned']
) AS
SELECT
  tenure,
  monthly_charges,
  total_charges,
  churned
FROM mydataset.customers;
```

Benefits:
- No data movement required
- SQL-familiar syntax for analysts
- Automatic feature preprocessing
- Integrated with BigQuery storage and security

### Supported Model Types

| Category | Models |
|----------|--------|
| Regression | Linear, XGBoost, DNN |
| Classification | Logistic, XGBoost, DNN, Random Forest |
| Clustering | K-means |
| Time Series | ARIMA_PLUS |
| Recommendation | Matrix Factorization |
| Deep Learning | TensorFlow imports, Vertex AI models |

### Model Training

```sql
-- Train XGBoost classifier
CREATE OR REPLACE MODEL mydataset.fraud_detector
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  num_parallel_tree=10,
  max_iterations=50,
  input_label_cols=['is_fraud']
) AS
SELECT * FROM mydataset.training_data;
```

### Model Evaluation

```sql
-- Evaluate model performance
SELECT *
FROM ML.EVALUATE(MODEL mydataset.churn_model,
  (SELECT * FROM mydataset.test_data));
```

Returns metrics appropriate for model type (AUC, accuracy, RMSE, etc.).

### Predictions

```sql
-- Generate predictions
SELECT *
FROM ML.PREDICT(MODEL mydataset.churn_model,
  (SELECT * FROM mydataset.new_customers));
```

### Feature Engineering

```sql
-- Automatic feature transforms
CREATE OR REPLACE MODEL mydataset.model
TRANSFORM(
  ML.STANDARD_SCALER(numeric_col) OVER() AS scaled_numeric,
  ML.ONE_HOT_ENCODER(category_col) OVER() AS encoded_category,
  label
)
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['label'])
AS SELECT * FROM training_data;
```

### Importing External Models

Import models from TensorFlow, ONNX, or other frameworks:

```sql
CREATE OR REPLACE MODEL mydataset.imported_model
OPTIONS(
  model_type='TENSORFLOW',
  model_path='gs://bucket/saved_model/*'
);
```

### Vertex AI Integration

Connect to Vertex AI for advanced capabilities:

```sql
CREATE OR REPLACE MODEL mydataset.gemini_model
REMOTE WITH CONNECTION `project.region.connection`
OPTIONS(endpoint='gemini-pro');

-- Use for text generation
SELECT *
FROM ML.GENERATE_TEXT(
  MODEL mydataset.gemini_model,
  (SELECT prompt FROM mydataset.prompts)
);
```

## Performance Optimization

### Query Optimization

**Avoid SELECT ***
```sql
-- Bad: scans all columns
SELECT * FROM large_table;

-- Good: only scan needed columns
SELECT id, name, value FROM large_table;
```

**Use Approximate Functions**
```sql
-- Exact count (expensive)
SELECT COUNT(DISTINCT user_id) FROM events;

-- Approximate count (faster, 1% error)
SELECT APPROX_COUNT_DISTINCT(user_id) FROM events;
```

**Optimize Joins**
```sql
-- Broadcast smaller table
SELECT /*+ BROADCAST(small_table) */
  l.*, s.category
FROM large_table l
JOIN small_table s ON l.id = s.id;
```

### BI Engine

In-memory acceleration for dashboards:
- Sub-second query response
- Automatic caching of frequently accessed data
- Integration with Looker, Data Studio, Tableau
- Reserved capacity pricing

### Query Plan Analysis

```sql
-- View query execution details
SELECT *
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE job_id = 'your-job-id';
```

Check:
- Bytes scanned vs bytes billed
- Slot utilization
- Stage timing
- Shuffle bytes

## Data Management

### Loading Data

```sql
-- Load from Cloud Storage
LOAD DATA INTO mydataset.table
FROM FILES (
  format = 'PARQUET',
  uris = ['gs://bucket/data/*.parquet']
);

-- Load from query
CREATE TABLE mydataset.new_table AS
SELECT * FROM mydataset.source_table;
```

### Time Travel

Query historical data up to 7 days:

```sql
-- Query data from specific time
SELECT * FROM mydataset.table
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR);

-- Restore deleted table
CREATE TABLE mydataset.restored_table AS
SELECT * FROM mydataset.deleted_table
FOR SYSTEM_TIME AS OF '2024-01-15 10:00:00 UTC';
```

### Snapshots

Create point-in-time copies:

```sql
CREATE SNAPSHOT TABLE mydataset.table_snapshot
CLONE mydataset.source_table
FOR SYSTEM_TIME AS OF CURRENT_TIMESTAMP();
```

Snapshots share storage with source until divergence.

## Comparison with Other Warehouses

### BigQuery vs Snowflake

| Aspect | BigQuery | Snowflake |
|--------|----------|-----------|
| Management | Fully serverless | Virtual warehouse management |
| Pricing | Per TB or slots | Per-second compute |
| Compute control | Limited | Fine-grained |
| Multi-cloud | GCP only | AWS, Azure, GCP |
| Nested data | Native STRUCT/ARRAY | VARIANT type |
| ML integration | BQML (SQL) | Snowpark ML (Python) |

Choose BigQuery when: You want zero infrastructure management, GCP-native integration, or SQL-based ML.

Choose Snowflake when: You need multi-cloud deployment, fine-grained compute control, or extensive data sharing.

### BigQuery vs Redshift

| Aspect | BigQuery | Redshift |
|--------|----------|----------|
| Serverless | Fully | Serverless option available |
| Scaling | Automatic | Manual or scheduled |
| Pricing | Query-based or slots | Node-based |
| Maintenance | None | Vacuum, analyze needed |
| Concurrency | Very high | Requires scaling |

Choose BigQuery when: You want truly serverless operation or variable workload patterns.

Choose Redshift when: You have AWS-centric architecture or predictable, sustained workloads.

## Security and Governance

### Access Control

```sql
-- Grant table access
GRANT SELECT ON TABLE project.dataset.table TO 'user:analyst@company.com';

-- Column-level security
CREATE TABLE secure_table (
  id INT64,
  name STRING,
  ssn STRING OPTIONS(policy_tags='projects/proj/locations/us/taxonomies/123/policyTags/456')
);
```

### Data Masking

Use data masking rules with policy tags:
- Define sensitivity levels
- Create masking rules
- Apply to columns via policy tags
- Users see masked or unmasked based on roles

### Audit Logging

All queries logged to Cloud Audit Logs:
- Who ran what query when
- Data accessed
- Bytes processed
- Integration with SIEM tools

## When to Use BigQuery

BigQuery is well-suited for:
- Organizations wanting minimal infrastructure management
- Variable or unpredictable query workloads
- GCP-native architectures
- Teams preferring SQL-based machine learning
- Large-scale analytics with infrequent queries

Consider alternatives when:
- You need fine-grained compute control (Snowflake)
- Multi-cloud deployment is required (Snowflake)
- Predictable high-volume workloads favor reserved compute (Redshift, Snowflake)
- Real-time streaming is primary focus (consider streaming-first solutions)
