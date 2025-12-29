# Amazon Redshift

## Summary

Amazon Redshift is AWS's fully managed, petabyte-scale data warehouse service built on massively parallel processing (MPP) architecture. It offers both provisioned clusters with explicit node management and a serverless option that automatically scales compute resources. Redshift's deep integration with the AWS ecosystem makes it a natural choice for organizations with significant AWS investments.

Key points to remember:

- MPP architecture distributes queries across multiple compute nodes
- RA3 node types separate compute and managed storage for independent scaling
- Redshift Serverless eliminates cluster management with pay-per-use pricing
- Columnar storage with automatic compression optimizes analytical workloads
- Spectrum enables queries against S3 data lakes without loading data
- Redshift ML integrates with SageMaker for SQL-based machine learning
- AQUA (Advanced Query Accelerator) pushes computation to storage layer
- Compared to Snowflake, Redshift requires more operational management but offers tighter AWS integration
- Compared to BigQuery, Redshift provides more control over infrastructure

## Architecture

### Cluster Components

A Redshift cluster consists of a leader node and one or more compute nodes:

**Leader Node**
- Receives queries from client applications
- Parses SQL and creates execution plans
- Compiles SQL to optimized C++ code
- Distributes work to compute nodes
- Aggregates results from compute nodes
- Manages cluster metadata

**Compute Nodes**
- Execute query fragments in parallel
- Store data in columnar format
- Each node divided into slices (parallel processing units)
- Communicate with leader and each other for redistributions

### Node Types

**RA3 (Recommended)**

RA3 nodes separate compute and managed storage:

| Type | vCPU | Memory | Slices | Storage |
|------|------|--------|--------|---------|
| ra3.xlplus | 4 | 32 GiB | 2 | Up to 32 TB |
| ra3.4xlarge | 12 | 96 GiB | 4 | Up to 128 TB |
| ra3.16xlarge | 48 | 384 GiB | 16 | Up to 128 TB |

Storage features:
- Hot data cached on local SSD
- Cold data automatically tiered to S3
- Intelligent prefetching based on access patterns
- Pay only for storage used

**DC2 (Dense Compute)**

Compute-intensive nodes with local SSD:

| Type | vCPU | Memory | Storage |
|------|------|--------|---------|
| dc2.large | 2 | 15 GiB | 160 GB |
| dc2.8xlarge | 32 | 244 GiB | 2.56 TB |

Best for:
- Datasets under 1 TB compressed
- Maximum local I/O performance
- Cost-effective for smaller warehouses

### Redshift Serverless

Fully managed compute without cluster provisioning:

```sql
-- Create serverless namespace and workgroup
-- (Typically done via console or SDK)
```

Characteristics:
- Automatic scaling based on workload
- Measured in Redshift Processing Units (RPUs)
- 1 RPU = 16 GiB memory
- Base capacity: 8-1024 RPUs
- Pay only for compute used

Best for:
- Unpredictable or intermittent workloads
- Development and testing
- Organizations wanting minimal administration

### Data Distribution

Redshift distributes table data across nodes using styles:

```sql
-- Even distribution (default)
CREATE TABLE orders (
  order_id INT,
  customer_id INT
) DISTSTYLE EVEN;

-- Key distribution (colocate related rows)
CREATE TABLE order_items (
  item_id INT,
  order_id INT
) DISTSTYLE KEY DISTKEY(order_id);

-- All distribution (replicate to all nodes)
CREATE TABLE regions (
  region_id INT,
  region_name VARCHAR(50)
) DISTSTYLE ALL;
```

Distribution strategies:
- EVEN: Round-robin distribution for balanced storage
- KEY: Rows with same key value on same node (optimizes joins)
- ALL: Full table copy on every node (for small dimension tables)
- AUTO: Redshift chooses based on table size

### Sort Keys

Organize data within each slice for efficient filtering:

```sql
-- Compound sort key (ordered filtering)
CREATE TABLE events (
  event_id INT,
  event_date DATE,
  event_type VARCHAR(50)
) COMPOUND SORTKEY(event_date, event_type);

-- Interleaved sort key (multiple filter columns)
CREATE TABLE sales (
  sale_id INT,
  region VARCHAR(50),
  product VARCHAR(100)
) INTERLEAVED SORTKEY(region, product);
```

Compound sort keys:
- Optimal when filtering by prefix columns
- Efficient range scans
- Lower maintenance overhead

Interleaved sort keys:
- Equal weight to all key columns
- Better for ad-hoc filtering patterns
- Higher vacuum overhead

## Key Features

### Columnar Storage

Data stored column by column rather than row by row:
- Only relevant columns read for queries
- Better compression ratios (similar values together)
- Automatic encoding selection per column

Encoding types include LZO, Zstandard, Delta, Runlength, and more. Redshift's ANALYZE COMPRESSION recommends optimal encodings.

### Workload Management (WLM)

Control resource allocation across query types:

```sql
-- Query with queue assignment
SET query_group TO 'etl';
INSERT INTO target SELECT * FROM source;

-- Reset
RESET query_group;
```

WLM features:
- Multiple queues with priority and memory allocation
- Automatic WLM adjusts dynamically
- Concurrency scaling for burst capacity
- Short query acceleration (SQA) prioritizes quick queries

### Concurrency Scaling

Automatically add cluster capacity for concurrent queries:
- Seamless scaling during usage spikes
- First hour of daily concurrency scaling free
- Additional clusters spin up in seconds
- Queries routed transparently

### Redshift Spectrum

Query data directly in S3 without loading:

```sql
-- Create external schema
CREATE EXTERNAL SCHEMA spectrum_schema
FROM DATA CATALOG
DATABASE 'my_glue_database'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftSpectrumRole'
REGION 'us-east-1';

-- Query external table
SELECT * FROM spectrum_schema.logs
WHERE log_date > '2024-01-01';
```

Benefits:
- Query data lake without ETL
- Join S3 data with Redshift tables
- Support for Parquet, ORC, JSON, CSV
- Pushdown predicates to S3 layer
- Pay per TB scanned

### AQUA (Advanced Query Accelerator)

Hardware-accelerated cache for RA3 nodes:
- Pushes filtering and aggregation to storage layer
- Up to 10x performance improvement for scans
- Automatic, no configuration required
- Available on RA3 nodes

### Materialized Views

Precomputed query results with automatic refresh:

```sql
CREATE MATERIALIZED VIEW daily_sales AS
SELECT
  sale_date,
  region,
  SUM(amount) as total_sales
FROM sales
GROUP BY sale_date, region;

-- Refresh manually or automatically
REFRESH MATERIALIZED VIEW daily_sales;
```

Features:
- Automatic refresh on base table changes
- Query rewrite optimization
- Incremental refresh when possible

### Data Sharing

Share live data across Redshift clusters:

```sql
-- Producer: Create datashare
CREATE DATASHARE sales_share;
ALTER DATASHARE sales_share ADD SCHEMA public;
ALTER DATASHARE sales_share ADD TABLE sales;
GRANT USAGE ON DATASHARE sales_share TO NAMESPACE 'consumer-namespace-id';

-- Consumer: Create database from datashare
CREATE DATABASE shared_sales FROM DATASHARE sales_share
OF NAMESPACE 'producer-namespace-id';
```

Benefits:
- Real-time data access without copying
- Cross-account and cross-region sharing
- Governed access control
- No storage duplication

## Redshift ML

### Overview

Train machine learning models using SQL with SageMaker integration:

```sql
CREATE MODEL customer_churn
FROM (
  SELECT
    tenure,
    monthly_charges,
    total_charges,
    churned
  FROM customers
)
TARGET churned
FUNCTION predict_churn
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftML'
SETTINGS (
  S3_BUCKET 'redshift-ml-bucket',
  MAX_RUNTIME 3600
);
```

How it works:
1. Redshift exports training data to S3
2. SageMaker Autopilot trains and tunes models
3. Model compiled and deployed back to Redshift
4. SQL function created for predictions

### Model Types

```sql
-- Auto-detect (default)
CREATE MODEL my_model FROM training_data TARGET label;

-- Explicit algorithm
CREATE MODEL xgb_model
FROM training_data
TARGET label
MODEL_TYPE XGBOOST;
```

Supported algorithms:
- XGBoost (classification, regression)
- Linear Learner
- Multilayer Perceptron (MLP)
- K-Means clustering
- Auto (SageMaker Autopilot selection)

### Making Predictions

```sql
-- Use the generated function
SELECT
  customer_id,
  predict_churn(tenure, monthly_charges, total_charges) as churn_probability
FROM new_customers;
```

Predictions execute locally in Redshift without SageMaker calls.

### Bring Your Own Model

Import pre-trained SageMaker models:

```sql
CREATE MODEL imported_model
FROM 'arn:aws:sagemaker:us-east-1:123456789:model/my-trained-model'
FUNCTION imported_predict
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftML';
```

### Remote Inference

Call SageMaker endpoints for complex models:

```sql
CREATE MODEL remote_model
FUNCTION remote_predict(text VARCHAR)
RETURNS VARCHAR
SAGEMAKER 'my-endpoint'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftML';
```

Remote inference useful for:
- Deep learning models
- Models too large for local deployment
- Real-time model updates

## Performance Optimization

### Table Design

**Distribution Key Selection**
```sql
-- Choose columns used in joins and GROUP BY
CREATE TABLE orders DISTKEY(customer_id);
CREATE TABLE order_items DISTKEY(order_id);

-- Enables collocated joins
SELECT * FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id;
```

**Sort Key Selection**
```sql
-- Use columns from WHERE and ORDER BY
CREATE TABLE events
SORTKEY(event_date)
AS SELECT * FROM raw_events;
```

### Vacuum and Analyze

Unlike Snowflake and BigQuery, Redshift requires maintenance:

```sql
-- Reclaim space and re-sort data
VACUUM FULL my_table;

-- Update statistics for query planner
ANALYZE my_table;

-- Combined operation
VACUUM FULL ANALYZE my_table;
```

Automatic maintenance:
- Auto vacuum runs during low activity
- Auto analyze updates statistics
- Monitoring via STL_ALERT_EVENT_LOG

### Query Optimization

```sql
-- Check query execution plan
EXPLAIN SELECT * FROM orders WHERE order_date > '2024-01-01';

-- Review actual execution
SELECT * FROM STL_QUERY WHERE query = 12345;
SELECT * FROM SVL_QUERY_REPORT WHERE query = 12345;
```

Optimization tips:
- Use appropriate distribution keys to minimize data movement
- Apply sort keys matching common filter patterns
- Avoid SELECT * when possible
- Use COPY for bulk loading (not INSERT)
- Review SVL_QUERY_SUMMARY for bottlenecks

### Compression

```sql
-- Analyze compression recommendations
ANALYZE COMPRESSION my_table;

-- Apply recommended encodings
ALTER TABLE my_table
ALTER COLUMN my_column ENCODE ZSTD;
```

Automatic compression:
- COPY command auto-selects encodings
- CREATE TABLE AS applies compression
- Manual override when needed

## Data Loading

### COPY Command

Primary method for bulk loading:

```sql
COPY orders
FROM 's3://my-bucket/orders/'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftLoadRole'
FORMAT AS PARQUET;

-- With options
COPY events
FROM 's3://my-bucket/events/'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftLoadRole'
CSV
GZIP
IGNOREHEADER 1
DATEFORMAT 'auto'
MAXERROR 100;
```

Best practices:
- Use multiple input files (parallel loading)
- Match file count to cluster slices
- Compress files (GZIP, LZO, BZIP2)
- Use manifest files for precise control

### Streaming Ingestion

Real-time ingestion from Kinesis:

```sql
CREATE EXTERNAL SCHEMA kinesis_schema
FROM KINESIS
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftStreamRole';

CREATE MATERIALIZED VIEW streaming_events AS
SELECT
  json_parse(kinesis_data) as event_data,
  approximate_arrival_timestamp
FROM kinesis_schema.my_stream;
```

Features:
- Near real-time data availability
- No intermediate storage required
- Automatic refresh of materialized views

### Zero-ETL Integration

Replicate data from operational databases:

```sql
-- Aurora PostgreSQL zero-ETL integration
-- (Configured via AWS console/API)
-- Data automatically synced to Redshift
```

Supported sources:
- Amazon Aurora PostgreSQL
- Amazon Aurora MySQL
- Amazon DynamoDB
- Amazon RDS

## Comparison with Other Warehouses

### Redshift vs Snowflake

| Aspect | Redshift | Snowflake |
|--------|----------|-----------|
| Management | Cluster provisioning (or Serverless) | Virtual warehouse management |
| Storage/Compute | RA3 separates; DC2 coupled | Fully separated |
| Maintenance | Vacuum/Analyze needed | Zero maintenance |
| AWS Integration | Native | Via PrivateLink |
| Data Sharing | Datashares within AWS | Cross-cloud shares |
| Scaling | Resize or Serverless | Instant warehouse resize |

Choose Redshift when: Deep AWS integration, RA3 managed storage, or streaming ingestion from Kinesis.

Choose Snowflake when: Zero maintenance, multi-cloud deployment, or simpler operations.

### Redshift vs BigQuery

| Aspect | Redshift | BigQuery |
|--------|----------|----------|
| Infrastructure | Cluster-based or Serverless | Fully serverless |
| Pricing | Node/hour or RPU | Per TB or slots |
| Control | Distribution and sort keys | Limited tuning |
| Ecosystem | AWS-native | GCP-native |
| Scaling | Manual resize or auto | Automatic |

Choose Redshift when: AWS-centric architecture, Kinesis integration, or need for fine-grained tuning.

Choose BigQuery when: Truly serverless operation, GCP ecosystem, or variable query patterns.

## Security and Governance

### Network Isolation

```sql
-- VPC configuration
-- Clusters deployed within VPC subnets
-- Enhanced VPC routing for all COPY/UNLOAD
```

Options:
- Private subnets with NAT
- VPC endpoints for S3 access
- Security groups for access control

### Encryption

```sql
-- Enable encryption at cluster creation
-- Managed via AWS KMS or HSM
```

Features:
- Encryption at rest (AES-256)
- Encryption in transit (SSL)
- Key rotation support
- AWS KMS or CloudHSM

### Access Control

```sql
-- Create user and grant permissions
CREATE USER analyst PASSWORD 'SecurePass123';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;

-- Row-level security
CREATE RLS POLICY region_policy
WITH (region VARCHAR(50))
USING (region = current_setting('app.user_region'));
```

## Cost Optimization

### Right-Sizing

- Monitor CloudWatch metrics for utilization
- Use RA3 for datasets over 1 TB
- Consider Serverless for variable workloads
- Reserved instances for predictable usage (up to 75% savings)

### Storage Costs

- RA3 managed storage: $0.024/GB/month
- Automatic tiering reduces costs
- Spectrum: $5/TB scanned
- Compression reduces storage 3-4x

### Compute Costs

- Concurrency scaling: first hour free daily
- Pause/resume clusters during off-hours
- Serverless: pay only for RPU-seconds used

## When to Use Redshift

Redshift is well-suited for:
- AWS-centric data architectures
- Workloads requiring fine-grained performance tuning
- Kinesis streaming integration
- Organizations with DBA expertise for optimization
- Predictable, sustained analytical workloads

Consider alternatives when:
- Zero maintenance is required (Snowflake)
- Multi-cloud deployment needed (Snowflake)
- Truly serverless operation preferred (BigQuery)
- Variable workloads without cluster management (BigQuery, Redshift Serverless)
