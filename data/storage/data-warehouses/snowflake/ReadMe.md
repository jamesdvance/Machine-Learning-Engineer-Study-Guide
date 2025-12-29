# Snowflake

## Summary

Snowflake is a cloud-native data warehouse built on a multi-cluster shared data architecture that completely separates storage, compute, and services into independently scalable layers. This separation enables elastic scaling, workload isolation, and consumption-based pricing that differs fundamentally from traditional on-premises data warehouses.

Key points to remember:

- Three-layer architecture: storage, compute (virtual warehouses), and cloud services
- Virtual warehouses are independent compute clusters that can scale up/down and suspend when idle
- Data is automatically partitioned into micro-partitions with columnar compression
- Near-zero administration: no index tuning, no vacuum operations, no capacity planning
- Native support for semi-structured data (JSON, Avro, Parquet) via VARIANT type
- Snowpark enables Python, Java, and Scala workloads directly in Snowflake
- Snowflake ML provides integrated machine learning with model registry and feature store
- Time travel and fail-safe provide data recovery without manual backups
- Compared to BigQuery, Snowflake offers more control over compute resources
- Compared to Redshift, Snowflake provides easier scaling and lower operational overhead

## Architecture

### Three-Layer Design

Snowflake's architecture separates concerns into three distinct layers that scale independently:

**Storage Layer**
- All data stored in cloud object storage (S3, Azure Blob, GCS)
- Snowflake manages organization, compression, and metadata
- Data stored in proprietary columnar format optimized for analytics
- Automatic micro-partitioning without manual intervention
- Users never interact directly with storage

**Compute Layer (Virtual Warehouses)**
- Clusters of compute resources for query execution
- Multiple warehouses can query the same data simultaneously
- Scale from X-Small to 6X-Large (doubling compute with each size)
- Auto-suspend when idle, auto-resume on query
- Complete workload isolation between warehouses

**Cloud Services Layer**
- Manages authentication, access control, metadata
- Query parsing, optimization, and compilation
- Transaction management and concurrency control
- Backed by FoundationDB for metadata storage
- Always available, no warehouse needed for metadata operations

### Virtual Warehouses

Virtual warehouses are the compute engines that execute queries:

```sql
-- Create a warehouse
CREATE WAREHOUSE analytics_wh
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 3;
```

Warehouse sizing follows t-shirt sizes:
- X-Small: 1 node
- Small: 2 nodes
- Medium: 4 nodes
- Large: 8 nodes
- X-Large: 16 nodes
- (continues doubling)

Multi-cluster warehouses automatically add clusters under load:
- Scale out for concurrent query volume
- Scale up (larger size) for complex single queries
- Scaling policy: Standard (conservative) or Economy (aggressive)

### Micro-Partitions

Snowflake automatically divides tables into micro-partitions:

- Contiguous units of 50-500 MB compressed
- Stored immutably in columnar format
- Metadata tracks min/max values per column per partition
- Enables partition pruning without manual partitioning
- No partition keys to define or maintain

Query optimization leverages micro-partition metadata for pruning without scanning data.

## Key Features

### Zero-Copy Cloning

Create instant copies of databases, schemas, or tables without duplicating data:

```sql
CREATE DATABASE dev_db CLONE prod_db;
CREATE TABLE test_data CLONE production_data;
```

Clones share underlying storage until modifications occur. Useful for:
- Development and testing environments
- Point-in-time snapshots
- Experimentation without impacting production

### Time Travel

Query historical data without explicit backups:

```sql
-- Query data as of specific time
SELECT * FROM my_table AT(TIMESTAMP => '2024-01-15 10:00:00'::TIMESTAMP);

-- Query data as of offset
SELECT * FROM my_table AT(OFFSET => -60*60); -- 1 hour ago

-- Restore dropped table
UNDROP TABLE accidentally_deleted;
```

Retention configurable from 1-90 days (Enterprise edition) affecting storage costs.

### Fail-Safe

Seven-day recovery period after Time Travel expires. Accessible only by Snowflake support for disaster recovery. Provides additional protection layer beyond Time Travel.

### Semi-Structured Data

Native VARIANT type stores JSON, Avro, ORC, Parquet, and XML:

```sql
-- Create table with variant column
CREATE TABLE events (
  event_id INTEGER,
  event_data VARIANT,
  event_time TIMESTAMP
);

-- Query nested JSON
SELECT
  event_data:user.name::STRING AS user_name,
  event_data:action::STRING AS action
FROM events;
```

No schema definition required. Query semi-structured data with familiar SQL syntax using colon notation for path access.

### Data Sharing

Share live data across Snowflake accounts without copying:

```sql
-- Create share
CREATE SHARE analytics_share;
GRANT USAGE ON DATABASE analytics_db TO SHARE analytics_share;
GRANT SELECT ON TABLE analytics_db.public.reports TO SHARE analytics_share;
```

Consumers see real-time data. No ETL, no data movement, no staleness. Foundation for Snowflake Marketplace data products.

### Streams and Tasks

Capture and process data changes:

```sql
-- Create stream to track changes
CREATE STREAM orders_stream ON TABLE orders;

-- Create task to process changes
CREATE TASK process_orders
  WAREHOUSE = etl_wh
  SCHEDULE = '1 MINUTE'
AS
  INSERT INTO orders_processed
  SELECT * FROM orders_stream;
```

Streams provide CDC (change data capture) without external tools. Tasks enable scheduled SQL execution.

## Snowflake for Machine Learning

### Snowpark

Snowpark enables Python, Java, and Scala code execution within Snowflake:

```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

session = Session.builder.configs(connection_params).create()

# DataFrame API similar to PySpark
df = session.table("sales")
result = df.filter(col("amount") > 100) \
           .group_by("region") \
           .agg({"amount": "sum"})
```

Key benefits:
- Pushdown execution: code runs in Snowflake, not locally
- No data movement required
- Familiar DataFrame API for data scientists
- User-defined functions (UDFs) in Python

### Snowpark ML

Integrated machine learning toolkit:

```python
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.ml.modeling.xgboost import XGBClassifier

# Preprocessing with distributed execution
scaler = StandardScaler(input_cols=features, output_cols=features)
scaled_df = scaler.fit(train_df).transform(train_df)

# Model training
model = XGBClassifier(input_cols=features, label_cols=["target"])
model.fit(scaled_df)
```

Features:
- Distributed preprocessing and feature engineering
- scikit-learn, XGBoost, LightGBM compatibility
- Hyperparameter optimization at scale
- Model registry for versioning and deployment

### Snowpark-Optimized Warehouses

Specialized warehouses for ML workloads:
- High memory configurations for model training
- Optimized for single-node intensive computation
- Suitable for custom Python model training
- Higher cost per credit than standard warehouses

### Model Registry

Central repository for ML models:

```python
from snowflake.ml.registry import Registry

reg = Registry(session=session)

# Log model
model_ref = reg.log_model(
    model,
    model_name="churn_predictor",
    version_name="v1",
    sample_input_data=sample_df
)

# Deploy for inference
model_ref.run(test_df, function_name="predict")
```

### Feature Store

Snowflake Feature Store manages ML features:
- Centralized feature definitions
- Point-in-time correct feature retrieval
- Feature sharing across teams
- Integration with Model Registry

### Snowflake Notebooks

Interactive development environment:
- Cell-based Python and SQL execution
- Streamlit visualization integration
- Direct Snowpark ML access
- Version control and collaboration

## Performance Optimization

### Clustering Keys

For large tables with specific query patterns:

```sql
ALTER TABLE sales CLUSTER BY (sale_date, region);
```

Clustering reorders data within micro-partitions to improve pruning. Unlike traditional indexes:
- Automatic maintenance (no rebuild required)
- Additional storage cost for metadata
- Best for columns with high cardinality used in filters

### Search Optimization

Accelerates point lookups and substring searches:

```sql
ALTER TABLE customers ADD SEARCH OPTIMIZATION ON EQUALITY(customer_id);
ALTER TABLE logs ADD SEARCH OPTIMIZATION ON SUBSTRING(message);
```

Creates additional data structures for fast lookups. Incurs storage and maintenance costs.

### Materialized Views

Precomputed query results automatically maintained:

```sql
CREATE MATERIALIZED VIEW daily_sales AS
SELECT sale_date, SUM(amount) as total
FROM sales
GROUP BY sale_date;
```

Automatically refreshed when base tables change. Useful for expensive aggregations queried frequently.

### Query Optimization Tips

1. Use appropriate warehouse size for query complexity
2. Leverage partition pruning via date/time filters
3. Avoid SELECT * when possible
4. Use clustering for frequently filtered large tables
5. Monitor query profile for optimization opportunities

## Cost Management

### Credit Consumption

Snowflake charges credits for compute:
- Virtual warehouse runtime (per-second billing, 60-second minimum)
- Cloud services (free up to 10% of daily compute)
- Serverless features (Snowpipe, tasks, etc.)

Storage charged separately:
- Compressed data storage
- Time Travel data retention
- Fail-Safe storage

### Cost Control Strategies

```sql
-- Set warehouse auto-suspend
ALTER WAREHOUSE my_wh SET AUTO_SUSPEND = 60;

-- Use resource monitors
CREATE RESOURCE MONITOR monthly_limit
  WITH CREDIT_QUOTA = 1000
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;
```

Best practices:
- Right-size warehouses for workloads
- Aggressive auto-suspend for interactive warehouses
- Resource monitors with alerts and limits
- Separate warehouses for different teams/workloads
- Review Query History for optimization opportunities

### Storage Optimization

- Shorter Time Travel retention reduces storage costs
- Transient tables skip Fail-Safe (7-day recovery)
- Temporary tables exist only for session duration
- Regular TRUNCATE instead of DELETE for full table refreshes

## Comparison with Other Warehouses

### Snowflake vs BigQuery

| Aspect | Snowflake | BigQuery |
|--------|-----------|----------|
| Pricing model | Credit-based (compute time) | On-demand (data scanned) or slots |
| Compute control | Explicit warehouse sizing | Automatic scaling |
| Idle costs | None (auto-suspend) | Slot reservations if used |
| Semi-structured | VARIANT type | STRUCT/ARRAY types |
| ML integration | Snowpark ML | BigQuery ML |
| Multi-cloud | Yes (AWS, Azure, GCP) | GCP only |

Choose Snowflake when: You need fine-grained compute control, multi-cloud deployment, or extensive data sharing capabilities.

Choose BigQuery when: You prefer fully serverless operation, GCP-native integration, or pay-per-query pricing.

### Snowflake vs Redshift

| Aspect | Snowflake | Redshift |
|--------|-----------|----------|
| Scaling | Instant, independent | Resize requires time |
| Storage/Compute | Fully separated | Coupled (Serverless separates) |
| Concurrency | Multi-cluster auto-scaling | Concurrency scaling add-on |
| Maintenance | Zero administration | Vacuum, analyze required |
| Pricing | Consumption-based | Reserved or on-demand |

Choose Snowflake when: You need elastic scaling, minimal administration, or frequent workload variation.

Choose Redshift when: You have predictable workloads, AWS-native requirements, or existing Redshift investment.

## Security and Governance

### Access Control

Role-based access control (RBAC):

```sql
CREATE ROLE data_analyst;
GRANT USAGE ON WAREHOUSE analytics_wh TO ROLE data_analyst;
GRANT USAGE ON DATABASE analytics TO ROLE data_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics.public TO ROLE data_analyst;
```

### Column and Row-Level Security

```sql
-- Column masking
CREATE MASKING POLICY email_mask AS (val STRING)
RETURNS STRING ->
  CASE WHEN CURRENT_ROLE() IN ('ADMIN') THEN val
       ELSE '***@***'
  END;

-- Row access policies
CREATE ROW ACCESS POLICY region_policy AS (region STRING)
RETURNS BOOLEAN ->
  region = CURRENT_USER_REGION();
```

### Network Security

- Private Link for private connectivity
- Network policies to restrict IP access
- Integration with identity providers (SAML, OAuth)
- Multi-factor authentication support

## When to Use Snowflake

Snowflake is well-suited for:
- Cloud-native analytics with variable workloads
- Organizations needing minimal administrative overhead
- Multi-cloud or cloud-agnostic strategies
- Data sharing and marketplace participation
- Teams wanting integrated ML capabilities via Snowpark

Consider alternatives when:
- Workloads are small and consistent (simpler solutions may suffice)
- Tight GCP integration is required (BigQuery may be better)
- Heavy existing AWS investment with stable workloads (Redshift may be more cost-effective)
- Real-time streaming is primary use case (consider streaming-first solutions)
