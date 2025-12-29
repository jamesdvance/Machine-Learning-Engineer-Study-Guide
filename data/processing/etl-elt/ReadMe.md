# ETL and ELT

## Summary

ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform) are data integration patterns for moving and preparing data for analysis. The key difference is where transformation occurs: ETL transforms data before loading into the target, while ELT loads raw data first and transforms in-place using the target system's compute power. Modern cloud data warehouses have made ELT the dominant pattern, with tools like Fivetran and Airbyte handling extraction/loading and dbt handling transformations.

Key points to remember:

- ETL: Transform during transit (legacy, complex integrations)
- ELT: Transform after loading (modern, cloud-native)
- Fivetran: Managed connectors, minimal configuration, highest cost
- Airbyte: Open-source connectors, self-hosted or cloud, lower cost
- dbt: SQL-based transformations in the warehouse, version-controlled
- The modern data stack: Fivetran/Airbyte (E/L) + dbt (T)
- For ML: ELT enables feature engineering in familiar SQL

## ETL vs ELT

### Architecture Comparison

**ETL (Extract, Transform, Load)**
```
Source Systems --> ETL Tool --> Transformed Data --> Data Warehouse
                     |
                Transform logic
                (external compute)
```

**ELT (Extract, Load, Transform)**
```
Source Systems --> EL Tool --> Raw Data --> Transform --> Analytics-Ready
                               (warehouse)    (in warehouse)
```

### When to Use Each

| Pattern | Use Case |
|---------|----------|
| ETL | Complex transformations requiring external compute |
| ETL | Legacy systems without SQL transformation capability |
| ETL | Data reduction before loading (cost savings on storage) |
| ELT | Cloud data warehouses (Snowflake, BigQuery, Redshift) |
| ELT | Exploratory analysis needing access to raw data |
| ELT | Iterative transformation development |
| ELT | Modern data stack with dbt |

### Trade-offs

| Aspect | ETL | ELT |
|--------|-----|-----|
| Transform compute | External (ETL tool) | Target (warehouse) |
| Raw data access | No (transformed only) | Yes |
| Transform flexibility | Limited after loading | Highly flexible |
| Development speed | Slower | Faster (SQL-based) |
| Cost profile | ETL tool compute | Warehouse compute |
| Schema changes | Require pipeline updates | Handle in transforms |

## Modern Data Stack

The modern data stack separates extraction/loading from transformation:

```
+-------------+     +------------------+     +---------------+
|   Sources   |     |   Data Warehouse |     |   Consumers   |
|-------------|     |------------------|     |---------------|
| - Databases |     |                  |     | - BI Tools    |
| - SaaS APIs |---->| Raw Schema       |     | - ML Models   |
| - Files     |     |    |             |---->| - Reverse ETL |
| - Events    |     |    v             |     | - Analytics   |
+-------------+     | Transformed      |     +---------------+
      |             | (dbt models)     |
      |             +------------------+
      |                    ^
      v                    |
+------------------+       |
| Fivetran/Airbyte |-------+
| (Extract & Load) |
+------------------+
```

### Component Roles

| Component | Role | Examples |
|-----------|------|----------|
| Extract & Load | Move data from sources to warehouse | Fivetran, Airbyte, Stitch |
| Transform | Clean, model, aggregate data | dbt, Dataform |
| Orchestration | Schedule and monitor pipelines | Airflow, Dagster, Prefect |
| Warehouse | Store and compute | Snowflake, BigQuery, Redshift |

## Fivetran vs Airbyte

### Comparison

| Aspect | Fivetran | Airbyte |
|--------|----------|---------|
| Model | Managed SaaS | Open-source + Cloud |
| Connectors | 300+ pre-built | 300+ community |
| Pricing | Per MAR (monthly active rows) | Free (self-host) or usage-based |
| Setup | Zero-config | More configuration |
| Maintenance | Fully managed | Self-managed or cloud |
| Customization | Limited | Highly customizable |

### When to Choose Fivetran

- Enterprise with budget for managed solutions
- Minimal engineering overhead required
- Need for guaranteed SLAs
- Predominantly common SaaS sources
- Compliance/security certifications needed

### When to Choose Airbyte

- Cost-sensitive or startup environment
- Custom connector requirements
- Self-hosting preference
- Open-source alignment
- Need for source code access

### Common Connectors

Both platforms support:
- **Databases**: PostgreSQL, MySQL, MongoDB, SQL Server
- **SaaS**: Salesforce, HubSpot, Stripe, Shopify
- **Files**: S3, GCS, SFTP
- **Events**: Segment, Amplitude
- **APIs**: REST, GraphQL (custom)

## dbt (Data Build Tool)

### What dbt Does

dbt enables SQL-based transformations with software engineering practices:

```
Raw Tables (from Fivetran/Airbyte)
        |
        v
+------------------+
| dbt Models (SQL) |
| - Staging        |
| - Intermediate   |
| - Marts          |
+------------------+
        |
        v
Analytics-Ready Tables
```

### Key Concepts

**Models**: SQL SELECT statements that dbt materializes as tables/views

```sql
-- models/staging/stg_orders.sql
SELECT
    id AS order_id,
    customer_id,
    order_date,
    amount,
    status
FROM {{ source('raw', 'orders') }}
WHERE order_date >= '2020-01-01'
```

**Materializations**: How models are persisted

| Type | Description | Use Case |
|------|-------------|----------|
| view | SQL view | Light transformations |
| table | Physical table | Heavy queries |
| incremental | Append/merge | Large tables |
| ephemeral | CTE (not persisted) | Intermediate logic |

**Sources and Refs**: Dependency management

```sql
-- Reference source (raw data)
{{ source('raw', 'orders') }}

-- Reference another model
{{ ref('stg_orders') }}
```

### Project Structure

```
dbt_project/
+-- dbt_project.yml
+-- models/
|   +-- staging/           # Clean raw data
|   |   +-- stg_orders.sql
|   |   +-- stg_customers.sql
|   +-- intermediate/      # Business logic
|   |   +-- int_order_items.sql
|   +-- marts/             # Analytics-ready
|       +-- dim_customers.sql
|       +-- fct_orders.sql
+-- tests/
|   +-- schema.yml
+-- macros/
+-- seeds/
```

### Running dbt

```bash
# Run all models
dbt run

# Run specific models
dbt run --select stg_orders

# Run models and downstream dependencies
dbt run --select stg_orders+

# Test data quality
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

## ML-Specific Patterns

### Feature Engineering with dbt

```sql
-- models/features/fct_user_features.sql
{{ config(materialized='incremental') }}

WITH user_events AS (
    SELECT
        user_id,
        event_type,
        event_timestamp,
        properties
    FROM {{ ref('stg_events') }}
    {% if is_incremental() %}
    WHERE event_timestamp > (SELECT MAX(computed_at) FROM {{ this }})
    {% endif %}
),

user_aggregates AS (
    SELECT
        user_id,
        COUNT(*) AS total_events,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchases,
        COUNT(CASE WHEN event_type = 'view' THEN 1 END) AS views,
        SUM(CASE WHEN event_type = 'purchase' THEN properties:amount::FLOAT END) AS total_spend,
        MIN(event_timestamp) AS first_seen,
        MAX(event_timestamp) AS last_seen,
        DATEDIFF('day', MIN(event_timestamp), MAX(event_timestamp)) AS tenure_days
    FROM user_events
    GROUP BY user_id
)

SELECT
    user_id,
    total_events,
    purchases,
    views,
    COALESCE(total_spend, 0) AS total_spend,
    COALESCE(purchases::FLOAT / NULLIF(views, 0), 0) AS purchase_rate,
    tenure_days,
    CURRENT_TIMESTAMP() AS computed_at
FROM user_aggregates
```

### Training Data Preparation

```sql
-- models/ml/training_data_churn.sql
WITH features AS (
    SELECT * FROM {{ ref('fct_user_features') }}
),

labels AS (
    SELECT
        user_id,
        CASE WHEN last_activity < DATEADD('day', -30, CURRENT_DATE()) THEN 1 ELSE 0 END AS churned
    FROM {{ ref('stg_users') }}
)

SELECT
    f.user_id,
    f.total_events,
    f.purchases,
    f.views,
    f.total_spend,
    f.purchase_rate,
    f.tenure_days,
    l.churned AS label,
    CURRENT_DATE() AS snapshot_date
FROM features f
JOIN labels l ON f.user_id = l.user_id
WHERE f.tenure_days >= 30  -- Minimum observation period
```

### Incremental Feature Updates

```sql
{{ config(
    materialized='incremental',
    unique_key='user_id',
    incremental_strategy='merge'
) }}

SELECT
    user_id,
    -- Latest features overwrite previous
    total_events,
    last_seen,
    CURRENT_TIMESTAMP() AS updated_at
FROM {{ ref('user_aggregates') }}

{% if is_incremental() %}
WHERE last_seen > (SELECT MAX(updated_at) FROM {{ this }})
{% endif %}
```

## Orchestration

### dbt with Airflow

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    'dbt_pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1)
)

# Run dbt models
dbt_run = BashOperator(
    task_id='dbt_run',
    bash_command='cd /dbt && dbt run --profiles-dir .',
    dag=dag
)

# Test data quality
dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command='cd /dbt && dbt test --profiles-dir .',
    dag=dag
)

dbt_run >> dbt_test
```

### Fivetran + dbt Integration

```yaml
# Fivetran triggers dbt after sync completes
# Configure in Fivetran UI or API

# dbt Cloud webhook:
# POST https://cloud.getdbt.com/api/v2/accounts/{account_id}/jobs/{job_id}/run/
```

### Full Pipeline

```
Scheduled Trigger (daily)
        |
        v
+------------------+
| Fivetran/Airbyte |  (Extract & Load)
| - Sync sources   |
+------------------+
        |
        v
+------------------+
| dbt              |  (Transform)
| - Run models     |
| - Test quality   |
+------------------+
        |
        v
+------------------+
| ML Pipeline      |
| - Feature store  |
| - Model training |
+------------------+
```

## Data Quality

### dbt Tests

```yaml
# models/schema.yml
version: 2

models:
  - name: stg_orders
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'completed', 'cancelled']
```

### Custom Tests

```sql
-- tests/assert_no_orphan_orders.sql
SELECT order_id
FROM {{ ref('stg_orders') }}
WHERE customer_id NOT IN (SELECT customer_id FROM {{ ref('stg_customers') }})
```

### Freshness Checks

```yaml
# models/sources.yml
sources:
  - name: raw
    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}
    tables:
      - name: orders
        loaded_at_field: _fivetran_synced
```

## Best Practices

### Naming Conventions

```
Staging models:     stg_<source>_<table>      (stg_stripe_payments)
Intermediate:       int_<entity>_<action>     (int_orders_pivoted)
Marts:              dim_<entity>              (dim_customers)
                    fct_<event>               (fct_orders)
Features:           fct_<entity>_features     (fct_user_features)
```

### Incremental Strategy

```sql
{{ config(
    materialized='incremental',
    unique_key='event_id',
    incremental_strategy='delete+insert',  -- or 'merge'
    on_schema_change='append_new_columns'
) }}

SELECT *
FROM {{ source('raw', 'events') }}

{% if is_incremental() %}
WHERE event_timestamp > (SELECT MAX(event_timestamp) FROM {{ this }})
{% endif %}
```

### Documentation

```yaml
# models/schema.yml
models:
  - name: fct_user_features
    description: "User-level features for ML models. Updated daily."
    columns:
      - name: user_id
        description: "Unique user identifier"
      - name: purchase_rate
        description: "Purchases divided by views. Null if no views."
```

## Further Reading

For detailed information on each tool, see:

- [Fivetran](fivetran/ReadMe.md) - Managed data integration
- [Airbyte](airbyte/ReadMe.md) - Open-source data integration
- [dbt](dbt/ReadMe.md) - SQL transformation framework
