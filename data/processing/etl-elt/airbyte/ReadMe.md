# Airbyte

## Summary

Airbyte is an open-source data integration platform for ELT (Extract, Load, Transform) pipelines. It provides 600+ pre-built connectors to move data from sources (APIs, databases, files) to destinations (data warehouses, lakes, databases). Unlike Fivetran's fully managed approach, Airbyte offers deployment flexibility: self-host for free or use their managed cloud. Its open-source nature enables custom connector development and full control over your data pipeline infrastructure.

Key points to remember:

- Open-source ELT platform with 600+ connectors
- Deployment options: self-hosted (free), cloud (paid), enterprise
- No-Code Connector Builder for custom sources without programming
- Sync modes: full refresh, incremental (append, deduped)
- Change Data Capture (CDC) support for database sources
- Schema management handles source schema changes automatically
- PyAirbyte enables Python-native data extraction for notebooks and scripts
- Compared to Fivetran: lower cost, more control, more setup
- Compared to custom scripts: maintained connectors, schema handling, monitoring

## Architecture

### Core Components

```
Sources                  Airbyte Platform              Destinations
--------                 ----------------              ------------
APIs                     +----------------+            Warehouses
 - Stripe       ------>  |                |  ------>   - Snowflake
 - Salesforce            |   Scheduler    |            - BigQuery
 - HubSpot               |   (Temporal)   |            - Redshift
                         |                |
Databases               |   Workers      |            Lakes
 - PostgreSQL   ------>  |   (K8s pods)   |  ------>   - S3/Delta
 - MySQL                 |                |            - Databricks
 - MongoDB               |   Connector    |
                         |   Containers   |            Databases
Files                    |                |            - PostgreSQL
 - S3           ------>  |   Config DB    |  ------>   - MySQL
 - GCS                   |   (Postgres)   |
 - SFTP                  +----------------+
```

### How Syncs Work

1. **Discovery**: Connector reads source schema (tables, columns, types)
2. **Configuration**: User selects streams and sync mode
3. **Extraction**: Source connector pulls data as JSON records
4. **Normalization**: Optional flattening of nested JSON (deprecated in favor of destination transforms)
5. **Loading**: Destination connector writes to target

### Sync Modes

| Source Mode | Destination Mode | Behavior |
|-------------|------------------|----------|
| Full Refresh | Overwrite | Replace all data each sync |
| Full Refresh | Append | Add full dataset as new records |
| Incremental | Append | Add only new records |
| Incremental | Append + Deduped | Upsert based on primary key |

**Incremental sync** requires a cursor field (timestamp or incrementing ID) to track progress:

```
First sync:  Extract all records where updated_at <= 2024-01-15 10:00
Second sync: Extract records where updated_at > 2024-01-15 10:00
```

### Change Data Capture (CDC)

For databases, CDC captures inserts, updates, and deletes by reading the transaction log:

```
PostgreSQL: Logical replication (pgoutput, wal2json)
MySQL: Binary log (binlog)
MongoDB: Change streams
SQL Server: Change tracking or CDC
```

CDC advantages:
- Captures deletes (cursor-based incremental cannot)
- Lower impact on source database
- More accurate change tracking

CDC requirements:
- Database configuration changes
- Replication slot management
- More complex setup

## Deployment Options

### Self-Hosted (Airbyte OSS)

Free deployment on your infrastructure:

```bash
# Docker Compose (development/small scale)
git clone https://github.com/airbytehq/airbyte.git
cd airbyte
./run-ab-platform.sh
```

```yaml
# Kubernetes (production)
# Using Helm chart
helm repo add airbyte https://airbytehq.github.io/helm-charts
helm install airbyte airbyte/airbyte \
  --namespace airbyte \
  --create-namespace
```

Self-hosted considerations:
- Full control over data (never leaves your infrastructure)
- No per-connector or per-row fees
- You manage upgrades, scaling, monitoring
- Community support only (Enterprise for SLA)

### Airbyte Cloud

Managed service with usage-based pricing:

- No infrastructure management
- Automatic updates and scaling
- SSO, audit logs (higher tiers)
- Connector support SLAs
- Credits-based pricing

### Self-Managed Enterprise

Enterprise features on your infrastructure:
- High availability
- RBAC and SSO
- Dedicated support
- Data never leaves your environment

## Configuration

### Connections

A connection links a source to a destination with sync settings:

```yaml
# Example connection configuration
source:
  name: "Production PostgreSQL"
  connector: "source-postgres"
  config:
    host: "db.example.com"
    port: 5432
    database: "production"
    username: "${POSTGRES_USER}"
    password: "${POSTGRES_PASSWORD}"
    replication_method:
      method: "CDC"
      replication_slot: "airbyte_slot"
      publication: "airbyte_publication"

destination:
  name: "Analytics Warehouse"
  connector: "destination-snowflake"
  config:
    host: "account.snowflakecomputing.com"
    warehouse: "COMPUTE_WH"
    database: "RAW"
    schema: "POSTGRES"
    username: "${SNOWFLAKE_USER}"
    password: "${SNOWFLAKE_PASSWORD}"

sync:
  schedule:
    type: "cron"
    expression: "0 */6 * * *"  # Every 6 hours
  streams:
    - name: "orders"
      sync_mode: "incremental"
      destination_sync_mode: "append_dedup"
      cursor_field: "updated_at"
      primary_key: ["id"]
    - name: "products"
      sync_mode: "full_refresh"
      destination_sync_mode: "overwrite"
```

### Stream Configuration

For each stream (table), configure:

| Setting | Options | Notes |
|---------|---------|-------|
| Sync mode | full_refresh, incremental | Incremental needs cursor |
| Destination mode | overwrite, append, append_dedup | Dedup needs primary key |
| Cursor field | Column name | For incremental (e.g., updated_at) |
| Primary key | Column(s) | For deduplication |
| Selected fields | Column list | Optional field selection |

### Schema Handling

Airbyte tracks source schema and handles changes:

| Change Type | Default Behavior | Options |
|-------------|------------------|---------|
| New column | Auto-add to destination | Propagate or ignore |
| Removed column | Keep in destination | Remove or keep |
| Type change | Fail sync | Allow or fail |
| New stream | Ignore | Auto-add or ignore |

Configure in connection settings:

```yaml
schema_change_handling:
  non_breaking_changes: "propagate"  # auto-apply
  breaking_changes: "disable"         # pause sync for review
```

## Connector Development

### No-Code Connector Builder

Build connectors visually for REST APIs:

1. Enter API documentation URL (AI parses specs)
2. Configure authentication (API key, OAuth, etc.)
3. Define streams (endpoints) and pagination
4. Map response to schema
5. Test and publish

Supported auth methods:
- API Key (header, query param)
- Basic Auth
- OAuth 2.0 (various flows)
- Session tokens

### Low-Code CDK

YAML-based connector definition:

```yaml
# source-example-api/manifest.yaml
version: "0.79.0"

definitions:
  requester:
    type: HttpRequester
    url_base: "https://api.example.com/v1"
    http_method: GET
    authenticator:
      type: ApiKeyAuthenticator
      api_token: "{{ config['api_key'] }}"
      inject_into:
        type: RequestOption
        field_name: "X-API-Key"
        inject_into: header

  paginator:
    type: DefaultPaginator
    pagination_strategy:
      type: CursorPagination
      cursor_value: "{{ response.next_cursor }}"
    page_token_option:
      type: RequestOption
      field_name: "cursor"
      inject_into: request_parameter

streams:
  - name: "users"
    primary_key: "id"
    schema_loader:
      type: InlineSchemaLoader
      schema:
        type: object
        properties:
          id:
            type: string
          email:
            type: string
          created_at:
            type: string
    retriever:
      type: SimpleRetriever
      requester:
        $ref: "#/definitions/requester"
        path: "/users"
      paginator:
        $ref: "#/definitions/paginator"
```

### Python CDK

Full programmatic control:

```python
# source_example/source.py
from airbyte_cdk.sources import AbstractSource
from airbyte_cdk.sources.streams import Stream
from airbyte_cdk.sources.streams.http import HttpStream

class UsersStream(HttpStream):
    url_base = "https://api.example.com/v1/"
    primary_key = "id"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def path(self, **kwargs) -> str:
        return "users"

    def request_headers(self, **kwargs) -> dict:
        return {"X-API-Key": self.api_key}

    def parse_response(self, response, **kwargs):
        yield from response.json()["users"]

    def next_page_token(self, response) -> dict | None:
        data = response.json()
        if data.get("next_cursor"):
            return {"cursor": data["next_cursor"]}
        return None

class SourceExample(AbstractSource):
    def check_connection(self, logger, config) -> tuple[bool, str | None]:
        try:
            # Validate credentials
            return True, None
        except Exception as e:
            return False, str(e)

    def streams(self, config: dict) -> list[Stream]:
        return [
            UsersStream(api_key=config["api_key"]),
        ]
```

## PyAirbyte

Python library for using Airbyte connectors directly in code:

```python
import airbyte as ab

# Configure source
source = ab.get_source(
    "source-postgres",
    config={
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "username": "user",
        "password": "pass",
        "schemas": ["public"],
    },
)

# Check connection
source.check()

# Select streams
source.select_streams(["users", "orders"])

# Read into cache (DuckDB by default)
cache = ab.get_default_cache()
result = source.read(cache=cache)

# Access as pandas DataFrames
users_df = result["users"].to_pandas()
orders_df = result["orders"].to_pandas()

# Or as SQL
cache.execute("SELECT * FROM users WHERE created_at > '2024-01-01'")
```

PyAirbyte use cases:
- Jupyter notebooks for exploration
- Python scripts for ad-hoc data pulls
- CI/CD pipelines for testing
- Prototyping before full deployment

## Orchestration Integration

### Airflow

```python
from airflow import DAG
from airflow.providers.airbyte.operators.airbyte import AirbyteTriggerSyncOperator
from airflow.providers.airbyte.sensors.airbyte import AirbyteJobSensor
from datetime import datetime

dag = DAG(
    'airbyte_sync',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
)

trigger_sync = AirbyteTriggerSyncOperator(
    task_id='trigger_airbyte_sync',
    airbyte_conn_id='airbyte_default',
    connection_id='your-connection-uuid',
    asynchronous=True,
    dag=dag,
)

wait_for_sync = AirbyteJobSensor(
    task_id='wait_for_sync',
    airbyte_conn_id='airbyte_default',
    airbyte_job_id="{{ task_instance.xcom_pull(task_ids='trigger_airbyte_sync') }}",
    dag=dag,
)

trigger_sync >> wait_for_sync
```

### Dagster

```python
from dagster_airbyte import AirbyteResource, load_assets_from_airbyte_instance

airbyte_instance = AirbyteResource(
    host="localhost",
    port="8000",
    username="airbyte",
    password="password",
)

# Auto-generate assets from Airbyte connections
airbyte_assets = load_assets_from_airbyte_instance(
    airbyte_instance,
    key_prefix=["raw"],
)
```

### Terraform

```hcl
# Configure Airbyte provider
provider "airbyte" {
  host = "http://localhost:8000"
}

# Define source
resource "airbyte_source_postgres" "production" {
  name = "Production Database"
  configuration = {
    host     = "db.example.com"
    port     = 5432
    database = "production"
    username = var.postgres_user
    password = var.postgres_password
  }
}

# Define destination
resource "airbyte_destination_snowflake" "warehouse" {
  name = "Analytics Warehouse"
  configuration = {
    host      = "account.snowflakecomputing.com"
    warehouse = "COMPUTE_WH"
    database  = "RAW"
    schema    = "POSTGRES"
    username  = var.snowflake_user
    password  = var.snowflake_password
  }
}

# Define connection
resource "airbyte_connection" "postgres_to_snowflake" {
  name           = "Production to Warehouse"
  source_id      = airbyte_source_postgres.production.source_id
  destination_id = airbyte_destination_snowflake.warehouse.destination_id

  schedule = {
    schedule_type = "cron"
    cron_expression = "0 */6 * * *"
  }
}
```

## Common Patterns

### Multi-Tenant Data Isolation

Separate data by customer using prefixes or schemas:

```yaml
# Per-tenant destination schema
destination:
  connector: "destination-snowflake"
  config:
    schema: "tenant_${TENANT_ID}"
```

Or using raw tables with tenant column:

```sql
-- dbt model to filter by tenant
SELECT *
FROM {{ source('airbyte', 'orders') }}
WHERE _airbyte_meta.tenant_id = '{{ var("tenant_id") }}'
```

### Handling API Rate Limits

Configure in connector settings:

```yaml
source:
  connector: "source-hubspot"
  config:
    # ... credentials
    request_options:
      requests_per_second: 10
      retry_attempts: 5
      backoff_strategy: "exponential"
```

### Incremental Sync with Backfill

For large historical loads:

```yaml
# Initial backfill: full refresh
streams:
  - name: "events"
    sync_mode: "full_refresh"
    destination_sync_mode: "overwrite"

# After backfill: switch to incremental
streams:
  - name: "events"
    sync_mode: "incremental"
    destination_sync_mode: "append_dedup"
    cursor_field: "event_time"
```

## Comparison with Fivetran

| Aspect | Airbyte | Fivetran |
|--------|---------|----------|
| Deployment | Self-hosted or cloud | Cloud only |
| Pricing | Free (OSS) or usage-based | Per MAR (expensive) |
| Connectors | 600+ (mixed quality) | 300+ (high quality) |
| Custom connectors | Builder, CDK, AI | Limited |
| Setup effort | More configuration | Turnkey |
| Support | Community or paid | Enterprise SLA |
| Compliance | Self-managed | SOC 2, HIPAA |
| Schema handling | Good | Excellent |

### When to Choose Airbyte

- Cost-sensitive environments
- Need for custom connectors
- Self-hosting requirement (data sovereignty)
- Open-source preference
- Willing to manage infrastructure

### When to Choose Fivetran

- Enterprise with budget for managed service
- Need guaranteed SLAs
- Minimal engineering overhead
- Compliance certifications required
- Predominantly standard sources

## Monitoring and Troubleshooting

### Sync Status

Monitor via UI, API, or webhooks:

```python
# API check
import requests

response = requests.get(
    "http://localhost:8000/api/v1/jobs",
    params={"connection_id": "uuid"},
    headers={"Authorization": "Bearer token"}
)

for job in response.json()["jobs"]:
    print(f"{job['id']}: {job['status']} - {job['bytes_synced']} bytes")
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Sync stuck | Resource exhaustion | Increase worker memory |
| Missing records | Cursor misconfigured | Verify cursor field updates |
| Schema mismatch | Breaking change | Review and approve changes |
| Auth failures | Token expired | Refresh OAuth tokens |
| Rate limited | Too aggressive | Reduce sync frequency |

### Logs

```bash
# Docker logs
docker logs airbyte-worker

# Kubernetes
kubectl logs -n airbyte deployment/airbyte-worker
```

## Best Practices

### Source Configuration

- Use CDC for databases when possible (captures deletes)
- Set appropriate sync frequency (not always real-time)
- Select only needed columns to reduce load
- Use dedicated read replicas for production databases

### Destination Configuration

- Use staging schema for raw data
- Enable schema change notifications
- Set appropriate warehouse size for syncs
- Use clustering/partitioning on timestamp columns

### Operations

- Monitor sync durations and volumes
- Set up alerts for failed syncs
- Regularly review and prune unused connections
- Version control Terraform/API configurations
- Test connector updates in staging first

## When to Use Airbyte

Airbyte is well-suited for:
- ELT pipelines from APIs and databases to warehouses
- Teams wanting open-source data integration
- Custom connector development needs
- Cost-conscious organizations
- Self-hosting requirements

Consider alternatives when:
- Need fully managed with zero ops (Fivetran)
- Real-time streaming required (Kafka, Flink)
- Complex transformations during transit (Spark)
- Simple one-off data moves (scripts)
