# Fivetran

## Summary

Fivetran is a fully managed ELT (Extract, Load, Transform) platform that automates data pipeline creation and maintenance. Unlike self-hosted alternatives, Fivetran handles all infrastructure, connector updates, and schema management. With 700+ pre-built connectors and zero-maintenance operation, it targets enterprises that prioritize reliability and minimal engineering overhead over cost optimization.

Key points to remember:

- Fully managed cloud-only ELT platform
- 700+ connectors including databases, SaaS apps, ERPs, files
- Pricing based on Monthly Active Rows (MAR) per connector
- Automatic schema change handling and propagation
- Sync frequencies from 1 minute to 24 hours depending on plan
- No infrastructure to manage (true SaaS)
- High reliability with automated retries and idempotent delivery
- SOC 2 Type II, HIPAA, and GDPR compliant
- Compared to Airbyte: higher cost, less configuration, more reliable
- Compared to custom pipelines: maintained connectors, no code, enterprise support

## Architecture

### How Fivetran Works

```
Data Sources              Fivetran Cloud              Destinations
------------              --------------              ------------
SaaS APIs                 +-----------------+         Warehouses
 - Salesforce   ------->  |                 |  ---->  - Snowflake
 - HubSpot                |  Connector      |         - BigQuery
 - Stripe                 |  Fleet          |         - Redshift
                          |  (managed)      |         - Databricks
Databases                 |                 |
 - PostgreSQL   ------->  |  Schema         |         Lakes
 - MySQL                  |  Tracking       |  ---->  - S3
 - Oracle                 |                 |         - Azure Data Lake
                          |  Sync           |
Files                     |  Orchestration  |         Databases
 - S3           ------->  |                 |  ---->  - PostgreSQL
 - SFTP                   |  Delivery       |         - MySQL
 - Azure Blob             |  Guarantees     |
                          +-----------------+
                                 |
                          Transformations
                          (dbt Cloud, SQL)
```

### Sync Process

1. **Initial Sync**: Full historical load of source data
2. **Schema Detection**: Automatic discovery of tables, columns, types
3. **Incremental Updates**: Capture changes using cursor or CDC
4. **Transformation**: Optional in-warehouse transforms
5. **Delivery**: Write to destination with retries and deduplication

### Sync Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Incremental | Track changes via cursor or CDC | Most tables |
| Full Resync | Reload entire table | Small reference tables |
| History | Preserve change history | Audit trails, SCD Type 2 |

Fivetran automatically selects the optimal mode based on source capabilities.

## Connector Types

### Standard Connectors

Fully maintained by Fivetran with guaranteed SLAs:

**SaaS Applications**
- CRM: Salesforce, HubSpot, Pipedrive
- Marketing: Google Ads, Facebook Ads, LinkedIn Ads
- Finance: Stripe, QuickBooks, NetSuite
- Support: Zendesk, Intercom, Freshdesk
- HR: Workday, BambooHR, Greenhouse

**Databases**
- Relational: PostgreSQL, MySQL, SQL Server, Oracle
- Cloud: Amazon RDS, Azure SQL, Cloud SQL
- NoSQL: MongoDB, DynamoDB, Cosmos DB
- Warehouses: Snowflake, BigQuery, Redshift (as source)

**Files and Events**
- Cloud Storage: S3, GCS, Azure Blob
- SFTP servers
- Event streams: Kafka, Kinesis, Webhooks

**ERPs**
- SAP: SAP ECC, S/4HANA, Business ByDesign
- Oracle: E-Business Suite, Fusion Applications
- Microsoft: Dynamics 365, Business Central

### Lite Connectors

AI-generated connectors for niche SaaS apps:
- Built using Fivetran's Connector SDK
- Available through "By Request" program (30-day delivery)
- Lower priority support than Standard
- Ideal for long-tail applications

### Partner-Built Connectors

Developed by technology partners:
- Maintained by partner organizations
- Certified by Fivetran
- May have different support SLAs

### Custom Connectors (Connector SDK)

Build your own when needed:

```python
# Fivetran Connector SDK example
from fivetran_connector_sdk import Connector, Operations

def schema(configuration: dict):
    return [
        {
            "table": "users",
            "primary_key": ["id"],
            "columns": {
                "id": "STRING",
                "email": "STRING",
                "created_at": "UTC_DATETIME",
            },
        }
    ]

def update(configuration: dict, state: dict):
    # Fetch data from your API
    users = fetch_users(configuration["api_key"], state.get("cursor"))

    for user in users:
        yield Operations.upsert(
            table="users",
            data={
                "id": user["id"],
                "email": user["email"],
                "created_at": user["created_at"],
            },
        )

    # Update cursor for next sync
    if users:
        yield Operations.checkpoint({"cursor": users[-1]["id"]})

connector = Connector(schema=schema, update=update)
```

## Configuration

### Connection Setup

Configuration is primarily through the web UI or API:

```python
# Fivetran API example
import requests

# Create connector
response = requests.post(
    "https://api.fivetran.com/v1/connectors",
    auth=(api_key, api_secret),
    json={
        "service": "postgres",
        "group_id": "destination_group_id",
        "config": {
            "host": "db.example.com",
            "port": 5432,
            "database": "production",
            "user": "fivetran_user",
            "password": "secure_password",
            "connection_type": "Directly",
        },
        "trust_certificates": True,
        "run_setup_tests": True,
    },
)

connector_id = response.json()["data"]["id"]
```

### Schema Configuration

Select which tables and columns to sync:

```python
# Update schema configuration
requests.patch(
    f"https://api.fivetran.com/v1/connectors/{connector_id}/schemas",
    auth=(api_key, api_secret),
    json={
        "schemas": {
            "public": {
                "enabled": True,
                "tables": {
                    "users": {
                        "enabled": True,
                        "columns": {
                            "id": {"enabled": True},
                            "email": {"enabled": True},
                            "password_hash": {"enabled": False},  # Exclude sensitive
                        },
                    },
                    "audit_logs": {"enabled": False},  # Skip table
                },
            }
        }
    },
)
```

### Sync Scheduling

```python
# Configure sync frequency
requests.patch(
    f"https://api.fivetran.com/v1/connectors/{connector_id}",
    auth=(api_key, api_secret),
    json={
        "sync_frequency": 60,  # Minutes: 5, 15, 30, 60, 120, 180, 360, 480, 720, 1440
        "paused": False,
    },
)
```

Sync frequency options by plan:

| Plan | Minimum Frequency |
|------|-------------------|
| Free | 6 hours |
| Starter | 1 hour |
| Standard | 15 minutes |
| Enterprise | 1 minute |

## Schema Management

### Automatic Schema Handling

Fivetran tracks source schemas and handles changes:

| Change Type | Default Behavior | Options |
|-------------|------------------|---------|
| New column | Add to destination | Block, allow |
| Dropped column | Keep in destination | Mark deprecated |
| Type change | Widen or notify | Block, allow |
| New table | Enable automatically | Require approval |

### Schema Change Notifications

```python
# Configure notifications via API
requests.patch(
    f"https://api.fivetran.com/v1/connectors/{connector_id}",
    auth=(api_key, api_secret),
    json={
        "schema_change_handling": "BLOCK_ALL",  # ALLOW_ALL, ALLOW_COLUMNS
    },
)
```

### Column Hashing

For sensitive data, hash columns before loading:

```python
# Enable column hashing
requests.patch(
    f"https://api.fivetran.com/v1/connectors/{connector_id}/schemas",
    auth=(api_key, api_secret),
    json={
        "schemas": {
            "public": {
                "tables": {
                    "users": {
                        "columns": {
                            "email": {
                                "enabled": True,
                                "hashed": True,  # SHA-256 hash
                            }
                        }
                    }
                }
            }
        }
    },
)
```

## Transformations

### Quickstart Data Models

Pre-built dbt packages for common sources:

```yaml
# dbt_project.yml
packages:
  - package: fivetran/salesforce
    version: "0.7.0"
  - package: fivetran/hubspot
    version: "0.8.0"
```

These provide:
- Staging models with consistent naming
- Business logic transformations
- Common metrics and dimensions

### In-Warehouse Transformations

Fivetran's transformation layer (separate product):

```sql
-- Triggered after sync completion
CREATE OR REPLACE TABLE analytics.dim_customers AS
SELECT
    id AS customer_id,
    CONCAT(first_name, ' ', last_name) AS full_name,
    email,
    created_at AS customer_since,
    DATEDIFF('day', created_at, CURRENT_DATE) AS tenure_days
FROM raw.salesforce.contact
WHERE is_deleted = FALSE;
```

### dbt Cloud Integration

Native integration for post-sync dbt runs:

1. Connect Fivetran to dbt Cloud
2. Configure job triggers
3. Fivetran sync completion triggers dbt job

## Pricing

### Monthly Active Rows (MAR)

Billing based on rows synced per connector:

```
MAR = Rows inserted + Rows updated + Rows deleted (per month per connector)
```

Example calculation:
- Orders table: 10,000 new rows/month + 5,000 updates = 15,000 MAR
- Products table: 100 new rows + 500 updates = 600 MAR
- Connector MAR: 15,600

### Pricing Tiers (as of 2024)

| Plan | Base Price | Per Million MAR | Min Sync |
|------|------------|-----------------|----------|
| Free | $0 | Included (500K limit) | 6 hours |
| Starter | ~$500/mo | ~$500 | 1 hour |
| Standard | ~$1,000/mo | ~$400-500 | 15 minutes |
| Enterprise | Custom | Volume discounts | 1 minute |

### Cost Considerations

Factors affecting cost:
- Number of connectors (each billed separately)
- Data change velocity (updates count as MAR)
- Historical load size (initial sync)
- Sync frequency (more syncs = more MAR captured)

Cost optimization strategies:
- Select only needed tables/columns
- Use appropriate sync frequency (not always fastest)
- Exclude high-churn, low-value tables
- Consider Airbyte for cost-sensitive workloads

### Pricing Changes (2025)

Fivetran moved from bulk MAR to per-connector MAR:
- Each connector has individual MAR accounting
- Minimum MAR thresholds per connector
- Can increase costs for many small connectors

## API and Automation

### REST API

```python
import requests
from requests.auth import HTTPBasicAuth

# List all connectors
response = requests.get(
    "https://api.fivetran.com/v1/connectors",
    auth=HTTPBasicAuth(api_key, api_secret),
    params={"group_id": "destination_group_id"},
)

connectors = response.json()["data"]["items"]

# Trigger manual sync
requests.post(
    f"https://api.fivetran.com/v1/connectors/{connector_id}/force",
    auth=HTTPBasicAuth(api_key, api_secret),
)

# Check sync status
response = requests.get(
    f"https://api.fivetran.com/v1/connectors/{connector_id}",
    auth=HTTPBasicAuth(api_key, api_secret),
)
status = response.json()["data"]["status"]["sync_state"]
```

### Terraform Provider

```hcl
terraform {
  required_providers {
    fivetran = {
      source  = "fivetran/fivetran"
      version = "~> 1.0"
    }
  }
}

provider "fivetran" {
  api_key    = var.fivetran_api_key
  api_secret = var.fivetran_api_secret
}

resource "fivetran_group" "analytics" {
  name = "Analytics Warehouse"
}

resource "fivetran_destination" "snowflake" {
  group_id = fivetran_group.analytics.id

  service = "snowflake"
  config {
    host     = "account.snowflakecomputing.com"
    database = "RAW"
    auth     = "PASSWORD"
    user     = var.snowflake_user
    password = var.snowflake_password
  }
}

resource "fivetran_connector" "salesforce" {
  group_id = fivetran_group.analytics.id

  service = "salesforce"
  config {
    client_id     = var.salesforce_client_id
    client_secret = var.salesforce_client_secret
  }

  sync_frequency = 60
  paused         = false
}
```

### Webhooks

Receive notifications for sync events:

```python
# Flask webhook handler
from flask import Flask, request

app = Flask(__name__)

@app.route("/fivetran-webhook", methods=["POST"])
def handle_webhook():
    event = request.json

    if event["event_type"] == "sync_end":
        connector_id = event["connector_id"]
        status = event["data"]["status"]
        rows_synced = event["data"]["rows_synced"]

        if status == "SUCCESSFUL":
            trigger_downstream_processing(connector_id)
        else:
            alert_on_failure(connector_id, event["data"]["error"])

    return "OK", 200
```

## Orchestration Integration

### Airflow

```python
from airflow import DAG
from airflow.providers.fivetran.operators.fivetran import FivetranOperator
from airflow.providers.fivetran.sensors.fivetran import FivetranSensor
from datetime import datetime

dag = DAG(
    'fivetran_sync',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
)

# Trigger sync
start_sync = FivetranOperator(
    task_id='start_fivetran_sync',
    fivetran_conn_id='fivetran_default',
    connector_id='connector_id',
    dag=dag,
)

# Wait for completion
wait_for_sync = FivetranSensor(
    task_id='wait_for_sync',
    fivetran_conn_id='fivetran_default',
    connector_id='connector_id',
    poke_interval=60,
    dag=dag,
)

start_sync >> wait_for_sync
```

### Dagster

```python
from dagster_fivetran import FivetranResource, load_assets_from_fivetran_instance

fivetran_instance = FivetranResource(
    api_key=EnvVar("FIVETRAN_API_KEY"),
    api_secret=EnvVar("FIVETRAN_API_SECRET"),
)

fivetran_assets = load_assets_from_fivetran_instance(
    fivetran_instance,
    key_prefix=["raw"],
)
```

## Comparison with Airbyte

| Aspect | Fivetran | Airbyte |
|--------|----------|---------|
| Deployment | Cloud only | Self-hosted or cloud |
| Pricing | MAR-based (expensive) | Free (OSS) or usage |
| Connectors | 700+ (high quality) | 600+ (variable quality) |
| Maintenance | Zero | Self-managed or paid |
| Custom connectors | SDK available | Multiple options |
| Setup time | Minutes | Hours to days |
| Reliability | Guaranteed SLAs | Depends on deployment |
| Compliance | SOC 2, HIPAA, GDPR | Self-managed |
| Schema handling | Excellent | Good |
| Support | Enterprise SLA | Community or paid |

### When to Choose Fivetran

- Enterprise with budget for managed service
- Zero tolerance for maintenance overhead
- Need compliance certifications
- Standard sources covered by connectors
- Reliability is paramount
- Quick time-to-value needed

### When to Choose Airbyte

- Cost is a primary concern
- Need custom connector development
- Self-hosting required
- Open-source preference
- Willing to manage infrastructure

## Monitoring and Troubleshooting

### Dashboard Monitoring

Fivetran UI provides:
- Sync status and history
- Data volume metrics
- Schema change notifications
- Connector health alerts

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Sync failures | Credentials expired | Re-authenticate |
| Missing data | Schema exclusion | Check column settings |
| High MAR | Frequent updates | Reduce sync frequency |
| Slow syncs | Large tables | Enable history mode |
| Connection errors | Network/firewall | Check allow lists |

### Logs and Debugging

```python
# Get sync history
response = requests.get(
    f"https://api.fivetran.com/v1/connectors/{connector_id}/logs",
    auth=HTTPBasicAuth(api_key, api_secret),
)

for log in response.json()["data"]["items"]:
    print(f"{log['timestamp']}: {log['message']}")
```

## Best Practices

### Source Configuration

- Use service accounts with minimal permissions
- Enable CDC for databases when available
- Exclude high-churn system tables
- Set appropriate sync frequency for data freshness needs

### Destination Configuration

- Use dedicated schemas for raw data (e.g., `raw_salesforce`)
- Enable clustering on timestamp columns
- Set up alerts for sync failures
- Document excluded columns and reasons

### Cost Management

- Review MAR usage monthly
- Identify and pause unused connectors
- Select only needed tables and columns
- Consider longer sync intervals for stable data
- Negotiate annual contracts for discounts (15-20%)

### Operations

- Version control Terraform configurations
- Set up monitoring and alerting
- Document transformation dependencies
- Test schema changes in staging first

## When to Use Fivetran

Fivetran is well-suited for:
- Enterprise data integration with budget
- Teams prioritizing reliability over cost
- Standard SaaS and database sources
- Compliance requirements (SOC 2, HIPAA)
- Minimal engineering overhead needed

Consider alternatives when:
- Cost is the primary concern (Airbyte)
- Custom connectors are primary use case
- Self-hosting is required
- Real-time streaming needed (Kafka, Flink)
