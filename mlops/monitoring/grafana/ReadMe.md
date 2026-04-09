# Grafana

## Summary

Grafana is an open-source visualization and dashboarding platform that has become the standard front end for infrastructure and application monitoring. For ML engineers, Grafana provides the visualization layer needed to observe model performance, data drift, system latency, and resource utilization in production. It connects to a wide range of data sources, most notably Prometheus, and turns raw metrics into actionable dashboards with alerting capabilities.

Key points to remember:

- Grafana is a visualization layer, not a data store; it queries external data sources like Prometheus, InfluxDB, Elasticsearch, and CloudWatch
- Dashboards are composed of panels, each bound to a query against a data source, and can be templated with variables for reuse across models and environments
- Time series panels, heatmaps, histograms, and stat panels are the most relevant visualization types for ML monitoring
- Grafana alerting evaluates rules on a schedule, fires alerts when thresholds are breached, and routes notifications to channels like Slack, PagerDuty, and email
- Dashboards can be managed as code through JSON models, the Grafana Terraform provider, or Grafonnet (a Jsonnet library), enabling version control and reproducibility
- Grafana Cloud offers a managed experience with integrated Prometheus (Mimir), Loki, and Tempo backends, while self-hosted deployments provide full control

## Data Sources

### How Grafana Connects to Your Data

Grafana does not store metrics or logs itself. Instead, it acts as a query and visualization front end for external data sources. Each data source has a plugin that translates Grafana queries into the native query language of the backend.

Common data sources for ML monitoring:

- Prometheus: The most common pairing. Prometheus scrapes metrics from instrumented services and stores them as time series. Grafana queries Prometheus using PromQL. This is the standard stack for monitoring model serving latency, request rates, and custom ML metrics.
- InfluxDB: A time series database with its own query language (Flux or InfluxQL). Useful when you need longer retention or different performance characteristics than Prometheus.
- Elasticsearch: Commonly used for log-based monitoring. ML prediction logs, feature values, and audit trails stored in Elasticsearch can be visualized alongside metric dashboards.
- CloudWatch: For teams running inference on AWS (SageMaker, ECS, Lambda), CloudWatch metrics can be pulled directly into Grafana dashboards.
- PostgreSQL and MySQL: Relational databases storing prediction logs, batch evaluation results, or A/B test outcomes can serve as Grafana data sources using SQL queries.
- Google Cloud Monitoring and Azure Monitor: Cloud-native metric backends for GCP and Azure workloads respectively.
- JSON API and CSV: For custom or lightweight integrations, the JSON API data source plugin can query any REST endpoint returning JSON.

You configure data sources in Grafana either through the UI (Settings > Data Sources) or through provisioning files (YAML configuration that Grafana reads at startup). Provisioning is preferred in production because it is declarative and version-controllable.

Example provisioning configuration:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 15s

  - name: PostgreSQL
    type: postgres
    url: prediction-db:5432
    database: ml_predictions
    user: grafana_reader
    secureJsonData:
      password: ${PG_PASSWORD}
    jsonData:
      sslmode: require
```

### Mixed Data Sources

A single dashboard can query multiple data sources. This is important for ML monitoring because model metrics (from Prometheus), prediction logs (from Elasticsearch), and evaluation results (from PostgreSQL) often need to appear side by side. Each panel independently selects its data source.

## Dashboard Design

### Panels

Panels are the building blocks of Grafana dashboards. Each panel contains a visualization backed by one or more queries. Panels are arranged in rows and columns on a responsive grid.

A panel definition includes:

- Data source selection
- Query (PromQL, SQL, Flux, etc.)
- Visualization type (time series, gauge, stat, table, etc.)
- Display options (axes, legends, thresholds, colors)
- Overrides for specific series or fields

Example PromQL query for a model latency panel:

```promql
histogram_quantile(0.95,
  rate(model_prediction_duration_seconds_bucket{model="fraud_detector"}[5m])
)
```

This query computes the 95th percentile prediction latency for a model named "fraud_detector" over a 5-minute window.

### Variables and Templating

Dashboard variables make dashboards reusable across multiple models, environments, and time ranges. Instead of hardcoding a model name into every query, you define a variable and reference it with the `$variable` syntax.

Variable types:

- Query variables: Populated dynamically from a data source. For example, a variable `model` that queries Prometheus for all distinct `model` label values.
- Custom variables: A static list of values (e.g., "staging,production").
- Interval variables: Allow users to adjust aggregation windows.
- Data source variables: Let users switch between Prometheus instances (e.g., staging vs. production).

Example variable definition:

```
Name: model
Type: Query
Data source: Prometheus
Query: label_values(model_prediction_total, model)
```

With this variable, panel queries use `$model` instead of a hardcoded name:

```promql
rate(model_prediction_total{model="$model", environment="$environment"}[5m])
```

Users select the model and environment from dropdowns at the top of the dashboard, and all panels update simultaneously.

### Templating for Multi-Model Dashboards

For organizations running many models, the repeat feature is particularly useful. You can configure a row or panel to repeat for each value of a multi-value variable. This automatically generates one panel per deployed model, keeping the dashboard definition DRY.

### Annotations

Annotations mark events on time series panels. For ML workflows, useful annotations include:

- Model deployment timestamps (new version rolled out)
- Training pipeline completion
- Detected drift events
- Incident start and resolution times

Annotations can be added manually, pulled from a data source, or pushed via the Grafana HTTP API.

## Visualization Types for ML

### Time Series

The time series panel is the primary visualization for monitoring trends over time. For ML monitoring, common uses include:

- Prediction throughput (requests per second) over time
- Model latency percentiles (p50, p95, p99)
- Feature value distributions plotted as quantiles
- Accuracy or error rate metrics from batch evaluation jobs

Time series panels support multiple series, threshold lines, and alert state overlays.

### Heatmaps

Heatmaps display the distribution of values over time, with color intensity indicating frequency. They are useful for:

- Prediction latency distributions: See how the full latency distribution shifts, not just a single percentile
- Prediction score distributions: Monitor whether output score distributions change over time, which can indicate drift
- Feature value distributions: Track how a feature's distribution evolves across time buckets

Heatmaps are especially effective when backed by Prometheus histograms, as they can render the bucket boundaries directly.

### Histograms

Histogram panels show the distribution of values in a single time window. They complement heatmaps by providing a snapshot view. Uses include:

- Examining the shape of model prediction score distributions at a specific point in time
- Comparing latency distributions before and after a deployment
- Analyzing feature value distributions for drift detection

### Stat and Gauge Panels

Stat panels display a single large number, optionally with sparkline and color thresholds. Gauge panels show a value on a radial scale. These are suited for at-a-glance indicators:

- Current model accuracy or F1 score
- Request queue depth
- Active model count
- Error rate (with red/yellow/green thresholds)

### Tables

Table panels display query results in tabular form. For ML monitoring, tables work well for:

- Listing recent predictions with input features and scores
- Showing per-model summaries (version, last deployed, current accuracy)
- Displaying drift detection results per feature

Tables support sorting, filtering, and cell-level color coding based on thresholds.

## Building ML Monitoring Dashboards

### Model Performance Dashboard

A model performance dashboard tracks the accuracy and behavior of a model in production. The exact metrics depend on the model type, but the dashboard structure is consistent.

Key panels for a classification model:

- Prediction volume: `rate(model_prediction_total{model="$model"}[5m])` -- tracks throughput and helps identify traffic anomalies
- Prediction latency (p95): `histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket{model="$model"}[5m]))` -- latency regression often indicates infrastructure or model issues
- Class distribution: Track the rate of each predicted class. A shift in prediction distribution without a corresponding shift in traffic patterns can signal model degradation
- Error rate: `rate(model_prediction_errors_total{model="$model"}[5m]) / rate(model_prediction_total{model="$model"}[5m])` -- fraction of prediction requests that fail
- Ground truth metrics: If labels arrive with a delay, batch evaluation jobs can push accuracy, precision, recall, and F1 to Prometheus via the Pushgateway, and these can be overlaid on the dashboard

For regression models, replace classification-specific panels with prediction value distribution (mean, standard deviation), residual statistics, and MAE or RMSE trends.

### Data Drift Dashboard

Data drift monitoring tracks whether the distribution of incoming features has changed relative to the training data. This dashboard typically consumes metrics produced by a drift detection service (such as Evidently, Alibi Detect, or a custom solution) that exposes results as Prometheus metrics.

Key panels:

- Per-feature drift score: A time series showing the drift statistic (e.g., KL divergence, PSI, KS statistic) for each feature. Threshold lines mark warning and critical drift levels.
- Drift status table: A table listing each feature, its current drift score, and a status column (no drift, warning, drift detected) with cell coloring.
- Feature distribution heatmap: For high-importance features, show the value distribution over time as a heatmap. Visual shifts in the heatmap provide immediate feedback.
- Overall drift indicator: A stat panel showing a composite drift score or a count of features currently in drift state.

Example metric exposed by a drift detection service:

```promql
feature_drift_psi{model="fraud_detector", feature="transaction_amount"}
```

### Latency and Infrastructure Dashboard

This dashboard covers the operational health of the serving infrastructure:

- Request latency histogram: Shows the full latency distribution as a heatmap
- GPU utilization: `nvidia_gpu_utilization{instance="$instance"}` if using NVIDIA DCGM exporter
- GPU memory usage: Track how close the serving process is to GPU memory limits
- CPU and memory: Standard node exporter metrics for the serving hosts
- Queue depth: If using a request queue (Redis, RabbitMQ), track queue length to identify backpressure
- Container restarts: `kube_pod_container_status_restarts_total` for Kubernetes deployments

### Dashboard Organization

For teams managing multiple models, organize dashboards hierarchically:

1. Overview dashboard: High-level health of all models (one row per model, stat panels for key metrics, links to detail dashboards)
2. Per-model dashboard: Detailed metrics for a single model, using the `$model` variable
3. Infrastructure dashboard: Serving cluster health, resource utilization

Use Grafana folders to group dashboards by team or domain, and apply permissions so teams see only their relevant dashboards.

## Alerting in Grafana

### Alert Rules

Grafana Alerting (unified alerting, introduced in Grafana 8 and the default since Grafana 9) evaluates expressions on a schedule and fires alerts when conditions are met.

An alert rule consists of:

- A query against a data source
- Reduce and threshold expressions (e.g., "the last value of the query is above 0.1")
- An evaluation interval (how often to check)
- A pending period (how long the condition must be true before firing)
- Labels for routing

Example alert rule for model error rate:

```yaml
name: High Error Rate - Fraud Model
condition: error_rate > 0.05
query: |
  rate(model_prediction_errors_total{model="fraud_detector"}[5m])
  /
  rate(model_prediction_total{model="fraud_detector"}[5m])
for: 5m
labels:
  severity: critical
  team: ml-platform
```

This fires a critical alert if the fraud model's error rate exceeds 5% for more than 5 minutes.

### Useful Alert Rules for ML

- Prediction latency exceeds SLA threshold (e.g., p95 > 200ms)
- Prediction volume drops below expected baseline (possible upstream failure)
- Error rate exceeds threshold
- Data drift score crosses warning or critical threshold on a key feature
- GPU memory usage exceeds 90%
- Model serving container restarts
- Batch evaluation job has not reported results within expected window

### Notification Channels (Contact Points)

Alerts route to contact points, which define where notifications are sent:

- Slack: Channel messages with dashboard links and metric values
- PagerDuty: Creates incidents for on-call rotation
- Email: For less urgent notifications
- OpsGenie: Alternative incident management integration
- Webhook: Call a custom endpoint for programmatic response (e.g., trigger automatic model rollback)

### Notification Policies

Notification policies control routing, grouping, and silencing:

- Routing: Route alerts to different contact points based on labels (e.g., `team=ml-platform` goes to the ML Slack channel, `severity=critical` goes to PagerDuty)
- Grouping: Group related alerts into a single notification to reduce noise (e.g., group all drift alerts for the same model)
- Silencing: Temporarily suppress alerts during maintenance windows
- Mute timings: Define recurring periods when alerts should not fire (e.g., during scheduled retraining)

## Grafana Cloud vs Self-Hosted

### Self-Hosted

Self-hosted Grafana is free and open-source (AGPL v3 license for recent versions). You deploy Grafana alongside your own Prometheus, Loki, and other backends.

Advantages:

- No per-user or per-metric cost
- Full control over data residency and retention
- Can run in air-gapped or restricted environments
- No vendor dependency

Considerations:

- You manage upgrades, backups, and high availability
- Scaling Prometheus and long-term storage requires additional tooling (Thanos, Cortex, or Mimir)
- SSL, authentication, and authorization must be configured manually

### Grafana Cloud

Grafana Cloud is a managed offering from Grafana Labs. It includes hosted versions of Grafana, Mimir (Prometheus-compatible metrics), Loki (logs), and Tempo (traces).

Advantages:

- No infrastructure management for the monitoring stack itself
- Built-in high availability and long-term storage
- Includes Grafana Machine Learning (time series forecasting and anomaly detection features)
- Free tier available (up to 10k metrics, 50GB logs, 50GB traces)

Considerations:

- Cost scales with metric volume, which can grow quickly with per-feature drift metrics across many models
- Data leaves your infrastructure
- Some enterprise features require paid plans

### Choosing Between Them

For small teams or early-stage ML platforms, Grafana Cloud reduces operational burden. For larger organizations with existing Kubernetes infrastructure and strict data governance requirements, self-hosted deployments are more common. Many organizations use a hybrid approach: Grafana Cloud for the visualization layer with remote-write from self-hosted Prometheus instances.

## Dashboard as Code

### JSON Models

Every Grafana dashboard is represented internally as a JSON document. You can export any dashboard as JSON from the UI (Dashboard Settings > JSON Model) and store it in version control.

A minimal JSON panel definition:

```json
{
  "panels": [
    {
      "title": "Prediction Throughput",
      "type": "timeseries",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "rate(model_prediction_total{model=\"$model\"}[5m])",
          "legendFormat": "{{model}}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "reqps"
        }
      },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
    }
  ]
}
```

JSON dashboards can be loaded through Grafana's provisioning system or the HTTP API.

### Provisioning

Grafana reads dashboard provisioning configuration from YAML files:

```yaml
apiVersion: 1

providers:
  - name: ML Dashboards
    type: file
    options:
      path: /var/lib/grafana/dashboards/ml
      foldersFromFilesStructure: true
```

Place JSON dashboard files in the specified directory, and Grafana loads them on startup. Combined with a ConfigMap in Kubernetes, this enables fully declarative dashboard management.

### Terraform Provider

The Grafana Terraform provider manages dashboards, data sources, folders, alert rules, and notification policies as infrastructure code:

```hcl
resource "grafana_dashboard" "model_performance" {
  config_json = file("dashboards/model-performance.json")
  folder      = grafana_folder.ml_monitoring.id
}

resource "grafana_data_source" "prometheus" {
  type = "prometheus"
  name = "Prometheus"
  url  = "http://prometheus:9090"

  json_data_encoded = jsonencode({
    timeInterval = "15s"
  })
}

resource "grafana_folder" "ml_monitoring" {
  title = "ML Monitoring"
}
```

This approach integrates dashboard management into existing infrastructure-as-code workflows and ensures that monitoring configuration is reviewed and versioned alongside application code.

### Grafonnet

Grafonnet is a Jsonnet library for generating Grafana dashboard JSON programmatically. It is useful when you need to generate dashboards dynamically, for example creating identical panels for each model in a registry.

```jsonnet
local grafana = import 'grafonnet/grafana.libsonnet';
local dashboard = grafana.dashboard;
local prometheus = grafana.prometheus;
local graphPanel = grafana.graphPanel;

dashboard.new('Model Latency', tags=['ml', 'monitoring'])
+ dashboard.addPanel(
  graphPanel.new(
    'P95 Prediction Latency',
    datasource='Prometheus',
  )
  + graphPanel.addTarget(
    prometheus.target(
      'histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket{model="$model"}[5m]))',
      legendFormat='p95'
    )
  ),
  gridPos={ h: 8, w: 12, x: 0, y: 0 }
)
```

Grafonnet is particularly powerful when combined with CI/CD: a pipeline can read the model registry, generate dashboards for all registered models, and provision them automatically.

## Integration with Prometheus for ML Monitoring

### The Prometheus-Grafana Stack

Prometheus and Grafana are the most common monitoring pair in cloud-native environments. Prometheus handles metric collection and storage; Grafana handles visualization and alerting. For ML systems, the integration path is:

1. Instrument your model serving code with a Prometheus client library
2. Expose a `/metrics` endpoint from your serving container
3. Configure Prometheus to scrape that endpoint
4. Build Grafana dashboards that query Prometheus

### Instrumenting ML Serving Code

Using the Python Prometheus client:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
PREDICTION_COUNT = Counter(
    'model_prediction_total',
    'Total number of predictions',
    ['model', 'model_version', 'predicted_class']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Time spent processing prediction',
    ['model'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_SCORE = Histogram(
    'model_prediction_score',
    'Distribution of prediction scores',
    ['model'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FEATURE_VALUE = Gauge(
    'model_feature_value',
    'Current feature value statistics',
    ['model', 'feature', 'statistic']
)

# Use in prediction path
def predict(request):
    model_name = "fraud_detector"
    with PREDICTION_LATENCY.labels(model=model_name).time():
        result = model.predict(request.features)

    PREDICTION_COUNT.labels(
        model=model_name,
        model_version="v2.3",
        predicted_class=str(result.label)
    ).inc()

    PREDICTION_SCORE.labels(model=model_name).observe(result.score)

    return result
```

### Prometheus Metric Types for ML

Choose the right metric type:

- Counter: Monotonically increasing values. Use for prediction count, error count. Query with `rate()` to get per-second rates.
- Histogram: Distributions. Use for latency, prediction scores, feature values. Enables percentile calculations with `histogram_quantile()`.
- Gauge: Values that go up and down. Use for current queue depth, GPU utilization, active model count.
- Summary: Client-side calculated percentiles. Less flexible than histograms for aggregation; generally prefer histograms.

### Pushgateway for Batch Metrics

Batch evaluation jobs that run periodically (not long-lived services) cannot be scraped by Prometheus in the standard pull model. The Prometheus Pushgateway bridges this gap:

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

registry = CollectorRegistry()

accuracy = Gauge(
    'model_evaluation_accuracy',
    'Model accuracy from batch evaluation',
    ['model', 'dataset'],
    registry=registry
)

accuracy.labels(model='fraud_detector', dataset='validation_2024_q4').set(0.943)

push_to_gateway('pushgateway:9091', job='batch_eval', registry=registry)
```

Grafana can then query these batch metrics alongside real-time serving metrics for a complete picture.

### Recording Rules

For expensive PromQL queries used in dashboards, define Prometheus recording rules to pre-compute results:

```yaml
groups:
  - name: ml_model_metrics
    interval: 30s
    rules:
      - record: model:prediction_rate:5m
        expr: rate(model_prediction_total[5m])

      - record: model:prediction_error_rate:5m
        expr: |
          rate(model_prediction_errors_total[5m])
          /
          rate(model_prediction_total[5m])

      - record: model:prediction_latency_p95:5m
        expr: |
          histogram_quantile(0.95,
            rate(model_prediction_duration_seconds_bucket[5m])
          )
```

Recording rules improve dashboard load times and ensure consistency across panels and alerts that use the same underlying calculation.

## Practical Tips for Building Effective ML Dashboards

### Start with the Questions You Need to Answer

Before building panels, list the questions the dashboard should answer. For a model serving dashboard, those questions might be:

- Is the model healthy right now?
- Has latency increased since the last deployment?
- Has the prediction distribution shifted?
- Are there any features exhibiting drift?

Each question maps to one or more panels. Avoid adding metrics that do not answer a specific question.

### Use Consistent Naming and Labels

Establish a labeling convention across all models. Every metric should include at minimum `model` and `model_version` labels. Consistent labels enable the variable and templating system to work across all models without per-model dashboard customization.

### Set Meaningful Thresholds

Color-code panels with thresholds that correspond to SLAs and operational expectations:

- Green: Normal operation
- Yellow: Warning, investigate soon
- Red: Critical, requires immediate attention

Define thresholds based on historical baselines and business requirements, not arbitrary values. A p95 latency threshold of 200ms only makes sense if the SLA requires sub-200ms responses.

### Layer Dashboards by Audience

Different stakeholders need different views:

- Executive dashboard: High-level health indicators, model count, overall error rates. Stat panels and simple time series.
- ML engineer dashboard: Per-model deep dives, feature drift details, latency breakdowns, deployment annotations.
- Infrastructure dashboard: GPU utilization, container health, network throughput, disk usage.

### Use Annotations for Context

Mark deployments, retraining events, and incidents on time series panels. Correlating metric changes with events is one of the most valuable debugging tools. Automate annotation creation in your CI/CD pipeline:

```bash
curl -X POST http://grafana:3000/api/annotations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
  -d '{
    "text": "Deployed fraud_detector v2.4",
    "tags": ["deployment", "fraud_detector"],
    "time": '$(date +%s000)'
  }'
```

### Design for Incident Response

During an incident, engineers need to move fast. Design dashboards to support this:

- Place the most critical health indicators at the top of the dashboard
- Include links to related dashboards (from the model dashboard, link to the infrastructure dashboard)
- Add text panels with runbook links for common failure modes
- Use panel descriptions to explain what each metric means and what to do if it is abnormal

### Avoid Dashboard Sprawl

Dashboard sprawl occurs when teams create many one-off dashboards that are never maintained. Mitigate this by:

- Using variables and templating to make dashboards reusable
- Establishing a standard dashboard template for all models
- Managing dashboards as code so they go through code review
- Periodically auditing dashboards and archiving unused ones

### Monitor the Monitoring

Ensure that your monitoring stack itself is healthy:

- Alert if Prometheus scrape targets are down
- Alert if Grafana data source queries start timing out
- Track Prometheus storage consumption and set retention policies
- Monitor the drift detection service itself for failures
