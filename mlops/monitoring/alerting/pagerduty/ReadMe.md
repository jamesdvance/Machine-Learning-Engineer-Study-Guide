# PagerDuty

## Summary

PagerDuty is an incident management and alerting platform that sits at the end of the monitoring pipeline, responsible for routing alerts to the right people at the right time and orchestrating the incident response process. In a typical ML infrastructure stack, monitoring tools like Prometheus and Grafana detect anomalies and threshold violations, then forward actionable alerts to PagerDuty, which handles on-call scheduling, escalation, and notification delivery. For ML teams, PagerDuty is where model degradation alerts, data pipeline failures, and GPU resource issues ultimately land when they require human intervention.

Key points to remember:

- PagerDuty receives alerts from monitoring tools (Prometheus Alertmanager, Grafana, CloudWatch) and routes them to on-call engineers
- Escalation policies define chains of responsibility so that unacknowledged incidents automatically escalate to secondary responders or management
- On-call schedules support weekly, follow-the-sun, weekend, and custom rotations with override capabilities
- Alert severity levels (Critical/High, Warning/Medium, Low, Informational) determine notification urgency and timing
- The Events API v2 allows custom ML systems to send structured alerts directly to PagerDuty
- PagerDuty is often compared with Opsgenie, its closest competitor (now owned by Atlassian)
- Alert fatigue is the primary operational risk; every alert that pages an engineer should require immediate human action

## How PagerDuty Fits in the ML Monitoring Stack

The ML monitoring stack follows a layered architecture where each component has a distinct role. Understanding where PagerDuty fits prevents duplication of effort and ensures alerts reach engineers through a single, well-managed channel.

The typical flow looks like this:

```
ML Application / Model Serving
        |
        v
Metrics Collection (Prometheus, StatsD, CloudWatch Agent)
        |
        v
Visualization & Rules (Grafana, CloudWatch Alarms)
        |
        v
Alert Routing (Prometheus Alertmanager)
        |
        v
Incident Management (PagerDuty)
        |
        v
On-Call Engineer (SMS, push notification, phone call)
```

Prometheus scrapes metrics from model serving endpoints and training infrastructure. Grafana provides dashboards and can evaluate alert rules against those metrics. When a rule fires, the alert is sent to Prometheus Alertmanager or directly from Grafana to PagerDuty via webhook integration. PagerDuty then determines who is on call, how to notify them, and what happens if they do not respond.

PagerDuty is not a monitoring tool. It does not collect metrics, store time series data, or render dashboards. Its job begins when a monitoring tool has already determined that something is wrong and a human needs to act. This distinction matters because ML teams sometimes attempt to configure alerting logic inside PagerDuty itself, which leads to a fragmented alerting configuration spread across multiple systems.

### Services and Integrations

In PagerDuty, a Service represents a component of infrastructure that can generate incidents. For an ML team, services might include:

- Model Serving (the inference API)
- Training Pipeline (batch or continuous training jobs)
- Feature Store (feature computation and storage)
- Data Ingestion Pipeline (upstream data feeds)
- GPU Cluster (compute resources)

Each service has its own integrations (connections to monitoring tools), escalation policies, and notification rules. This separation ensures that a GPU alert routes to the infrastructure on-call engineer while a model accuracy alert routes to the ML engineer on call.

## Escalation Policies and On-Call Schedules

### Escalation Policies

An escalation policy defines what happens when an incident is triggered and who gets notified in sequence. A typical ML team escalation policy might look like:

```
Level 1: ML Engineer On-Call (notify immediately)
         Wait 10 minutes for acknowledgment
         
Level 2: ML Team Lead
         Wait 15 minutes for acknowledgment
         
Level 3: Engineering Manager + Senior ML Engineer
         Wait 20 minutes for acknowledgment
         
Level 4: VP of Engineering (last resort)
```

If the Level 1 responder acknowledges the incident, escalation stops. If they do not respond within the configured timeout, PagerDuty automatically escalates to Level 2, and so on. This guarantees that no incident ever falls through the cracks during off-hours.

Escalation policies can be reused across services. A common pattern is to have one escalation policy for high-severity production issues and a separate, less aggressive policy for lower-priority alerts that only notify during business hours.

### On-Call Schedules

On-call schedules define which team members are responsible for responding at any given time. PagerDuty supports several rotation types:

**Weekly rotation**: One engineer covers a full week, then hands off to the next person. Simple but can be burdensome for small teams.

**Daily rotation**: Shorter shifts distribute the load more evenly. Common for teams with four or more engineers.

**Follow-the-sun**: Teams distributed across time zones hand off to colleagues in the next zone. Engineers in San Francisco cover US business hours, then London covers European hours, then Bangalore covers Asian hours. No one gets paged at 3 AM.

**Weekend rotation**: A separate schedule specifically for weekends, which may rotate independently from weekday coverage.

**Custom rotation**: Any combination of the above, including split shifts within a single day.

Schedules integrate with calendar applications (Google Calendar, Outlook, iCal) so engineers can see their on-call commitments alongside regular meetings. Engineers can create overrides when they need to swap shifts, and PagerDuty sends automatic reminders before shifts begin.

### Schedule Layers

PagerDuty supports layered schedules where multiple rotation layers combine into a final schedule. This is useful for ML teams that want a primary and secondary on-call:

- Layer 1 (Primary): The engineer who gets paged first
- Layer 2 (Secondary/Shadow): A backup who can be escalated to, or a less experienced engineer shadowing for training purposes

Layers are evaluated in order, and the final schedule is the union of all layers with higher layers overriding lower ones where they overlap.

## Integration with Monitoring Tools

### Prometheus Alertmanager

Prometheus Alertmanager is the most common integration point for teams running Kubernetes-based ML infrastructure. Alertmanager receives firing alerts from Prometheus, groups them, and forwards them to PagerDuty.

Alertmanager configuration for PagerDuty:

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: 'pagerduty-critical'
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
    - match:
        severity: warning
      receiver: 'pagerduty-warning'

receivers:
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - routing_key: '<integration-key>'
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          dashboard: '{{ .CommonAnnotations.dashboard_url }}'
          runbook: '{{ .CommonAnnotations.runbook_url }}'

  - name: 'pagerduty-warning'
    pagerduty_configs:
      - routing_key: '<integration-key>'
        severity: warning
```

The `routing_key` (also called an integration key) is generated in PagerDuty when you add a Prometheus integration to a service. Alertmanager sends events using PagerDuty's Events API v2 format.

Key configuration decisions:

- **group_by**: Group related alerts to avoid flooding PagerDuty with dozens of individual incidents. Grouping by `alertname` and `service` is a good default.
- **group_wait**: How long to wait before sending the first notification for a new group. Set this high enough (30s) to batch related alerts.
- **repeat_interval**: How often to re-send a firing alert. Set this long enough (4h+) to avoid nagging but short enough that persistent issues remain visible.

### Grafana

Grafana can send alerts directly to PagerDuty without Alertmanager as an intermediary. This is simpler for teams not using Prometheus or for dashboard-driven alerting.

In Grafana, create a PagerDuty contact point by providing the integration key. Grafana alert rules evaluate queries against data sources (Prometheus, InfluxDB, CloudWatch, etc.) and route firing alerts through notification policies to the PagerDuty contact point.

Grafana is particularly useful for ML teams because alert rules can be defined alongside dashboard panels. An engineer looking at a model accuracy dashboard can create an alert rule directly from the panel query, keeping the alerting configuration close to the visualization.

### Amazon CloudWatch

For ML workloads running on AWS (SageMaker endpoints, ECS-hosted inference, Lambda functions), CloudWatch Alarms integrate with PagerDuty through Amazon SNS (Simple Notification Service) or Amazon EventBridge.

The flow is:

```
CloudWatch Alarm -> SNS Topic -> PagerDuty Integration (HTTPS endpoint)
```

CloudWatch metrics relevant to ML include SageMaker endpoint invocation errors, latency percentiles, GPU utilization on EC2/ECS instances, and custom metrics published from application code.

### Other Integrations

PagerDuty offers over 700 integrations. Those commonly used by ML teams include:

- **Datadog**: Full-featured monitoring with APM, logs, and metrics
- **New Relic**: Application performance monitoring
- **Slack**: Bidirectional integration for incident communication
- **Jira**: Automatic ticket creation for incidents
- **AWS EventBridge**: Serverless event routing from AWS services
- **Azure Monitor**: Alerts from Azure ML and other Azure services
- **Google Cloud Monitoring**: Alerts from Vertex AI and GCP infrastructure

## Alert Routing and Severity Levels

### Severity Levels

PagerDuty uses four severity levels that map to different notification behaviors:

**Critical**: Requires immediate action regardless of time. Triggers phone calls, SMS, and push notifications. Use for production model serving outages, complete data pipeline failures, or security incidents. The on-call engineer is expected to respond within minutes.

**Error/High**: Requires action but may be slightly less urgent. Triggers push and SMS notifications. Use for significant model degradation, partial pipeline failures, or resource exhaustion trending toward an outage.

**Warning**: Requires action within business hours. May suppress overnight notifications depending on configuration. Use for gradual drift, disk usage approaching thresholds, or certificate expiration warnings.

**Info**: Logged for awareness but does not page anyone. Use for successful deployments, completed training runs, or scheduled maintenance notifications.

The most common mistake ML teams make is treating warnings as critical. If a model's accuracy drops by 0.5% overnight, that is important but rarely requires waking someone up at 3 AM. Reserve critical severity for conditions where users are actively impacted or data is at risk.

### Event Rules and Routing

PagerDuty's Event Orchestration (formerly Event Rules) allows teams to transform, route, and suppress incoming events before they create incidents. This is where ML teams implement intelligent alert routing:

**Routing by source**: Send GPU alerts to the infrastructure team's service and model performance alerts to the ML team's service.

**Time-based suppression**: Suppress non-critical alerts during maintenance windows or known batch processing periods. Training jobs that spike GPU utilization every night at 2 AM should not page anyone.

**Content-based enrichment**: Add runbook links, dashboard URLs, or additional context to events based on pattern matching against the event payload.

**Deduplication**: Prevent the same underlying issue from creating multiple incidents. If 10 model replicas all report the same upstream data source failure, PagerDuty should create one incident, not 10.

**Threshold-based suppression**: Suppress alerts that fire and resolve quickly (flapping). If a model latency alert fires for 30 seconds then resolves, it probably does not warrant an incident.

## ML-Specific Alerting

ML systems have failure modes that traditional software does not. PagerDuty handles these through standard incident management, but the alerting rules themselves must be designed with ML-specific concerns in mind.

### Model Performance Degradation

Model accuracy, precision, recall, F1, or business metrics (click-through rate, conversion rate) can degrade silently. Monitoring these requires ground truth labels, which may arrive with significant delay.

Alerts to configure:

- **Prediction distribution shift**: The distribution of model outputs changes significantly from the baseline. This can be detected in real time without ground truth.
- **Feature value anomalies**: Input features fall outside expected ranges or exhibit unusual statistical properties.
- **Accuracy drop below threshold**: When ground truth labels arrive and computed accuracy falls below a defined threshold.
- **Prediction latency increase**: Model inference time exceeds SLA targets, possibly indicating resource contention or model complexity issues.

Example Prometheus alert rule for prediction distribution shift:

```yaml
groups:
  - name: ml-model-alerts
    rules:
      - alert: PredictionDistributionShift
        expr: |
          abs(
            model_prediction_mean{model="fraud-detector"}
            - model_prediction_baseline_mean{model="fraud-detector"}
          ) > 0.1
        for: 15m
        labels:
          severity: warning
          team: ml-engineering
        annotations:
          summary: "Prediction distribution shift detected for fraud-detector"
          runbook_url: "https://wiki.internal/runbooks/prediction-drift"
          dashboard_url: "https://grafana.internal/d/model-monitoring"
```

### Data Pipeline Failures

ML models are only as good as their input data. Data pipeline failures are among the most common causes of model issues in production.

Alerts to configure:

- **Pipeline job failure**: An Airflow DAG, Spark job, or feature computation task fails.
- **Data freshness**: Input data has not been updated within the expected window. If the feature store is stale by more than 2 hours, alert.
- **Schema violations**: Incoming data does not match expected schema (missing columns, type mismatches, unexpected null rates).
- **Data volume anomalies**: Significantly more or fewer records than expected, which may indicate upstream issues.

### GPU and Infrastructure Issues

GPU-accelerated workloads have unique failure modes:

- **GPU memory exhaustion**: CUDA out-of-memory errors during training or inference.
- **GPU utilization anomalies**: Sustained 0% utilization (job crashed but process is still running) or sustained 100% utilization (job is stuck).
- **GPU temperature**: Overheating can cause thermal throttling and hardware damage.
- **NVLink/interconnect errors**: Multi-GPU training failures due to communication errors.
- **Training job stalls**: No gradient updates or checkpoint writes for an extended period.

### Batch Prediction Failures

Many ML systems run batch predictions on a schedule. These require alerting on:

- **Job completion**: Alert if the batch job does not complete within its expected window.
- **Output validation**: Alert if the number of predictions generated is significantly different from expected.
- **Downstream consumption**: Alert if downstream systems have not consumed the batch output.

## Events API for Custom Integrations

PagerDuty's Events API v2 allows any system to create, update, and resolve incidents programmatically. This is essential for ML teams building custom monitoring that does not fit neatly into Prometheus or Grafana.

### Sending a Trigger Event

```python
import requests
import json

def trigger_pagerduty_alert(
    routing_key: str,
    summary: str,
    severity: str = "critical",
    source: str = "ml-pipeline",
    component: str = None,
    custom_details: dict = None
):
    """Send an alert to PagerDuty via Events API v2."""
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": f"{source}-{component}-{summary[:50]}",
        "payload": {
            "summary": summary,
            "severity": severity,
            "source": source,
            "component": component,
            "custom_details": custom_details or {}
        }
    }

    response = requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    return response.json()


# Example: alert on model accuracy drop
trigger_pagerduty_alert(
    routing_key="your-integration-key",
    summary="Fraud detector accuracy dropped below 0.90 (current: 0.84)",
    severity="critical",
    source="model-monitor",
    component="fraud-detector-v2",
    custom_details={
        "model_name": "fraud-detector-v2",
        "current_accuracy": 0.84,
        "threshold": 0.90,
        "dashboard": "https://grafana.internal/d/fraud-model",
        "runbook": "https://wiki.internal/runbooks/model-accuracy-drop"
    }
)
```

### Resolving an Event

Use the same `dedup_key` to resolve a previously triggered event:

```python
def resolve_pagerduty_alert(routing_key: str, dedup_key: str):
    """Resolve a PagerDuty incident via Events API v2."""
    payload = {
        "routing_key": routing_key,
        "event_action": "resolve",
        "dedup_key": dedup_key
    }

    response = requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    return response.json()
```

The `dedup_key` is critical. It links trigger, acknowledge, and resolve events together. Without it, PagerDuty creates a new incident for every event. For ML systems, a good dedup key pattern is `{source}-{model_name}-{alert_type}`, which ensures that repeated alerts about the same model issue consolidate into a single incident.

### Change Events

PagerDuty also accepts Change Events, which record deployments, configuration changes, and other non-incident events. These are useful for correlating model deployments with subsequent incidents:

```python
def send_change_event(routing_key: str, summary: str, source: str):
    """Record a change event in PagerDuty."""
    payload = {
        "routing_key": routing_key,
        "payload": {
            "summary": summary,
            "source": source,
            "timestamp": "2026-04-08T12:00:00Z",
            "custom_details": {
                "model_version": "v2.3.1",
                "deployed_by": "ci-pipeline"
            }
        }
    }

    response = requests.post(
        "https://events.pagerduty.com/v2/change/enqueue",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    return response.json()
```

When an incident occurs shortly after a deployment, PagerDuty surfaces the change event in the incident timeline, making it easier to identify whether the deployment caused the issue.

## Comparison with Opsgenie

Opsgenie, now part of Atlassian (acquired in 2018), is PagerDuty's closest competitor. Both platforms solve the same core problem but differ in their ecosystem and pricing.

### Feature Comparison

| Feature | PagerDuty | Opsgenie |
|---------|-----------|----------|
| On-call scheduling | Full-featured with layers | Full-featured with layers |
| Escalation policies | Multi-level with timeouts | Multi-level with timeouts |
| Integration count | 700+ | 200+ |
| Event orchestration | Advanced rules engine | Alert policies |
| Incident workflows | AIOps and automation | Basic automation |
| Mobile app | iOS and Android | iOS and Android |
| Analytics | Operational reviews, MTTA/MTTR | Basic reporting |
| Pricing tier entry | Higher | Lower |
| Ecosystem fit | Standalone, broad integrations | Best with Atlassian (Jira, Confluence, Statuspage) |

### When to Choose PagerDuty

- Your team is large (50+ engineers) and needs advanced analytics on incident response performance
- You need sophisticated event orchestration with complex routing rules
- You are integrating with a wide variety of monitoring tools beyond the Atlassian ecosystem
- You need AIOps features like intelligent alert grouping and noise reduction
- You want an established incident management platform with a long track record

### When to Choose Opsgenie

- Your organization already uses Atlassian products (Jira, Confluence, Bitbucket, Statuspage) and wants tight integration
- Budget is a primary concern; Opsgenie's entry-level pricing is more accessible for smaller teams
- Your alerting needs are straightforward and do not require advanced event orchestration
- You want incident management embedded in your existing Atlassian workflow rather than as a standalone tool

For most ML teams, the choice comes down to existing tooling. If you are already deep in the Atlassian ecosystem, Opsgenie is the natural fit. If you need a standalone incident management platform with the broadest integration ecosystem, PagerDuty is the standard choice.

## Practical Tips for ML Teams

### Avoiding Alert Fatigue

Alert fatigue is the single biggest operational risk in incident management. When engineers receive too many alerts, they start ignoring all of them, including the ones that matter. PagerDuty's own alerting principles state that "an alert is something which requires a human to perform an action." Everything else is a notification.

Concrete steps to reduce alert fatigue:

- **Audit alert volume weekly.** If an engineer receives more than one or two pages per on-call shift on average, you have too many alerts or they are set at the wrong severity.
- **Classify ruthlessly.** Only conditions that require immediate human action deserve critical severity. A model accuracy drop of 0.5% that will be investigated in the morning is a warning, not a critical alert.
- **Suppress known noise.** Scheduled training jobs, batch processing windows, and planned maintenance should have corresponding suppression rules in PagerDuty Event Orchestration.
- **Fix or delete flapping alerts.** If an alert fires and resolves repeatedly, either the threshold is wrong, the check interval is too short, or the underlying system is genuinely unstable. All three require investigation, not tolerance.
- **Review after every incident.** During post-incident review, ask whether the alert that triggered the response was actionable, timely, and routed correctly. If not, fix it immediately.

### Writing Runbooks

Every alert that can page an engineer should have an associated runbook. A runbook is a document that describes what to do when the alert fires. PagerDuty supports linking runbooks directly to services and incidents.

A good ML runbook includes:

1. **What this alert means**: One paragraph explaining the condition and why it matters.
2. **Impact assessment**: What is affected? Users, batch jobs, downstream systems?
3. **Immediate triage steps**: Quick checks to determine severity. Is the model still serving? Is the data pipeline running? What does the dashboard show?
4. **Resolution steps**: Step-by-step instructions for common root causes.
5. **Escalation criteria**: When to escalate and to whom.
6. **Related dashboards and logs**: Direct links to Grafana dashboards, Kibana queries, or CloudWatch log groups.

Example runbook structure for a model accuracy alert:

```
Alert: Model Accuracy Below Threshold
Model: fraud-detector-v2
Threshold: 0.90

1. IMPACT
   - False negatives increase, fraudulent transactions pass through
   - Estimated revenue impact: $X per hour at current traffic

2. IMMEDIATE TRIAGE
   - Check Grafana dashboard: [link]
   - Verify data pipeline status in Airflow: [link]
   - Check recent model deployments: [link to CI/CD]
   - Query recent prediction distribution: [SQL query]

3. COMMON ROOT CAUSES
   a. Data pipeline delivered stale or corrupt data
      - Check feature freshness in feature store
      - Verify upstream data source availability
   b. Recent model deployment introduced regression
      - Compare current model version with previous
      - Check deployment logs
   c. Genuine distribution shift in production traffic
      - Compare current feature distributions with training data
      - Assess whether retraining is needed

4. ESCALATION
   - If not resolved within 30 minutes, escalate to ML Team Lead
   - If data pipeline issue, page Data Engineering on-call
```

### Structuring Services for ML

Organize PagerDuty services to match your team's areas of responsibility rather than mirroring infrastructure components one-to-one:

- **Model Serving**: All inference-related incidents (latency, errors, availability)
- **Training Infrastructure**: Training job failures, GPU issues, resource exhaustion
- **Data Pipeline**: Feature computation, data ingestion, schema validation
- **ML Platform**: Shared infrastructure like the feature store, model registry, experiment tracking

Each service should have its own escalation policy mapping to the team that owns it. Avoid creating one monolithic "ML" service that routes everything to the same on-call rotation.

### On-Call Hygiene

- **Rotate fairly.** Use PagerDuty's scheduling to ensure equitable distribution of on-call burden. Review on-call hours monthly.
- **Staff for the load.** An on-call rotation needs a minimum of four to five people to be sustainable. Fewer than that leads to burnout.
- **Handoff explicitly.** At the end of each on-call shift, the outgoing engineer should brief the incoming engineer on any active or recent incidents.
- **Compensate on-call time.** Whether through additional pay, time off, or other means, acknowledge that on-call work has a real cost to engineers.

### Testing Alerts End-to-End

Before relying on an alert in production, test the entire chain:

1. Simulate the failure condition (inject bad data, deploy a degraded model to staging)
2. Verify the monitoring tool detects it and fires the alert
3. Verify Alertmanager or Grafana routes the alert to PagerDuty
4. Verify PagerDuty creates an incident on the correct service
5. Verify the on-call engineer receives the notification
6. Verify the runbook link works and the content is current
7. Simulate resolution and verify PagerDuty auto-resolves the incident

Untested alerts provide false confidence. An alert that exists but does not work is worse than no alert at all because the team believes they are covered.
