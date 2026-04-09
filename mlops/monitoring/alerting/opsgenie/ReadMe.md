# Opsgenie

## Summary

Opsgenie is Atlassian's incident management and alerting platform designed to ensure that critical alerts reach the right people through the right channels at the right time. Originally an independent company founded in 2012, Opsgenie was acquired by Atlassian in 2018 and has since been deeply integrated into the Atlassian ecosystem alongside Jira, Confluence, and Statuspage. For ML engineering teams, Opsgenie serves as the central nervous system for production alerting, routing model performance degradation alerts, infrastructure failures, and pipeline breakdowns to the appropriate on-call engineers with configurable escalation policies.

Key points to remember:

- Opsgenie centralizes alerts from hundreds of monitoring tools (Prometheus, Grafana, CloudWatch, Datadog) into a single pane of glass with deduplication and correlation
- Alert routing uses teams, schedules, and escalation policies to ensure the right engineer is notified based on time of day, alert severity, and domain ownership
- Deep integration with Jira, Confluence, and Statuspage makes it a natural choice for organizations already invested in the Atlassian ecosystem
- On-call scheduling supports complex rotation patterns including follow-the-sun, weekly rotations, and override schedules
- Opsgenie is generally more cost-effective than PagerDuty, especially for teams that already hold Atlassian licenses
- The platform provides a robust API and Terraform provider for infrastructure-as-code management of alerting configurations

## How Opsgenie Works

### Alert Lifecycle

Every alert in Opsgenie follows a defined lifecycle. When a monitoring tool detects an anomaly, it sends an alert to Opsgenie via API, email, or a native integration. Opsgenie then processes the alert through a series of steps before it reaches a human.

The alert lifecycle proceeds through these states:

- Open: The alert has been created and notifications are being sent
- Acknowledged: An engineer has seen the alert and is investigating
- Closed: The issue has been resolved

When an alert is created, Opsgenie evaluates it against the receiving team's routing rules. Routing rules determine which notification policy and escalation policy apply based on conditions like alert priority, tags, or source. For example, a P1 alert about model serving latency might trigger immediate phone calls, while a P4 alert about a non-critical batch pipeline delay might only send an email during business hours.

```
Alert Source (Prometheus, CloudWatch, etc.)
    |
    v
Opsgenie API Integration
    |
    v
Alert Processing (deduplication, enrichment, tagging)
    |
    v
Routing Rules (match conditions to policies)
    |
    v
Escalation Policy (who gets notified, in what order)
    |
    v
Notification Channels (SMS, phone, push, email, Slack)
```

### Alert Deduplication and Correlation

In ML systems, a single root cause often triggers a cascade of alerts. A failed feature store update might cause data drift alerts, prediction latency spikes, and accuracy degradation warnings simultaneously. Opsgenie handles this through deduplication and alert correlation.

Deduplication uses an alert's `alias` field. When multiple alerts arrive with the same alias, Opsgenie increments the count on the existing alert rather than creating new ones. This prevents alert fatigue during cascading failures.

Alert correlation goes further by grouping related alerts together. You can configure Opsgenie to treat alerts with matching tags or sources as related, presenting them as a single incident rather than a flood of individual notifications.

### Integrations with Monitoring Tools

Opsgenie provides native integrations with the monitoring tools commonly used in ML infrastructure. Each integration translates tool-specific alert formats into Opsgenie's standard alert schema.

Prometheus and Alertmanager:

Prometheus fires alerts based on PromQL rules, and Alertmanager routes them to Opsgenie via webhook. The integration maps Prometheus labels to Opsgenie alert fields.

```yaml
# Alertmanager configuration for Opsgenie
receivers:
  - name: 'opsgenie-critical'
    opsgenie_configs:
      - api_key: '<opsgenie-api-key>'
        message: '{{ .GroupLabels.alertname }}'
        priority: '{{ if eq .GroupLabels.severity "critical" }}P1{{ else }}P3{{ end }}'
        tags: 'prometheus,{{ .GroupLabels.team }}'
        details:
          description: '{{ .CommonAnnotations.description }}'
          runbook: '{{ .CommonAnnotations.runbook_url }}'
```

Grafana:

Grafana can send alerts directly to Opsgenie through its notification channels. This is useful when you have Grafana dashboards monitoring model metrics and want alerts based on visual thresholds.

CloudWatch:

AWS CloudWatch integrates through SNS topics. CloudWatch alarms trigger SNS notifications, which are forwarded to Opsgenie via an SNS integration. This covers infrastructure-level alerts for SageMaker endpoints, ECS tasks running model servers, and Lambda-based inference pipelines.

Datadog:

Datadog's integration sends monitors and events to Opsgenie, mapping Datadog priority levels to Opsgenie priorities. For teams using Datadog for application performance monitoring alongside ML model monitoring, this provides a unified alerting path.

Custom integrations are also straightforward via the REST API:

```python
import requests

def send_opsgenie_alert(message, priority, tags, details):
    url = "https://api.opsgenie.com/v2/alerts"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "GenieKey <api-key>"
    }
    payload = {
        "message": message,
        "priority": priority,
        "tags": tags,
        "details": details
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# Example: alert on model performance degradation
send_opsgenie_alert(
    message="Fraud model AUC dropped below 0.85",
    priority="P2",
    tags=["ml-models", "fraud-detection", "performance"],
    details={
        "model_name": "fraud-detector-v3",
        "current_auc": "0.82",
        "threshold": "0.85",
        "environment": "production",
        "runbook": "https://wiki.internal/runbooks/fraud-model-degradation"
    }
)
```

## Atlassian Ecosystem Integration

### Jira Integration

Opsgenie's integration with Jira is one of its strongest differentiators. When an alert fires, Opsgenie can automatically create a Jira ticket, and when the alert is resolved, it can transition the ticket. This bidirectional sync eliminates manual ticket creation during incidents.

Configuration options include:

- Automatic Jira issue creation when alerts are acknowledged or escalated
- Mapping alert priority to Jira priority or custom fields
- Linking Jira issues to Opsgenie incidents for traceability
- Closing alerts when corresponding Jira tickets are resolved

For ML teams, this means a model retraining failure can automatically create a Jira ticket assigned to the ML platform team, with all relevant context (model name, error logs, pipeline run ID) populated from the alert details.

### Confluence Integration

Opsgenie can publish post-incident reports to Confluence pages. After resolving an incident, the on-call engineer fills out a post-mortem template in Opsgenie, and the platform generates a formatted Confluence page in the team's space. This creates a searchable knowledge base of past incidents, which is valuable for understanding recurring model failures and their resolutions.

### Statuspage Integration

For teams that expose model availability to internal consumers or external customers, Opsgenie integrates with Atlassian Statuspage. When an incident is created in Opsgenie, it can automatically update a Statuspage component. For example, if the recommendation service is degraded, the Statuspage can reflect this without manual intervention, keeping downstream consumers informed.

## Alert Routing and Escalation

### Team-Based Alert Management

Opsgenie organizes alert management around teams. Each team has its own routing rules, escalation policies, and on-call schedules. This maps naturally to how ML organizations are structured.

A typical ML organization might configure teams like:

- ML Platform Team: Receives alerts for training infrastructure, feature stores, and model registries
- Model Serving Team: Receives alerts for inference latency, endpoint health, and scaling issues
- Data Engineering Team: Receives alerts for data pipeline failures, data quality issues, and feature freshness
- Applied ML Team: Receives alerts for model-specific performance degradation

Each team can have different notification preferences, escalation timelines, and routing logic. The ML Platform Team might want immediate phone calls for GPU cluster failures, while the Applied ML Team might prefer Slack notifications for gradual accuracy drift.

### Routing Rules

Routing rules determine which escalation policy handles an incoming alert. Rules are evaluated in order, and the first matching rule applies.

```
Rule 1: If tags contain "infrastructure" AND priority is P1 or P2
  -> Use "Infrastructure Critical" escalation policy

Rule 2: If tags contain "model-performance" AND priority is P1
  -> Use "ML On-Call Urgent" escalation policy

Rule 3: If tags contain "model-performance" AND priority is P3 or P4
  -> Use "ML On-Call Standard" escalation policy

Default: Use "Team Default" escalation policy
```

Rules can match on any alert field: priority, tags, source, message content, or custom properties in the details field. This flexibility allows fine-grained routing without modifying the alerting source.

### Escalation Policies

Escalation policies define who gets notified and when. A policy contains a series of steps, each specifying a recipient and a delay from the previous step.

A typical escalation policy for an ML serving team:

- Step 0 (immediately): Notify the on-call engineer via push notification and SMS
- Step 1 (after 5 minutes if not acknowledged): Notify the on-call engineer via phone call
- Step 2 (after 10 minutes): Notify the secondary on-call via push and SMS
- Step 3 (after 20 minutes): Notify the team lead via phone call
- Step 4 (after 30 minutes): Notify the engineering manager via phone call

Escalation ensures that critical alerts are never lost. If the primary on-call is unavailable, the alert automatically escalates to the next responder.

### On-Call Scheduling

Opsgenie provides flexible on-call scheduling that supports the patterns ML teams commonly need:

Weekly rotations: The simplest pattern, where one engineer is on call for a full week before rotating to the next person. Suitable for small teams.

Daily rotations: Each engineer takes one day. This distributes the burden more evenly and reduces fatigue.

Follow-the-sun: For distributed teams, on-call responsibility follows business hours across time zones. An engineer in London handles alerts during European hours, then hands off to a colleague in San Francisco. This avoids middle-of-the-night pages.

Custom rotations: Opsgenie supports arbitrary rotation intervals, restrictions (weekday only, weekend only), and time-of-day constraints.

Override schedules: Temporary overrides for vacations, conferences, or planned absences. An engineer can hand off their on-call slot without modifying the base schedule.

```
Schedule: ML Platform On-Call
  Rotation: Weekly, starting Monday 09:00 UTC
  Participants: [engineer-a, engineer-b, engineer-c, engineer-d]

  Override: 2026-04-10 to 2026-04-17
  engineer-b replaced by engineer-d (vacation)
```

## Infrastructure as Code

Opsgenie configurations should be managed as code, not through the UI. The Terraform provider is the standard approach for teams practicing infrastructure as code.

```hcl
resource "opsgenie_team" "ml_platform" {
  name        = "ML Platform"
  description = "ML infrastructure and platform team"
}

resource "opsgenie_user" "oncall_engineer" {
  username  = "engineer@company.com"
  full_name = "Jane Smith"
  role      = "User"
}

resource "opsgenie_schedule" "ml_platform_oncall" {
  name    = "ML Platform On-Call"
  team_id = opsgenie_team.ml_platform.id
  enabled = true
}

resource "opsgenie_schedule_rotation" "weekly" {
  schedule_id = opsgenie_schedule.ml_platform_oncall.id
  name        = "Weekly Rotation"
  type        = "weekly"
  start_date  = "2026-01-05T09:00:00Z"

  participants {
    type = "user"
    id   = opsgenie_user.oncall_engineer.id
  }
}

resource "opsgenie_escalation" "ml_platform_critical" {
  name    = "ML Platform Critical"
  team_id = opsgenie_team.ml_platform.id

  rules {
    condition   = "if-not-acked"
    notify_type = "default"
    delay       = 0

    recipient {
      type = "schedule"
      id   = opsgenie_schedule.ml_platform_oncall.id
    }
  }

  rules {
    condition   = "if-not-acked"
    notify_type = "default"
    delay       = 10

    recipient {
      type = "user"
      id   = opsgenie_user.oncall_engineer.id
    }
  }
}

resource "opsgenie_api_integration" "prometheus" {
  name = "Prometheus"
  type = "Prometheus"

  responders {
    type = "team"
    id   = opsgenie_team.ml_platform.id
  }
}
```

Managing Opsgenie through Terraform ensures that alerting configurations are versioned, reviewed, and reproducible. Changes go through pull requests rather than ad-hoc UI modifications, which is critical for audit trails in regulated ML environments.

## ML-Specific Considerations

### Model Performance Alerting

Traditional software alerting focuses on binary states: a service is up or down, latency is above or below a threshold. ML systems introduce a different category of alerts where degradation is gradual and context-dependent.

Model performance metrics (accuracy, precision, recall, AUC) tend to drift over time rather than fail abruptly. Opsgenie's priority system maps well to this:

- P1: Model predictions are clearly wrong or service is down (e.g., serving stale predictions, endpoint returning errors)
- P2: Significant performance degradation detected (e.g., AUC dropped below business-critical threshold)
- P3: Moderate drift detected (e.g., feature distribution shift, gradual accuracy decline)
- P4: Informational (e.g., retraining job completed, new model version registered)

Configure P3 and P4 alerts to notify only during business hours, since they rarely require immediate action. P1 and P2 should use aggressive escalation policies with phone calls.

### Training Pipeline Alerts

ML training pipelines have their own failure modes that differ from serving-side alerts:

- GPU out-of-memory errors during training
- Training loss divergence or NaN values
- Data pipeline failures upstream of training
- Feature store staleness beyond acceptable windows
- Model registry promotion failures

These alerts should route to the ML platform or training team rather than the serving team. Opsgenie's team-based routing makes this separation clean.

### Data Quality Alerts

Data quality issues are among the most common root causes of ML model failures in production. Monitoring tools like Great Expectations, Evidently, or custom validation scripts can send alerts to Opsgenie when:

- Input feature distributions shift beyond expected bounds
- Missing value rates exceed thresholds
- Data freshness falls below requirements
- Schema changes are detected in upstream data sources

These alerts benefit from rich context in the details field. Include the specific feature that drifted, the expected versus observed distribution statistics, and a link to the relevant monitoring dashboard.

### Alert Fatigue in ML Systems

ML systems are particularly prone to alert fatigue because models continuously produce metrics that fluctuate naturally. Not every metric dip is a genuine problem. To combat this:

- Use Opsgenie's deduplication aggressively, setting aliases based on the model name and failure type
- Set meaningful thresholds that account for normal variance rather than alerting on any deviation
- Use P4/P5 priorities for informational alerts and restrict notifications to dashboards or low-priority channels
- Implement alert suppression during known maintenance windows (scheduled retraining, data backfills)
- Review alert volume regularly and remove or adjust noisy alerts

## Opsgenie vs PagerDuty

PagerDuty and Opsgenie are the two dominant incident management platforms. Both provide alert routing, escalation, on-call management, and integrations with monitoring tools. The choice between them often depends on organizational context rather than raw feature comparison.

### Pricing

Opsgenie is generally less expensive than PagerDuty. Opsgenie offers a free tier for up to five users with basic alerting and on-call management. Its paid plans start lower than PagerDuty's equivalent tiers. For organizations already paying for Atlassian products (Jira, Confluence), Opsgenie may be bundled or discounted as part of Atlassian Cloud plans, making it significantly cheaper.

PagerDuty's pricing starts higher but includes features like event intelligence (ML-based alert grouping) and advanced analytics at its upper tiers. For large enterprises with complex incident management needs, PagerDuty's higher cost may be justified.

### Feature Comparison

Both platforms cover the core requirements well. The differences are at the margins:

PagerDuty advantages:
- More mature event intelligence with ML-based alert correlation and noise reduction
- Broader ecosystem of native integrations (over 700 versus Opsgenie's 200+)
- More advanced analytics and reporting for incident management metrics
- Stronger presence in enterprise incident management workflows
- Service dependency mapping and impact analysis
- Dedicated mobile app experience with more features

Opsgenie advantages:
- Deep Atlassian ecosystem integration (Jira, Confluence, Statuspage, Bitbucket)
- More cost-effective, especially for Atlassian shops
- Flexible alert routing with condition-based rules
- Built-in Terraform provider for infrastructure as code
- Simpler UI that is easier to onboard new team members
- Heartbeat monitoring for cron jobs and scheduled tasks out of the box

### Ecosystem Fit

This is often the deciding factor. If your organization uses Jira for project management, Confluence for documentation, and Bitbucket for source control, Opsgenie is the natural choice. The integrations are first-party, well-maintained, and bidirectional. Incident-to-ticket workflows are seamless.

If your organization is not invested in Atlassian, or uses tools like ServiceNow, GitHub, or Linear for project management, PagerDuty may integrate more smoothly. PagerDuty also has deeper partnerships with major cloud providers' enterprise support offerings.

### When to Choose Opsgenie

Choose Opsgenie when:

- Your organization already uses Atlassian products and wants tight integration
- Cost is a significant factor, especially for growing teams
- You need straightforward alert routing and escalation without advanced ML-based correlation
- Your team values infrastructure-as-code management of alerting configurations
- You want built-in heartbeat monitoring for scheduled ML pipelines
- Your incident management needs are well-served by team-based routing and standard escalation

### When to Choose PagerDuty

Choose PagerDuty when:

- You need advanced event intelligence to handle high alert volumes with automated correlation
- Your organization has complex incident management workflows involving multiple teams and services
- You require deep analytics on incident metrics (MTTA, MTTR, escalation frequency)
- Integration breadth is important because you use a wide variety of monitoring and operational tools
- You are in a large enterprise environment where PagerDuty is already the standard
- Service dependency mapping is critical for understanding blast radius of ML system failures

## Heartbeat Monitoring

Opsgenie's heartbeat monitoring is particularly useful for ML pipelines. A heartbeat is a periodic signal that an application sends to Opsgenie to confirm it is running. If Opsgenie does not receive a heartbeat within the configured interval, it creates an alert.

This is ideal for:

- Scheduled retraining pipelines that should run daily or weekly
- Batch prediction jobs that must complete within a time window
- Feature computation pipelines that need to stay current
- Data validation jobs that run on a cadence

```python
import requests

def send_heartbeat(heartbeat_name):
    url = f"https://api.opsgenie.com/v2/heartbeats/{heartbeat_name}/ping"
    headers = {"Authorization": "GenieKey <api-key>"}
    requests.get(url, headers=headers)

# Call at the end of your training pipeline
def run_daily_retraining():
    data = load_training_data()
    model = train_model(data)
    evaluate_and_register(model)
    send_heartbeat("daily-retraining-pipeline")
```

If the pipeline fails silently or hangs, the missing heartbeat triggers an alert. This catches failure modes that would otherwise go undetected until a downstream consumer notices stale predictions.

## Common Pitfalls

### Over-Alerting on Model Metrics

Setting alert thresholds too tightly on ML metrics leads to constant noise. Model accuracy naturally fluctuates with incoming data distribution. Set thresholds based on sustained degradation (e.g., 30-minute rolling average) rather than instantaneous values.

### Ignoring Alert Routing Configuration

Sending all alerts to a single on-call person regardless of type creates burnout and slow response times. Invest time in setting up proper team-based routing so that infrastructure alerts go to the platform team and model-specific alerts go to the team that owns the model.

### Not Using Runbook Links

Every alert should include a link to a runbook in the details field. During a 3 AM page, engineers should not need to figure out diagnosis steps from scratch. Runbooks are especially important for ML-specific alerts where the remediation (trigger retraining, roll back model version, switch to fallback model) may not be obvious to all on-call engineers.

### Manual Configuration Drift

Configuring Opsgenie through the UI leads to undocumented changes and configuration drift between environments. Use the Terraform provider or API to manage all configurations, and store them in version control alongside your ML infrastructure code.

### Not Testing Escalation Policies

Escalation policies should be tested regularly to confirm they work as expected. Opsgenie provides a "Send Test Notification" feature. Use it after any configuration change and during on-call onboarding for new team members.
