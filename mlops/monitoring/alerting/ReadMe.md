# Alerting

## Summary

Alerting is the bridge between automated monitoring and human action. In ML systems, monitoring tools like Prometheus, Grafana, and CloudWatch continuously collect metrics on model performance, data quality, infrastructure health, and serving latency. Alerting defines the rules and channels that determine when those metrics require human intervention, who gets notified, how urgently, and what they should do about it. Without a well-designed alerting strategy, teams either drown in noise or miss critical failures entirely.

ML systems present unique alerting challenges compared to traditional software. A web service is either up or down, but a model can silently degrade for days or weeks before anyone notices. Prediction distributions can drift, feature pipelines can deliver stale data, and accuracy can erode gradually without a single error log or crashed process. Alerting for ML must account for these softer failure modes while still handling the hard infrastructure failures that any production system can experience.

The two dominant platforms for alert routing and incident management are PagerDuty and Opsgenie. Both solve the same core problem of getting the right alert to the right person at the right time, but they differ in ecosystem integration, pricing, and advanced features. This chapter covers alerting principles that apply regardless of platform, then provides guidance on choosing between them.

Key points to remember:

- Every alert that pages an engineer should require immediate human action; everything else is a notification or a logged event
- Alert fatigue is the most dangerous failure mode in an alerting system because it causes teams to ignore all alerts, including the ones that matter
- ML systems need alerts for model performance degradation, data quality issues, and prediction distribution shifts in addition to standard infrastructure alerts
- Severity levels should map to response urgency: critical alerts wake people up, warnings wait for business hours, informational alerts go to dashboards
- Escalation policies ensure that unacknowledged alerts reach someone who can act, even when the primary on-call is unavailable
- Runbooks attached to every alert reduce mean time to resolution and make on-call sustainable for the team
- PagerDuty offers the broadest integration ecosystem and advanced event intelligence; Opsgenie offers deeper Atlassian integration and lower cost

## Alerting Principles for ML Systems

### What Deserves an Alert

The single most important principle in alerting is that an alert should require a human to take an action. This principle, articulated clearly in PagerDuty's own incident response documentation, separates actionable alerts from noise. If the system can heal itself, it does not need an alert. If the condition does not require action within the next few hours, it does not need to page someone. If the condition is interesting but not urgent, it belongs on a dashboard or in a weekly report.

For ML systems, this means carefully evaluating each potential alert against three criteria:

- Is a human needed? If the system can automatically roll back a bad model or retry a failed pipeline, a notification is sufficient. An alert is only warranted when automation cannot handle the situation.
- Is it time-sensitive? A 0.3% accuracy drop detected at 2 AM does not warrant waking someone up if it can wait until morning. A complete model serving outage at 2 AM does.
- Is it actionable? The alert must point toward something the engineer can investigate and fix. An alert that says "something might be wrong" without enough context to begin diagnosis is worse than no alert because it consumes attention without enabling action.

### ML-Specific Failure Modes

Traditional software alerting focuses on availability (is the service up?), latency (is it fast enough?), and error rates (are requests failing?). ML systems share these concerns but add an entire category of failures where the service is technically healthy but producing incorrect or degraded results.

Model performance degradation is the most distinctive ML failure mode. A model can serve predictions with low latency and zero errors while its accuracy steadily declines because the real-world data distribution has shifted away from the training distribution. Detecting this requires monitoring prediction distributions, feature value distributions, and (when ground truth is available) accuracy metrics over time.

Data pipeline failures sit upstream of models and are among the most common root causes of production ML issues. Stale features, schema changes in upstream data sources, missing values, and volume anomalies all feed bad data into models that then produce bad predictions. These failures are often silent from the model's perspective since it receives input and produces output without knowing the input is stale or corrupted.

Training pipeline failures affect model freshness. If a scheduled retraining job fails silently, the production model gradually becomes stale as it operates on an increasingly outdated understanding of the data. Heartbeat monitoring, where the pipeline sends a periodic signal to confirm it completed successfully, is an effective pattern for catching these silent failures.

GPU and infrastructure issues include out-of-memory errors, thermal throttling, interconnect failures in multi-GPU training, and training jobs that stall without crashing. These require infrastructure-level monitoring that is distinct from application-level model metrics.

### Thresholds and Sensitivity

Setting alert thresholds for ML metrics is harder than for traditional metrics because ML metrics have natural variance. A model's accuracy fluctuates with incoming data, and not every dip is a genuine problem. Alerting on instantaneous values leads to false positives, while alerting on long rolling averages leads to delayed detection.

Effective approaches include:

- Use rolling averages (15-minute or 30-minute windows) rather than instantaneous values to smooth out normal fluctuation
- Set thresholds based on business impact rather than statistical deviation. A 2% accuracy drop might be acceptable for a content recommendation model but catastrophic for a fraud detection model
- Use the `for` clause in Prometheus alert rules (or equivalent sustained-duration settings in other tools) to require that a condition persists before firing. A 15-minute `for` clause eliminates most transient spikes
- Establish separate thresholds for warning (investigate during business hours) and critical (act immediately) severity levels
- Revisit thresholds after model retraining, since a retrained model may have a different baseline performance level

## Alert Fatigue

Alert fatigue occurs when engineers receive so many alerts that they stop paying attention to any of them. It is the most dangerous failure mode in an alerting system because it is self-reinforcing: the more alerts fire unnecessarily, the less engineers trust the system, the more they ignore alerts, and the more likely they are to miss a genuine incident.

### Causes

The most common causes of alert fatigue in ML teams are:

- Thresholds set too aggressively on naturally variable ML metrics, causing alerts to fire on normal fluctuation
- Lack of deduplication, where a single root cause triggers dozens of individual alerts across related components
- No severity differentiation, where every alert pages the on-call engineer regardless of actual urgency
- Alerts that fire during known events like scheduled retraining windows, batch processing periods, or maintenance
- Alerts left in place after the underlying issue was fixed or the system was redesigned, creating orphaned alerts that no longer make sense
- Treating monitoring dashboards as alerting by converting every dashboard panel into an alert rule

### Prevention

Preventing alert fatigue requires ongoing discipline, not a one-time configuration. The following practices should be treated as operational requirements:

Audit alert volume regularly. Track how many alerts each on-call engineer receives per shift. If the average exceeds one or two pages per shift, the alerting configuration needs work. Both PagerDuty and Opsgenie provide analytics on alert volume and frequency that support this review.

Classify severity ruthlessly. Only conditions that require immediate human action at any hour deserve critical severity. Everything else should be warning (business hours) or informational (dashboard only). When in doubt, start at a lower severity and promote to critical only after confirming the alert is consistently actionable.

Deduplicate aggressively. Configure deduplication keys (called `dedup_key` in PagerDuty and `alias` in Opsgenie) so that related alerts from the same root cause consolidate into a single incident. If ten model replicas all detect the same upstream data issue, the on-call engineer needs one alert, not ten.

Suppress known noise. Use time-based suppression rules to silence alerts during scheduled maintenance windows, batch processing periods, and known high-variance intervals. Both PagerDuty's Event Orchestration and Opsgenie's alert policies support this.

Delete or fix flapping alerts. An alert that fires and resolves repeatedly (known as flapping) is either misconfigured or detecting a genuinely unstable system. Both require investigation. Tolerating flapping alerts trains the team to ignore alerts.

Conduct post-incident alert reviews. After every incident, ask whether the alert that triggered the response was actionable, timely, and correctly routed. If the answer to any of these is no, fix the alert configuration as part of the incident follow-up, not as a backlog item.

## Severity Levels and Escalation Patterns

### Severity Levels

Both PagerDuty and Opsgenie use tiered severity/priority systems. While the exact naming differs, the operational meaning is consistent:

Critical / P1: Production is down or severely degraded. Users are actively impacted. Requires immediate action regardless of time of day. Notification via phone call, SMS, and push notification. Examples: model serving endpoint returning errors, complete data pipeline failure blocking predictions, security breach.

High / P2: Significant degradation detected but the system is partially functional. Requires action within a short time window. Notification via SMS and push. Examples: model accuracy dropped below a business-critical threshold, partial pipeline failure affecting a subset of features, GPU cluster at capacity with queued training jobs.

Warning / P3: A condition that needs attention but can wait for business hours. Notification via push or email, suppressed overnight. Examples: gradual prediction distribution drift, disk usage approaching threshold, certificate expiring in 14 days, feature freshness approaching the staleness limit.

Informational / P4-P5: No action required. Logged for awareness and correlation. Examples: model deployment completed, retraining job finished, scheduled maintenance starting.

The most common mistake is treating P3 conditions as P1. When a team is new to alerting, there is a natural tendency to err on the side of caution by making everything critical. This rapidly leads to alert fatigue, which is far more dangerous than a slightly delayed response to a warning-level condition.

### Escalation Policies

Escalation policies define the chain of responsibility when an alert fires. A well-designed escalation policy guarantees that every critical alert reaches someone who can act on it, even when the primary on-call is asleep, in a meeting, or has their phone on silent.

A typical escalation chain for an ML serving team:

1. Primary on-call engineer, notified immediately via push and SMS. Wait 5-10 minutes for acknowledgment.
2. Primary on-call again via phone call if not acknowledged. Wait 5 minutes.
3. Secondary on-call engineer via push and SMS. Wait 10 minutes.
4. Team lead via phone call. Wait 15 minutes.
5. Engineering manager via phone call.

Escalation policies should be tested regularly. Both PagerDuty and Opsgenie support sending test notifications to verify that the chain works. Test after any configuration change and as part of onboarding new on-call engineers.

Different services should have different escalation policies. A GPU infrastructure alert might escalate to the platform team lead, while a model accuracy alert escalates to the ML team lead. Avoid routing all alerts through a single escalation policy, which both overloads one chain and means that alerts reach people who may not have the context to act on them.

### On-Call Scheduling

Sustainable on-call requires:

- A minimum of four to five people in the rotation to avoid burnout
- Fair distribution of on-call hours, reviewed monthly
- Follow-the-sun rotations for distributed teams so no one gets paged at 3 AM
- Override capabilities for vacations and planned absences
- Explicit handoff at shift boundaries, where the outgoing engineer briefs the incoming engineer on active or recent incidents
- Compensation for on-call time, whether through additional pay, time off in lieu, or other recognition

Both PagerDuty and Opsgenie support weekly, daily, follow-the-sun, and custom rotation patterns with layered schedules for primary and secondary on-call.

## Comparing PagerDuty and Opsgenie

PagerDuty and Opsgenie are the two dominant incident management platforms. Both provide alert routing, escalation, on-call scheduling, and integrations with monitoring tools. For most ML teams, the choice depends on organizational context more than raw feature differences.

### Feature Comparison

| Capability | PagerDuty | Opsgenie |
|---|---|---|
| On-call scheduling | Full-featured with layers | Full-featured with layers |
| Escalation policies | Multi-level with timeouts | Multi-level with timeouts |
| Native integrations | 700+ | 200+ |
| Event intelligence / AIOps | Advanced ML-based grouping and noise reduction | Basic alert correlation |
| Analytics | Operational reviews, MTTA/MTTR tracking | Basic reporting |
| Infrastructure as code | Terraform provider available | First-class Terraform provider |
| Heartbeat monitoring | Via integrations | Built-in |
| Atlassian integration | Available but third-party | Native (Jira, Confluence, Statuspage) |
| Pricing | Higher entry point | Lower entry point, free tier for small teams |
| Mobile experience | Full-featured iOS and Android | Full-featured iOS and Android |

### When to Choose PagerDuty

PagerDuty is the stronger choice when the team needs the broadest possible integration ecosystem (700+ native integrations), advanced event intelligence that uses ML-based alert grouping to reduce noise, deep analytics on incident response performance (mean time to acknowledge, mean time to resolve, escalation frequency), and service dependency mapping for understanding blast radius. It is well established in large enterprise environments and has a long track record. Teams with 50+ engineers and complex multi-team incident workflows benefit most from PagerDuty's advanced features.

### When to Choose Opsgenie

Opsgenie is the stronger choice when the organization already uses Atlassian products (Jira, Confluence, Bitbucket, Statuspage) and wants seamless integration. It is more cost-effective, especially for smaller teams or organizations that already hold Atlassian licenses. Its built-in heartbeat monitoring is particularly useful for ML teams that need to detect silent pipeline failures. The Terraform provider is well maintained for teams practicing infrastructure as code. Teams with straightforward alerting needs that do not require advanced event intelligence will find Opsgenie simpler to configure and manage.

### Practical Guidance

For most ML teams, the decision comes down to ecosystem. If Jira is your project management tool and Confluence is your documentation platform, Opsgenie integrates natively and the incident-to-ticket workflow is seamless. If you use a broader set of tools or need advanced noise reduction at scale, PagerDuty is the established choice. Both platforms are mature and capable; neither is a wrong choice.

For detailed coverage of each platform, including integration configuration, API usage, and platform-specific best practices, see the child chapters:

- [PagerDuty](pagerduty/ReadMe.md)
- [Opsgenie](opsgenie/ReadMe.md)

## Runbooks and Incident Response

### Why Runbooks Matter

Every alert that can page an engineer should have an associated runbook. A runbook is a document that describes what the alert means, how to assess impact, how to triage, and how to resolve common root causes. Runbooks are especially important in ML systems because the remediation steps (trigger retraining, roll back a model version, switch to a fallback model, invalidate a feature cache) may not be obvious to all on-call engineers, particularly those who rotate in from adjacent teams.

Without runbooks, each incident becomes a from-scratch investigation. With runbooks, mean time to resolution drops because the engineer starts with a structured triage path rather than guessing where to look.

### Runbook Structure

A good ML runbook contains the following sections:

1. What this alert means. A single paragraph explaining the condition and why it matters in business terms. This grounds the on-call engineer in the impact before they start investigating.

2. Impact assessment. What is affected: end users, batch jobs, downstream systems, revenue? Quantify where possible. "Estimated revenue impact: $X per hour at current traffic" is more useful than "this is important."

3. Immediate triage steps. Quick checks to narrow the problem. Is the model still serving? Is the data pipeline running? When was the last successful training run? What does the dashboard show? Include direct links to Grafana dashboards, Airflow DAGs, CloudWatch log groups, and model registry pages.

4. Common root causes and resolution steps. Organized as a decision tree or numbered list. For example: (a) if the data pipeline delivered stale data, check feature freshness and upstream availability; (b) if a recent deployment introduced a regression, compare the current model version with the previous one; (c) if this is genuine distribution shift, assess whether retraining is needed.

5. Escalation criteria. When to escalate and to whom. "If not resolved within 30 minutes, escalate to the ML team lead. If the root cause is a data pipeline issue, page the data engineering on-call."

6. Links. Dashboard URLs, log queries, CI/CD pipeline pages, model registry entries, and the relevant section of the internal wiki.

### Keeping Runbooks Current

Runbooks decay quickly if they are not maintained. After every incident, review the runbook that was used. Ask the on-call engineer: was the runbook accurate? Were there steps missing? Were any links broken? Update the runbook as part of the post-incident follow-up, not as a separate backlog task.

Both PagerDuty and Opsgenie support attaching runbook URLs directly to services and alert configurations. Prometheus alert rules and Grafana alert rules support a `runbook_url` annotation that flows through to the incident management platform. Use this consistently so that every alert arrives with a link to its runbook.

## Testing the Alerting Chain

An alert that exists but does not work is worse than no alert at all because it gives the team false confidence. Before relying on any alert in production, test the entire chain end to end:

1. Simulate the failure condition in a staging or test environment. Inject bad data, deploy a degraded model, or artificially spike latency.
2. Verify the monitoring tool detects the condition and evaluates the alert rule.
3. Verify the alert routes through Alertmanager, Grafana, or the relevant intermediary to the incident management platform.
4. Verify the platform creates an incident on the correct service with the correct severity.
5. Verify the on-call engineer receives the notification on the expected channel (push, SMS, phone).
6. Verify the runbook link works and the content is current.
7. Simulate resolution and verify the incident auto-resolves.

Repeat this test after any change to the alerting configuration, the monitoring stack, or the incident management platform. Schedule periodic fire drills (monthly or quarterly) to validate that the entire system still works as expected.

## Designing an Alerting Strategy for ML

Putting the principles together, a practical alerting strategy for an ML team involves the following steps:

Inventory your failure modes. List every way your ML system can fail: model serving errors, latency spikes, accuracy degradation, data pipeline failures, training job failures, feature staleness, infrastructure issues. For each failure mode, determine whether it requires immediate action, business-hours action, or is informational only.

Define services in your incident management platform. Organize services by team ownership: model serving, training infrastructure, data pipeline, ML platform. Each service gets its own escalation policy mapping to the team that owns it.

Set severity levels conservatively. Start with fewer critical alerts and more warnings. Promote alerts to critical only after confirming they are consistently actionable and time-sensitive. It is easier to increase severity later than to recover from alert fatigue.

Configure deduplication and suppression. Set deduplication keys based on the combination of source, component, and alert type. Configure suppression rules for scheduled events like nightly retraining, batch processing windows, and planned maintenance.

Write runbooks for every paging alert. Before an alert goes live, its runbook should exist, be linked, and be reviewed by at least one other engineer.

Test end to end. Validate the full chain from failure simulation through engineer notification for every critical and high-severity alert.

Review and iterate. Track alert volume, false positive rate, and mean time to resolution. Review weekly. Delete or adjust alerts that are not earning their keep.
