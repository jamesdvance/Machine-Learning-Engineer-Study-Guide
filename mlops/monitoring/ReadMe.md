# Monitoring for ML Systems

## Summary

Monitoring is the practice of collecting, analyzing, and acting on signals from production systems to ensure they behave correctly and perform within acceptable bounds. For machine learning systems, monitoring extends well beyond traditional infrastructure and application health checks. ML systems are uniquely vulnerable to silent failures where the service remains operational -- returning HTTP 200 responses with low latency -- while the model itself produces degraded or incorrect predictions due to changes in the underlying data. This makes monitoring both more important and more complex for ML workloads than for conventional software.

A complete ML monitoring stack typically consists of three layers: metrics collection, visualization, and alerting. Prometheus has become the standard open-source tool for collecting and storing time series metrics from instrumented services. Grafana provides the visualization layer, turning raw metrics into dashboards that make system behavior legible to engineers. Alerting systems like PagerDuty and Opsgenie close the loop by notifying the right people when metrics cross critical thresholds. On top of this infrastructure-level stack, ML systems require model-specific monitoring for phenomena like data drift, concept drift, and performance degradation -- problems that do not exist in traditional software and require specialized detection techniques.

Key points to remember:

- ML systems can fail silently: a model serving incorrect predictions will not trigger standard health checks or error rate alerts
- The monitoring stack for ML follows a layered architecture: collection (Prometheus) to visualization (Grafana) to alerting (PagerDuty/Opsgenie)
- Traditional software monitoring covers infrastructure and application health; ML monitoring adds data quality, feature distributions, prediction distributions, and model accuracy tracking
- Observability is a broader concept than monitoring, encompassing the ability to understand internal system state from external outputs through metrics, logs, and traces
- Model-specific monitoring requires tracking statistical properties of inputs and outputs over time, not just operational metrics
- Effective ML monitoring bridges two audiences: infrastructure/platform engineers who care about uptime and latency, and data scientists who care about model quality and drift

## How ML Monitoring Differs from Traditional Software Monitoring

Traditional software monitoring focuses on deterministic systems. A web application either processes a request correctly or it throws an error. The failure modes are well understood: the server crashes, a database query times out, memory is exhausted, a dependency is unreachable. Standard monitoring tools detect these failures reliably because they produce clear signals -- error codes, exceptions, timeouts.

ML systems have all of these traditional failure modes plus an entirely separate class of failures that are statistical in nature. A fraud detection model may start approving fraudulent transactions not because of a code bug or infrastructure failure, but because the distribution of incoming transaction data has shifted away from what the model was trained on. The service continues to operate normally from an infrastructure perspective. Latency is fine. Error rates are zero. But the model is producing predictions that no longer match reality.

This distinction creates a fundamental monitoring challenge: you cannot rely solely on traditional operational metrics to determine whether an ML system is working correctly. You must also monitor the statistical properties of the data flowing through the system and the predictions the model produces.

### The Three Layers of ML System Failure

Infrastructure failures are the most straightforward to detect. These include server outages, GPU memory exhaustion, network partitions, and disk space issues. Standard infrastructure monitoring catches these reliably. Prometheus node exporters and Kubernetes metrics provide complete visibility.

Application failures include bugs in preprocessing code, serialization errors, model loading failures, and dependency version mismatches. These typically produce error responses or exceptions that increment error counters. Standard application monitoring handles these well.

Model failures are unique to ML systems. These include data drift (input feature distributions changing over time), concept drift (the relationship between features and target changing), prediction distribution shifts, and gradual performance degradation. These failures produce no errors, no exceptions, and no infrastructure alerts. They require dedicated monitoring approaches that track statistical properties of data and predictions over time.

A robust ML monitoring strategy must address all three layers.

### Feedback Delay

Another challenge unique to ML monitoring is feedback delay. In traditional software, you know immediately whether a request was handled correctly -- the user sees the right page, the payment processes, the email sends. In ML systems, the ground truth label that tells you whether a prediction was correct may arrive hours, days, or weeks after the prediction was made. A fraud detection model makes a prediction at transaction time, but the true label (fraudulent or not) may not be confirmed for 30 to 90 days.

This delay means you cannot rely on real-time accuracy metrics alone. You need proxy metrics that can detect problems before ground truth arrives. Prediction distribution monitoring, feature distribution monitoring, and data quality checks serve as early warning signals that something may be wrong, even when you cannot yet confirm that predictions are incorrect.

## Observability vs Monitoring

Monitoring and observability are related but distinct concepts, and the distinction matters for ML systems.

Monitoring is the practice of collecting predefined metrics and setting alerts on known failure conditions. You decide in advance what to measure (latency, error rate, request count) and what thresholds indicate a problem. Monitoring answers the question: "Is something wrong?"

Observability is the property of a system that allows you to understand its internal state by examining its external outputs. An observable system produces enough telemetry -- metrics, logs, and traces -- that you can diagnose novel problems you did not anticipate. Observability answers the question: "Why is something wrong?"

For ML systems, the distinction is particularly important because many failure modes are not predictable in advance. You may not know which features will drift, how concept drift will manifest, or what novel data patterns will cause model degradation. A purely monitoring-based approach (predefined metrics with fixed thresholds) will miss unexpected failure modes. An observability approach that captures rich telemetry about inputs, outputs, feature values, and model behavior provides the raw material to investigate problems you did not foresee.

### The Three Pillars of Observability

Metrics are numeric measurements collected over time. They are efficient to store and query, support aggregation, and are the foundation of dashboards and alerts. Prometheus is a metrics system. For ML, metrics include request rates, latency percentiles, prediction score distributions, and drift statistics.

Logs are timestamped records of discrete events. They provide detailed context about individual requests or operations. For ML systems, prediction logs that record input features, model version, prediction output, and confidence scores are invaluable for debugging. Log aggregation systems like the ELK stack (Elasticsearch, Logstash, Kibana) or Grafana Loki handle log collection and search.

Traces track the flow of a request through multiple services or processing stages. For ML inference pipelines that involve feature retrieval, preprocessing, model inference, and postprocessing, traces reveal which stage is slow or failing. Distributed tracing systems like Jaeger or Grafana Tempo provide this capability.

A well-instrumented ML system produces all three signal types. Metrics tell you that latency increased. Logs tell you which requests were slow. Traces tell you which pipeline stage caused the slowdown.

### OpenTelemetry

OpenTelemetry (OTel) is a CNCF project that provides a vendor-neutral standard for instrumenting applications with metrics, logs, and traces. Rather than using separate libraries for each signal type and each backend, OpenTelemetry provides unified SDKs that can export to Prometheus (for metrics), Loki or Elasticsearch (for logs), and Jaeger or Tempo (for traces).

For ML systems, OpenTelemetry is increasingly relevant because it allows you to instrument once and export to multiple backends. The OpenTelemetry Collector can receive telemetry from applications and route it to the appropriate storage backends. Prometheus compatibility is built in, so teams can adopt OpenTelemetry instrumentation while continuing to use Prometheus as their metrics backend.

## The Monitoring Stack

The standard open-source monitoring stack for ML systems follows a clear pipeline: collect metrics, store them, visualize them, and alert on them. Each component has a specific role.

### Collection: Prometheus

Prometheus is the metrics collection and storage layer. It uses a pull-based architecture where the Prometheus server scrapes HTTP endpoints exposed by instrumented services at regular intervals. Services expose a `/metrics` endpoint that returns current metric values in a text format. Prometheus stores these as time series data, indexed by metric name and label key-value pairs.

For ML systems, Prometheus collects both infrastructure metrics (CPU, memory, GPU utilization via exporters) and application metrics (request rate, latency, prediction distributions via instrumented code). The `prometheus_client` Python library makes it straightforward to instrument ML serving code.

Prometheus provides PromQL, a powerful query language for analyzing time series data. PromQL supports rate calculations, histogram quantile computation, aggregation across dimensions, and mathematical operations on time series. Alerting rules are also written in PromQL and evaluated by the Prometheus server at regular intervals.

Key characteristics relevant to ML monitoring:

- Pull-based scraping works well for long-running inference services
- The Pushgateway handles short-lived batch jobs like training runs and evaluation pipelines
- Four metric types (counter, gauge, histogram, summary) cover the full range of ML monitoring needs
- Label-based data model naturally maps to ML concepts like model name, version, and predicted class
- Default 15-day retention is sufficient for operational monitoring but not for long-term model performance tracking

See the [Prometheus chapter](prometheus/ReadMe.md) for detailed coverage of architecture, PromQL, Python instrumentation, and scaling.

### Visualization: Grafana

Grafana is the visualization layer. It does not store data itself but connects to data sources like Prometheus and renders their data as dashboards. Grafana dashboards consist of panels, each displaying a visualization backed by a query against a data source.

For ML monitoring, Grafana provides:

- Time series panels for tracking prediction throughput, latency, and error rates over time
- Heatmaps for visualizing latency and prediction score distributions
- Stat panels for at-a-glance health indicators
- Table panels for per-feature drift status and model metadata
- Dashboard variables and templating for reusable, multi-model dashboards
- Annotations for marking deployments and retraining events on charts

Grafana also has built-in alerting that can route to Slack, PagerDuty, OpsGenie, email, and webhooks. However, for Prometheus-based monitoring, it is generally recommended to define alerting rules in Prometheus Alertmanager and use Grafana primarily for visualization.

Dashboards can be managed as code through JSON exports, Terraform providers, or Grafonnet (a Jsonnet library), enabling version control and automated provisioning.

See the [Grafana chapter](grafana/ReadMe.md) for detailed coverage of data sources, dashboard design, alerting configuration, and dashboard-as-code patterns.

### Alerting: PagerDuty and Opsgenie

Alerting systems are the action layer that turns detected problems into human responses. When Prometheus detects that a metric crosses a threshold, it sends an alert to Alertmanager, which deduplicates, groups, and routes alerts to notification channels. PagerDuty and Opsgenie are the two most widely used incident management platforms for this purpose.

These platforms provide capabilities beyond simple notification:

- On-call scheduling and rotation management
- Escalation policies that ensure alerts reach someone even if the primary responder does not acknowledge
- Incident tracking and lifecycle management
- Runbook integration for standardized response procedures
- Alert deduplication and intelligent grouping to reduce noise
- Analytics on alert volume, response times, and incident trends

For ML systems, alerting needs are layered. Infrastructure alerts (service down, GPU failure) should page immediately and go to platform engineers. Model quality alerts (drift detected, prediction distribution shift) may have different urgency levels and route to data science teams. Well-designed alert routing ensures that the right team receives the right alerts at the right urgency level.

See the [Alerting chapter](alerting/ReadMe.md) for detailed coverage of PagerDuty, Opsgenie, alert routing strategies, and ML-specific alerting patterns.

### Model Monitoring

Model monitoring is the layer that addresses the uniquely ML aspects of monitoring: data drift, concept drift, and performance degradation. This is where ML monitoring fundamentally diverges from traditional software monitoring.

Data drift occurs when the statistical distribution of input features changes relative to the training data distribution. A model trained on transactions averaging $50 may perform poorly when the average shifts to $500. Drift detection techniques like Population Stability Index (PSI), Kolmogorov-Smirnov tests, and Jensen-Shannon divergence quantify how much feature distributions have shifted.

Concept drift occurs when the relationship between input features and the target variable changes, even if feature distributions remain stable. The same transaction patterns that were once legitimate may become fraudulent as fraud tactics evolve. Concept drift is harder to detect because it requires ground truth labels.

Performance degradation is the downstream consequence of drift or other changes: the model's accuracy, precision, recall, or other task-specific metrics decline over time. Detecting performance degradation directly requires ground truth labels, which may be delayed.

Model monitoring tools like Evidently, Whylabs, and Arize provide specialized drift detection, visualization, and alerting. These can integrate with Prometheus and Grafana -- exposing drift statistics as Prometheus metrics that appear on Grafana dashboards alongside operational metrics.

See the [Model Monitoring chapter](model-monitoring/ReadMe.md) for detailed coverage of data drift detection, concept drift, and performance degradation.

## What to Monitor in ML Systems

### Infrastructure Metrics

These are standard metrics that apply to any production service:

- CPU utilization and memory usage on serving hosts
- GPU utilization and GPU memory usage for GPU-accelerated inference
- Disk I/O and available storage
- Network throughput and error rates
- Container health and restart counts (in Kubernetes: pod status, OOMKilled events)
- Node availability and resource allocation

### Application Metrics

These track the operational behavior of the inference service:

- Request rate (total predictions per second, by model and version)
- Inference latency (p50, p95, p99 broken down by model)
- Error rate (by error type: validation errors, model errors, timeout errors)
- Requests in progress (concurrency/queue depth)
- Model load time on startup or after hot-reload
- Payload sizes (input and output)

### Model Quality Metrics

These are specific to ML and track whether the model is producing good predictions:

- Prediction score/confidence distributions (tracked as histograms over time)
- Predicted class distributions (the ratio of each predicted class should remain roughly stable unless the real world has changed)
- Feature value distributions (per-feature mean, standard deviation, null rate, out-of-range counts)
- Drift statistics (PSI, KS statistic, JS divergence per feature and for the overall input distribution)
- Ground truth metrics when labels are available (accuracy, precision, recall, F1, AUC, MAE, RMSE depending on task)
- Prediction-label agreement rate for online evaluation

### Data Pipeline Metrics

These track the health of data flowing into the model:

- Feature freshness (time since last update for each feature or feature group)
- Feature completeness (null rates, missing value rates)
- Schema validation failures (unexpected types, new categorical values)
- Data volume (number of records processed per interval)
- Feature store latency (time to retrieve features for a prediction)

### Business Metrics

These connect model predictions to business outcomes:

- Revenue impact of model decisions (approvals, rejections, recommendations)
- User engagement metrics influenced by model outputs
- False positive/negative costs
- Manual review rates (what fraction of predictions require human review)

Business metrics often cannot be tracked in Prometheus directly but can be computed in batch and pushed via the Pushgateway or tracked in a separate analytics system and correlated with model metrics in Grafana using mixed data sources.

## Architecture Patterns

### Basic Stack

The simplest production monitoring setup:

```
ML Service (instrumented with prometheus_client)
    |
    v
Prometheus (scrapes /metrics endpoint)
    |
    v
Grafana (queries Prometheus, renders dashboards)
    |
    v
Alertmanager -> PagerDuty/Opsgenie (routes alerts)
```

This covers infrastructure and application metrics well. For model-specific monitoring, you add a drift detection service that computes drift statistics and exposes them as Prometheus metrics, feeding into the same pipeline.

### Extended Stack with Model Monitoring

```
ML Service
    |-- /metrics --> Prometheus --> Grafana
    |-- prediction logs --> Log Store (Elasticsearch/Loki)
    |
Feature Store --> Drift Detection Service
    |                    |
    v                    v
Training Data       Prometheus (drift metrics)
Reference                |
                         v
                    Grafana (drift dashboard)
                         |
                         v
                    Alertmanager -> PagerDuty/Opsgenie
```

This pattern separates operational monitoring (direct from the ML service) from model quality monitoring (computed by a dedicated drift detection service). The drift detection service compares current feature distributions against reference distributions from training data and exposes the results as Prometheus metrics.

### Enterprise Pattern with Dedicated Model Monitoring Platform

```
ML Service --> Prometheus --> Grafana (operational dashboards)
    |
    v
Prediction Logger --> Model Monitoring Platform
                      (Evidently / Arize / Whylabs)
                           |
                           v
                      Platform Dashboards + Alerts
                           |
                           v
                      PagerDuty/Opsgenie
```

Larger organizations often adopt dedicated model monitoring platforms that provide built-in drift detection, root cause analysis, and model-specific visualization. These platforms complement rather than replace the Prometheus/Grafana stack. Operational metrics stay in Prometheus/Grafana. Model quality metrics flow through the specialized platform. Both feed into the same alerting infrastructure.

## Designing a Monitoring Strategy

### Start with SLOs

Service Level Objectives define what "good enough" means for your system. Before instrumenting anything, define your SLOs:

- Availability: The model serving endpoint should be available 99.9% of the time.
- Latency: 95% of predictions should complete within 100ms; 99% within 200ms.
- Accuracy: Model accuracy should remain above 92% (measured via delayed ground truth).
- Freshness: Features should be no more than 5 minutes stale.

SLOs drive what you monitor and what thresholds you set for alerts. Without SLOs, alert thresholds are arbitrary and either too noisy (alerting on insignificant fluctuations) or too lenient (missing real problems).

### Alert Design Principles

Not every metric needs an alert. Effective alert design follows several principles:

Alert on symptoms, not causes. Alert on "prediction error rate is above 5%" rather than "CPU usage is above 80%." High CPU usage is not necessarily a problem; high error rate always is.

Require sustained conditions. Use the `for` clause in Prometheus alerting rules to require a condition to persist before firing. A latency spike lasting 30 seconds may be normal; one lasting 10 minutes is not.

Route alerts appropriately. Infrastructure alerts go to platform engineers. Model quality alerts go to data scientists. Business metric alerts may go to product managers.

Include context. Alert annotations should include the current metric value, the threshold, a link to the relevant Grafana dashboard, and a link to the runbook.

Reduce noise aggressively. Every noisy alert that gets ignored trains the team to ignore all alerts. If an alert fires frequently and is routinely dismissed, either fix the underlying issue or adjust the threshold.

### Monitoring Lifecycle

ML monitoring is not a one-time setup. It evolves with the system:

Day one: Instrument the service with basic operational metrics (request rate, latency, errors). Set up Prometheus scraping and a basic Grafana dashboard. Configure critical alerts (service down, high error rate).

Week one: Add model-specific metrics (prediction distributions, feature statistics). Build model performance dashboards. Configure drift detection.

Month one: Tune alert thresholds based on observed baseline behavior. Add recording rules for expensive queries. Build per-model dashboards using variables and templating.

Ongoing: Review alert noise regularly. Update drift detection baselines after each model retrain. Add monitoring for new models. Archive dashboards for retired models. Conduct post-incident reviews to identify monitoring gaps.

## Comparison of Child Chapters

The four child chapters cover distinct components of the monitoring stack. Each addresses a specific layer, and together they form a complete monitoring solution for ML systems.

### Prometheus

Covers metrics collection and storage. Topics include Prometheus architecture (pull-based scraping, time series database), the four metric types (counter, gauge, histogram, summary), PromQL query language, Python instrumentation with `prometheus_client`, alerting rules, service discovery in Kubernetes, scaling with federation and remote storage (Thanos, Mimir), and ML-specific metrics instrumentation patterns.

Use Prometheus when you need to understand how to collect and query metrics from your ML services.

### Grafana

Covers visualization and dashboard design. Topics include data source configuration, panel types relevant to ML (time series, heatmaps, histograms, stat panels, tables), dashboard variables and templating for multi-model dashboards, annotation for deployment tracking, dashboard-as-code patterns (JSON, Terraform, Grafonnet), and Grafana alerting configuration.

Use Grafana when you need to understand how to build effective dashboards for ML services or manage dashboard infrastructure as code.

### Model Monitoring

Covers the ML-specific monitoring concerns that distinguish ML monitoring from traditional software monitoring. Subtopics include data drift detection (PSI, KS test, JS divergence, feature-level and dataset-level drift), concept drift (types, detection methods, relationship to ground truth), and performance degradation (tracking accuracy metrics over time, proxy metrics when ground truth is delayed).

Use Model Monitoring when you need to understand why your model's predictions may be degrading and how to detect it.

### Alerting

Covers incident management and alert routing. Subtopics include PagerDuty (on-call scheduling, escalation policies, incident lifecycle) and Opsgenie (similar capabilities with different integration patterns). Topics span alert routing strategies, severity classification for ML alerts, runbook design, and reducing alert fatigue.

Use Alerting when you need to understand how to route detected problems to the right people with the right urgency.

## Chapter Overview

### [Prometheus](prometheus/ReadMe.md)

Metrics collection and storage for ML systems:
- Pull-based architecture and scrape configuration
- Four metric types and when to use each
- PromQL for querying and alerting rules
- Instrumenting Python ML services
- Scaling with federation, Thanos, and Mimir

### [Grafana](grafana/ReadMe.md)

Visualization and dashboarding:
- Data source configuration and mixed-source dashboards
- Panel types for ML monitoring (time series, heatmaps, histograms, tables)
- Variables and templating for multi-model dashboards
- Dashboard-as-code with JSON, Terraform, and Grafonnet
- Grafana Cloud vs self-hosted deployment

### [Model Monitoring](model-monitoring/ReadMe.md)

ML-specific monitoring concerns:
- [Data Drift Detection](model-monitoring/data-drift-detection/ReadMe.md): Detecting changes in input feature distributions
- [Concept Drift](model-monitoring/concept-drift/ReadMe.md): Detecting changes in the feature-target relationship
- [Performance Degradation](model-monitoring/performance-degradation/ReadMe.md): Tracking model accuracy and quality over time

### [Alerting](alerting/ReadMe.md)

Incident management and notification routing:
- [PagerDuty](alerting/pagerduty/ReadMe.md): On-call management and incident response
- [Opsgenie](alerting/opsgenie/ReadMe.md): Alert routing and escalation policies
