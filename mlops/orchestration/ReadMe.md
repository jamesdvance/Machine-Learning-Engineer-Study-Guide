# Orchestration

## Summary

Workflow orchestration is the backbone of production ML systems. While a single training script can run end-to-end on a laptop, production ML involves multi-step pipelines spanning data ingestion, validation, feature engineering, training, evaluation, model registration, deployment, and monitoring. These steps have dependencies, require different compute resources, fail in different ways, and must run on schedules or in response to events. An orchestrator manages the execution order, handles failures, provides observability, and tracks artifacts across these steps so that engineers can reason about pipelines as coherent systems rather than collections of disconnected scripts.

The orchestration landscape for ML has matured considerably. Apache Airflow established the directed acyclic graph (DAG) as the dominant paradigm for expressing workflow dependencies. Newer tools like Prefect, Dagster, and Argo Workflows have introduced alternative abstractions -- code-first decorators, software-defined assets, and container-native execution -- that address limitations teams encountered with first-generation orchestrators. Choosing the right tool depends on your infrastructure, team skills, pipeline complexity, and whether you think primarily in terms of tasks, data assets, or containerized steps.

Key points to remember:

- ML pipelines differ from data engineering pipelines: they involve heterogeneous compute (CPU, GPU, TPU), long-running tasks, large artifacts, conditional logic based on metrics, and feedback loops between training and serving
- The DAG paradigm models workflows as nodes (tasks) with directed edges (dependencies) and no cycles, ensuring a deterministic execution order
- Core orchestration concerns include scheduling, dependency resolution, retries with backoff, artifact management, observability, and resource allocation
- Airflow is the most widely deployed orchestrator with the largest ecosystem; Prefect offers a Python-native, dynamic-first alternative; Dagster introduces asset-centric orchestration with built-in lineage; Argo Workflows provides Kubernetes-native container orchestration
- The choice of orchestrator is rarely about raw capability -- most can run your pipeline -- but about which abstraction matches your team's mental model and infrastructure

## Child Chapters

This chapter provides the conceptual foundation and comparative framework. Detailed coverage of each orchestrator is in the following child chapters:

- [Airflow](./airflow/ReadMe.md) -- The industry standard for batch workflow orchestration, with Python-defined DAGs, a massive operator ecosystem, and managed offerings on every major cloud
- [Argo Workflows](./argo/ReadMe.md) -- Kubernetes-native orchestration where every step runs in its own container, defined in YAML as Kubernetes custom resources
- [Prefect](./prefect/ReadMe.md) -- Code-first Python orchestration with decorator-based flows, dynamic task generation, and flexible infrastructure through work pools
- [Dagster](./dagster/ReadMe.md) -- Asset-oriented orchestration that models pipelines around the data objects they produce, with automatic lineage tracking and built-in data validation

## Why ML Pipelines Need Orchestration

A production ML system is not a single script. It is a collection of interdependent steps that must execute reliably, repeatedly, and observably. Consider what a typical model retraining pipeline involves:

1. Data extraction from one or more source systems (warehouses, lakes, APIs)
2. Data validation to confirm schema, volume, and distribution expectations
3. Feature engineering to transform raw data into model inputs
4. Data splitting for train, validation, and test sets
5. Model training, potentially across multiple configurations or hyperparameter sets
6. Model evaluation against held-out data and comparison to the current production model
7. Conditional promotion: register the model only if it meets quality thresholds
8. Deployment to a serving infrastructure (API endpoint, batch inference job)
9. Post-deployment smoke tests and monitoring setup

Each step can fail independently. Data sources go down. Feature engineering code has bugs. Training runs out of memory. Evaluation reveals regression. Without orchestration, you are left with cron jobs chained by file system signals, manual reruns after failures, and no unified view of what ran, when, and why it failed.

Orchestration solves several specific problems:

**Dependency management.** Steps must execute in the correct order. Training cannot start before feature engineering completes. Deployment should not proceed if evaluation fails a quality gate. An orchestrator encodes these dependencies explicitly and enforces them at runtime.

**Failure handling.** Transient failures are common in distributed systems: network timeouts, spot instance preemption, rate-limited APIs. Orchestrators provide configurable retry policies with exponential backoff, so a temporary S3 outage does not require manual intervention.

**Scheduling.** Retraining pipelines run on schedules (nightly, weekly) or in response to events (new data arrival, drift detection). Orchestrators provide cron-based scheduling, event-driven triggers, or both.

**Resource allocation.** Different pipeline steps need different resources. Data validation runs on a small CPU instance. Training requires GPUs. Feature engineering needs high memory. Orchestrators route tasks to appropriate compute, either through Kubernetes resource requests, cloud-specific operators, or infrastructure configuration.

**Observability.** When a pipeline fails at 3 AM, you need to know which step failed, what inputs it received, how long it ran, and what logs it produced. Orchestration UIs provide DAG visualization, task-level logs, timing analysis, and historical run comparisons.

**Artifact tracking.** ML pipelines produce intermediate and final artifacts: processed datasets, trained model files, evaluation reports, feature importance plots. Orchestrators manage the flow of these artifacts between steps and, in some cases, provide built-in artifact storage and versioning.

**Reproducibility.** Given the same code version and input data, the pipeline should produce the same results. Orchestrators contribute to reproducibility by fixing execution order, recording parameters, and (in container-based systems like Argo) isolating each step's runtime environment.

## The DAG Paradigm

The directed acyclic graph is the foundational abstraction in workflow orchestration. A DAG consists of nodes representing computational tasks and directed edges representing dependencies between them. The "acyclic" constraint means there are no circular dependencies: if task A depends on task B, then B cannot also depend on A, directly or transitively. This constraint guarantees that there exists at least one valid execution order (a topological sort) for the tasks.

In an ML pipeline DAG, nodes typically represent operations like "extract data," "compute features," "train model," and "evaluate." Edges encode the data and execution dependencies: training depends on feature engineering, which depends on data extraction. The orchestrator walks the DAG, executing tasks whose upstream dependencies are satisfied, potentially running independent tasks in parallel.

### Strengths of the DAG Model

The DAG model is intuitive for batch pipelines. Most ML workflows are naturally acyclic: data flows from raw sources through transformations to model artifacts. The DAG provides a clear visual representation of the pipeline, makes parallel execution opportunities obvious (independent branches), and enables the orchestrator to determine the minimal set of tasks to re-execute after a failure.

### Limitations and Alternatives

The static DAG model has known limitations that newer orchestrators address in different ways:

**Dynamic workflows.** In some pipelines, the number of tasks is not known until runtime. A hyperparameter sweep might generate a variable number of training runs based on a search space. Airflow addressed this with dynamic task mapping in version 2.3. Prefect handles it naturally because the execution graph is determined at runtime from Python control flow. Argo supports fan-out with `withItems` and `withParam`.

**Feedback loops.** Reinforcement learning training loops, active learning cycles, and iterative optimization do not fit the acyclic constraint. These are typically handled by embedding the loop within a single task node or by having the orchestrator trigger new pipeline runs based on the output of previous ones.

**Asset-centric thinking.** The DAG models tasks (what to do), but some teams find it more natural to model data (what should exist). Dagster's software-defined asset model is an explicit alternative to the task DAG. Instead of defining "run preprocessing then training," you define "the feature table depends on raw data, the trained model depends on the feature table." The execution graph is derived from the data dependencies.

**Event-driven execution.** Pure DAG orchestration is pull-based: the scheduler checks whether it is time to run. Event-driven patterns (trigger retraining when new data arrives, when drift is detected, when a model is approved) require additional mechanisms. Airflow uses sensors, Prefect uses automations, Dagster uses sensors, and Argo pairs with Argo Events.

## Key Orchestration Concerns

### Scheduling

Most ML pipelines run on time-based schedules: daily feature engineering, weekly retraining, monthly model audits. All four orchestrators support cron-based scheduling. Beyond cron, event-driven triggers are increasingly important for ML: retraining when new data lands, when monitoring detects drift, or when an upstream pipeline completes.

Airflow has the most mature scheduler with support for timetables, data-aware scheduling, and complex schedule expressions. Dagster provides both schedules and sensors, with sensors polling for external conditions. Prefect supports cron schedules on deployments and automations for event-driven triggers. Argo uses CronWorkflow resources for scheduled execution and Argo Events for event-driven triggering.

### Retries and Failure Handling

ML pipelines interact with external systems that fail transiently. A good retry strategy includes configurable retry counts, exponential backoff, and the ability to distinguish transient failures (network timeouts, spot preemption) from permanent failures (invalid data, code bugs). All four orchestrators support task-level retries with backoff:

- Airflow: `retries` and `retry_delay` in default_args or per-task
- Prefect: `retries` and `retry_delay_seconds` on `@task` and `@flow` decorators
- Dagster: retry policies on ops and assets
- Argo: `retryStrategy` with `retryPolicy`, `limit`, and `backoff` configuration

Beyond retries, orchestrators differ in how they handle partial failures. Airflow can retry individual tasks within a DAG run. Argo can retry a workflow from the point of failure. Prefect tracks task states and allows flows to branch based on failure. Dagster can re-materialize specific assets without re-running the entire pipeline.

### Dependency Resolution

Dependencies come in several forms:

- **Intra-pipeline dependencies:** Task B waits for Task A within the same pipeline run. All orchestrators handle this through DAG edges or implicit data dependencies.
- **Cross-pipeline dependencies:** A training pipeline depends on a feature engineering pipeline completing. Airflow uses ExternalTaskSensor. Dagster handles this natively through asset dependencies that can span code locations. Prefect can use automations triggered by flow completion events.
- **Data dependencies:** A task depends on specific data being available, not just a prior task completing. Dagster's asset model handles this most naturally. Airflow's data-aware scheduling (introduced in 2.4) and sensors provide alternatives.

### Observability

Observability in orchestration means answering questions like: What is running right now? What failed last night? How long does the training step typically take? Is this run's duration abnormal?

All four tools provide web UIs with DAG visualization, run history, and task-level logs. They differ in depth:

- Airflow provides Gantt charts for timing analysis, task instance history, and log viewing. Its UI is mature and well understood.
- Argo Server shows the DAG visually with pod-level logs, artifact inspection, and real-time workflow watching.
- Prefect's UI focuses on flow runs, task states, and timeline views. Prefect Cloud adds automations for alerting.
- Dagster's UI (Dagit) stands apart by showing the asset graph with materialization status, staleness indicators, and asset check results, reflecting its data-centric philosophy.

For production alerting, all four integrate with external notification systems (Slack, PagerDuty, email), though the integration depth varies. Dagster+ and Prefect Cloud have the most built-in alerting capabilities.

### Artifact Management

ML pipelines produce large artifacts: datasets, model checkpoints, evaluation reports, feature importance files. Orchestrators handle artifacts differently:

- **Argo** has first-class artifact support. Each step declares input and output artifacts with storage locations (S3, GCS, HDFS). Artifacts flow between steps explicitly and are viewable in the Argo Server UI.
- **Airflow** uses XComs for small data (metadata, paths, metrics) and expects large artifacts to be managed externally (S3, GCS) with paths passed through XComs. The XCom backend can be customized for larger payloads.
- **Prefect** uses result storage and blocks. Tasks can persist results to configured storage backends. Large artifacts are typically stored externally with references passed between tasks.
- **Dagster** handles artifacts through IO managers, which abstract the serialization and storage layer. Each asset's output is automatically persisted by its assigned IO manager, and upstream assets are loaded transparently. This is the most opinionated and integrated approach.

### Resource Management

ML workloads have diverse resource requirements. Orchestrators vary in how they manage compute allocation:

- **Argo** excels here because each step is a Kubernetes pod with explicit resource requests and limits. GPU scheduling, node affinity, and tolerations are native Kubernetes concepts that Argo exposes directly.
- **Airflow** with KubernetesPodOperator provides similar per-task resource control. The KubernetesExecutor launches each task as a pod. With CeleryExecutor, tasks share worker resources.
- **Prefect** delegates resource management to the infrastructure layer (work pools). A Kubernetes work pool can specify resource requirements per deployment or per flow.
- **Dagster** with its Kubernetes executor launches each step as a Kubernetes job with configurable resources. The resource abstraction also manages connections and external services.

## Comparing the Four Orchestrators

### At a Glance

| Dimension | Airflow | Argo Workflows | Prefect | Dagster |
|---|---|---|---|---|
| Core abstraction | Task (Operator) | Container step | Python function (Task/Flow) | Software-defined asset |
| Definition language | Python | YAML | Python | Python |
| Infrastructure | Any (VM, K8s, managed) | Kubernetes only | Any (local, Docker, K8s, serverless) | Any (local, Docker, K8s) |
| Scheduling | Built-in, mature | CronWorkflow + Argo Events | Cron on deployments, automations | Schedules + sensors |
| Dynamic workflows | Dynamic task mapping (2.3+) | withItems/withParam fan-out | Native Python control flow | Partitions, dynamic partitions |
| Data lineage | Not built-in | Not built-in | Not built-in | First-class (asset graph) |
| Artifact handling | XComs (small), external storage | First-class (S3, GCS) | Result storage, blocks | IO managers |
| Testing | Requires mocking Airflow context | Requires cluster or mocks | Functions callable directly | Functions callable directly, resource mocking |
| Managed offerings | MWAA, Cloud Composer, Astronomer | Limited | Prefect Cloud | Dagster+ |
| Ecosystem size | Largest (1000+ operators) | Kubernetes ecosystem | Growing | Growing |
| Container isolation | KubernetesPodOperator | Default (every step) | Via work pools | Via Kubernetes executor |
| Learning curve | Moderate | Low-moderate (high if new to K8s) | Low | Moderate |

### Workflow Definition Philosophy

The four tools represent three distinct philosophies for defining workflows:

**Task-centric (Airflow, Prefect).** You define what to do (tasks) and in what order (dependencies). The orchestrator executes tasks and tracks their state. Airflow uses operators within a DAG context manager; Prefect uses decorated Python functions with implicit dependencies determined at runtime.

**Container-centric (Argo).** You define what containers to run and how they connect. Each step is a container image with explicit inputs and outputs. The orchestrator manages pod lifecycle, artifact transfer, and dependency resolution. Workflow definitions are declarative YAML, not imperative code.

**Asset-centric (Dagster).** You define what data should exist and how each piece derives from other data. The orchestrator materializes assets in dependency order, tracks lineage, and detects staleness. The execution graph is a consequence of data dependencies, not something you define directly.

### Operational Complexity

The operational burden of running each tool varies significantly:

**Airflow** requires a scheduler, webserver, metadata database, and workers. In production, this typically means PostgreSQL, Redis (for Celery), and multiple worker processes or pods. Managed services (MWAA, Cloud Composer, Astronomer) eliminate most of this burden.

**Argo** requires a Kubernetes cluster with the Argo controller and optionally the Argo Server. If you already run Kubernetes, the incremental operational cost is low. If you do not, standing up Kubernetes for orchestration alone is a significant investment.

**Prefect** requires a server (self-hosted or Prefect Cloud) and workers in your infrastructure. The server is a single process with a database. Prefect Cloud eliminates the server management entirely. Workers are lightweight polling processes.

**Dagster** requires a webserver (Dagit), daemon, and database. Similar to Airflow in component count but generally considered simpler to operate. Dagster+ Serverless mode eliminates infrastructure management entirely.

### Integration Ecosystem

Airflow has the largest integration ecosystem by a wide margin, with over 1000 operators across AWS, GCP, Azure, Snowflake, dbt, Spark, and hundreds of other systems. This ecosystem is Airflow's strongest moat.

Argo integrates naturally with the Kubernetes ecosystem: KServe, Kubeflow, Seldon, Spark on Kubernetes, and any containerized workload. Its integration model is "if it runs in a container, Argo can orchestrate it."

Prefect and Dagster have growing integration libraries (prefect-aws, prefect-dbt, dagster-dbt, dagster-snowflake, etc.) but neither matches Airflow's breadth. Both compensate by making it easy to call any Python library directly from tasks or assets.

## Decision Framework for Choosing an Orchestrator

Choosing an orchestrator is a significant architectural decision. The following framework organizes the decision around the factors that matter most in practice.

### Start with Infrastructure

If your team already operates Kubernetes and your ML workloads run in containers, Argo Workflows is the natural starting point. It requires no additional infrastructure and gives you container isolation, GPU scheduling, and artifact management natively.

If you do not run Kubernetes, eliminate Argo from consideration unless you are willing to adopt Kubernetes as part of this decision.

### Consider Your Team's Mental Model

How does your team think about pipelines?

- If you think in terms of "run this, then run that" -- task-centric tools (Airflow, Prefect) match your mental model.
- If you think in terms of "this data depends on that data" -- Dagster's asset model will feel natural.
- If you think in terms of "run these containers in this order" -- Argo's container-centric model fits.

### Evaluate Ecosystem Needs

If your pipelines need to coordinate across many external systems (cloud services, databases, SaaS APIs), Airflow's operator ecosystem is a genuine advantage. Writing custom integrations for every service in Prefect or Dagster is possible but adds development cost.

If your pipelines are primarily Python code interacting with a few well-known services (S3, a database, an experiment tracker), the ecosystem difference matters less.

### Assess Operational Appetite

- **Minimal operations:** Prefect Cloud or Dagster+ Serverless. Managed orchestration with no infrastructure to run.
- **Managed but configurable:** MWAA, Cloud Composer, or Astronomer for Airflow. Dagster+ Hybrid.
- **Self-managed:** Any of the four, with Argo being the simplest if you already have Kubernetes, and Dagster or Prefect being simpler than Airflow for self-hosted deployments.

### Factor in Team Skills

- Python-heavy data science teams with limited infrastructure experience: Prefect or Dagster.
- Platform engineering teams comfortable with Kubernetes: Argo.
- Teams with existing Airflow expertise and DAGs: Airflow (migration cost is real).
- Teams starting from scratch with strong Python skills: Prefect for simplicity, Dagster for data-awareness.

### Decision Summary

| If your priority is... | Consider... |
|---|---|
| Breadth of integrations across cloud services | Airflow |
| Kubernetes-native container orchestration | Argo Workflows |
| Python-native simplicity and dynamic workflows | Prefect |
| Data lineage and asset lifecycle management | Dagster |
| Managed service with minimal ops | Prefect Cloud or Dagster+ |
| Existing Airflow investment | Stay with Airflow |
| GPU-heavy training on Kubernetes | Argo or Airflow with KubernetesPodOperator |

## Other Orchestration Tools

While this guide focuses on the four most relevant orchestrators for ML, several others deserve mention:

**Luigi.** Developed at Spotify, Luigi was one of the earliest Python workflow frameworks. It introduced the concept of targets (output artifacts) that determine whether a task needs to run. Luigi is largely superseded by Airflow and newer tools but still appears in legacy systems. It lacks a built-in scheduler, has no web UI for management (only monitoring), and has a smaller community than Airflow.

**AWS Step Functions.** A serverless orchestrator from AWS that defines workflows as state machines in JSON (Amazon States Language). Step Functions integrates deeply with AWS services (Lambda, ECS, SageMaker, Glue) and is a strong choice for AWS-native ML pipelines that do not need portability. Its pay-per-transition pricing and tight IAM integration make it attractive for serverless architectures. The visual workflow editor and built-in error handling are polished. The limitation is vendor lock-in: Step Functions workflows are not portable to other clouds or on-premise environments.

**Flyte.** Developed at Lyft, Flyte is a Kubernetes-native workflow automation platform designed specifically for ML and data workloads. It uses a typed Python SDK where tasks are containerized and the type system enforces data contracts between steps. Flyte's distinguishing features include built-in data versioning, strong typing across task boundaries, and a caching mechanism tied to input hashes. It sits somewhere between Argo (container-native, Kubernetes-first) and Prefect (Python SDK, ML-focused). Union.ai offers a managed Flyte service. Flyte is worth evaluating if you want Kubernetes-native execution with a Python SDK and stronger typing than Argo provides.

**Kubeflow Pipelines.** A component of the Kubeflow ML platform that historically used Argo Workflows as its execution backend (KFP v2 introduced its own backend). KFP provides a Python SDK for defining pipelines that compile to Argo Workflow YAML. It adds ML-specific features like experiment tracking, pipeline versioning, and a metadata store. KFP is most relevant if you are already using the broader Kubeflow platform; otherwise, using Argo directly is simpler.

**SageMaker Pipelines and Vertex AI Pipelines.** Cloud-native ML pipeline services from AWS and GCP, respectively. These are tightly integrated with their cloud platforms' training, serving, and feature store offerings. They are the right choice when you are fully committed to a single cloud's ML platform and want the tightest integration, but they offer limited portability.

**Mage.** A newer orchestration tool aimed at data engineering with a notebook-like development experience. It supports Python, SQL, and R within a visual pipeline editor. Less established than the big four but gaining traction for teams that want a more interactive development experience.

**Temporal.** A workflow engine for long-running, stateful applications. Temporal is not ML-specific but is used by some teams for ML pipelines that require complex failure handling, human-in-the-loop approval steps, or workflows that span days or weeks. Its programming model (workflow and activity functions with durable execution) differs from the DAG paradigm.

## Patterns Across Orchestrators

Regardless of which orchestrator you choose, certain patterns recur in ML pipeline design:

**Idempotency.** Every pipeline step should be safe to retry without side effects. Write outputs to partitioned locations that overwrite on re-execution. Use upserts instead of inserts. This is not an orchestrator feature but a design discipline that orchestration makes critical because retries are automatic.

**Small metadata, external artifacts.** Pass file paths, URIs, and metrics between tasks, not large data objects. Every orchestrator has limits on inter-task data size (Airflow XComs in the metadata database, Argo parameters in etcd, Prefect results in the server database). Store actual data in object storage and pass references.

**Quality gates.** Insert evaluation steps that check data quality or model performance before proceeding. Use conditional execution (Airflow branching, Argo `when` clauses, Prefect `if` statements, Dagster asset checks) to prevent bad data or models from propagating downstream.

**Timeouts on every step.** ML tasks can hang: training loops that do not converge, data sources that stop responding, GPU deadlocks. Set explicit timeouts on every task to prevent zombie runs from consuming resources indefinitely.

**Separation of concerns.** Keep data pipelines, training pipelines, and deployment pipelines as separate orchestrated workflows that communicate through well-defined interfaces (artifact stores, model registries, feature stores) rather than monolithic mega-pipelines. This improves maintainability, allows independent scheduling, and limits blast radius when things fail.
