# Argo Workflows

## Summary

Argo Workflows is a Kubernetes-native workflow orchestration engine for running complex, multi-step pipelines as directed acyclic graphs (DAGs) on Kubernetes. Each step in an Argo workflow runs inside its own container, which means every task gets full isolation, its own dependencies, and reproducible execution. For ML engineers, Argo provides a natural way to orchestrate training pipelines, hyperparameter sweeps, model evaluation, and deployment steps directly on the same Kubernetes infrastructure that often already hosts model serving workloads.

Argo Workflows is a graduated project of the Cloud Native Computing Foundation (CNCF), which speaks to its maturity and community support. It is often used alongside Argo Events (for event-driven triggering), Argo CD (for GitOps-based deployment), and Argo Rollouts (for progressive delivery), though each is an independent project that can be adopted separately.

Key points to remember:

- Kubernetes-native: workflows are defined as Custom Resource Definitions (CRDs), managed by kubectl
- Each step runs in its own container with full isolation and explicit resource requests
- DAG and step-based execution models for expressing complex dependency graphs
- First-class support for artifacts (S3, GCS, HDFS, Git) and parameter passing between steps
- Argo Events enables event-driven workflow triggering from webhooks, message queues, schedules, and more
- Argo Server provides a web UI for monitoring, managing, and debugging workflows
- Compared to Airflow: Argo is container-first and Kubernetes-native; Airflow is Python-first with broader scheduler options
- Best suited for teams already on Kubernetes who want containerized, reproducible pipeline steps
- Workflow templates enable reusable, parameterized pipeline definitions across teams
- Supports retry strategies, timeouts, conditional execution, and resource-aware scheduling

## Architecture

### Core Components

Argo Workflows consists of three main components that run inside your Kubernetes cluster:

```
Workflow CRDs (Custom Resource Definitions)
   Workflow         - a single pipeline execution
   WorkflowTemplate - reusable workflow definition
   CronWorkflow     - scheduled workflow execution
   ClusterWorkflowTemplate - cluster-wide reusable template

Workflow Controller
   Watches for Workflow CRDs
   Schedules pods for each step
   Manages step dependencies and data flow
   Handles retries, timeouts, and failure policies
   Runs as a Kubernetes Deployment

Argo Server
   Web UI for workflow visualization
   REST API for programmatic access
   SSO integration for authentication
   Artifact viewing and log streaming
   Runs as a Kubernetes Deployment + Service
```

The workflow controller is the engine. It watches the Kubernetes API server for Workflow custom resources, creates pods for each step, monitors their completion, and orchestrates the overall execution. The Argo Server is optional but provides the web UI and API that most teams find indispensable for debugging.

### How a Workflow Executes

When you submit a workflow (via `argo submit` or `kubectl apply`), the following happens:

1. The Workflow CRD is created in the Kubernetes API server
2. The workflow controller detects the new resource
3. For each ready step (no unmet dependencies), the controller creates a pod
4. An init container or sidecar (the "wait" container) handles artifact collection and parameter output
5. The main container runs your step logic
6. On completion, artifacts are saved to configured storage (S3, GCS, etc.)
7. Output parameters are captured and made available to downstream steps
8. The controller evaluates which steps are now ready and repeats

Each workflow gets a unique name and all its pods are labeled for easy identification. The controller maintains the workflow status in the CRD itself, so the state is always queryable via the Kubernetes API.

### Installation

```bash
# Create namespace
kubectl create namespace argo

# Install Argo Workflows (cluster-wide)
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

# Or namespace-scoped installation
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/namespace-install.yaml

# Install the Argo CLI
# macOS
brew install argo

# Linux
curl -sLO https://github.com/argoproj/argo-workflows/releases/latest/download/argo-linux-amd64.gz
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
mv argo-linux-amd64 /usr/local/bin/argo

# Verify
argo version
```

## Workflow Specification

### Templates

Templates are the fundamental building blocks of Argo workflows. There are several template types:

```yaml
# Container template - runs a single container
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: hello-world-
spec:
  entrypoint: say-hello
  templates:
  - name: say-hello
    container:
      image: alpine:3.18
      command: [echo]
      args: ["Hello from Argo"]
```

```yaml
# Script template - inline script execution
templates:
- name: generate-data
  script:
    image: python:3.11-slim
    command: [python]
    source: |
      import json
      import random
      data = [random.gauss(0, 1) for _ in range(100)]
      print(json.dumps({"mean": sum(data)/len(data)}))
```

```yaml
# Resource template - create/patch Kubernetes resources
templates:
- name: create-training-job
  resource:
    action: create
    manifest: |
      apiVersion: batch/v1
      kind: Job
      metadata:
        name: training-job
      spec:
        template:
          spec:
            containers:
            - name: trainer
              image: myorg/trainer:v1
              resources:
                limits:
                  nvidia.com/gpu: 1
```

### Steps vs DAGs

Argo offers two ways to express dependencies between templates: steps (sequential/parallel groups) and DAGs (explicit dependency graph).

```yaml
# Steps: sequential groups, parallel within groups
spec:
  entrypoint: ml-pipeline
  templates:
  - name: ml-pipeline
    steps:
    - - name: preprocess        # Step group 1 (runs first)
        template: preprocess-data
    - - name: train-model-a     # Step group 2 (runs in parallel)
        template: train
        arguments:
          parameters:
          - name: model-type
            value: "xgboost"
      - name: train-model-b
        template: train
        arguments:
          parameters:
          - name: model-type
            value: "lightgbm"
    - - name: evaluate           # Step group 3 (runs after group 2)
        template: evaluate-models
```

```yaml
# DAG: explicit dependency declarations
spec:
  entrypoint: ml-pipeline
  templates:
  - name: ml-pipeline
    dag:
      tasks:
      - name: preprocess
        template: preprocess-data
      - name: train-xgboost
        dependencies: [preprocess]
        template: train
        arguments:
          parameters:
          - name: model-type
            value: "xgboost"
      - name: train-lightgbm
        dependencies: [preprocess]
        template: train
        arguments:
          parameters:
          - name: model-type
            value: "lightgbm"
      - name: evaluate
        dependencies: [train-xgboost, train-lightgbm]
        template: evaluate-models
      - name: deploy
        dependencies: [evaluate]
        template: deploy-model
        when: "{{tasks.evaluate.outputs.parameters.accuracy}} > 0.95"
```

DAGs are generally preferred for ML pipelines because dependencies are explicit and the graph structure is easier to reason about. The `when` clause enables conditional execution based on outputs from upstream tasks.

### Parameters

Parameters allow you to pass data between steps and parameterize workflows:

```yaml
spec:
  entrypoint: train-pipeline
  arguments:
    parameters:
    - name: learning-rate
      value: "0.001"
    - name: epochs
      value: "100"
    - name: dataset-path
      value: "s3://ml-data/training/v3"

  templates:
  - name: train-pipeline
    dag:
      tasks:
      - name: train
        template: train-model
        arguments:
          parameters:
          - name: lr
            value: "{{workflow.parameters.learning-rate}}"
          - name: epochs
            value: "{{workflow.parameters.epochs}}"

  - name: train-model
    inputs:
      parameters:
      - name: lr
      - name: epochs
    container:
      image: myorg/trainer:v2
      command: [python, train.py]
      args:
        - "--lr={{inputs.parameters.lr}}"
        - "--epochs={{inputs.parameters.epochs}}"
    outputs:
      parameters:
      - name: accuracy
        valueFrom:
          path: /tmp/metrics.json
          jqFilter: '.accuracy'
```

Output parameters are extracted from files written by the container. The `jqFilter` option lets you extract specific fields from JSON output, which is useful for passing metrics between steps.

### Artifacts

Artifacts handle large data objects (datasets, model files, checkpoints) that are too large for parameters:

```yaml
templates:
- name: preprocess
  container:
    image: myorg/preprocessor:v1
    command: [python, preprocess.py]
  outputs:
    artifacts:
    - name: processed-data
      path: /tmp/processed_data.parquet
      s3:
        endpoint: s3.amazonaws.com
        bucket: ml-artifacts
        key: "{{workflow.name}}/processed_data.parquet"
        accessKeySecret:
          name: aws-credentials
          key: accessKey
        secretKeySecret:
          name: aws-credentials
          key: secretKey

- name: train
  inputs:
    artifacts:
    - name: training-data
      path: /tmp/data/training.parquet
      s3:
        endpoint: s3.amazonaws.com
        bucket: ml-artifacts
        key: "{{workflow.name}}/processed_data.parquet"
  container:
    image: myorg/trainer:v2
    command: [python, train.py]
    args: ["--data-path=/tmp/data/training.parquet"]
```

You can configure a default artifact repository to avoid repeating S3/GCS configuration on every artifact:

```yaml
# ConfigMap: workflow-controller-configmap
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-controller-configmap
data:
  artifactRepository: |
    s3:
      bucket: ml-artifacts
      endpoint: s3.amazonaws.com
      accessKeySecret:
        name: aws-credentials
        key: accessKey
      secretKeySecret:
        name: aws-credentials
        key: secretKey
```

With a default repository configured, artifact definitions simplify to just a name and path:

```yaml
outputs:
  artifacts:
  - name: model
    path: /tmp/model.pt
```

## ML Pipeline Examples

### Full Training Pipeline

This example shows a realistic ML pipeline with preprocessing, training, evaluation, and conditional deployment:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-training-pipeline-
spec:
  entrypoint: training-pipeline
  arguments:
    parameters:
    - name: dataset-version
      value: "v3.2"
    - name: model-name
      value: "fraud-detector"
    - name: min-accuracy
      value: "0.92"

  volumes:
  - name: shared-data
    emptyDir: {}

  templates:
  - name: training-pipeline
    dag:
      tasks:
      - name: validate-data
        template: data-validation
      - name: preprocess
        dependencies: [validate-data]
        template: preprocess-data
      - name: train
        dependencies: [preprocess]
        template: train-model
      - name: evaluate
        dependencies: [train]
        template: evaluate-model
      - name: register
        dependencies: [evaluate]
        template: register-model
        when: >-
          {{tasks.evaluate.outputs.parameters.accuracy}} >=
          {{workflow.parameters.min-accuracy}}
      - name: deploy
        dependencies: [register]
        template: deploy-model

  - name: data-validation
    container:
      image: myorg/data-validator:v1
      command: [python, validate.py]
      args:
        - "--dataset={{workflow.parameters.dataset-version}}"
      resources:
        requests:
          memory: "2Gi"
          cpu: "1"

  - name: preprocess-data
    container:
      image: myorg/preprocessor:v1
      command: [python, preprocess.py]
      resources:
        requests:
          memory: "8Gi"
          cpu: "4"
    outputs:
      artifacts:
      - name: features
        path: /tmp/features.parquet

  - name: train-model
    inputs:
      artifacts:
      - name: features
        path: /tmp/features.parquet
        from: "{{tasks.preprocess.outputs.artifacts.features}}"
    container:
      image: myorg/trainer:v2
      command: [python, train.py]
      args:
        - "--model-name={{workflow.parameters.model-name}}"
      resources:
        requests:
          memory: "16Gi"
          cpu: "4"
        limits:
          nvidia.com/gpu: 1
    outputs:
      artifacts:
      - name: model
        path: /tmp/model/

  - name: evaluate-model
    inputs:
      artifacts:
      - name: model
        path: /tmp/model/
        from: "{{tasks.train.outputs.artifacts.model}}"
    container:
      image: myorg/evaluator:v1
      command: [python, evaluate.py]
    outputs:
      parameters:
      - name: accuracy
        valueFrom:
          path: /tmp/metrics.json
          jqFilter: '.accuracy'
      - name: f1-score
        valueFrom:
          path: /tmp/metrics.json
          jqFilter: '.f1_score'

  - name: register-model
    container:
      image: myorg/model-registry-client:v1
      command: [python, register.py]
      args:
        - "--name={{workflow.parameters.model-name}}"
        - "--version={{workflow.name}}"

  - name: deploy-model
    resource:
      action: apply
      manifest: |
        apiVersion: serving.kserve.io/v1beta1
        kind: InferenceService
        metadata:
          name: {{workflow.parameters.model-name}}
        spec:
          predictor:
            model:
              modelFormat:
                name: pytorch
              storageUri: "s3://ml-models/{{workflow.parameters.model-name}}/{{workflow.name}}"
```

### Hyperparameter Sweep with `withItems`

Argo can fan out across parameter combinations for hyperparameter searches:

```yaml
templates:
- name: hp-sweep
  dag:
    tasks:
    - name: train-variants
      template: train-with-params
      arguments:
        parameters:
        - name: lr
          value: "{{item.lr}}"
        - name: batch-size
          value: "{{item.batch_size}}"
      withItems:
      - { lr: "0.001", batch_size: "32" }
      - { lr: "0.001", batch_size: "64" }
      - { lr: "0.01",  batch_size: "32" }
      - { lr: "0.01",  batch_size: "64" }
      - { lr: "0.1",   batch_size: "32" }
      - { lr: "0.1",   batch_size: "64" }
    - name: select-best
      dependencies: [train-variants]
      template: model-selection
```

Each item spawns a separate pod, all running in parallel (subject to Kubernetes resource limits). For larger sweeps, use `withParam` to pass a JSON list generated by a prior step.

### Reusable WorkflowTemplates

WorkflowTemplates let you define reusable pipelines that can be invoked with different parameters:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: standard-training-pipeline
spec:
  arguments:
    parameters:
    - name: image
    - name: dataset-path
    - name: model-output-path
  entrypoint: pipeline
  templates:
  - name: pipeline
    steps:
    - - name: train
        template: train-step
    - - name: evaluate
        template: evaluate-step
  # ... template definitions
```

```yaml
# Invoke the template
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: invoke-training-
spec:
  workflowTemplateRef:
    name: standard-training-pipeline
  arguments:
    parameters:
    - name: image
      value: "myorg/trainer:v3"
    - name: dataset-path
      value: "s3://data/latest"
    - name: model-output-path
      value: "s3://models/experiment-42"
```

### CronWorkflows for Scheduled Retraining

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: weekly-retrain
spec:
  schedule: "0 2 * * 0"    # Every Sunday at 2 AM
  timezone: "America/New_York"
  concurrencyPolicy: "Replace"
  workflowSpec:
    entrypoint: retrain-pipeline
    workflowTemplateRef:
      name: standard-training-pipeline
    arguments:
      parameters:
      - name: dataset-path
        value: "s3://data/weekly-snapshot"
```

## Argo Events

Argo Events is a separate project that provides event-driven workflow triggering. It consists of three key concepts:

```
EventSource
   Listens for events from external systems
   Types: webhook, S3, Kafka, AMQP, SNS/SQS, GitHub, cron, etc.

Sensor
   Evaluates event filters and dependencies
   Triggers actions when conditions are met
   Can require multiple events before triggering

Trigger
   The action taken when a sensor fires
   Can submit Argo Workflows, create K8s resources, send HTTP requests, etc.
```

### Example: Trigger Training on New Data

```yaml
# EventSource: watch S3 bucket for new data
apiVersion: argoproj.io/v1alpha1
kind: EventSource
metadata:
  name: data-upload
spec:
  s3:
    new-dataset:
      bucket:
        name: ml-data
      events:
        - s3:ObjectCreated:Put
      filter:
        prefix: "datasets/"
        suffix: ".parquet"
      endpoint: s3.amazonaws.com
      accessKey:
        name: aws-credentials
        key: accessKey
      secretKey:
        name: aws-credentials
        key: secretKey
---
# Sensor: trigger workflow on new data event
apiVersion: argoproj.io/v1alpha1
kind: Sensor
metadata:
  name: retrain-on-data
spec:
  dependencies:
  - name: new-data
    eventSourceName: data-upload
    eventName: new-dataset
  triggers:
  - template:
      name: trigger-training
      argoWorkflow:
        operation: submit
        source:
          resource:
            apiVersion: argoproj.io/v1alpha1
            kind: Workflow
            metadata:
              generateName: auto-retrain-
            spec:
              workflowTemplateRef:
                name: standard-training-pipeline
              arguments:
                parameters:
                - name: dataset-path
                  src:
                    dependencyName: new-data
                    dataKey: notification.object.key
```

This pattern is powerful for ML: new data lands in S3, Argo Events detects it, and a training pipeline kicks off automatically. You can also combine multiple event dependencies, for example requiring both new data and a passing data quality check before triggering retraining.

## Comparison with Other Orchestrators

### Argo vs Airflow

Airflow is the most widely deployed workflow orchestrator and the most common comparison point.

| Aspect | Argo Workflows | Apache Airflow |
|---|---|---|
| Execution model | Each step is a container/pod | Tasks run via operators (Python, Bash, K8s, etc.) |
| Definition format | YAML (Kubernetes CRDs) | Python (DAG files) |
| Infrastructure | Requires Kubernetes | Runs anywhere (VM, K8s, managed services) |
| Scheduling | CronWorkflows + Argo Events | Built-in scheduler with rich cron and sensor support |
| UI | Workflow-centric, shows DAG and pod logs | Mature UI with Gantt charts, task instance history |
| Ecosystem | CNCF ecosystem, Kubernetes-native tooling | Massive provider ecosystem (AWS, GCP, Snowflake, dbt) |
| Dynamic workflows | withItems/withParam for fan-out | Dynamic task mapping, branching |
| Isolation | Full container isolation per step by default | KubernetesPodOperator for isolation, otherwise shared |
| State management | Kubernetes etcd (workflow CRDs) | Metadata database (Postgres/MySQL) |
| Managed offerings | Few (Argo on GKE, Couler) | MWAA, Cloud Composer, Astronomer |

Choose Airflow when you need broad integration with data tools, want a Python-native definition experience, or need a mature managed service. Choose Argo when container isolation, Kubernetes-native operation, and YAML-defined pipelines are priorities.

### Argo vs Prefect

Prefect offers a Python-native, decorator-based workflow definition that feels more like writing regular Python code. Prefect 2+ uses a hybrid model where the orchestration server tracks state while execution happens on your infrastructure.

Key differences: Argo provides stronger container isolation and is Kubernetes-native, while Prefect is more ergonomic for Python-heavy data science teams. Prefect has a managed cloud offering (Prefect Cloud) that reduces operational burden. Argo requires operating Kubernetes, which is a significant investment if you do not already have it.

### Argo vs Dagster

Dagster is asset-oriented rather than task-oriented. Its core abstraction is the Software-Defined Asset, which represents a persistent data artifact (a table, a model, a file). This makes it particularly strong for data lineage and dependency tracking.

Key differences: Dagster provides richer type checking and data lineage out of the box. Argo is more infrastructure-focused and container-centric. Dagster works well for teams that think in terms of data assets; Argo works well for teams that think in terms of computational steps running in containers.

### When to Choose Argo

Argo Workflows is the right choice when:

- Your team already operates Kubernetes and wants to keep ML orchestration on the same infrastructure
- You need strong container isolation between pipeline steps (different runtimes, dependencies, or GPU requirements per step)
- Your pipelines involve heterogeneous workloads: some steps are Python, some are Spark jobs, some create Kubernetes resources
- You want to leverage the broader Argo ecosystem (Argo CD for deployment, Argo Events for triggering, Argo Rollouts for canary deployments)
- You prefer declarative YAML definitions over imperative Python code
- You need to integrate with Kubernetes-native ML tools (KServe, Kubeflow, Seldon, Volcano for batch scheduling)

Argo is not the best fit when:

- You do not run Kubernetes and do not want to start
- Your team is Python-centric and prefers defining workflows in code
- You need a managed service with minimal operational overhead
- Your pipelines are primarily data transformations that integrate heavily with warehouse/lake tooling

## Practical Tips and Gotchas

### Resource Management

Always set resource requests and limits on your workflow steps. Without them, the Kubernetes scheduler may place training pods on nodes without enough memory or GPU, leading to OOM kills or degraded performance.

```yaml
container:
  resources:
    requests:
      memory: "8Gi"
      cpu: "2"
    limits:
      memory: "16Gi"
      nvidia.com/gpu: 1
```

For GPU workloads, use node selectors or tolerations to ensure pods land on GPU nodes:

```yaml
container:
  image: myorg/trainer:v1
nodeSelector:
  accelerator: nvidia-a100
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

### Retry Strategies

Network failures, spot instance preemption, and transient errors are common in ML pipelines. Configure retry strategies:

```yaml
templates:
- name: train-model
  retryStrategy:
    limit: 3
    retryPolicy: "Always"
    backoff:
      duration: "30s"
      factor: 2
      maxDuration: "10m"
  container:
    image: myorg/trainer:v1
```

Use `retryPolicy: "OnError"` to retry only on infrastructure failures (not application-level exit codes), or `"OnTransientError"` for pod-level transient failures.

### Workflow Garbage Collection

Old workflow CRDs accumulate in etcd and can degrade Kubernetes API server performance. Configure TTL-based cleanup:

```yaml
spec:
  ttlStrategy:
    secondsAfterCompletion: 86400   # 1 day
    secondsAfterSuccess: 86400
    secondsAfterFailure: 259200     # 3 days (keep failures longer for debugging)
```

### Avoiding Common Pitfalls

**Large artifacts through parameters.** Parameters are stored in the workflow CRD in etcd, which has a default object size limit of 1.5 MB. Pass large data through artifacts (S3/GCS), not parameters.

**Missing artifact repository configuration.** If you forget to configure the default artifact repository, every artifact declaration needs full S3/GCS credentials inline, which is verbose and error-prone.

**Workflow names too long.** Kubernetes resource names have a 253-character limit, and Argo appends step names to create pod names. Use `generateName` instead of `name` and keep template names short.

**Not setting activeDeadlineSeconds.** Without a timeout, a hung training step can consume GPU resources indefinitely. Always set a workflow-level or step-level timeout:

```yaml
spec:
  activeDeadlineSeconds: 86400   # 24-hour workflow timeout

templates:
- name: train-model
  activeDeadlineSeconds: 14400   # 4-hour step timeout
```

**Ignoring pod garbage collection.** Completed pods remain in the cluster by default. Set `podGC` to clean them up:

```yaml
spec:
  podGC:
    strategy: OnWorkflowCompletion
```

### Security Considerations

- Use Kubernetes ServiceAccounts with minimal RBAC permissions for workflow execution
- Store credentials (S3 keys, registry tokens) in Kubernetes Secrets, never in workflow specs
- Enable SSO on Argo Server to control who can submit and view workflows
- Consider network policies to restrict pod-to-pod communication within workflows
- Use `securityContext` to run containers as non-root

### Debugging Workflows

```bash
# List workflows
argo list -n argo

# Get workflow status
argo get my-workflow -n argo

# Watch a workflow in real time
argo watch my-workflow -n argo

# View logs for a specific step
argo logs my-workflow my-step-name -n argo

# View logs for the entire workflow
argo logs my-workflow -n argo

# Retry a failed workflow from the point of failure
argo retry my-workflow -n argo

# Resubmit a workflow (creates a new instance)
argo resubmit my-workflow -n argo
```

The Argo Server UI is often more practical for debugging than the CLI. It shows the DAG visually, lets you click into individual step pods, view logs, and inspect input/output parameters and artifacts.

### Integration with Kubeflow Pipelines

Kubeflow Pipelines (KFP) uses Argo Workflows as its default backend (though KFP v2 introduced its own backend as well). If you use KFP, you are using Argo under the hood. The KFP SDK provides a Python-native way to define pipelines that compile down to Argo Workflow YAML. This can be a good middle ground for teams that want Python-based pipeline definitions but Argo-based execution.

```python
# Kubeflow Pipelines SDK compiles to Argo Workflow YAML
from kfp import dsl

@dsl.pipeline(name="training-pipeline")
def training_pipeline(lr: float = 0.001):
    preprocess_task = preprocess_op()
    train_task = train_op(
        data=preprocess_task.outputs["data"],
        lr=lr
    )
    eval_task = eval_op(model=train_task.outputs["model"])
```
