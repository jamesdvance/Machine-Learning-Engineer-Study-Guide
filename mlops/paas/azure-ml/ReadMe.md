# Azure Machine Learning

## Summary

Azure Machine Learning is Microsoft's fully managed cloud platform for building, training, deploying, and managing machine learning models at scale. It provides a unified environment that spans the entire ML lifecycle, from data preparation and experimentation through to production deployment and monitoring. Azure ML integrates deeply with the broader Azure ecosystem, making it the natural choice for organizations already invested in Microsoft's cloud infrastructure.

The platform has evolved significantly since its early drag-and-drop Studio days. The current generation, built around SDK v2 and CLI v2, offers a code-first experience that treats ML assets as declarative, versioned resources. This shift aligns Azure ML more closely with modern MLOps practices where reproducibility, automation, and infrastructure-as-code are priorities.

Key points to remember:

- The Workspace is the central organizational unit; all assets (compute, data, models, endpoints) live within a workspace
- SDK v2 and CLI v2 use a unified, YAML-based asset definition model that treats everything as a resource
- Training is organized around jobs: command jobs for single runs, sweep jobs for hyperparameter tuning, pipeline jobs for multi-step workflows
- Managed endpoints handle both real-time (online) and large-scale (batch) inference without requiring you to manage infrastructure
- The Model Registry provides versioned model storage with lineage tracking and staging workflows
- Responsible AI tooling is built in, not bolted on, with a dashboard for fairness, interpretability, and error analysis

## Core Concepts

### Workspace

The workspace is the top-level resource in Azure ML. It acts as a container for all ML assets and is the boundary for access control, networking, and billing. A workspace is backed by several Azure resources that are provisioned alongside it:

- Azure Storage Account: stores datasets, logs, model artifacts, and job snapshots
- Azure Key Vault: manages secrets, connection strings, and credentials
- Azure Container Registry: stores Docker images for training and inference environments
- Azure Application Insights: collects telemetry from deployed endpoints

Workspaces can be configured with private endpoints and virtual network integration to meet enterprise networking requirements. For organizations running multiple teams, a common pattern is one workspace per team or per project, with Azure RBAC controlling who can access what.

### Compute

Azure ML provides several compute options:

Compute Instances are single-node VMs intended for interactive development. They come pre-configured with common ML frameworks and integrate with Jupyter, VS Code, and RStudio. Think of them as managed dev machines. They support scheduling (auto-shutdown at night) to control costs.

Compute Clusters are multi-node pools that auto-scale based on job demand. They scale down to zero nodes when idle, so you only pay for active training time. Clusters are the primary target for training jobs. You configure minimum and maximum node counts, VM size (including GPU SKUs like Standard_NC6s_v3 or Standard_ND96amsr_A100_v4), and idle timeout.

Serverless Compute removes the need to pre-create and manage clusters entirely. When you submit a job targeting serverless compute, Azure ML provisions the required resources on-demand and releases them when the job completes. This is useful for sporadic workloads where maintaining a cluster is wasteful.

Attached Compute lets you bring your own resources. You can attach Azure Databricks clusters, Azure Arc-enabled Kubernetes clusters, or remote VMs as compute targets. This is how teams integrate existing infrastructure with Azure ML's job submission and tracking.

### Datastores and Data Assets

Datastores are references to Azure storage services. They store connection information (not the data itself) and handle credential management. Supported backing stores include Azure Blob Storage, Azure Data Lake Storage Gen1 and Gen2, Azure Files, Azure SQL Database, and Azure Database for PostgreSQL.

Data Assets are versioned references to specific data. They come in three types:

- URI File: points to a single file
- URI Folder: points to a directory
- MLTable: defines a tabular schema over one or more files, supporting column selection, type casting, and partitioning

Data assets provide lineage tracking. When a training job consumes a data asset, that relationship is recorded, so you can trace any model back to the exact data version it was trained on.

### Environments

Environments define the software dependencies for training and inference. They are versioned and reproducible. You can define them from:

- A conda specification file
- A Docker image
- A Dockerfile

Azure ML maintains a set of curated environments with pre-installed frameworks (PyTorch, TensorFlow, scikit-learn, and others) that are tested and optimized for Azure infrastructure. Custom environments are built and cached in the workspace's container registry.

## SDK v2 and CLI v2

Azure ML's second-generation interfaces represent a fundamental shift in how you interact with the platform. Both SDK v2 (Python) and CLI v2 (az ml commands) share the same underlying REST API and YAML schema.

### Why v2 Matters

SDK v1 used an imperative, object-oriented approach where you constructed Python objects and called methods. SDK v2 uses a declarative model: you define what you want (a job, an endpoint, a compute cluster) as a YAML specification or Python object, and the system figures out how to create it. This approach has several advantages:

- YAML definitions can be version-controlled alongside code
- The same YAML works with both CLI and SDK
- Assets are reusable across projects and pipelines
- The interface is more predictable and less stateful

### SDK v2 Example

```python
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace",
)

# Define a command job
job = command(
    code="./src",
    command="python train.py --data ${{inputs.training_data}} --lr ${{inputs.learning_rate}}",
    inputs={
        "training_data": Input(type="uri_folder", path="azureml:my-dataset:1"),
        "learning_rate": 0.01,
    },
    environment="azureml:AzureML-sklearn-1.0:1",
    compute="gpu-cluster",
    experiment_name="sklearn-experiment",
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
```

### CLI v2 Example

The equivalent YAML-based approach using the CLI:

```yaml
# job.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
code: ./src
command: python train.py --data ${{inputs.training_data}} --lr ${{inputs.learning_rate}}
inputs:
  training_data:
    type: uri_folder
    path: azureml:my-dataset:1
  learning_rate: 0.01
environment: azureml:AzureML-sklearn-1.0:1
compute: azureml:gpu-cluster
experiment_name: sklearn-experiment
```

```bash
az ml job create --file job.yml --resource-group my-rg --workspace-name my-ws
```

The CLI v2 is particularly useful in CI/CD pipelines where you want to submit jobs from shell scripts or GitHub Actions workflows without writing Python.

## Training

### Command Jobs

A command job is the basic unit of work. It runs a single command in a specified environment on a specified compute target. The command can be any shell command, but it is typically a Python script invocation. Command jobs track:

- Input and output data references
- Environment (Docker image plus conda/pip dependencies)
- Compute target
- Code snapshot (the local code directory is uploaded and versioned)
- Metrics logged during execution
- Output artifacts

Every command job gets a unique run ID and its inputs, outputs, metrics, and logs are recorded in the workspace for later comparison and lineage tracking.

### Sweep Jobs

Sweep jobs perform hyperparameter tuning by running multiple trials of a command job with different parameter combinations. You define:

- A search space: the parameters and their distributions (choice, uniform, loguniform, and others)
- A sampling method: grid, random, or Bayesian
- An early termination policy: bandit, median stopping, or truncation selection
- A primary metric and optimization goal (minimize or maximize)

```python
from azure.ai.ml.sweep import Choice, Uniform, BanditPolicy

job_for_sweep = job(
    learning_rate=Uniform(min_value=0.001, max_value=0.1),
    batch_size=Choice(values=[16, 32, 64, 128]),
)

sweep_job = job_for_sweep.sweep(
    sampling_algorithm="bayesian",
    primary_metric="validation_accuracy",
    goal="maximize",
    max_total_trials=50,
    max_concurrent_trials=10,
    early_termination_policy=BanditPolicy(
        slack_factor=0.1, evaluation_interval=2
    ),
)
sweep_job.compute = "gpu-cluster"

returned_sweep = ml_client.jobs.create_or_update(sweep_job)
```

Sweep jobs distribute trials across cluster nodes automatically. The Bayesian sampler learns from completed trials to make smarter suggestions, which typically converges faster than random search for moderate-dimensionality search spaces.

### Pipeline Jobs

Pipeline jobs chain multiple steps into a directed acyclic graph (DAG). Each step is a command job (or another component type) with defined inputs and outputs. Outputs from one step become inputs to the next. This is how you build end-to-end workflows:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
experiment_name: training-pipeline

jobs:
  preprocess:
    type: command
    component: azureml:preprocess_component:1
    inputs:
      raw_data:
        type: uri_folder
        path: azureml:raw-dataset:1
    outputs:
      processed_data:
        type: uri_folder

  train:
    type: command
    component: azureml:train_component:1
    inputs:
      training_data: ${{parent.jobs.preprocess.outputs.processed_data}}
      learning_rate: 0.01
    outputs:
      model_output:
        type: mlflow_model

  evaluate:
    type: command
    component: azureml:evaluate_component:1
    inputs:
      model: ${{parent.jobs.train.outputs.model_output}}
      test_data:
        type: uri_folder
        path: azureml:test-dataset:1
```

Components are reusable, versioned building blocks. You define a component once (its interface, environment, and code), register it, and then reference it from any pipeline. This promotes modularity and prevents copy-pasting job definitions across projects.

### Distributed Training

Azure ML supports distributed training across multiple nodes and GPUs. For PyTorch, you specify a distribution configuration:

```python
from azure.ai.ml import command
from azure.ai.ml.entities import ResourceConfiguration

job = command(
    code="./src",
    command="python train_distributed.py",
    environment="azureml:AzureML-pytorch-2.0:1",
    compute="gpu-cluster",
    instance_count=4,
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 1,
    },
)
```

Azure ML handles the orchestration: it provisions nodes, sets up the distributed communication (NCCL for GPUs), populates environment variables (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK), and tears everything down when training completes. TensorFlow distribution strategies and MPI-based approaches (for Horovod) are also supported.

## Model Registry

The Model Registry is the central catalog for trained models in a workspace. It stores model artifacts along with metadata:

- Version number (auto-incremented or manually specified)
- Description and tags
- Framework and format (MLflow, custom, ONNX, and others)
- Lineage: which job produced the model, from which data
- Metrics from the training run

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path="azureml://jobs/{job_id}/outputs/model_output",
    name="fraud-detection-model",
    description="XGBoost fraud detection model trained on Q1 2026 data",
    type=AssetTypes.MLFLOW_MODEL,
)

ml_client.models.create_or_update(model)
```

MLflow model format is the recommended standard. Models logged as MLflow models carry their dependencies, input/output signatures, and inference code, which makes deployment straightforward since Azure ML can auto-generate the serving container.

For cross-workspace sharing, Azure ML supports Registry (with a capital R), a centralized, workspace-independent asset store. Teams register models to a shared Registry and consume them in their own workspaces without copying artifacts.

## Managed Endpoints

### Online Endpoints

Online endpoints serve models for real-time inference via REST APIs. You deploy one or more models behind an endpoint with traffic splitting for A/B testing or canary rollouts.

```yaml
# endpoint.yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: fraud-detection-endpoint
auth_mode: key

# deployment.yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: fraud-detection-endpoint
model: azureml:fraud-detection-model:3
instance_type: Standard_DS3_v2
instance_count: 2
```

Key features of online endpoints:

- Auto-scaling based on CPU/memory utilization or request rate
- Built-in load balancing across instances
- Blue/green deployments with percentage-based traffic routing
- Authentication via key or Azure AD token
- Integration with Application Insights for logging and monitoring
- Native support for MLflow models (no custom scoring script needed)

For MLflow models, Azure ML auto-generates the scoring code and Docker image. For custom models, you write a scoring script with `init()` and `run()` functions.

### Batch Endpoints

Batch endpoints handle large-scale offline inference. You submit a dataset, and Azure ML processes it across a compute cluster, writing results to storage.

```bash
az ml batch-endpoint invoke \
  --name fraud-batch-endpoint \
  --input azureml:transactions-to-score:1
```

Batch endpoints are ideal for:

- Scoring millions of records overnight
- Processing data too large for real-time inference
- Scenarios where latency is not critical but throughput is
- Regular scheduled scoring jobs

They support parallel processing across nodes, automatic retries on failure, and output to Azure storage.

## Responsible AI Dashboard

Azure ML integrates responsible AI tooling directly into the model evaluation workflow. The Responsible AI dashboard provides a unified interface for:

Error Analysis: identifies cohorts of data where the model underperforms. It builds a decision tree over error rates, helping you find systematic failure modes (such as the model performing poorly on a specific demographic or data segment).

Fairness Assessment: measures model performance across sensitive groups using metrics like demographic parity, equalized odds, and selection rate. Powered by Fairlearn under the hood.

Model Interpretability: provides feature importance at the global and local level using techniques like SHAP values. Helps explain why the model makes specific predictions.

Counterfactual Analysis: shows what minimal changes to input features would flip a prediction. Useful for understanding decision boundaries.

Causal Inference: estimates causal effects of features on outcomes, moving beyond correlation-based explanations.

The dashboard is generated by submitting a Responsible AI pipeline job that runs these analyses on a test dataset, then viewing results in the Azure ML Studio UI.

## Integration with Azure Services

### Azure Synapse Analytics

Azure ML and Synapse integrate through linked services. Data engineers working in Synapse can submit Azure ML training jobs from Synapse notebooks, and ML engineers can access Synapse SQL pools and Spark pools as data sources. This is useful when feature engineering happens in Synapse and training happens in Azure ML.

### Azure Databricks

Databricks clusters can be attached as compute targets in Azure ML, allowing you to submit jobs to Databricks from the Azure ML interface. More commonly, teams use MLflow's integration: train in Databricks, log experiments to an Azure ML workspace via the MLflow tracking URI, and register models in Azure ML's Model Registry for deployment through managed endpoints.

### Azure DevOps

Azure DevOps Pipelines provide CI/CD for ML workflows. A common setup:

- Code changes trigger an Azure DevOps pipeline
- The pipeline validates code and runs unit tests
- If on the main branch, it submits an Azure ML training pipeline job
- A gated step checks model metrics against thresholds
- If the model passes, the pipeline deploys it to a staging endpoint
- After manual or automated approval, traffic shifts to the new deployment

The Azure ML CLI v2 integrates cleanly with Azure DevOps since pipeline tasks just call `az ml` commands.

### Azure Monitor and Application Insights

Deployed endpoints emit telemetry to Application Insights: request rates, latencies, error rates, and custom metrics. Azure Monitor can trigger alerts when metrics cross thresholds, enabling automated responses to model degradation.

### Microsoft Fabric

Microsoft Fabric, the unified analytics platform, includes a built-in integration with Azure ML. Data scientists working in Fabric notebooks can access Azure ML experiments, register models, and deploy endpoints. Fabric's OneLake provides a unified data layer that Azure ML can reference via datastores.

## Comparison with Other Platforms

### Azure ML vs. AWS SageMaker

SageMaker and Azure ML are the closest competitors in terms of scope and target audience. Both cover the full ML lifecycle.

SageMaker strengths relative to Azure ML:
- SageMaker Studio provides a more polished notebook IDE experience
- Built-in algorithms are more extensive and optimized for AWS infrastructure
- SageMaker Pipelines has tighter integration with the AWS Step Functions ecosystem
- Processing jobs for data preparation are more mature

Azure ML strengths relative to SageMaker:
- YAML-based asset definitions are more CI/CD-friendly than SageMaker's Python-heavy approach
- Responsible AI tooling is more comprehensive out of the box
- Azure ML's Registry provides better cross-workspace model sharing
- Integration with Active Directory and enterprise governance is stronger
- Serverless compute is simpler to set up

### Azure ML vs. Google Vertex AI

Vertex AI is Google's unified ML platform, consolidating several earlier services (AI Platform, AutoML).

Vertex AI strengths relative to Azure ML:
- Tighter integration with BigQuery for feature engineering
- Vertex AI Feature Store is more mature
- AutoML capabilities are generally stronger
- TensorFlow support is more deeply optimized

Azure ML strengths relative to Vertex AI:
- Better support for diverse frameworks (PyTorch, scikit-learn) without bias toward TensorFlow
- More flexible compute options (attached compute, Kubernetes)
- Enterprise features (RBAC, networking, compliance) are more mature
- Wider geographic availability of Azure regions

### Azure ML vs. Databricks

Databricks is less of a direct competitor and more of an adjacent platform. Databricks excels at data engineering and collaborative notebook-based exploration with Spark at its core. Azure ML excels at structured ML workflows, deployment, and governance.

Teams often use both: Databricks for data preparation and experimentation, Azure ML for production deployment and model management. The MLflow integration makes this workflow smooth since MLflow is native to both platforms.

Where Databricks directly competes is in its Mosaic AI and Model Serving capabilities. Databricks Model Serving provides real-time endpoints, and Mosaic AI offers training infrastructure for large models. For teams already deep in the Databricks ecosystem, staying in Databricks for deployment can be simpler than introducing Azure ML.

## When to Choose Azure ML

Azure ML is the strongest choice when:

- Your organization is already on Azure. The integration with Azure AD, Azure networking, Azure storage, and Azure governance is seamless. Fighting this by choosing a non-Azure ML platform adds unnecessary friction.
- Enterprise governance is a priority. Azure ML's RBAC, private endpoints, managed virtual networks, and compliance certifications meet the requirements of regulated industries.
- You need a managed deployment story. The managed endpoint experience is mature and requires minimal infrastructure knowledge to get models serving in production.
- Your ML team uses diverse frameworks. Azure ML is not opinionated about frameworks. PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, and others are all equally supported.
- You want a code-first, GitOps-friendly workflow. The YAML + CLI v2 approach integrates naturally with Git-based CI/CD pipelines.

Azure ML is a weaker choice when:

- Your data and infrastructure live in AWS or GCP. Cross-cloud ML platforms add latency, complexity, and egress costs.
- You are a small team that wants minimal operational overhead. Simpler platforms or managed notebook services may be more appropriate.
- Your primary workload is large-scale data processing. Databricks or Synapse may be better starting points, with Azure ML added for deployment.

## Practical Tips

Start with curated environments. Building custom Docker images from scratch is time-consuming and error-prone. Begin with Azure ML's curated environments and add packages on top using conda or pip specifications.

Use MLflow for experiment tracking and model logging. MLflow is the de facto standard in Azure ML. Models logged with MLflow can be deployed to managed endpoints without writing scoring scripts. The MLflow tracking integration also works from any compute, including local machines and Databricks.

Version everything. Data assets, environments, models, and components are all versioned. Reference specific versions in production pipelines rather than relying on "latest." This makes pipelines reproducible and rollbacks straightforward.

Design for cost. Compute clusters scale to zero when idle, but compute instances do not by default. Set up auto-shutdown schedules on instances. Use serverless compute for sporadic jobs to avoid paying for idle clusters. Monitor costs through Azure Cost Management and set budgets with alerts.

Keep pipelines modular. Build components that do one thing well. A preprocessing component, a training component, and an evaluation component can be composed into different pipelines for different projects. This reduces duplication and makes testing easier.

Use managed virtual networks in production. Azure ML supports workspace-level managed virtual networks that restrict outbound traffic and ensure data does not leave the network boundary. This is not just a security nicety but a compliance requirement in many industries.

Test endpoints locally before deploying. The `az ml local-endpoint` commands let you test your scoring script and environment on your local machine using Docker. This catches environment and code issues before consuming cloud compute.

Leverage the Responsible AI dashboard early. Do not treat responsible AI as a checkbox at the end of the project. Run the dashboard during model development to catch fairness and reliability issues while you can still adjust your approach.

Automate with CI/CD from day one. Even if your first iteration is a single training job, set it up to run from a CI/CD pipeline. The cost of adding automation later, when the project has grown in complexity, is much higher than starting with it.

Monitor deployed models. Set up data drift detection and model performance monitoring through Azure ML's built-in capabilities. A model that performed well at deployment will degrade as data distributions shift. Schedule regular retraining or set up alerts that trigger retraining when drift exceeds thresholds.
