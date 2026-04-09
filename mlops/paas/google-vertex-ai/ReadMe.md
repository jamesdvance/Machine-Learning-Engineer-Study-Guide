# Google Vertex AI

## Summary

Google Vertex AI is Google Cloud's unified machine learning platform for building, training, deploying, and managing ML models and AI applications at scale. It consolidates what were previously separate services (AI Platform Training, AI Platform Prediction, AutoML) into a single control plane with a consistent API surface. For ML engineers, Vertex AI provides managed infrastructure for the full lifecycle: data preparation, experiment tracking, training (both AutoML and custom), model registry, serving, monitoring, and pipeline orchestration. Its deep integration with BigQuery and Google Cloud Storage makes it especially compelling for teams already operating within the GCP ecosystem.

Key points to remember:

- Vertex AI unifies AutoML and custom training under one platform, with shared endpoints, model registry, and monitoring
- Pipelines are built on Kubeflow Pipelines (KFP) v2, giving portability while benefiting from managed execution
- Feature Store provides both batch and online serving of features with point-in-time correctness
- Model Garden offers access to first-party models (Gemini, Imagen), open models (Gemma, Llama), and third-party models (Claude) through a single catalog
- BigQuery ML integration allows training and inference directly from SQL, bridging the gap between data analysts and ML engineers
- Vertex AI Experiments and TensorBoard integration provide native experiment tracking without additional tooling
- The platform targets GCP-native teams but supports hybrid workflows through its KFP-based pipeline system

## Platform Architecture and Core Components

Vertex AI is organized around a set of modular services that can be used independently or composed together. Understanding how these services relate to each other is essential for designing production ML systems on GCP.

### Training

Vertex AI Training supports two primary modes: AutoML and custom training. Both modes produce models that land in the same Model Registry and can be deployed to the same Endpoint infrastructure.

**AutoML** trains models on tabular, image, text, and video data without requiring you to write model code. You provide a dataset, specify the prediction target and optimization objective, and Vertex AI handles architecture search, hyperparameter tuning, and ensembling. AutoML is appropriate for baseline modeling, for teams without deep ML expertise on a particular modality, or when the problem is well-suited to standard architectures. AutoML tabular models, for example, use an ensemble of gradient-boosted trees and neural networks that often perform competitively with hand-tuned models.

**Custom training** gives full control over the training process. You provide a training script or container image, specify the machine type and accelerator configuration, and Vertex AI manages the infrastructure. Custom training supports pre-built containers for TensorFlow, PyTorch, XGBoost, and scikit-learn, or you can bring your own container. Distributed training is supported natively, including multi-node GPU training with frameworks like PyTorch DDP or TensorFlow's distribution strategies. Custom training jobs can be configured with hyperparameter tuning, where Vertex AI runs multiple trials with different hyperparameter configurations using Bayesian optimization or grid search.

The key distinction is control versus convenience. AutoML gets you to a deployable model quickly but limits architectural choices. Custom training requires more engineering effort but allows arbitrary model architectures, custom loss functions, and non-standard training loops.

### Prediction and Endpoints

Vertex AI provides two serving modes: online prediction and batch prediction.

**Online prediction** deploys models behind managed endpoints with autoscaling, traffic splitting, and health monitoring. You create an Endpoint resource, deploy one or more Model versions to it, and configure traffic routing. This supports A/B testing and canary deployments natively. Endpoints handle TLS termination, request logging, and autoscaling based on CPU, GPU utilization, or request rate. Models can be served using pre-built serving containers (TensorFlow Serving, TorchServe, Triton Inference Server) or custom containers.

**Batch prediction** processes large datasets stored in BigQuery or GCS without standing up a persistent endpoint. You specify an input source, a model, and an output destination, and Vertex AI provisions ephemeral compute, runs inference, and writes results. This is appropriate for offline scoring workloads, periodic recomputation of recommendations, or any scenario where latency requirements are measured in minutes or hours rather than milliseconds.

Online endpoints also support private endpoints for VPC-native deployments, which is important for organizations with strict network isolation requirements. Dedicated endpoints provide reserved capacity for latency-sensitive workloads.

### Pipelines

Vertex AI Pipelines provides managed, serverless execution of ML workflows defined using the Kubeflow Pipelines (KFP) v2 SDK. Pipelines orchestrate multi-step workflows such as data preprocessing, training, evaluation, and conditional deployment. Each step runs in its own container, enabling reproducibility and isolation.

The KFP v2 SDK uses a Python-based DSL where you define components as Python functions decorated with `@kfp.component` and compose them into pipelines with `@kfp.pipeline`. Compiled pipelines produce a YAML or JSON specification that Vertex AI executes on managed infrastructure. You do not provision or manage Kubernetes clusters.

Pipelines integrate with Vertex AI's other services. A pipeline can trigger a custom training job, evaluate the resulting model, register it in the Model Registry, and deploy it to an endpoint, all as connected steps with artifact lineage tracked automatically. Pipeline runs, their parameters, and output artifacts are stored and queryable, supporting audit and reproducibility requirements.

For teams already using TFX (TensorFlow Extended), Vertex AI supports running TFX pipelines natively. TFX provides opinionated components for data validation (TFDV), data transformation (TFT), training, evaluation (TFMA), and serving, with built-in schema enforcement and data drift detection.

### Feature Store

Vertex AI Feature Store is a managed service for storing, serving, and sharing ML features. It addresses common feature engineering challenges: ensuring consistency between training and serving (avoiding training-serving skew), providing low-latency online serving for real-time models, and enabling feature reuse across teams and models.

Feature Store organizes features into Feature Groups and Features. Features are ingested from BigQuery tables or views, and the store maintains both an online store (backed by Bigtable for low-latency lookups) and an offline store (backed by BigQuery for batch retrieval). Point-in-time lookups ensure that training data uses only features that were available at the time of each training example, preventing data leakage.

Feature monitoring detects drift in feature distributions over time. When the distribution of incoming feature values deviates significantly from the baseline, alerts can be triggered for investigation.

### Model Registry

The Model Registry is a central catalog of trained models with versioning, metadata, and lineage tracking. Every model trained through Vertex AI (whether via AutoML or custom training) can be registered here, and externally trained models can be imported as well. Each model version records its training pipeline, input data, evaluation metrics, and container image.

The registry serves as the handoff point between training and serving. Deployment to an endpoint references a specific model version in the registry. This separation enables workflows where data scientists produce model versions and ML engineers or automated pipelines handle deployment.

### Experiments and Metadata

Vertex AI Experiments provides native experiment tracking. Each experiment contains multiple runs, and each run records parameters, metrics, and artifacts. This integrates with the Vertex AI SDK so that logging metrics from a training script requires only a few lines of code. Vertex AI also provides managed TensorBoard instances for visualizing training curves, comparing runs, and sharing results.

The Metadata service tracks lineage across the platform: which dataset was used to train which model, which pipeline produced which artifact. This lineage graph supports debugging production issues (tracing a misprediction back to its training data), compliance audits, and reproducibility.

### Model Garden

Model Garden is a curated catalog of models available on Vertex AI. It includes three categories:

**First-party models** are Google's own foundation models, including Gemini (multimodal LLM), Imagen (image generation), Chirp (speech), Codey (code generation), and Embeddings models. These are accessible through managed APIs with no infrastructure to provision.

**Open models** include community models like Gemma, Llama, Mistral, and Falcon. These can be deployed to Vertex AI endpoints with one click or fine-tuned on your data. The platform handles the infrastructure for serving these large models, including multi-GPU configurations.

**Third-party models** from partners like Anthropic (Claude) and Meta are available through the Model Garden, providing a single billing and governance layer across model providers.

## Vertex AI for Large Language Models

Vertex AI has become a primary platform for deploying and customizing large language models. The Gemini API, accessible through Vertex AI, provides managed access to Google's frontier models with enterprise features like data residency, VPC Service Controls, and customer-managed encryption keys.

### Model Tuning

Vertex AI supports several approaches to customizing foundation models:

**Supervised fine-tuning** adjusts model weights on labeled examples. You provide a JSONL dataset of prompt-completion pairs, and Vertex AI handles the distributed training infrastructure. This is available for Gemini models and select open models.

**Reinforcement learning from human feedback (RLHF)** tuning is supported for certain models, allowing alignment with specific quality criteria beyond what supervised tuning provides.

**Distillation** allows you to train a smaller, faster model to mimic the behavior of a larger model. This is useful when you need the quality of a large model at the latency and cost of a smaller one.

**Prompt tuning and adapter tuning** provide parameter-efficient alternatives that modify a small number of parameters while keeping the base model frozen. These approaches are faster and cheaper than full fine-tuning while often achieving comparable results for domain-specific tasks.

### Vertex AI Studio

Vertex AI Studio provides an interactive interface for prototyping with foundation models. You can test prompts, adjust parameters (temperature, top-k, top-p), configure safety filters, and evaluate outputs before committing to an API integration. Prompts can be saved, versioned, and shared.

### Grounding and RAG

Vertex AI supports grounding model responses in external data through integration with Vertex AI Search (formerly Enterprise Search) and custom grounding sources. This enables retrieval-augmented generation (RAG) patterns where the model's response is anchored to specific documents or data, reducing hallucination and improving factual accuracy.

## Integration with the GCP Ecosystem

One of Vertex AI's strongest differentiators is its integration depth with other Google Cloud services.

### BigQuery

BigQuery ML (BQML) allows training and running inference on models directly from SQL. You can train linear models, boosted trees, deep neural networks, and even invoke Vertex AI models (including Gemini) from BigQuery using remote model references. The `ML.GENERATE_TEXT`, `AI.GENERATE`, and `AI.EMBED` functions bring generative AI capabilities directly into SQL workflows.

For ML engineers, this means that data teams who live in BigQuery can produce baseline models without leaving their SQL environment. When those models need to graduate to production with proper serving infrastructure, they can be exported to the Vertex AI Model Registry and deployed to endpoints.

BigQuery also serves as the offline store for Feature Store and as a common data source for Vertex AI Pipelines and training jobs. The `google-cloud-bigquery` Python client and BigQuery I/O connectors in Apache Beam provide programmatic access from training scripts and pipeline components.

### Google Cloud Storage

GCS is the default artifact store for Vertex AI. Training data, model artifacts, pipeline outputs, and exported models all reside in GCS buckets. Training jobs mount GCS paths directly, and the Vertex AI SDK handles data staging transparently.

### TFX Integration

TFX pipelines run natively on Vertex AI Pipelines. This is significant for teams that have invested in TFX's data validation, transformation, and evaluation components. Running TFX on Vertex AI provides the orchestration and managed infrastructure of Vertex AI while retaining TFX's opinionated, production-grade pipeline components.

### Dataflow and Pub/Sub

For real-time feature engineering, Vertex AI Feature Store integrates with Dataflow (managed Apache Beam) for stream processing and Pub/Sub for event ingestion. This enables patterns where features are computed from streaming data and served in real time through Feature Store's online serving layer.

## Comparison with Other ML Platforms

Vertex AI competes with AWS SageMaker, Azure Machine Learning, and Databricks as a managed ML platform. Each has distinct strengths.

### Versus AWS SageMaker

SageMaker offers broader compute options, including custom silicon (Inferentia, Trainium) optimized for inference and training cost. SageMaker's ecosystem is larger by market share, with more third-party integrations and community examples. SageMaker Studio provides a more full-featured notebook-based IDE. However, Vertex AI offers tighter integration with BigQuery (which has no direct SageMaker equivalent), a more streamlined developer experience for common workflows, and access to Gemini and Google's foundation models natively. SageMaker Pipelines use a proprietary SDK, while Vertex AI uses the more portable KFP standard.

### Versus Azure Machine Learning

Azure ML provides strong enterprise integration with the Microsoft ecosystem (Active Directory, Power BI, Azure DevOps). Its Responsible AI dashboard is more mature than Vertex AI's equivalent tooling. Azure ML's designer provides a visual pipeline builder that has no direct Vertex AI counterpart. Vertex AI's advantage is in its BigQuery integration, TPU access for training, and Model Garden breadth. Azure ML's MLflow integration gives it an edge in open-source experiment tracking portability.

### Versus Databricks

Databricks is a data-platform-first offering that has added ML capabilities (MLflow, Feature Store, Model Serving). Its strength is in unifying data engineering, analytics, and ML on a single lakehouse platform. For organizations where Spark-based data processing is central, Databricks provides tighter integration between data pipelines and ML workflows. Vertex AI is a stronger choice when the ML workload is the primary concern rather than an extension of data engineering, when BigQuery is the data warehouse, or when access to Google's foundation models is important.

### Cross-Platform Considerations

The choice between platforms is rarely made on ML capabilities alone. It is driven by existing cloud infrastructure, data gravity (where the data already lives), team expertise, and enterprise agreements. Multi-cloud ML is possible but introduces complexity in data movement, identity management, and artifact portability.

## When to Choose Vertex AI

Vertex AI is the strongest choice in these scenarios:

**GCP-native organizations.** If your data lives in BigQuery and GCS, your infrastructure runs on GKE, and your team is proficient with GCP IAM and networking, Vertex AI avoids the friction of cross-cloud data movement and identity management.

**BigQuery-heavy workflows.** The depth of integration between BigQuery and Vertex AI, including BQML, remote model invocation, and Feature Store backed by BigQuery, makes Vertex AI the natural choice when BigQuery is the center of your data platform.

**Foundation model access.** If your use case requires Gemini, Imagen, or other Google first-party models, Vertex AI provides the most direct and fully supported path to these models with enterprise controls.

**Teams that value simplicity.** Vertex AI's unified experience and managed defaults reduce the configuration surface compared to SageMaker. For mid-sized teams that want to move quickly without managing infrastructure, this simplicity is valuable.

**KFP portability.** If your team has invested in Kubeflow Pipelines or wants the option to run pipelines on self-managed Kubernetes clusters, Vertex AI's KFP-based pipeline system provides a migration path in both directions.

## Practical Tips

**Start with AutoML for baselines.** Before investing in custom model development, train an AutoML model to establish a performance baseline. AutoML tabular models in particular are often surprisingly competitive and can inform whether custom modeling effort is justified.

**Use managed TensorBoard for visibility.** Vertex AI's managed TensorBoard instances persist beyond individual training jobs and can be shared across a team. Point all training jobs at the same TensorBoard instance to enable cross-experiment comparison.

**Structure pipelines for reuse.** Define pipeline components as independent, parameterized units. A data-loading component should work with any dataset path, and an evaluation component should work with any model. This modularity pays off quickly as the number of models and experiments grows.

**Leverage prebuilt containers when possible.** Vertex AI provides pre-built training and serving containers for major frameworks. These are optimized, patched, and tested by Google. Use custom containers only when your dependencies or serving logic requires it.

**Use Workbench for development, Pipelines for production.** Vertex AI Workbench (managed JupyterLab) is appropriate for interactive development and prototyping. Once a workflow is stable, convert it to a Vertex AI Pipeline for reproducible, scheduled execution.

**Monitor model performance in production.** Configure Vertex AI Model Monitoring to detect prediction drift and feature skew. Set up alerts through Cloud Monitoring so that model degradation is caught before it impacts business metrics.

**Control costs with autoscaling minimums.** Online endpoints can scale to zero when using certain model types, but most configurations require a minimum of one replica. For development and staging endpoints, use the smallest available machine type and set maximum replica counts conservatively.

**Use VPC Service Controls for sensitive data.** Vertex AI supports VPC-SC, which restricts data from leaving a defined security perimeter. This is essential for healthcare, financial services, and other regulated industries.

**Pin SDK versions in production.** The Vertex AI Python SDK and KFP SDK evolve rapidly. Pin versions in your requirements files and test upgrades deliberately to avoid breaking changes in production pipelines.

**Separate projects for environments.** Use distinct GCP projects for development, staging, and production Vertex AI resources. This provides clean IAM boundaries, separate billing, and independent quotas.
