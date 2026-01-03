# Model Registry

## Summary

A model registry is a centralized repository for storing, versioning, and managing machine learning models throughout their lifecycle. It serves as the bridge between model development and production deployment, providing lineage tracking, stage management, and access control for model artifacts.

Key points to remember:

- Model registries store model artifacts with metadata, not just weights
- Version management tracks model evolution and enables rollback
- Stage transitions (staging, production, archived) formalize the deployment process
- Lineage tracking connects models to training data, code, and experiments
- Access control ensures only authorized models reach production

## Why Model Registries Matter

Without a model registry, teams typically store models in:
- Local file systems (no sharing, no versioning)
- Shared drives (versioning via file names, no metadata)
- Cloud storage (better, but lacks ML-specific features)
- Ad-hoc databases (custom, maintenance burden)

These approaches create problems:
- Which model version is currently in production?
- What training data produced this model?
- Who approved this model for deployment?
- How do we roll back to the previous version?
- What experiments led to this model?

Model registries answer these questions with a purpose-built system.

## Core Capabilities

### Model Versioning

Each registered model maintains a version history:

- Automatic version numbering on registration
- Immutable versions (no overwriting)
- Version comparison and diff capabilities
- Semantic versioning support (major.minor.patch)

Versions enable rollback when new models underperform and comparison of model evolution over time.

### Artifact Storage

Model registries store more than just weights:

- Model weights (PyTorch state_dict, TensorFlow SavedModel, ONNX, etc.)
- Preprocessing artifacts (scalers, encoders, vocabularies)
- Configuration files (hyperparameters, architecture specs)
- Requirements and environment specifications
- Custom artifacts (feature importance, calibration curves)

All artifacts needed to reproduce inference are stored together.

### Metadata Management

Rich metadata enables discovery and governance:

Training metadata:
- Training dataset version or hash
- Training code commit
- Hyperparameters
- Training duration and resources
- Framework and library versions

Performance metadata:
- Evaluation metrics on standard test sets
- Latency benchmarks
- Model size and memory requirements

Operational metadata:
- Creator and timestamp
- Approval history
- Deployment history
- Tags and descriptions

### Stage Management

Models progress through lifecycle stages:

- Development: Active experimentation, not ready for serving
- Staging: Candidate for production, undergoing validation
- Production: Actively serving requests
- Archived: Retired from active use, retained for reference

Stage transitions can require approvals and trigger automated validations.

### Lineage Tracking

Lineage connects models to their origins:

- Experiment runs that produced the model
- Training datasets and versions
- Feature engineering pipelines
- Upstream model dependencies (for ensemble or multi-stage systems)

Lineage enables debugging (why did this model fail?) and compliance (prove model provenance).

## Model Registry Platforms

### MLflow Model Registry

MLflow is open-source and widely adopted:

- Integrates with MLflow Tracking for experiment lineage
- Supports multiple model flavors (sklearn, pytorch, tensorflow, etc.)
- REST API for programmatic access
- Stage management with webhooks
- Can be self-hosted or used as managed service

Model registration example:
```python
import mlflow

with mlflow.start_run():
    # Training code here
    mlflow.pytorch.log_model(model, "model", registered_model_name="my_model")
```

### Weights and Biases Model Registry

W&B provides a managed registry integrated with their experiment tracking:

- Automatic artifact versioning
- Aliases for semantic versioning
- Lineage graphs connecting runs to artifacts
- Team collaboration features
- Integration with W&B Sweeps for hyperparameter tuning

### Amazon SageMaker Model Registry

SageMaker provides AWS-native model management:

- Tight integration with SageMaker training and endpoints
- Model package groups for organization
- Approval workflows
- Integration with AWS IAM for access control
- Model cards for documentation

### Azure ML Model Registry

Azure ML provides Microsoft cloud-native model management:

- Integration with Azure ML pipelines
- Model profiling and packaging
- Deployment to Azure Kubernetes Service
- Enterprise security and compliance features

### Google Vertex AI Model Registry

Vertex AI provides GCP-native model management:

- Unified registry for all model types
- Integration with Vertex AI Training and Prediction
- Model versioning and aliasing
- IAM-based access control

## Implementation Patterns

### Registration Workflow

Standard registration process:

1. Train model and log to experiment tracking
2. Evaluate on holdout test sets
3. Generate model card with documentation
4. Register model with metadata
5. Request review and approval
6. Transition to staging for validation
7. After validation, transition to production

Automate steps 1-4 in training pipelines. Steps 5-7 often require human involvement.

### Naming Conventions

Consistent naming enables discovery:

```
{team}/{project}/{model_type}

examples:
fraud/transaction_scoring/xgboost
recommendations/user_embedding/two_tower
search/ranking/bert_reranker
```

Include environment context in stage, not name:
- Good: model "fraud/scorer" in stage "production"
- Bad: model "fraud/scorer_prod"

### Metadata Standards

Define required metadata:

```yaml
required_metadata:
  - training_data_version
  - evaluation_metrics
  - model_framework
  - input_schema
  - output_schema

optional_metadata:
  - training_duration
  - gpu_type
  - feature_importance
  - calibration_metrics
```

Enforce metadata requirements in registration pipelines.

### Model Cards

Model cards document model details:

- Model description and intended use
- Training data description
- Performance metrics across demographics
- Limitations and failure modes
- Ethical considerations

Model cards promote responsible AI practices and enable informed deployment decisions.

## Integration with CI/CD

### Automated Registration

CI/CD pipelines can automate registration:

```yaml
register_model:
  stage: register
  script:
    - python register.py \
        --model outputs/model.pt \
        --name $MODEL_NAME \
        --metrics outputs/metrics.json \
        --commit $CI_COMMIT_SHA
```

Automated registration ensures all models are properly tracked.

### Promotion Pipelines

Automate stage transitions with validation:

```yaml
promote_to_production:
  stage: deploy
  script:
    - python validate_model.py --model $MODEL_NAME --version $VERSION
    - mlflow models transition-stage --name $MODEL_NAME --version $VERSION --stage Production
  when: manual
```

Validation gates prevent bad models from reaching production.

### Rollback Automation

Enable rapid rollback when issues arise:

```yaml
rollback:
  stage: emergency
  script:
    - PREVIOUS=$(mlflow models get-latest-versions --name $MODEL_NAME --stages Archived | head -1)
    - mlflow models transition-stage --name $MODEL_NAME --version $PREVIOUS --stage Production
  when: manual
```

Rollback scripts should be tested before they are needed.

## Access Control

### Role-Based Access

Define roles for model management:

- Data Scientist: Register models, view all models
- ML Engineer: Promote to staging, approve for production
- Admin: Configure registry, manage access

Enforce separation of duties for production deployments.

### Approval Workflows

Production transitions require approvals:

- Automated validation must pass
- Peer review of model card
- Product owner sign-off for business-critical models
- Compliance review where required

Document approval decisions for audit trails.

### Audit Logging

Track all registry operations:

- Model registration events
- Stage transitions
- Metadata modifications
- Access and download events

Audit logs support compliance and incident investigation.

## Serving Integration

### Model Loading

Serving systems load models from the registry:

```python
import mlflow

model = mlflow.pytorch.load_model(
    model_uri="models:/my_model/Production"
)
```

Loading by stage ensures serving always uses the current production model.

### Feature Store Integration

Connect models to feature stores:

- Store feature schema with model
- Validate feature availability before promotion
- Link to feature store entities

Feature-model integration prevents serving errors from missing features.

### A/B Testing Integration

Registry supports A/B testing:

- Deploy multiple model versions simultaneously
- Track which version serves each request
- Compare performance across versions

Registry stages can include "Shadow" or "Canary" for gradual rollout.

## Common Pitfalls

### Skipping Registration

Deploying models without registration creates shadow IT:
- No visibility into what is running
- No ability to rollback
- No audit trail

Enforce that all deployed models must be registered.

### Insufficient Metadata

Metadata added later is often inaccurate. Capture metadata at training time when it is readily available.

### Version Sprawl

Many registered versions with unclear differences create confusion. Archive old versions and maintain clear naming.

### Manual Stage Management

Manual stage transitions are error-prone. Automate transitions with validation pipelines.

### Ignoring Model Size

Large models may exceed registry storage limits or cause slow loading. Consider model optimization before registration.
