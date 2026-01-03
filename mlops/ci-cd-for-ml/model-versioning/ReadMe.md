# Model Versioning

## Summary

Model versioning tracks changes to trained models over time, enabling reproducibility, rollback, and comparison. Unlike code versioning with Git, model versioning must handle large binary artifacts, associated metadata, and the complex relationships between models, data, and training code.

Key points to remember:

- Models require both artifact versioning (weights) and metadata versioning (configuration, metrics)
- Immutable versions prevent overwriting; create new versions instead of modifying existing ones
- Link model versions to code versions, data versions, and experiment runs
- Storage strategy matters: large models need specialized artifact storage
- Version naming should be meaningful and consistent across the organization

## Versioning Challenges for ML

### Large Binary Artifacts

Model weights are large binary files (megabytes to gigabytes) that do not diff meaningfully. Traditional version control systems like Git struggle with:

- Storage costs for many large files
- Clone times when pulling full history
- No meaningful diff between versions
- Poor performance with binary files

Solutions include Git LFS, DVC, and model registries with dedicated artifact storage.

### Multi-Artifact Versioning

A deployable model often includes multiple artifacts:

- Model weights (e.g., model.pt, model.h5)
- Preprocessing objects (scaler.pkl, tokenizer.json)
- Configuration files (config.yaml)
- Feature schema definitions
- Calibration artifacts

All artifacts must be versioned together as a coherent unit.

### Dependency Tracking

Model versions depend on:

- Training code version
- Training data version
- Hyperparameter configuration
- Library versions
- Hardware environment

Complete reproducibility requires tracking all dependencies.

## Versioning Approaches

### Semantic Versioning

Semantic versioning (major.minor.patch) communicates change significance:

- Major: Breaking changes to model interface or significant behavior changes
- Minor: Improved performance, new capabilities, backwards compatible
- Patch: Bug fixes, minor improvements

Example: Model v2.1.3
- v2: New architecture (breaking change from v1)
- .1: Added new input feature support
- .3: Fixed edge case prediction bug

Semantic versioning works well when humans manage version numbers and changes are discrete.

### Sequential Versioning

Simple incrementing integers (v1, v2, v3) or timestamps:

- Easy to implement and understand
- Clear ordering of versions
- Works well with automated training pipelines

Most model registries use sequential versioning by default.

### Hash-Based Versioning

Content-addressable versioning using artifact hashes:

- Guarantees version refers to exact content
- Enables deduplication of identical artifacts
- Automatic version identification without manual tagging

DVC uses hash-based versioning for data and models.

### Branch-Based Versioning

Parallel version lines for different purposes:

- main: Production-ready models
- experiment/new-feature: Testing new capabilities
- client/customer-a: Client-specific variants

Branch-based approaches work well with Git-based workflows but can become complex.

## Storage Strategies

### Git LFS

Git Large File Storage extends Git for large files:

- Files tracked by Git, stored externally
- Works with existing Git workflows
- Transparent to users after setup

Limitations:
- Storage costs can grow quickly
- Bandwidth limits on hosted services
- Still requires pulling large files

Best for: Teams already using Git who need simple large file handling.

### DVC (Data Version Control)

DVC provides Git-like versioning for data and models:

- .dvc files in Git reference external storage
- Multiple backend support (S3, GCS, Azure, local)
- Pipeline tracking with dvc.yaml
- Metrics tracking and comparison

DVC workflow:
```bash
dvc add model.pt           # Track model
git add model.pt.dvc       # Commit reference
dvc push                   # Upload to remote storage
```

Best for: Teams wanting data and model versioning integrated with Git workflows.

### Model Registry Storage

Model registries provide integrated artifact storage:

- MLflow stores artifacts in configurable backends
- W&B provides managed artifact storage
- SageMaker, Vertex AI use cloud-native storage

Registry storage advantages:
- Integrated with experiment tracking
- Stage management built-in
- Centralized access control

Best for: Teams using model registries as primary model management tool.

### Object Storage Direct

Direct use of S3, GCS, or Azure Blob with custom metadata:

- Full control over storage structure
- Use bucket versioning for artifact versions
- Store metadata in sidecar files or database

Requires more custom tooling but offers maximum flexibility.

## Implementation Patterns

### Version Naming Convention

Consistent naming across the organization:

```
{model_name}/{version}/{artifact}

examples:
fraud_detector/v12/model.pt
fraud_detector/v12/config.yaml
fraud_detector/v12/metadata.json
```

Include timestamp or commit hash when helpful:

```
fraud_detector/20240115_abc123/model.pt
```

### Metadata Structure

Standard metadata for all model versions:

```json
{
  "version": "v12",
  "created_at": "2024-01-15T10:30:00Z",
  "created_by": "user@company.com",
  "training": {
    "code_commit": "abc123",
    "data_version": "2024-01",
    "hyperparameters": {
      "learning_rate": 0.001,
      "epochs": 100
    }
  },
  "evaluation": {
    "test_accuracy": 0.95,
    "test_f1": 0.93
  },
  "artifacts": [
    "model.pt",
    "config.yaml",
    "preprocessor.pkl"
  ]
}
```

### Immutability

Versions should be immutable once created:

- Never overwrite existing versions
- Create new versions for any changes
- Use aliases for mutable references (latest, production)

Immutability ensures reproducibility and enables safe rollback.

### Linking Versions

Connect model versions to related artifacts:

```json
{
  "model_version": "v12",
  "links": {
    "experiment_run": "mlflow://runs/abc123",
    "training_data": "dvc://data/train.csv@v5",
    "code_commit": "github://org/repo@abc123",
    "parent_model": "models://fraud_detector/v11"
  }
}
```

Links enable lineage traversal and debugging.

## Version Comparison

### Metric Comparison

Compare versions on key metrics:

```
Version  Accuracy  F1     Latency  Size
v10      0.93      0.91   12ms     45MB
v11      0.94      0.92   15ms     52MB
v12      0.95      0.93   14ms     48MB
```

Automated comparison in CI/CD prevents deploying regressions.

### Behavioral Comparison

Compare predictions on standard test sets:

- Identify where versions differ
- Analyze error patterns
- Check for bias changes

Tools like model-diff can visualize prediction differences.

### Resource Comparison

Compare operational characteristics:

- Inference latency
- Memory usage
- Model file size
- GPU requirements

Resource changes affect deployment decisions.

## Version Lifecycle

### Creation

Version creation typically happens:

- At end of training pipeline
- Manually by data scientist
- Through automated retraining

Include all artifacts and metadata at creation time.

### Validation

New versions undergo validation:

- Automated metric checks
- Behavioral tests
- Performance benchmarks
- Human review for significant changes

Only validated versions proceed to staging.

### Deployment

Validated versions deploy to production:

- Stage transitions in model registry
- Deployment pipeline updates serving infrastructure
- Canary or blue-green deployment for safety

Record deployment events with version.

### Archival

Old versions are archived, not deleted:

- Retain for rollback capability
- Maintain audit trail
- Reference for debugging

Set retention policies based on compliance requirements.

## Rollback

### Rollback Strategy

Enable rapid rollback when issues arise:

1. Identify current production version
2. Identify target rollback version
3. Validate rollback version is compatible
4. Switch serving to rollback version
5. Monitor for success
6. Investigate original issue

### Rollback Automation

Automate rollback for speed:

```bash
# Example rollback script
CURRENT=$(get_production_version)
PREVIOUS=$(get_previous_version)
deploy_version $PREVIOUS
notify_team "Rolled back from $CURRENT to $PREVIOUS"
```

Test rollback procedures before they are needed.

### Compatibility Considerations

Not all versions are rollback-compatible:

- Schema changes in model inputs/outputs
- Feature store changes
- Dependency updates

Track compatibility metadata:

```json
{
  "version": "v12",
  "compatible_with": ["v11", "v10"],
  "breaking_changes_from": ["v9"]
}
```

## Best Practices

### Version Everything Together

When training produces a new model, version all artifacts together:
- Model weights
- Preprocessing artifacts
- Configuration
- Requirements

Partial updates create inconsistency risks.

### Automate Version Creation

Manual versioning is error-prone. Integrate versioning into training pipelines so every training run creates a proper version.

### Keep Versions Lightweight

Avoid bloating versions with unnecessary files. Store training logs and intermediate checkpoints separately from production artifacts.

### Document Breaking Changes

When versions introduce breaking changes, document clearly:
- What changed
- Migration path for dependent systems
- Deprecation timeline for old versions

### Clean Up Old Versions

Accumulating versions consumes storage and creates confusion. Establish retention policies:
- Keep production and staging versions
- Keep last N versions for each model
- Archive beyond retention period
