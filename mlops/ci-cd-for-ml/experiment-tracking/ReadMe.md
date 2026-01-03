# Experiment Tracking

## Summary

Experiment tracking systematically records the inputs, configurations, and outputs of ML experiments to enable reproducibility, comparison, and collaboration. It captures the relationship between code versions, data versions, hyperparameters, and resulting metrics, providing a complete audit trail of model development.

Key points to remember:

- Track everything needed to reproduce an experiment: code, data, config, environment, and results
- Use structured logging with consistent schemas across experiments
- Integrate experiment tracking into training scripts, not as an afterthought
- Compare experiments along multiple dimensions: metrics, resources, and artifacts
- Experiment tracking is the foundation for model registry and deployment pipelines

## Why Experiment Tracking Matters

ML development is inherently experimental. Engineers try different architectures, hyperparameters, data preprocessing steps, and training procedures. Without systematic tracking:

- Successful experiments cannot be reproduced
- Comparisons between runs are manual and error-prone
- Collaboration requires sharing notebooks and spreadsheets
- Debugging production issues lacks historical context
- Compliance and audit requirements are difficult to meet

Experiment tracking transforms ad-hoc experimentation into a rigorous, reproducible process.

## What to Track

### Code and Environment

- Git commit hash for training code
- Git diff for uncommitted changes
- Python environment (requirements.txt, poetry.lock, conda.yaml)
- Docker image tag if containerized
- Library versions for key dependencies

Code tracking enables reproducing the exact training logic. Environment tracking ensures the same libraries and versions are available.

### Data

- Dataset identifiers and versions
- Data file hashes or DVC commits
- Train/validation/test split parameters
- Data preprocessing configuration
- Feature engineering parameters

Data provenance is critical for reproducibility. A model cannot be reproduced without knowing exactly which data was used.

### Configuration

- Model architecture specification
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Training configuration (optimizer, loss function, regularization)
- Random seeds
- Hardware configuration (GPU type, distributed training setup)

Configuration should be captured automatically from config files or argument parsers.

### Metrics and Artifacts

- Training metrics over time (loss curves, validation metrics per epoch)
- Final evaluation metrics on test sets
- Model checkpoints and final weights
- Generated artifacts (plots, confusion matrices, sample predictions)
- Inference latency and resource usage

Metrics enable comparison. Artifacts provide deeper insight and debugging capability.

### Metadata

- Experiment name and description
- Author and timestamp
- Tags for organization
- Parent experiment (for hyperparameter sweeps or continuation)
- Notes and observations

Metadata helps organize and find experiments later.

## Experiment Tracking Platforms

### MLflow Tracking

MLflow is open-source and widely adopted. Its tracking component provides:

- Parameter and metric logging
- Artifact storage
- Experiment organization
- Search and comparison UI
- REST API for programmatic access

MLflow integrates with most ML frameworks and can be self-hosted or used as a managed service.

### Weights and Biases

W&B is a managed service with strong visualization capabilities:

- Rich dashboards and plots
- Automatic hyperparameter importance analysis
- Report generation for sharing
- Team collaboration features
- Integration with hyperparameter sweep tools

W&B is particularly strong for deep learning experiments with its live training visualization.

### Neptune

Neptune focuses on experiment management and collaboration:

- Flexible metadata structure
- Strong collaboration features
- Integration with notebooks
- Customizable dashboards

### Comet

Comet provides comprehensive experiment tracking with:

- Automatic environment capture
- Code diffing
- Model registry integration
- Production monitoring

### TensorBoard

TensorBoard is TensorFlow's visualization toolkit but works with PyTorch and other frameworks:

- Training curve visualization
- Hyperparameter comparison
- Model graph visualization
- Profiling tools

TensorBoard is lightweight but lacks the collaboration and management features of full platforms.

## Implementation Best Practices

### Automatic Logging

Reduce manual tracking burden with automatic logging:

- Auto-log hyperparameters from argparse or config files
- Auto-log metrics from training loops
- Auto-capture git commit and environment
- Auto-log model architecture summaries

Most platforms provide auto-logging integrations for common frameworks.

### Consistent Naming Conventions

Establish naming conventions for:

- Experiment names: project/task/variant pattern
- Parameter names: consistent across experiments
- Metric names: standardized across models
- Tag taxonomy: predefined categories

Consistent naming enables search, comparison, and aggregation.

### Hierarchical Organization

Organize experiments hierarchically:

- Projects: top-level grouping by business objective
- Experiments: groups of related runs
- Runs: individual training executions

This hierarchy helps navigate large numbers of experiments.

### Checkpointing Strategy

Save checkpoints at meaningful intervals:

- Best validation checkpoint
- Last checkpoint for resumption
- Periodic checkpoints for analysis
- Checkpoint metadata (step, metrics)

Link checkpoints to experiment tracking so they can be retrieved later.

### Configuration Management

Use structured configuration:

- YAML or JSON config files
- Hydra, OmegaConf, or similar libraries
- Config file versioning
- Environment variable overrides

Configuration management tools integrate well with experiment tracking, automatically logging config parameters.

## Experiment Comparison

### Metric Comparison

Compare experiments along key metrics:

- Parallel coordinates plots for hyperparameter analysis
- Scatter plots for metric correlations
- Tables sorted by target metrics
- Statistical significance tests for close results

### Resource Comparison

Consider efficiency alongside accuracy:

- Training time per epoch
- Total training time
- GPU memory usage
- Final model size
- Inference latency

A slightly less accurate model might be preferable if it trains in half the time or runs twice as fast.

### Artifact Comparison

Compare qualitative outputs:

- Confusion matrices side-by-side
- Sample predictions on the same examples
- Learning curves overlaid
- Attention visualizations

Artifact comparison often reveals insights that metrics alone miss.

## Integration with CI/CD

### Triggered Experiments

CI/CD can trigger experiments:

- Retrain models when data updates
- Run hyperparameter sweeps on schedule
- Validate experiments on PR changes

Link CI/CD job IDs to experiment runs for traceability.

### Experiment-Based Promotion

Use experiment tracking as the source of truth for promotion:

- Query tracking system for best-performing experiments
- Compare candidate against production baseline
- Promote based on defined criteria
- Record promotion decision and rationale

### Reproducibility Checks

Periodically verify reproducibility:

- Re-run random sample of historical experiments
- Compare metrics within tolerance
- Flag experiments that cannot be reproduced

Reproducibility checks catch environment drift and data staleness.

## Common Pitfalls

### Tracking Too Little

Minimal tracking seems sufficient until you need to reproduce or debug. Track comprehensively from the start; storage is cheap, missing information is costly.

### Tracking Too Late

Adding experiment tracking after development leads to untracked historical experiments. Integrate tracking at project start.

### Inconsistent Tracking

Experiments tracked differently are hard to compare. Establish conventions and enforce them through code review and templates.

### Ignoring Failed Experiments

Failed experiments contain valuable information. Track them with notes about why they failed. Negative results prevent repeating mistakes.

### Manual Logging

Manual logging is tedious and error-prone. Automate logging through framework integrations and wrapper functions.

### Orphaned Artifacts

Artifacts stored separately from experiment records become disconnected. Use experiment tracking artifact storage or maintain clear links.
