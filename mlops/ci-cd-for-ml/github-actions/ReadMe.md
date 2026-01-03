# GitHub Actions for ML

## Summary

GitHub Actions provides CI/CD automation that integrates directly with GitHub repositories. For ML projects, it enables automated testing, model training, validation, and deployment triggered by code changes, schedules, or manual dispatch. The workflow-as-code approach makes ML pipelines version-controlled and reproducible.

Key points to remember:

- Workflows are YAML files defining triggers, jobs, and steps
- Self-hosted runners enable GPU access and custom environments
- Matrix builds parallelize hyperparameter sweeps and cross-platform testing
- Caching reduces build times for large dependencies
- Secrets management handles API keys and cloud credentials securely

## Core Concepts

### Workflows

Workflows are YAML files in `.github/workflows/` that define automated processes. A workflow contains:

- Triggers: Events that start the workflow (push, PR, schedule, manual)
- Jobs: Groups of steps that run on a single runner
- Steps: Individual commands or actions

Workflows execute in isolated environments. Each job gets a fresh virtual machine or container.

### Runners

Runners are machines that execute workflow jobs:

GitHub-hosted runners:
- Ubuntu, Windows, macOS available
- Pre-installed common tools
- Clean environment each run
- Limited to CPU workloads
- Usage limits on free tier

Self-hosted runners:
- Your own machines or cloud instances
- GPU access for training
- Custom software installations
- Persistent environment between runs
- No GitHub usage limits

For ML workloads requiring GPUs, self-hosted runners are typically necessary.

### Actions

Actions are reusable workflow components:

- Published actions from GitHub Marketplace
- Repository-local actions
- Docker container actions
- JavaScript actions

Common ML-related actions:
- setup-python: Configure Python version
- cache: Cache dependencies between runs
- upload-artifact/download-artifact: Share files between jobs
- actions/checkout: Clone repository

## ML Workflow Patterns

### Testing on Pull Requests

Run tests when PRs are opened or updated:

```yaml
name: ML Tests
on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
      - name: Run data validation
        run: python scripts/validate_data.py
```

Keep PR checks fast (under 10 minutes) for rapid feedback.

### Model Training Workflows

Trigger training on data updates or manual dispatch:

```yaml
name: Train Model
on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name'
        required: true
  push:
    paths:
      - 'data/**'

jobs:
  train:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Train model
        run: python train.py --name ${{ github.event.inputs.experiment_name }}
      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: outputs/model.pt
```

Self-hosted runners with GPUs handle the actual training.

### Model Validation Pipeline

Validate models before deployment:

```yaml
name: Validate Model
on:
  workflow_run:
    workflows: ["Train Model"]
    types: [completed]

jobs:
  validate:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download model
        uses: actions/download-artifact@v4
        with:
          name: model
          run-id: ${{ github.event.workflow_run.id }}
      - name: Run validation
        run: python scripts/validate_model.py
      - name: Check metrics
        run: python scripts/check_thresholds.py
```

Chain workflows to create multi-stage pipelines.

### Scheduled Retraining

Retrain models on a schedule:

```yaml
name: Scheduled Retrain
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly at 2am Sunday

jobs:
  retrain:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Fetch latest data
        run: python scripts/fetch_data.py
      - name: Train model
        run: python train.py
      - name: Validate and deploy
        run: python scripts/validate_and_deploy.py
```

Scheduled workflows ensure models stay current with fresh data.

### Hyperparameter Sweeps

Use matrix builds for parallel experiments:

```yaml
name: Hyperparameter Sweep
on: workflow_dispatch

jobs:
  sweep:
    runs-on: self-hosted
    strategy:
      matrix:
        learning_rate: [0.001, 0.01, 0.1]
        batch_size: [32, 64, 128]
    steps:
      - uses: actions/checkout@v4
      - name: Train
        run: |
          python train.py \
            --lr ${{ matrix.learning_rate }} \
            --batch-size ${{ matrix.batch_size }}
```

Matrix builds run experiments in parallel, limited by available runners.

## Self-Hosted Runners for ML

### Setting Up GPU Runners

1. Provision a machine with GPU (cloud instance or on-premises)
2. Install NVIDIA drivers and CUDA
3. Install the GitHub Actions runner software
4. Register the runner with your repository or organization
5. Configure labels for routing (e.g., `gpu`, `high-memory`)

### Runner Management

For production ML, consider:

- Autoscaling runner pools based on queue depth
- Ephemeral runners that terminate after each job
- Container-based runners for isolation
- Monitoring runner health and utilization

Tools like actions-runner-controller (for Kubernetes) automate runner management.

### Security Considerations

Self-hosted runners have security implications:

- Runners have access to secrets and code
- Malicious PRs could execute arbitrary code
- Network access to internal resources

Mitigations:
- Require PR approval before running workflows
- Use ephemeral runners that reset after each job
- Network isolation for runners
- Separate runners for different trust levels

## Caching and Optimization

### Dependency Caching

Cache Python dependencies to speed up workflows:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

Caching cuts minutes from workflows that would otherwise reinstall large packages.

### Model and Data Caching

Cache large files that do not change frequently:

```yaml
- uses: actions/cache@v4
  with:
    path: data/preprocessed
    key: preprocessed-data-${{ hashFiles('data/raw/**') }}
```

Be careful with data caching; stale cached data can cause subtle bugs.

### Artifact Management

Use artifacts to pass files between jobs:

```yaml
jobs:
  train:
    steps:
      - name: Upload model
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: model.pt
          retention-days: 7

  deploy:
    needs: train
    steps:
      - name: Download model
        uses: actions/download-artifact@v4
        with:
          name: model
```

Artifacts are temporary; use external storage (S3, GCS) for long-term model storage.

## Secrets Management

Store sensitive values as repository or organization secrets:

```yaml
- name: Deploy to cloud
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: python deploy.py
```

Secret best practices:
- Use least-privilege credentials
- Rotate secrets regularly
- Use OIDC authentication instead of long-lived credentials where possible
- Audit secret access

## Comparison with GitLab CI

GitHub Actions and GitLab CI offer similar capabilities with different syntax and features:

GitHub Actions advantages:
- Tighter GitHub integration
- Larger marketplace of reusable actions
- Matrix builds are more flexible
- Easier self-hosted runner setup

GitLab CI advantages:
- Built-in container registry
- Integrated Kubernetes deployment
- More sophisticated DAG pipelines
- Better for air-gapped environments

Both can handle ML workflows effectively. Choose based on where your code lives and team familiarity.

## Common Patterns

### Conditional Execution

Run steps conditionally:

```yaml
- name: Deploy to production
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: python deploy.py
```

### Timeout Handling

Set timeouts for long-running jobs:

```yaml
jobs:
  train:
    timeout-minutes: 360  # 6 hours
    runs-on: self-hosted
```

Training jobs need longer timeouts than typical CI jobs.

### Failure Handling

Continue workflow despite step failures:

```yaml
- name: Run optional validation
  continue-on-error: true
  run: python optional_check.py
```

Use sparingly; most failures should halt the workflow.

## Common Pitfalls

### Exceeding Usage Limits

GitHub-hosted runners have monthly minute limits. Long training jobs can exhaust limits quickly. Use self-hosted runners for heavy workloads.

### Missing GPU Support

GitHub-hosted runners lack GPUs. Workflows assuming GPU availability will fail. Use self-hosted GPU runners or CPU-compatible code paths.

### Large File Handling

GitHub has file size limits. Large models or datasets should be stored externally (S3, GCS, Hugging Face Hub) and downloaded during workflows.

### Workflow Complexity

Complex workflows become hard to maintain. Extract reusable components into custom actions. Keep individual workflows focused on single purposes.
