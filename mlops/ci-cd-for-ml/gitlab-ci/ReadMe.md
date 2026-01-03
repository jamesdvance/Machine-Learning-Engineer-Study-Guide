# GitLab CI for ML

## Summary

GitLab CI/CD provides integrated continuous integration and deployment within GitLab repositories. For ML projects, it offers pipeline automation, container registry integration, Kubernetes deployment, and artifact management. The `.gitlab-ci.yml` configuration file defines pipelines as code.

Key points to remember:

- Pipelines are defined in `.gitlab-ci.yml` with stages, jobs, and scripts
- GitLab runners execute jobs; self-hosted runners enable GPU access
- Built-in container registry simplifies Docker workflow for ML
- DAG pipelines allow complex job dependencies beyond linear stages
- Integration with Kubernetes enables scaled training and serving deployment

## Core Concepts

### Pipelines and Stages

A pipeline is a collection of jobs organized into stages. Stages run sequentially; jobs within a stage run in parallel.

```yaml
stages:
  - test
  - train
  - validate
  - deploy

test:
  stage: test
  script:
    - pytest tests/

train:
  stage: train
  script:
    - python train.py

validate:
  stage: validate
  script:
    - python validate.py

deploy:
  stage: deploy
  script:
    - python deploy.py
```

Jobs in later stages only run if all jobs in earlier stages succeed.

### Jobs

Jobs define what to execute:

```yaml
train_model:
  stage: train
  image: pytorch/pytorch:latest
  script:
    - pip install -r requirements.txt
    - python train.py
  artifacts:
    paths:
      - model.pt
    expire_in: 1 week
  timeout: 6 hours
  tags:
    - gpu
```

Key job properties:
- `image`: Docker image to run in
- `script`: Commands to execute
- `artifacts`: Files to preserve after job
- `timeout`: Maximum job duration
- `tags`: Runner selection criteria

### Runners

GitLab runners execute jobs. Types include:

Shared runners:
- Managed by GitLab.com or your GitLab instance
- Good for standard CI tasks
- CPU-only on GitLab.com

Specific runners:
- Dedicated to your project or group
- Custom configuration and software
- GPU support through your infrastructure

Runner selection uses tags:

```yaml
train:
  tags:
    - gpu
    - high-memory
```

### Variables

Variables configure jobs:

```yaml
variables:
  BATCH_SIZE: "64"
  LEARNING_RATE: "0.001"

train:
  script:
    - python train.py --batch-size $BATCH_SIZE --lr $LEARNING_RATE
```

Variable sources:
- Defined in `.gitlab-ci.yml`
- Project/group settings (for secrets)
- Predefined CI/CD variables
- Pipeline triggers

Sensitive values should be masked and protected in project settings.

## ML Workflow Patterns

### Testing Pipeline

Run tests on merge requests:

```yaml
stages:
  - lint
  - test
  - validate

lint:
  stage: lint
  image: python:3.11
  script:
    - pip install ruff
    - ruff check .

unit_tests:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/unit/

data_validation:
  stage: validate
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python scripts/validate_data.py
```

### Training Pipeline

Training workflows with GPU support:

```yaml
stages:
  - prepare
  - train
  - evaluate

prepare_data:
  stage: prepare
  script:
    - python scripts/prepare_data.py
  artifacts:
    paths:
      - data/processed/

train_model:
  stage: train
  tags:
    - gpu
  image: nvcr.io/nvidia/pytorch:24.01-py3
  script:
    - python train.py
  artifacts:
    paths:
      - outputs/model.pt
      - outputs/metrics.json
  timeout: 8 hours

evaluate:
  stage: evaluate
  script:
    - python evaluate.py
  dependencies:
    - train_model
```

### Hyperparameter Search

Parallel job matrix for hyperparameter sweeps:

```yaml
train:
  stage: train
  tags:
    - gpu
  parallel:
    matrix:
      - LEARNING_RATE: ["0.001", "0.01", "0.1"]
        BATCH_SIZE: ["32", "64"]
  script:
    - python train.py --lr $LEARNING_RATE --batch $BATCH_SIZE
  artifacts:
    paths:
      - outputs/
```

The parallel matrix creates 6 jobs (3 learning rates x 2 batch sizes).

### DAG Pipelines

Use `needs` for complex dependencies:

```yaml
stages:
  - prepare
  - train
  - aggregate

prepare_a:
  stage: prepare
  script: python prepare_a.py

prepare_b:
  stage: prepare
  script: python prepare_b.py

train_a:
  stage: train
  needs: [prepare_a]
  script: python train_a.py

train_b:
  stage: train
  needs: [prepare_b]
  script: python train_b.py

aggregate:
  stage: aggregate
  needs: [train_a, train_b]
  script: python aggregate.py
```

DAG pipelines start jobs as soon as dependencies complete, rather than waiting for entire stages.

### Scheduled Retraining

Schedule pipelines with GitLab schedules:

```yaml
scheduled_train:
  stage: train
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
  script:
    - python scripts/fetch_data.py
    - python train.py
```

Configure schedules in GitLab UI: Settings > CI/CD > Pipeline schedules.

## Container Registry Integration

GitLab provides an integrated container registry.

### Building Training Images

```yaml
build_image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

train:
  stage: train
  image: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  script:
    - python train.py
```

The built image ensures training runs in a reproducible environment.

### Multi-Stage Builds for ML

Efficient Dockerfiles for ML:

```dockerfile
# Build stage
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip wheel -r requirements.txt -w /wheels

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
COPY . /app
WORKDIR /app
```

Multi-stage builds reduce image size by excluding build dependencies.

## Artifact Management

### Model Artifacts

Preserve trained models:

```yaml
train:
  artifacts:
    paths:
      - models/
    expire_in: 30 days
    reports:
      metrics: metrics.txt
```

Metrics reports appear in merge request UI.

### Artifact Dependencies

Control artifact flow between jobs:

```yaml
train:
  stage: train
  artifacts:
    paths:
      - model.pt

deploy:
  stage: deploy
  dependencies:
    - train  # Only get artifacts from train job
  script:
    - python deploy.py model.pt
```

Use `dependencies: []` to skip artifact download for jobs that do not need them.

### External Artifact Storage

For large models, use external storage:

```yaml
train:
  script:
    - python train.py
    - aws s3 cp model.pt s3://bucket/models/$CI_COMMIT_SHA/

deploy:
  script:
    - aws s3 cp s3://bucket/models/$CI_COMMIT_SHA/model.pt .
    - python deploy.py
```

External storage avoids GitLab artifact size limits and provides permanent storage.

## Kubernetes Integration

### GitLab Kubernetes Agent

Connect GitLab to Kubernetes clusters for deployment:

1. Install the GitLab agent in your cluster
2. Configure repository access
3. Define deployment jobs

```yaml
deploy_to_k8s:
  stage: deploy
  image: bitnami/kubectl
  script:
    - kubectl apply -f k8s/deployment.yaml
```

### Kubernetes Executors

Run CI jobs in Kubernetes pods:

```yaml
train:
  tags:
    - kubernetes
  image: pytorch/pytorch
  script:
    - python train.py
```

Kubernetes executors enable autoscaling and resource management.

## Caching

### Dependency Caching

Cache Python packages:

```yaml
variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

.python_template:
  cache:
    key: $CI_COMMIT_REF_SLUG
    paths:
      - .cache/pip/
      - venv/
  before_script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
```

### Data Caching

Cache preprocessed data:

```yaml
preprocess:
  cache:
    key: data-$CI_COMMIT_SHA
    paths:
      - data/processed/
  script:
    - python preprocess.py
```

Be cautious with data caching; stale cached data causes subtle bugs.

## Security

### Protected Variables

Store secrets as protected variables in project settings:

```yaml
deploy:
  script:
    - echo $AWS_SECRET_ACCESS_KEY  # Never do this
    - aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
```

Protected variables are only exposed to protected branches.

### Protected Branches

Restrict who can push to production branches and run deployments. Protected branches only allow maintainers to merge.

### Container Scanning

GitLab includes security scanning:

```yaml
include:
  - template: Security/Container-Scanning.gitlab-ci.yml

container_scanning:
  variables:
    CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## Comparison with GitHub Actions

Feature differences:

GitLab CI advantages:
- Integrated container registry
- Native Kubernetes integration
- More sophisticated pipeline DAGs
- Better for self-managed instances
- Built-in security scanning

GitHub Actions advantages:
- Larger action marketplace
- Matrix builds more flexible
- Easier self-hosted runner setup
- Tighter GitHub ecosystem integration

Both handle ML workflows effectively. The choice typically depends on where your code lives.

## Common Pitfalls

### Runner Capacity

GPU runners are expensive and limited. Long training jobs can queue if runners are busy. Monitor runner utilization and scale appropriately.

### Artifact Size Limits

GitLab has artifact size limits (configurable for self-managed). Large models should use external storage.

### Pipeline Complexity

Complex pipelines become hard to maintain. Use includes and templates:

```yaml
include:
  - local: .gitlab/ci/test.yml
  - local: .gitlab/ci/train.yml
  - local: .gitlab/ci/deploy.yml
```

### Cache Invalidation

Stale caches cause subtle bugs. Include cache-busting keys:

```yaml
cache:
  key: $CI_COMMIT_REF_SLUG-$CI_COMMIT_SHA
```

Or manually clear caches when needed.
