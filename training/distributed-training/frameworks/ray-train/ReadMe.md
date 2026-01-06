# Ray Train

## Summary

Ray Train is the distributed training component of the Ray ecosystem, providing a flexible and scalable approach to distributed machine learning. It abstracts away the complexity of distributed training while integrating with popular frameworks like PyTorch, TensorFlow, and Hugging Face. Ray Train excels in dynamic environments where training resources may change and in organizations that use Ray for other ML workloads.

Key points to remember:

- Part of the Ray ecosystem for distributed computing
- Framework-agnostic with native integrations
- Fault tolerant with automatic recovery
- Integrates with Ray Tune for hyperparameter optimization
- Supports heterogeneous hardware and dynamic scaling
- Good for multi-cloud and Kubernetes deployments
- Less specialized optimization than DeepSpeed/Megatron
- Strong for end-to-end ML pipelines

## Installation

```bash
pip install "ray[train]"

# With specific framework support
pip install "ray[train]" torch
pip install "ray[train]" tensorflow
```

## Basic Usage

### PyTorch Training

```python
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func():
    # Get distributed training context
    context = train.get_context()

    # Setup is handled by Ray Train
    model = create_model()
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = train.get_dataset_shard("train")

    for epoch in range(10):
        for batch in dataset.iter_torch_batches(batch_size=32):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Report metrics to Ray Train
        train.report({"loss": loss.item(), "epoch": epoch})

# Configure training
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True
    ),
    datasets={"train": train_dataset}
)

result = trainer.fit()
```

### Scaling Configuration

```python
scaling_config = ScalingConfig(
    num_workers=8,              # Number of training workers
    use_gpu=True,               # Use GPUs
    resources_per_worker={      # Resources per worker
        "CPU": 4,
        "GPU": 1
    },
    trainer_resources={         # Resources for driver
        "CPU": 1
    }
)
```

## Data Loading

### Ray Data Integration

```python
import ray

# Create Ray Dataset
dataset = ray.data.read_parquet("s3://bucket/data/")

# Preprocess with Ray Data
dataset = dataset.map_batches(preprocess_fn)

# Pass to trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    datasets={"train": dataset},
    ...
)

# Access in training function
def train_func():
    dataset_shard = train.get_dataset_shard("train")
    for batch in dataset_shard.iter_torch_batches(batch_size=32):
        # Training step
        pass
```

### Streaming Data

```python
# Stream data without loading entirely into memory
dataset = ray.data.read_parquet("s3://bucket/large_data/")

for batch in dataset.iter_torch_batches(batch_size=32, prefetch_batches=2):
    # Process batch
    pass
```

## Checkpointing

### Automatic Checkpointing

```python
from ray.train import Checkpoint

def train_func():
    checkpoint = train.get_checkpoint()
    if checkpoint:
        # Resume from checkpoint
        state = checkpoint.to_dict()
        start_epoch = state["epoch"]
        model.load_state_dict(state["model"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        # Training loop
        ...

        # Save checkpoint
        checkpoint = Checkpoint.from_dict({
            "epoch": epoch,
            "model": model.state_dict(),
        })
        train.report({"loss": loss}, checkpoint=checkpoint)
```

### Checkpoint Configuration

```python
from ray.train import CheckpointConfig, RunConfig

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,                    # Keep last 3 checkpoints
        checkpoint_score_attribute="loss",
        checkpoint_score_order="min"      # Keep best checkpoints
    )
)

trainer = TorchTrainer(
    ...,
    run_config=run_config
)
```

## Fault Tolerance

### Automatic Recovery

```python
from ray.train import FailureConfig, RunConfig

run_config = RunConfig(
    failure_config=FailureConfig(
        max_failures=3  # Retry up to 3 times
    )
)

trainer = TorchTrainer(
    ...,
    run_config=run_config
)
```

Training automatically resumes from the last checkpoint on failure.

## Integration with Ray Tune

### Hyperparameter Tuning

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_func(config):
    lr = config["lr"]
    batch_size = config["batch_size"]

    model = create_model()
    model = train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(100):
        # Training with config hyperparameters
        ...
        train.report({"loss": loss})

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)

tuner = tune.Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64])
        }
    },
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=ASHAScheduler()
    )
)

results = tuner.fit()
```

## Hugging Face Integration

### Transformers Trainer

```python
from ray.train.huggingface import TransformersTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = TransformersTrainer(
    trainer_init_per_worker=lambda: get_trainer(training_args),
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    datasets={"train": train_dataset, "eval": eval_dataset}
)

result = trainer.fit()
```

### With DeepSpeed

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",
    ...
)

# Ray Train handles distributed setup
trainer = TransformersTrainer(
    trainer_init_per_worker=lambda: get_trainer(training_args),
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True)
)
```

## Distributed Data Parallel

### PyTorch DDP with Ray Train

```python
from ray.train.torch import TorchTrainer

def train_func():
    # Ray Train automatically wraps model with DDP
    model = create_model()
    model = train.torch.prepare_model(model)

    # Model is now DDP-wrapped
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)
```

### FSDP with Ray Train

```python
from ray.train.torch.fsdp import FSDPStrategy

def train_func():
    model = create_model()
    # Ray Train wraps with FSDP
    model = train.torch.prepare_model(
        model,
        parallel_strategy=FSDPStrategy()
    )
    ...

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    torch_config=TorchConfig(fsdp=True)
)
```

## Cluster Deployment

### Ray Cluster Setup

```yaml
# cluster.yaml
cluster_name: training-cluster

provider:
  type: aws
  region: us-west-2

available_node_types:
  head:
    node_config:
      InstanceType: m5.xlarge
    resources: {"CPU": 4}

  gpu-worker:
    node_config:
      InstanceType: p3.8xlarge
    resources: {"CPU": 32, "GPU": 4}
    min_workers: 2
    max_workers: 8
```

```bash
# Start cluster
ray up cluster.yaml

# Submit training job
ray job submit --address ray://cluster:10001 -- python train.py
```

### Kubernetes Deployment

```yaml
apiVersion: ray.io/v1alpha1
kind: RayJob
metadata:
  name: training-job
spec:
  entrypoint: python train.py
  runtimeEnv:
    pip:
      - torch
      - ray[train]
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              resources:
                limits:
                  cpu: "4"
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 4
        template:
          spec:
            containers:
              - name: ray-worker
                resources:
                  limits:
                    nvidia.com/gpu: "1"
```

## Monitoring

### Ray Dashboard

Access at `http://localhost:8265` when Ray cluster is running.

Shows:
- Worker status
- Resource utilization
- Training progress
- Logs

### Metrics Reporting

```python
def train_func():
    for epoch in range(num_epochs):
        # Training step
        ...

        # Report metrics
        train.report({
            "loss": loss.item(),
            "accuracy": accuracy,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0]
        })
```

### TensorBoard Integration

```python
from ray.train import RunConfig

run_config = RunConfig(
    storage_path="s3://bucket/experiments",
    name="my_experiment"
)

# TensorBoard logs saved to storage_path
```

## Best Practices

1. **Use Ray Data** for large datasets and preprocessing
2. **Enable checkpointing** for fault tolerance
3. **Configure failure recovery** for long training runs
4. **Use Ray Tune** for hyperparameter optimization
5. **Monitor via Ray Dashboard** during training
6. **Start simple** with basic TorchTrainer before adding complexity

## Comparison

| Aspect | Ray Train | PyTorch DDP | DeepSpeed |
|--------|-----------|-------------|-----------|
| Ecosystem integration | Excellent | Native | Good |
| Fault tolerance | Built-in | Manual | Manual |
| Dynamic scaling | Yes | No | No |
| HPO integration | Native | Separate | Separate |
| Memory optimization | Basic | FSDP | ZeRO |
| Setup complexity | Moderate | Low | Moderate |
