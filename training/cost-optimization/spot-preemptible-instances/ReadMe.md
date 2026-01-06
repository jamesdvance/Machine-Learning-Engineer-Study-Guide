# Spot and Preemptible Instances

## Summary

Spot instances (AWS), preemptible VMs (GCP), and spot VMs (Azure) offer significant cost savings (60-90%) compared to on-demand pricing by using spare cloud capacity. The tradeoff is that instances can be interrupted with little notice. For ML training workloads with proper checkpointing, spot instances are an excellent choice for reducing costs.

Key points to remember:

- 60-90% cost savings vs on-demand pricing
- Can be interrupted with 30s-2min notice
- Essential: frequent checkpointing for recovery
- Use instance types with high availability
- Consider spot-friendly architectures
- Bid strategies affect availability
- Multi-zone/region for better availability
- Not suitable for time-critical workloads

## Cloud Provider Options

### AWS Spot Instances

```python
# SageMaker with Spot
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p4d.24xlarge',
    instance_count=4,
    use_spot_instances=True,
    max_wait=72 * 3600,  # Max time to wait for spot
    max_run=48 * 3600,   # Max training time
    checkpoint_s3_uri='s3://bucket/checkpoints/',
)

estimator.fit()
```

### GCP Preemptible VMs

```yaml
# Vertex AI custom job
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-8g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 8
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 100
  scheduling:
    preemptible: true  # Use preemptible
```

### Azure Spot VMs

```python
# Azure ML
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute

compute = AmlCompute(
    name="spot-cluster",
    size="Standard_NC96ads_A100_v4",
    min_instances=0,
    max_instances=8,
    tier="low_priority",  # Spot instances
)

ml_client.compute.begin_create_or_update(compute)
```

## Interruption Handling

### Basic Checkpointing

```python
import signal
import torch

# Handle interruption signal
def handle_interrupt(signum, frame):
    print("Received interrupt, saving checkpoint...")
    save_checkpoint(model, optimizer, epoch, step)
    exit(0)

signal.signal(signal.SIGTERM, handle_interrupt)

# Checkpoint frequently
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        train_step(batch)

        # Save every N steps
        if step % 100 == 0:
            save_checkpoint(model, optimizer, epoch, step)
```

### Robust Checkpoint System

```python
import os
import torch
from pathlib import Path

class SpotCheckpointManager:
    def __init__(self, checkpoint_dir, max_to_keep=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.checkpoints = []

    def save(self, state, step):
        """Save checkpoint with rotation."""
        path = self.checkpoint_dir / f"checkpoint_{step}.pt"

        # Atomic save
        tmp_path = path.with_suffix('.tmp')
        torch.save(state, tmp_path)
        tmp_path.rename(path)

        # Rotate old checkpoints
        self.checkpoints.append(path)
        while len(self.checkpoints) > self.max_to_keep:
            old = self.checkpoints.pop(0)
            if old.exists():
                old.unlink()

    def load_latest(self):
        """Load most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            return torch.load(checkpoints[-1])
        return None

    def sync_to_cloud(self, cloud_path):
        """Sync checkpoints to cloud storage."""
        # Use cloud SDK to upload
        pass
```

### DeepSpeed Checkpointing

```python
import deepspeed

# DeepSpeed handles spot-friendly checkpointing
model_engine.save_checkpoint(
    save_dir='checkpoints/',
    tag='latest',
    exclude_frozen_parameters=True
)

# Automatic resume
model_engine.load_checkpoint(
    load_dir='checkpoints/',
    tag='latest'
)
```

## Spot-Friendly Training

### Training Loop with Recovery

```python
class SpotTrainer:
    def __init__(self, model, optimizer, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.start_epoch = 0
        self.start_step = 0

        # Try to resume
        self._try_resume()

    def _try_resume(self):
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            self.start_step = checkpoint['step']
            print(f"Resumed from epoch {self.start_epoch}, step {self.start_step}")

    def train(self, dataloader, num_epochs):
        for epoch in range(self.start_epoch, num_epochs):
            for step, batch in enumerate(dataloader):
                if epoch == self.start_epoch and step < self.start_step:
                    continue  # Skip already processed steps

                self._train_step(batch)

                if step % 100 == 0:
                    self._save_checkpoint(epoch, step)

    def _train_step(self, batch):
        # Training logic
        pass

    def _save_checkpoint(self, epoch, step):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        self.checkpoint_manager.save(state, epoch * 10000 + step)
```

## Availability Strategies

### Multi-Zone Deployment

```python
# AWS - specify multiple availability zones
from sagemaker import Session

session = Session()
estimator = PyTorch(
    # ...
    subnets=['subnet-az1', 'subnet-az2', 'subnet-az3'],
    security_group_ids=['sg-xxx'],
)
```

### Instance Type Fallback

```python
# Try multiple instance types
INSTANCE_PREFERENCES = [
    'ml.p4d.24xlarge',  # First choice
    'ml.p3.16xlarge',   # Fallback 1
    'ml.g5.48xlarge',   # Fallback 2
]

def launch_with_fallback(estimator, preferences):
    for instance_type in preferences:
        try:
            estimator.instance_type = instance_type
            estimator.fit()
            return
        except Exception as e:
            if "InsufficientCapacity" in str(e):
                continue
            raise
    raise Exception("No capacity available")
```

### Bid Strategies

```
AWS Spot:
- On-demand price cap (recommended)
- Custom max price
- Lowest price allocation

GCP:
- Preemptible (fixed discount)
- Spot (variable, lower)

Azure:
- Max price: -1 (on-demand cap)
- Custom max price
```

## Framework Integration

### PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    save_top_k=3,
    every_n_train_steps=100,  # Frequent saves for spot
)

trainer = Trainer(
    callbacks=[checkpoint_callback],
    resume_from_checkpoint='checkpoints/last.ckpt',
)
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./checkpoints',
    save_strategy='steps',
    save_steps=100,  # Frequent saves
    save_total_limit=3,
    load_best_model_at_end=True,
    resume_from_checkpoint=True,  # Auto-resume
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# Automatically resumes if checkpoint exists
trainer.train()
```

### Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Save state for interruption
accelerator.save_state('checkpoint/')

# Load state on resume
accelerator.load_state('checkpoint/')
```

## Cost Comparison

### Typical Savings

| Instance | On-Demand | Spot | Savings |
|----------|-----------|------|---------|
| p4d.24xlarge | $32.77/hr | $9.83/hr | 70% |
| p3.16xlarge | $24.48/hr | $7.34/hr | 70% |
| g5.48xlarge | $16.29/hr | $4.89/hr | 70% |

### Total Cost Calculation

```python
def calculate_spot_savings(
    training_hours,
    on_demand_rate,
    spot_rate,
    interruption_rate=0.05,  # 5% of hours lost to interruption
    checkpoint_overhead=0.02  # 2% overhead for checkpointing
):
    """Calculate expected savings with spot instances."""
    on_demand_cost = training_hours * on_demand_rate

    # Account for interruptions and overhead
    effective_hours = training_hours * (1 + interruption_rate + checkpoint_overhead)
    spot_cost = effective_hours * spot_rate

    savings = on_demand_cost - spot_cost
    savings_pct = (savings / on_demand_cost) * 100

    print(f"On-demand cost: ${on_demand_cost:.2f}")
    print(f"Spot cost: ${spot_cost:.2f}")
    print(f"Savings: ${savings:.2f} ({savings_pct:.1f}%)")
    return savings

calculate_spot_savings(
    training_hours=100,
    on_demand_rate=32.77,
    spot_rate=9.83
)
```

## Best Practices

1. **Checkpoint frequently**: Every 10-30 minutes
2. **Use cloud storage**: S3/GCS for checkpoint durability
3. **Multiple instance types**: Increase availability
4. **Multiple zones**: Better spot availability
5. **Monitor interruption rate**: Adjust strategy if high
6. **Set reasonable max price**: Usually on-demand cap
7. **Test recovery**: Verify checkpoint/resume works
8. **Use managed services**: SageMaker, Vertex handle complexity

## When NOT to Use Spot

- Urgent deadlines (unpredictable interruptions)
- Short runs (< 1 hour, may not complete)
- Debugging (interruptions disrupt workflow)
- Low checkpoint frequency (high rework cost)
- Interactive development
