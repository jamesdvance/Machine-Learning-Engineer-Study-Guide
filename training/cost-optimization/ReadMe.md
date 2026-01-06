# Cost Optimization for ML Training

## Summary

Training machine learning models at scale requires significant compute resources, making cost optimization essential. Effective strategies include choosing the right hardware, using spot instances, implementing early stopping, and proper budget planning. Understanding the cost-performance tradeoffs enables training better models within budget constraints.

Key points to remember:

- Spot/preemptible instances offer 60-90% savings
- Right-sizing GPU selection prevents overspending
- Early stopping avoids wasted compute on poor runs
- Budget estimation prevents cost overruns
- Learning rate scheduling affects training efficiency
- Auto-scaling optimizes resource utilization
- Monitor cost per training step
- Balance training speed vs cost

## Cost Breakdown

### Typical Training Costs

| Component | % of Total | Notes |
|-----------|------------|-------|
| GPU compute | 70-90% | Main cost driver |
| Storage | 5-15% | Data, checkpoints |
| Networking | 2-10% | Multi-node communication |
| CPU/Memory | 2-5% | Data preprocessing |

### GPU Pricing (Approximate)

| GPU | On-Demand ($/hr) | Spot ($/hr) | Memory |
|-----|-----------------|-------------|--------|
| T4 | $0.35-0.50 | $0.10-0.15 | 16 GB |
| A10G | $1.00-1.50 | $0.30-0.45 | 24 GB |
| A100 40GB | $3.00-4.00 | $0.90-1.20 | 40 GB |
| A100 80GB | $4.00-5.00 | $1.20-1.50 | 80 GB |
| H100 | $4.50-5.50 | $1.35-1.65 | 80 GB |

*Prices vary by cloud provider and region*

## Cost Optimization Strategies

### Hardware-Level

```
1. Use spot/preemptible instances (60-90% savings)
2. Right-size GPU selection (don't over-provision)
3. Use GPU memory efficiently (maximize batch size)
4. Consider older GPU generations for small models
```

### Training-Level

```
1. Early stopping for unpromising runs
2. Efficient learning rate scheduling
3. Mixed precision training (faster = cheaper)
4. Gradient checkpointing (fit larger batches)
```

### Infrastructure-Level

```
1. Auto-scaling based on queue depth
2. Multi-tenant GPU sharing
3. Preemption-aware training
4. Efficient data loading (no GPU idle time)
```

## Quick Cost Calculations

### Training Cost Estimation

```python
def estimate_training_cost(
    model_size_params,
    dataset_tokens,
    throughput_tokens_per_sec,
    gpu_cost_per_hour,
    num_gpus
):
    """Estimate training cost."""
    # Training tokens (6x for forward + backward + extra)
    total_flops = 6 * model_size_params * dataset_tokens

    # Time calculation
    seconds = dataset_tokens / (throughput_tokens_per_sec * num_gpus)
    hours = seconds / 3600

    # Cost
    total_cost = hours * gpu_cost_per_hour * num_gpus

    print(f"Training time: {hours:.1f} hours")
    print(f"Total cost: ${total_cost:.2f}")
    return total_cost

# Example: 7B model, 100B tokens
estimate_training_cost(
    model_size_params=7e9,
    dataset_tokens=100e9,
    throughput_tokens_per_sec=10000,  # Per GPU
    gpu_cost_per_hour=4.0,  # A100
    num_gpus=8
)
```

### Cost per Step

```python
def cost_per_step(gpu_cost_per_hour, step_time_seconds, num_gpus):
    """Calculate cost per training step."""
    cost_per_second = gpu_cost_per_hour / 3600
    return cost_per_second * step_time_seconds * num_gpus

# Example
cost = cost_per_step(
    gpu_cost_per_hour=4.0,
    step_time_seconds=0.5,
    num_gpus=8
)
print(f"Cost per step: ${cost:.4f}")
```

## Cost Monitoring

### Track Training Costs

```python
import wandb

class CostTracker:
    def __init__(self, gpu_cost_per_hour, num_gpus):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.num_gpus = num_gpus
        self.start_time = time.time()

    def get_cost(self):
        hours = (time.time() - self.start_time) / 3600
        return hours * self.gpu_cost_per_hour * self.num_gpus

    def log(self, step):
        cost = self.get_cost()
        wandb.log({
            "cost/total_usd": cost,
            "cost/per_step": cost / (step + 1),
        }, step=step)

# Usage
tracker = CostTracker(gpu_cost_per_hour=4.0, num_gpus=8)
for step in range(num_steps):
    train_step()
    if step % 100 == 0:
        tracker.log(step)
```

### Set Cost Alerts

```python
MAX_BUDGET = 1000  # USD

def check_budget(tracker, step):
    cost = tracker.get_cost()
    if cost > MAX_BUDGET * 0.8:
        wandb.alert(
            title="Budget warning",
            text=f"Training cost ${cost:.2f} exceeds 80% of budget",
            level=wandb.AlertLevel.WARN
        )
    if cost > MAX_BUDGET:
        raise Exception(f"Budget exceeded: ${cost:.2f}")
```

## Decision Framework

### When to Use Spot Instances

```
Use Spot:
- Long training runs (hours to days)
- Fault-tolerant training (checkpointing)
- Hyperparameter sweeps
- Development and experimentation

Use On-Demand:
- Production deadlines
- Short runs (< 1 hour)
- Debugging and testing
- When availability is critical
```

### GPU Selection Guide

```
Model < 1B parameters:
  -> T4 (16GB) or A10G (24GB)

Model 1B - 7B parameters:
  -> A10G (24GB) or A100 40GB

Model 7B - 70B parameters:
  -> A100 80GB or H100

Model > 70B parameters:
  -> Multi-GPU with model parallelism
```

## Best Practices

1. **Start small**: Validate on small scale before scaling
2. **Use spot instances**: Default for training workloads
3. **Checkpoint frequently**: Enable preemption recovery
4. **Monitor cost metrics**: Track $/step and $/epoch
5. **Right-size hardware**: Don't over-provision
6. **Early stopping**: Kill unpromising runs
7. **Efficient data loading**: No GPU idle time
8. **Reserved capacity**: For predictable workloads

## Further Reading

- [Spot/Preemptible Instances](spot-preemptible-instances/ReadMe.md): 60-90% cost savings
- [Right-Sizing GPU Selection](right-sizing-gpu-selection/ReadMe.md): Optimal hardware choice
- [Early Stopping](early-stopping/ReadMe.md): Avoid wasted compute
- [Training Budget Estimation](training-budget-estimation/ReadMe.md): Cost planning
- [Learning Rate Scheduling](learning-rate-scheduling/ReadMe.md): Training efficiency
- [Auto-Scaling](auto-scaling/ReadMe.md): Resource optimization
