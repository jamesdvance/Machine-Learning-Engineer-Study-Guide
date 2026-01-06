# Training Budget Estimation

## Summary

Accurate training budget estimation prevents cost overruns and enables proper resource planning. Estimation involves calculating compute requirements from model size, dataset size, and training configuration, then translating to infrastructure costs. Good estimates account for failed runs, hyperparameter tuning, and iteration cycles.

Key points to remember:

- FLOPs = 6 x parameters x tokens (transformer training)
- Include optimizer state and activation memory
- Account for failed runs and restarts
- Plan for hyperparameter search costs
- Consider spot vs on-demand tradeoffs
- Include storage and networking costs
- Add buffer for unexpected issues (20-50%)
- Track actual vs estimated for future improvement

## Compute Requirements

### FLOPs Estimation

```python
def estimate_training_flops(num_params, num_tokens, num_epochs=1):
    """
    Estimate total FLOPs for training.

    Rule of thumb for transformers:
    - Forward pass: ~2 * params * tokens FLOPs
    - Backward pass: ~4 * params * tokens FLOPs
    - Total: ~6 * params * tokens FLOPs
    """
    flops_per_token = 6 * num_params
    total_flops = flops_per_token * num_tokens * num_epochs

    print(f"Parameters: {num_params / 1e9:.1f}B")
    print(f"Tokens: {num_tokens / 1e9:.1f}B")
    print(f"Total FLOPs: {total_flops / 1e21:.2f} ZFLOPs")

    return total_flops

# Example: Train 7B model on 1T tokens
estimate_training_flops(
    num_params=7e9,
    num_tokens=1e12
)
```

### GPU Hours from FLOPs

```python
def flops_to_gpu_hours(total_flops, gpu_type='A100'):
    """Convert FLOPs to GPU hours."""
    # Peak FP16 TFLOPS
    gpu_tflops = {
        'T4': 65,
        'A10G': 125,
        'A100': 312,
        'H100': 990,
    }

    # Assume 40% utilization (typical for training)
    utilization = 0.4
    effective_tflops = gpu_tflops[gpu_type] * utilization

    # Convert to hours
    tflops_per_hour = effective_tflops * 3600
    flops_per_hour = tflops_per_hour * 1e12

    gpu_hours = total_flops / flops_per_hour

    print(f"GPU: {gpu_type}")
    print(f"Effective TFLOPS: {effective_tflops:.0f}")
    print(f"GPU hours: {gpu_hours:.0f}")

    return gpu_hours

# Example
flops = estimate_training_flops(7e9, 100e9)
gpu_hours = flops_to_gpu_hours(flops, 'A100')
```

## Cost Calculation

### Basic Cost Estimation

```python
def estimate_training_cost(
    num_params,
    num_tokens,
    gpu_type='A100',
    num_gpus=8,
    spot_pricing=True,
    num_epochs=1
):
    """Estimate total training cost."""

    # GPU pricing ($/hour)
    prices = {
        'T4': {'spot': 0.12, 'on_demand': 0.35},
        'A10G': {'spot': 0.35, 'on_demand': 1.00},
        'A100': {'spot': 1.00, 'on_demand': 4.00},
        'H100': {'spot': 1.50, 'on_demand': 5.00},
    }

    # Calculate FLOPs and GPU hours
    total_flops = 6 * num_params * num_tokens * num_epochs
    single_gpu_hours = flops_to_gpu_hours(total_flops, gpu_type)
    wall_clock_hours = single_gpu_hours / num_gpus

    # Account for communication overhead
    efficiency = 0.85 if num_gpus > 1 else 1.0
    actual_hours = wall_clock_hours / efficiency

    # Calculate cost
    price_type = 'spot' if spot_pricing else 'on_demand'
    hourly_cost = prices[gpu_type][price_type] * num_gpus
    total_cost = actual_hours * hourly_cost

    print(f"\n=== Training Cost Estimate ===")
    print(f"Model: {num_params / 1e9:.1f}B parameters")
    print(f"Dataset: {num_tokens / 1e9:.1f}B tokens")
    print(f"Hardware: {num_gpus}x {gpu_type}")
    print(f"Wall clock time: {actual_hours:.1f} hours")
    print(f"Pricing: {price_type} (${prices[gpu_type][price_type]}/GPU/hr)")
    print(f"Total cost: ${total_cost:.2f}")

    return {
        'hours': actual_hours,
        'cost': total_cost,
        'gpu_hours': single_gpu_hours
    }

# Example
estimate_training_cost(
    num_params=7e9,
    num_tokens=100e9,
    gpu_type='A100',
    num_gpus=8,
    spot_pricing=True
)
```

### Full Project Budget

```python
def estimate_project_budget(
    base_training_cost,
    num_hparam_trials=20,
    hparam_budget_fraction=0.2,
    num_full_runs=3,
    failure_rate=0.2,
    storage_cost_per_month=100,
    project_months=3
):
    """Estimate full project budget including iteration."""

    # Hyperparameter search (reduced budget per trial)
    hparam_cost = num_hparam_trials * base_training_cost * hparam_budget_fraction

    # Full training runs
    full_run_cost = num_full_runs * base_training_cost

    # Account for failures and restarts
    failure_overhead = (hparam_cost + full_run_cost) * failure_rate

    # Storage costs
    storage = storage_cost_per_month * project_months

    # Networking and misc (10% of compute)
    misc = (hparam_cost + full_run_cost) * 0.1

    # Total
    compute_total = hparam_cost + full_run_cost + failure_overhead
    total = compute_total + storage + misc

    # Add safety buffer
    buffer = total * 0.25
    final_budget = total + buffer

    print(f"\n=== Project Budget ===")
    print(f"Hyperparameter search: ${hparam_cost:.2f}")
    print(f"Full training runs: ${full_run_cost:.2f}")
    print(f"Failure overhead: ${failure_overhead:.2f}")
    print(f"Storage: ${storage:.2f}")
    print(f"Misc: ${misc:.2f}")
    print(f"Subtotal: ${total:.2f}")
    print(f"Buffer (25%): ${buffer:.2f}")
    print(f"Total budget: ${final_budget:.2f}")

    return final_budget

# Example
base_cost = estimate_training_cost(7e9, 100e9, 'A100', 8, True)['cost']
estimate_project_budget(base_cost)
```

## Memory Requirements

### Estimate GPU Memory

```python
def estimate_memory_requirements(
    num_params,
    batch_size,
    seq_len,
    hidden_dim,
    num_layers,
    precision='fp16'
):
    """Estimate GPU memory requirements."""

    bytes_per_param = 2 if precision == 'fp16' else 4

    # Parameters
    params_mem = num_params * bytes_per_param

    # Gradients
    grads_mem = num_params * bytes_per_param

    # Optimizer (Adam: 2 states per param in FP32)
    optimizer_mem = num_params * 8

    # Activations (rough estimate)
    activation_mem = (
        batch_size * seq_len * hidden_dim * num_layers *
        bytes_per_param * 12  # Factor for intermediates
    )

    total = params_mem + grads_mem + optimizer_mem + activation_mem

    print(f"\n=== Memory Requirements ===")
    print(f"Parameters: {params_mem / 1e9:.2f} GB")
    print(f"Gradients: {grads_mem / 1e9:.2f} GB")
    print(f"Optimizer: {optimizer_mem / 1e9:.2f} GB")
    print(f"Activations: {activation_mem / 1e9:.2f} GB")
    print(f"Total: {total / 1e9:.2f} GB")

    # Recommend GPU
    if total < 16e9:
        print("Recommended: T4 (16GB)")
    elif total < 24e9:
        print("Recommended: A10G (24GB)")
    elif total < 40e9:
        print("Recommended: A100 40GB")
    elif total < 80e9:
        print("Recommended: A100 80GB or gradient checkpointing")
    else:
        print("Recommended: Multi-GPU with model parallelism")

    return total
```

## Training Time Estimation

### Tokens Per Second

```python
def estimate_training_time(
    num_params,
    num_tokens,
    num_gpus,
    tokens_per_second_per_gpu
):
    """Estimate training time."""

    total_tps = tokens_per_second_per_gpu * num_gpus

    # Account for communication overhead
    if num_gpus > 1:
        efficiency = 0.85
        total_tps *= efficiency

    seconds = num_tokens / total_tps
    hours = seconds / 3600
    days = hours / 24

    print(f"\n=== Training Time ===")
    print(f"Throughput: {total_tps:.0f} tokens/sec")
    print(f"Time: {hours:.1f} hours ({days:.2f} days)")

    return hours

# Typical tokens/sec per GPU
# T4: ~5,000 tokens/sec (7B model)
# A100: ~15,000 tokens/sec (7B model)
# H100: ~30,000 tokens/sec (7B model)
```

## Scaling Laws

### Compute-Optimal Training

```python
def chinchilla_optimal_training(compute_budget_flops):
    """
    Chinchilla scaling law: optimal model size and data.

    C = 6 * N * D
    N ~= 0.73 * sqrt(C/6)
    D ~= 1.37 * sqrt(C/6)
    """
    import math

    # Optimal parameters and tokens
    optimal_params = 0.73 * math.sqrt(compute_budget_flops / 6)
    optimal_tokens = 1.37 * math.sqrt(compute_budget_flops / 6)

    print(f"\n=== Compute-Optimal Training ===")
    print(f"Compute budget: {compute_budget_flops / 1e21:.2f} ZFLOPs")
    print(f"Optimal params: {optimal_params / 1e9:.1f}B")
    print(f"Optimal tokens: {optimal_tokens / 1e9:.1f}B")

    return optimal_params, optimal_tokens
```

## Budget Tracking

### Track Actual vs Estimated

```python
class BudgetTracker:
    def __init__(self, budget_usd, gpu_cost_per_hour, num_gpus):
        self.budget = budget_usd
        self.hourly_cost = gpu_cost_per_hour * num_gpus
        self.start_time = None
        self.spent = 0

    def start(self):
        import time
        self.start_time = time.time()

    def update(self):
        import time
        hours = (time.time() - self.start_time) / 3600
        self.spent = hours * self.hourly_cost
        remaining = self.budget - self.spent
        pct_used = (self.spent / self.budget) * 100

        return {
            'spent': self.spent,
            'remaining': remaining,
            'pct_used': pct_used
        }

    def check_budget(self, warn_threshold=0.8):
        status = self.update()
        if status['pct_used'] > warn_threshold * 100:
            print(f"WARNING: {status['pct_used']:.1f}% of budget used")
        return status
```

## Best Practices

1. **Start with estimates**: Calculate before training
2. **Run small experiments**: Validate throughput assumptions
3. **Track actual costs**: Compare to estimates
4. **Include all costs**: Storage, networking, failed runs
5. **Add buffer**: 20-50% for unexpected issues
6. **Use spot instances**: Reduce compute costs
7. **Monitor continuously**: Alert on budget thresholds
8. **Document assumptions**: For future reference
