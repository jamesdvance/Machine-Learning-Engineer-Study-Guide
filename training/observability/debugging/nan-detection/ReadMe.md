# NaN Detection

## Summary

NaN (Not a Number) and Inf (Infinity) values in training indicate numerical instability that will prevent model convergence. Detecting these issues early is critical as they often propagate through the network, corrupting all parameters. Understanding the causes and implementing proper detection enables rapid debugging and prevention.

Key points to remember:

- NaN usually means division by zero or log of negative
- Inf typically means exponential overflow or very large values
- Use torch.autograd.set_detect_anomaly(True) for debugging
- Check loss, gradients, and activations
- Common causes: high LR, bad initialization, unstable operations
- Mixed precision can trigger with improper loss scaling
- Early detection saves compute
- Gradient clipping prevents gradient-related NaN

## Detecting NaN/Inf

### Basic Checks

```python
import torch

def check_tensor(tensor, name):
    """Check tensor for NaN/Inf."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan:
        raise ValueError(f"NaN detected in {name}")
    if has_inf:
        raise ValueError(f"Inf detected in {name}")

# In training loop
loss = model(batch)
check_tensor(loss, "loss")
```

### Comprehensive Check

```python
def check_model(model, check_params=True, check_grads=True):
    """Check model for NaN/Inf in parameters and gradients."""
    issues = []

    for name, param in model.named_parameters():
        if check_params:
            if torch.isnan(param).any():
                issues.append(f"NaN in param: {name}")
            if torch.isinf(param).any():
                issues.append(f"Inf in param: {name}")

        if check_grads and param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN in grad: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"Inf in grad: {name}")

    if issues:
        for issue in issues:
            print(issue)
        return False
    return True
```

### Anomaly Detection Mode

```python
import torch

# Enable anomaly detection (shows where NaN originated)
torch.autograd.set_detect_anomaly(True)

# Training will throw error with stack trace when NaN occurs
try:
    loss = model(batch)
    loss.backward()
except RuntimeError as e:
    print(f"Anomaly detected: {e}")
    # Stack trace shows which operation caused NaN

# Disable for production (overhead)
torch.autograd.set_detect_anomaly(False)
```

## Common Causes and Fixes

### Division by Zero

```python
# Problem
x = tensor / another_tensor  # another_tensor may contain zeros

# Fix: Add epsilon
eps = 1e-8
x = tensor / (another_tensor + eps)

# Or use clamp
x = tensor / another_tensor.clamp(min=eps)
```

### Log of Zero/Negative

```python
# Problem
log_prob = torch.log(prob)  # prob may be 0 or negative

# Fix: Clamp input
log_prob = torch.log(prob.clamp(min=1e-8))

# Or use log-softmax directly (more stable)
log_prob = F.log_softmax(logits, dim=-1)
```

### Exponential Overflow

```python
# Problem
exp_x = torch.exp(x)  # x may be very large

# Fix: Clamp before exp
exp_x = torch.exp(x.clamp(max=80))  # ~10^34

# Or use log-sum-exp trick
def stable_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    return torch.exp(x - x_max) / torch.exp(x - x_max).sum(dim=-1, keepdim=True)
```

### Large Gradients

```python
# Problem: Gradient explosion causes Inf then NaN

# Fix: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### Bad Initialization

```python
# Problem: Random init with wrong scale
linear = nn.Linear(1000, 1000)
# Default init may be too large for deep networks

# Fix: Use appropriate initialization
nn.init.xavier_uniform_(linear.weight)
nn.init.zeros_(linear.bias)

# For ReLU networks
nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
```

### High Learning Rate

```python
# Problem: LR too high causes weight explosion

# Fix: Use learning rate warmup
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)

# Or use lower initial LR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

## Mixed Precision NaN

### Loss Scaling Issues

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        loss = model(batch).loss

    # Scaler handles Inf gradients
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Check for Inf gradients
    if not check_model(model, check_params=False, check_grads=True):
        print("Skipping step due to Inf gradients")
        scaler.update()
        continue

    # Clip gradients after unscaling
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

### Disable Mixed Precision for Debugging

```python
# If NaN occurs with mixed precision, test without it
# to isolate the issue

# Disable mixed precision
model = model.float()  # FP32
input = input.float()

# If NaN still occurs, issue is not precision-related
```

## Activation Monitoring

### Hook-Based Monitoring

```python
class NaNMonitor:
    def __init__(self, model):
        self.hooks = []
        self.nan_detected = False
        self.nan_layer = None

        for name, module in model.named_modules():
            self.hooks.append(
                module.register_forward_hook(self.make_hook(name))
            )

    def make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    self.nan_detected = True
                    self.nan_layer = name
                    print(f"NaN detected in {name}")
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()

# Usage
monitor = NaNMonitor(model)
output = model(input)
if monitor.nan_detected:
    print(f"First NaN at: {monitor.nan_layer}")
monitor.remove()
```

### Track Statistics

```python
def monitor_activations(model, input):
    """Track activation statistics to catch issues early."""
    stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
        return hook

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(input)

    for hook in hooks:
        hook.remove()

    return stats
```

## Debugging Workflow

### Step-by-Step Isolation

```python
def debug_nan(model, batch):
    """Step through training to find NaN source."""

    # Step 1: Check input
    print("Checking input...")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN in input: {key}")
                return

    # Step 2: Check forward pass
    print("Checking forward pass...")
    torch.autograd.set_detect_anomaly(True)
    try:
        loss = model(batch).loss
        if torch.isnan(loss):
            print("NaN in loss")
    except RuntimeError as e:
        print(f"Error in forward: {e}")
        return

    # Step 3: Check backward pass
    print("Checking backward pass...")
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"Error in backward: {e}")
        return

    # Step 4: Check gradients
    print("Checking gradients...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in: {name}")

    print("Debug complete")
```

### Recovery Strategy

```python
class NaNRecovery:
    def __init__(self, model, checkpoint_freq=100):
        self.model = model
        self.checkpoint_freq = checkpoint_freq
        self.last_good_state = None
        self.step = 0

    def save_checkpoint(self):
        self.last_good_state = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

    def recover(self):
        if self.last_good_state is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.last_good_state[name])
            print("Recovered from last good checkpoint")
            return True
        return False

    def step_and_check(self, loss):
        self.step += 1

        if self.step % self.checkpoint_freq == 0:
            self.save_checkpoint()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf detected at step {self.step}")
            return self.recover()

        return True
```

## Best Practices

1. **Check early**: Validate inputs and first forward pass
2. **Use anomaly detection**: During development
3. **Gradient clipping**: Always use for stability
4. **Learning rate warmup**: Prevents early instability
5. **Monitor statistics**: Track mean/std of activations
6. **Add epsilon**: To all divisions and logs
7. **Test without mixed precision**: Isolate precision issues
8. **Save checkpoints**: Enable recovery from NaN

## Quick Reference

| Issue | Symptom | Common Fix |
|-------|---------|------------|
| Log(0) | NaN | Add epsilon |
| Exp overflow | Inf -> NaN | Clamp input |
| Division by zero | NaN | Add epsilon |
| Large gradients | Inf -> NaN | Gradient clipping |
| High LR | Explosion | Warmup, reduce LR |
| Bad init | Early NaN | Xavier/Kaiming init |
| Underflow (FP16) | NaN | Loss scaling |
