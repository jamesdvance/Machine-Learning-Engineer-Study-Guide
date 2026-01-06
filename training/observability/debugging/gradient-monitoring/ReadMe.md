# Gradient Monitoring

## Summary

Gradient monitoring tracks the magnitude and distribution of gradients during training to detect issues like vanishing or exploding gradients. Healthy gradients are essential for effective learning - too small and the model won't update; too large and training becomes unstable. Regular gradient monitoring enables early detection of training problems.

Key points to remember:

- Monitor gradient norms per step and per layer
- Gradient explosion: norms grow exponentially
- Gradient vanishing: norms near zero, no learning
- Use gradient clipping to prevent explosion
- Log gradient histograms for distribution analysis
- Per-layer analysis identifies problematic layers
- Residual connections and normalization help gradient flow
- Compare gradient magnitude to parameter magnitude

## Gradient Norm Monitoring

### Total Gradient Norm

```python
import torch

def compute_grad_norm(model, norm_type=2):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(norm_type).item() ** norm_type
    return total_norm ** (1.0 / norm_type)

# In training loop
loss.backward()
grad_norm = compute_grad_norm(model)
print(f"Gradient norm: {grad_norm:.4f}")

# Log to experiment tracker
wandb.log({"grad_norm": grad_norm})
```

### Per-Layer Gradient Norms

```python
def compute_per_layer_grad_norms(model):
    """Compute gradient norm for each layer."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
    return norms

# Log per-layer norms
layer_norms = compute_per_layer_grad_norms(model)
for name, norm in layer_norms.items():
    print(f"{name}: {norm:.6f}")
```

### Gradient-to-Weight Ratio

```python
def compute_grad_to_weight_ratio(model):
    """Compute ratio of gradient norm to weight norm per layer."""
    ratios = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            weight_norm = param.data.norm(2).item()
            if weight_norm > 0:
                ratios[name] = grad_norm / weight_norm
    return ratios

# Healthy ratio typically 0.001 to 0.01
# Too small: vanishing gradients
# Too large: potential instability
```

## Gradient Statistics

### Distribution Statistics

```python
def gradient_statistics(model):
    """Compute detailed gradient statistics."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': grad.norm(2).item(),
                'zero_pct': (grad == 0).float().mean().item() * 100,
            }
    return stats

# Check for issues
stats = gradient_statistics(model)
for name, s in stats.items():
    if s['norm'] < 1e-8:
        print(f"Warning: Near-zero gradients in {name}")
    if s['norm'] > 1000:
        print(f"Warning: Large gradients in {name}")
    if s['zero_pct'] > 50:
        print(f"Warning: Many zero gradients in {name}")
```

### Gradient Histogram

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def log_gradient_histograms(model, step):
    """Log gradient histograms to TensorBoard."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, step)

# Log periodically
if step % 100 == 0:
    log_gradient_histograms(model, step)
```

## Detecting Gradient Problems

### Vanishing Gradients

```python
def detect_vanishing_gradients(model, threshold=1e-7):
    """Detect layers with vanishing gradients."""
    vanishing = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm < threshold:
                vanishing.append((name, grad_norm))
    return vanishing

# Check after backward
loss.backward()
vanishing = detect_vanishing_gradients(model)
if vanishing:
    print("Vanishing gradients detected:")
    for name, norm in vanishing:
        print(f"  {name}: {norm:.2e}")
```

### Exploding Gradients

```python
def detect_exploding_gradients(model, threshold=1000):
    """Detect layers with exploding gradients."""
    exploding = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm > threshold or torch.isnan(param.grad).any():
                exploding.append((name, grad_norm))
    return exploding

# Check after backward
loss.backward()
exploding = detect_exploding_gradients(model)
if exploding:
    print("Exploding gradients detected:")
    for name, norm in exploding:
        print(f"  {name}: {norm:.2e}")
```

### Gradient Flow Verification

```python
def verify_gradient_flow(model, input):
    """Verify gradients flow to all parameters."""
    model.zero_grad()
    output = model(input)
    output.sum().backward()

    no_grad = []
    small_grad = []

    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
        elif param.grad.abs().max() < 1e-10:
            small_grad.append(name)

    if no_grad:
        print("No gradient:")
        for name in no_grad:
            print(f"  {name}")

    if small_grad:
        print("Very small gradient:")
        for name in small_grad:
            print(f"  {name}")

    return len(no_grad) == 0 and len(small_grad) == 0
```

## Gradient Clipping

### Clip by Norm

```python
import torch

# After backward, before optimizer step
loss.backward()

# Clip gradients to max norm
max_norm = 1.0
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=max_norm
)

# Log whether clipping occurred
if grad_norm > max_norm:
    print(f"Gradients clipped: {grad_norm:.2f} -> {max_norm}")

optimizer.step()
```

### Clip by Value

```python
# Clip each gradient element to [-clip_value, clip_value]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### Adaptive Clipping

```python
class AdaptiveGradientClipper:
    def __init__(self, percentile=95, min_clip=0.1, max_clip=10.0):
        self.history = []
        self.percentile = percentile
        self.min_clip = min_clip
        self.max_clip = max_clip

    def clip(self, model):
        # Compute current norm
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2) ** 2
        current_norm = total_norm ** 0.5

        self.history.append(current_norm.item())

        # Compute adaptive threshold
        if len(self.history) > 100:
            threshold = np.percentile(self.history[-1000:], self.percentile)
            threshold = np.clip(threshold, self.min_clip, self.max_clip)

            if current_norm > threshold:
                torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)
                return threshold

        return current_norm.item()
```

## Hook-Based Monitoring

### Gradient Hooks

```python
class GradientMonitor:
    def __init__(self, model):
        self.gradients = {}
        self.hooks = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self.make_hook(name))
                self.hooks.append(hook)

    def make_hook(self, name):
        def hook(grad):
            self.gradients[name] = {
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
            }
            return grad  # Return unchanged
        return hook

    def get_stats(self):
        return self.gradients

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Usage
monitor = GradientMonitor(model)
loss.backward()
stats = monitor.get_stats()
monitor.remove_hooks()
```

### Activation Gradient Monitoring

```python
class ActivationGradientMonitor:
    def __init__(self, model):
        self.activation_grads = {}
        self.hooks = []

        for name, module in model.named_modules():
            hook = module.register_full_backward_hook(self.make_hook(name))
            self.hooks.append(hook)

    def make_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.activation_grads[name] = grad_output[0].norm().item()
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

## Visualization

### Gradient Flow Plot

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_gradient_flow(model):
    """Plot gradient flow through layers."""
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.grad is not None and 'bias' not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label='max')
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, label='mean')
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel('Layers')
    plt.ylabel('Gradient magnitude')
    plt.title('Gradient flow')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
```

### Training Gradient History

```python
class GradientHistory:
    def __init__(self):
        self.norms = []
        self.layer_norms = {}

    def record(self, model):
        total_norm = compute_grad_norm(model)
        self.norms.append(total_norm)

        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.layer_norms:
                    self.layer_norms[name] = []
                self.layer_norms[name].append(param.grad.norm().item())

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.norms)
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm History')
        plt.savefig('gradient_history.png')
```

## Best Practices

1. **Log gradient norms**: Every step or every N steps
2. **Use gradient clipping**: Prevents explosion
3. **Check per-layer**: Identifies problematic layers
4. **Monitor ratio**: Gradient norm vs weight norm
5. **Use proper initialization**: Prevents early issues
6. **Add skip connections**: Improves gradient flow
7. **Use normalization**: Stabilizes gradients
8. **Track history**: Detect trends over time

## Quick Reference

| Issue | Symptom | Fix |
|-------|---------|-----|
| Vanishing | Norm < 1e-7 | Skip connections, better init |
| Exploding | Norm > 1000 | Gradient clipping, lower LR |
| Dead neurons | Many zero grads | LeakyReLU, better init |
| Uneven flow | Layer variance | Layer normalization |
| Oscillation | Norm jumps | Lower LR, warmup |
