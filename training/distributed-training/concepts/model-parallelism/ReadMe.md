# Model Parallelism

## Summary

Model parallelism splits a neural network across multiple devices by assigning different layers to different GPUs. This approach is necessary when a model is too large to fit in the memory of a single GPU. Unlike data parallelism where each device has a complete model copy, model parallelism partitions the model itself, with each device responsible for a subset of the computation.

Key points to remember:

- Model is split by layers across devices (also called layer parallelism)
- Required when model parameters exceed single GPU memory
- Activations are passed between devices during forward pass
- Gradients flow back through devices during backward pass
- Naive implementation leads to significant GPU idle time (pipeline bubble)
- Typically combined with pipeline parallelism to improve utilization
- Inter-device communication bandwidth becomes critical
- Balancing compute across devices is challenging

## Why Model Parallelism

### Memory Constraints

A model with P parameters requires approximately:

| Component | Memory (FP32) | Memory (Mixed Precision) |
|-----------|---------------|--------------------------|
| Parameters | 4P bytes | 6P bytes (FP16 + master) |
| Gradients | 4P bytes | 2P bytes |
| Optimizer (Adam) | 8P bytes | 8P bytes |
| Total | 16P bytes | 16P bytes |

For a 10B parameter model: 16 x 10B = 160GB, exceeding any single GPU.

### When Data Parallelism Is Not Enough

Data parallelism replicates the entire model on each device. If the model itself does not fit, data parallelism cannot help. Model parallelism is the fundamental solution to the memory constraint.

## Basic Model Parallelism

### Manual Layer Assignment

```python
import torch
import torch.nn as nn

class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First half of layers on GPU 0
        self.layer1 = nn.Linear(1024, 2048).to('cuda:0')
        self.layer2 = nn.Linear(2048, 2048).to('cuda:0')

        # Second half on GPU 1
        self.layer3 = nn.Linear(2048, 2048).to('cuda:1')
        self.layer4 = nn.Linear(2048, 512).to('cuda:1')

    def forward(self, x):
        x = x.to('cuda:0')
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        # Transfer activations to GPU 1
        x = x.to('cuda:1')
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x
```

### The Pipeline Bubble Problem

With naive model parallelism, GPUs sit idle while waiting:

```
Time:    T1    T2    T3    T4    T5    T6
GPU 0: [Fwd ] [    ] [    ] [    ] [Bwd ] [    ]
GPU 1: [    ] [Fwd ] [    ] [Bwd ] [    ] [    ]
GPU 2: [    ] [    ] [Fwd ] [Bwd ] [    ] [    ]
```

GPU utilization is approximately 1/N for N devices, making scaling inefficient.

### Calculating Bubble Size

For N pipeline stages and batch of size B:
- Forward time per stage: F
- Backward time per stage: approximately 2F
- Total ideal time: N x 3F
- Actual time with bubble: (N-1)F + 3F + (N-1) x 2F = 3NF - F
- Bubble fraction: (N-1) / N

For 4 GPUs: 75% of GPU time is bubble. This is why pipeline parallelism is essential.

## Layer Assignment Strategies

### Even Distribution

Split model into equal-sized chunks:

```python
def assign_layers_even(model, num_devices):
    layers = list(model.children())
    per_device = len(layers) // num_devices

    device_assignment = {}
    for i, layer in enumerate(layers):
        device = min(i // per_device, num_devices - 1)
        device_assignment[layer] = device

    return device_assignment
```

### Memory-Balanced Distribution

Account for varying layer sizes:

```python
def get_layer_memory(layer):
    """Estimate memory requirement for a layer."""
    params = sum(p.numel() * p.element_size() for p in layer.parameters())
    # Activations depend on input size, estimate conservatively
    return params * 4  # params + grads + optimizer states

def assign_layers_balanced(model, num_devices, device_memory):
    layers = list(model.children())
    layer_memories = [get_layer_memory(l) for l in layers]

    device_assignment = {}
    current_device = 0
    current_memory = 0

    for layer, mem in zip(layers, layer_memories):
        if current_memory + mem > device_memory and current_device < num_devices - 1:
            current_device += 1
            current_memory = 0
        device_assignment[layer] = current_device
        current_memory += mem

    return device_assignment
```

### Compute-Balanced Distribution

Balance FLOPs rather than memory:

```python
def get_layer_flops(layer, input_shape):
    """Estimate FLOPs for a layer."""
    if isinstance(layer, nn.Linear):
        return 2 * layer.in_features * layer.out_features * input_shape[0]
    elif isinstance(layer, nn.Conv2d):
        # More complex calculation
        pass
    return 0

# Assign layers to balance compute time across devices
```

## Communication Patterns

### Forward Pass Communication

```python
class ModelParallelStage(nn.Module):
    def __init__(self, layers, prev_device, curr_device, next_device):
        super().__init__()
        self.layers = nn.Sequential(*layers).to(curr_device)
        self.prev_device = prev_device
        self.curr_device = curr_device
        self.next_device = next_device

    def forward(self, x):
        # Receive from previous device
        if self.prev_device is not None:
            x = x.to(self.curr_device)

        # Compute
        x = self.layers(x)

        # Send to next device
        if self.next_device is not None:
            x = x.to(self.next_device)

        return x
```

### Backward Pass Communication

Gradients flow automatically through the .to() operations:

```python
# Forward: GPU0 -> GPU1 -> GPU2 -> loss
# Backward: loss -> GPU2 -> GPU1 -> GPU0
# PyTorch autograd handles this automatically
```

### Activation Memory

Each device must store activations for backward pass:

```
GPU 0: Store activations from layers 0-3
GPU 1: Store activations from layers 4-7
...
```

This can be reduced with activation checkpointing.

## Practical Considerations

### Handling Different Tensor Devices

```python
class SafeModelParallel(nn.Module):
    def forward(self, x, labels=None):
        x = x.to('cuda:0')
        x = self.encoder(x)

        x = x.to('cuda:1')
        x = self.decoder(x)

        if labels is not None:
            # Ensure labels are on same device as output
            labels = labels.to(x.device)
            loss = self.criterion(x, labels)
            return loss
        return x
```

### Synchronization Points

```python
# Explicit synchronization when needed
torch.cuda.synchronize(device='cuda:0')
torch.cuda.synchronize(device='cuda:1')
```

### Memory Management

```python
# Clear cache on specific devices
torch.cuda.empty_cache()

# Monitor memory per device
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
```

## Comparison with Other Approaches

| Approach | Memory Scaling | Communication | Implementation |
|----------|---------------|---------------|----------------|
| Model Parallel | Linear with devices | High (activations) | Manual |
| Tensor Parallel | Linear with devices | Very high (within layers) | Complex |
| Data Parallel | No reduction | Moderate (gradients) | Simple |
| ZeRO/FSDP | Linear with devices | Moderate | Framework support |

### When to Use Model Parallelism

**Use model parallelism when**:
- Model does not fit in single GPU
- Combined with pipeline parallelism for efficiency
- Tensor parallelism is not applicable (small hidden dims)

**Consider alternatives when**:
- Model fits in GPU with ZeRO/FSDP
- Can reduce model size (quantization, pruning)
- Gradient checkpointing is sufficient

## Integration with Pipeline Parallelism

Model parallelism becomes practical with pipeline parallelism:

```python
# Instead of processing one batch through all stages:
#   Forward all stages -> Backward all stages

# Pipeline parallelism processes micro-batches:
#   GPU0: F(mb0), F(mb1), F(mb2), F(mb3), B(mb0), B(mb1), ...
#   GPU1:        F(mb0), F(mb1), F(mb2), B(mb0), B(mb1), ...
```

This significantly improves GPU utilization by overlapping computation.

## Common Issues

### Memory Imbalance

First and last stages often have different memory requirements:
- First stage: Embedding layers
- Last stage: Output projections, loss computation

**Solution**: Adjust layer assignment or use asymmetric stages.

### Communication Bottleneck

Large activation tensors can bottleneck training:

```python
# Measure activation sizes
def forward_with_logging(self, x):
    x = self.layer1(x)
    print(f"After layer1: {x.shape}, {x.element_size() * x.numel() / 1e9:.2f} GB")
    x = x.to('cuda:1')  # This transfer takes time
    ...
```

**Solutions**:
- Reduce batch size
- Use activation compression
- Ensure fast interconnect (NVLink)

### Deadlocks

Incorrect synchronization can cause hangs:

```python
# Wrong: Asymmetric execution paths
if some_condition:
    x = x.to('cuda:1')  # One path transfers
# Other path does not - devices get out of sync

# Right: Symmetric execution
x = x.to('cuda:1')  # Always transfer
```

## Framework Support

### PyTorch Native

Manual implementation as shown above. Full control but more code.

### GPipe (torch.distributed.pipeline)

```python
from torch.distributed.pipeline.sync import Pipe

# Define sequential model
model = nn.Sequential(layer1, layer2, layer3, layer4)

# Create pipeline
model = Pipe(model, chunks=8)  # 8 micro-batches
```

### Megatron-LM

Provides model parallelism for transformer architectures with tensor parallelism integration.

### DeepSpeed Pipeline

```python
from deepspeed.pipe import PipelineModule

model = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=loss_fn
)
```

## Best Practices

1. **Start with profiling**: Understand memory usage per layer before partitioning

2. **Balance carefully**: Unbalanced stages create bottlenecks

3. **Use pipeline parallelism**: Pure model parallelism has poor utilization

4. **Monitor all devices**: Check utilization and memory on each GPU

5. **Consider alternatives**: ZeRO/FSDP may be simpler for your use case

6. **Test at small scale**: Verify correctness before scaling up

7. **Use checkpointing**: Reduce activation memory per device
