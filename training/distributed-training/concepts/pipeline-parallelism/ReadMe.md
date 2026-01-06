# Pipeline Parallelism

## Summary

Pipeline parallelism divides a model into sequential stages across devices and processes multiple micro-batches concurrently to maximize GPU utilization. This approach addresses the fundamental inefficiency of naive model parallelism where GPUs sit idle waiting for activations. By splitting a batch into smaller micro-batches and pipelining them through stages, pipeline parallelism achieves much better hardware utilization.

Key points to remember:

- Model is split into stages, each assigned to different devices
- Batches are divided into micro-batches for pipelining
- GPipe and 1F1B are the two main scheduling strategies
- Pipeline bubble (idle time) is inevitable but can be minimized
- More micro-batches reduce bubble fraction but increase memory
- 1F1B schedule limits memory by interleaving forward and backward
- Activation checkpointing often combined to reduce memory further
- Inter-stage communication passes activations and gradients

## The Pipeline Bubble Problem

### Naive Model Parallelism

Without pipelining, GPUs are mostly idle:

```
Time steps:    1   2   3   4   5   6   7   8
Stage 0 (GPU0): F   -   -   -   B   -   -   -
Stage 1 (GPU1): -   F   -   -   -   B   -   -
Stage 2 (GPU2): -   -   F   -   -   -   B   -
Stage 3 (GPU3): -   -   -   F   -   -   -   B

F = Forward, B = Backward, - = Idle
```

For 4 stages, each GPU is active only 2/8 = 25% of the time.

### Pipeline Parallelism Solution

Process multiple micro-batches concurrently:

```
Micro-batches: m0, m1, m2, m3

Stage 0: F0  F1  F2  F3  B3  B2  B1  B0
Stage 1: --  F0  F1  F2  B3  F3  B2  B1  B0
Stage 2: --  --  F0  F1  B3  F2  B2  F3  B1  B0
Stage 3: --  --  --  F0  B0  F1  B1  F2  B2  F3  B3
```

More micro-batches means better utilization.

## Scheduling Strategies

### GPipe Schedule

All forwards, then all backwards (fill-drain schedule):

```
4 stages, 8 micro-batches:

Stage 0: F0 F1 F2 F3 F4 F5 F6 F7 -- -- -- -- -- -- -- B7 B6 B5 B4 B3 B2 B1 B0
Stage 1: -- F0 F1 F2 F3 F4 F5 F6 F7 -- -- -- -- -- -- B7 B6 B5 B4 B3 B2 B1 B0
Stage 2: -- -- F0 F1 F2 F3 F4 F5 F6 F7 -- -- -- -- B7 B6 B5 B4 B3 B2 B1 B0 --
Stage 3: -- -- -- F0 F1 F2 F3 F4 F5 F6 F7 -- -- B7 B6 B5 B4 B3 B2 B1 B0 -- --
```

**Bubble fraction**: (p-1) / m where p = stages, m = micro-batches

For 4 stages, 8 micro-batches: 3/8 = 37.5% bubble

**Memory**: Must store activations for all micro-batches during forward phase.

### 1F1B Schedule (One Forward One Backward)

Interleave forward and backward passes:

```
4 stages, 4 micro-batches:

Stage 0: F0 F1 F2 F3 B0 B1 B2 B3
Stage 1: -- F0 F1 F2 B0 F3 B1 B2 B3
Stage 2: -- -- F0 F1 B0 F2 B1 F3 B2 B3
Stage 3: -- -- -- F0 B0 F1 B1 F2 B2 F3 B3
```

**Key insight**: Once warmup completes, each stage does one forward followed by one backward.

**Memory advantage**: Only need to store activations for p micro-batches (one per stage in flight), not all micro-batches.

**Bubble fraction**: Same as GPipe: (p-1) / m

### Interleaved 1F1B

Virtual pipeline stages allow more micro-batches in flight:

```
Physical GPU 0 handles virtual stages 0 and 4
Physical GPU 1 handles virtual stages 1 and 5
...

This allows twice as many stages without more GPUs.
```

**Benefit**: Reduces bubble by increasing virtual pipeline depth.

**Cost**: More complex scheduling, some redundant computation.

## Implementation

### Basic Pipeline Stage

```python
class PipelineStage(nn.Module):
    def __init__(self, layers, stage_id, num_stages):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first = stage_id == 0
        self.is_last = stage_id == num_stages - 1

    def forward(self, x):
        return self.layers(x)
```

### GPipe-Style Training Loop

```python
def gpipe_forward_backward(stages, micro_batches, loss_fn):
    num_stages = len(stages)
    num_micro_batches = len(micro_batches)

    # Storage for activations and outputs
    activations = [[None] * num_stages for _ in range(num_micro_batches)]
    losses = []

    # Forward pass for all micro-batches
    for mb_idx in range(num_micro_batches):
        x = micro_batches[mb_idx]
        for stage_idx in range(num_stages):
            activations[mb_idx][stage_idx] = x.detach().requires_grad_()
            x = stages[stage_idx](x)
            if stage_idx < num_stages - 1:
                x = x.to(f'cuda:{stage_idx + 1}')

        loss = loss_fn(x)
        losses.append(loss)

    # Backward pass for all micro-batches
    for mb_idx in reversed(range(num_micro_batches)):
        losses[mb_idx].backward()

        # Propagate gradients back through stages
        for stage_idx in reversed(range(num_stages - 1)):
            # Get gradient from next stage
            grad = activations[mb_idx][stage_idx + 1].grad
            grad = grad.to(f'cuda:{stage_idx}')
            activations[mb_idx][stage_idx].backward(grad)

    return sum(losses) / num_micro_batches
```

### 1F1B Training Loop

```python
def one_f_one_b(stages, micro_batches, loss_fn):
    num_stages = len(stages)
    num_micro_batches = len(micro_batches)

    input_tensors = [[] for _ in range(num_stages)]
    output_tensors = [[] for _ in range(num_stages)]

    # Warmup: Fill the pipeline
    warmup_steps = num_stages - 1
    for mb_idx in range(warmup_steps):
        x = micro_batches[mb_idx]
        for stage_idx in range(min(mb_idx + 1, num_stages)):
            if stage_idx > 0:
                x = recv_from_prev_stage(stage_idx)

            x = x.detach().requires_grad_()
            input_tensors[stage_idx].append(x)

            output = stages[stage_idx](x)
            output_tensors[stage_idx].append(output)

            if stage_idx < num_stages - 1:
                send_to_next_stage(output, stage_idx)

    # Steady state: 1F1B
    for mb_idx in range(warmup_steps, num_micro_batches):
        # Forward
        forward_micro_batch(mb_idx, stages, input_tensors, output_tensors)

        # Backward (for micro-batch that completed earlier)
        backward_mb_idx = mb_idx - warmup_steps
        backward_micro_batch(backward_mb_idx, stages, input_tensors, output_tensors)

    # Cooldown: Finish remaining backwards
    for mb_idx in range(num_micro_batches - warmup_steps, num_micro_batches):
        backward_micro_batch(mb_idx, stages, input_tensors, output_tensors)
```

## Communication Patterns

### Activation Passing

```python
def send_activation(tensor, dst_rank):
    """Send activation to next pipeline stage."""
    dist.send(tensor, dst=dst_rank)

def recv_activation(shape, dtype, src_rank, device):
    """Receive activation from previous pipeline stage."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank)
    return tensor
```

### Gradient Passing

```python
def send_gradient(tensor, dst_rank):
    """Send gradient to previous pipeline stage."""
    dist.send(tensor, dst=dst_rank)

def recv_gradient(shape, dtype, src_rank, device):
    """Receive gradient from next pipeline stage."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank)
    return tensor
```

### Communication Overlap

Overlap communication with computation:

```python
# Non-blocking send
req = dist.isend(output, dst=next_rank)

# Continue with computation
intermediate = some_local_computation()

# Wait for send to complete
req.wait()
```

## Memory Optimization

### Activation Checkpointing per Stage

```python
class CheckpointedStage(nn.Module):
    def __init__(self, layers, checkpoint_every=2):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0 and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

### Memory Budget per Stage

```
Peak memory = max(forward_peak, backward_peak)

Forward peak:
  - Stage parameters
  - Input activation
  - Stored activations (num_micro_batches_in_flight)
  - Intermediate activations

Backward peak:
  - Stage parameters + gradients
  - Stored activations
  - Gradient tensors
```

1F1B limits micro-batches in flight, reducing peak memory.

## Bubble Analysis

### Theoretical Bubble

For p stages and m micro-batches:

**GPipe**:
- Bubble time: (p-1) * (forward_time + backward_time)
- Compute time: m * (forward_time + backward_time) * p
- Bubble fraction: (p-1) / (m * p) approximately (p-1) / m for large m

**1F1B**:
- Same bubble fraction
- But better memory characteristics

### Minimizing Bubble

1. **Increase micro-batches**: More m reduces bubble fraction
2. **Balance stages**: Unequal stage times increase effective bubble
3. **Interleaved stages**: Double virtual stages to halve bubble
4. **Reduce pipeline depth**: Fewer stages means less bubble

### Practical Bubble Targets

| Stages | Micro-batches | Bubble Fraction |
|--------|---------------|-----------------|
| 4 | 8 | 37.5% |
| 4 | 16 | 18.75% |
| 4 | 32 | 9.4% |
| 8 | 32 | 21.9% |
| 8 | 64 | 10.9% |

Target: Less than 20% bubble for reasonable efficiency.

## Framework Support

### PyTorch Pipe

```python
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(
    stage_0_layers,
    stage_1_layers,
    stage_2_layers,
    stage_3_layers
)

model = Pipe(model, chunks=8, checkpoint='never')
output = model(input)
```

### DeepSpeed Pipeline

```python
from deepspeed.pipe import PipelineModule, LayerSpec

layers = [
    LayerSpec(EmbeddingLayer, vocab_size, hidden_size),
    *[LayerSpec(TransformerLayer, hidden_size) for _ in range(num_layers)],
    LayerSpec(OutputLayer, hidden_size, vocab_size)
]

model = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=loss_fn,
    partition_method='uniform'
)

engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

### Megatron-LM Pipeline

Megatron-LM provides sophisticated pipeline parallelism with:
- Interleaved 1F1B scheduling
- Tensor parallel integration
- Optimized communication

```python
# Megatron partitions based on configuration
# num_layers_per_stage = num_layers / pipeline_parallel_size
```

## Combining with Other Parallelisms

### Data + Pipeline

```
Data Parallel Group 0          Data Parallel Group 1
[Stage0] -> [Stage1]          [Stage0] -> [Stage1]
   |           |                  |           |
   +-----------+                  +-----------+
   All-reduce gradients          All-reduce gradients
```

Each pipeline replica processes different data, gradients synchronized within stages.

### Tensor + Pipeline

```
Node 0 (TP=4)         Node 1 (TP=4)
[Stage 0: TP Group]   [Stage 1: TP Group]
GPU0 GPU1 GPU2 GPU3   GPU0 GPU1 GPU2 GPU3
    |     |              |     |
    +-----+    ->        +-----+
    Activations passed between nodes
```

Tensor parallel within nodes (fast interconnect), pipeline across nodes.

### 3D Parallelism

```
                    Pipeline Stages
                    Stage 0    Stage 1
Data Parallel 0:    [TP=4]  -> [TP=4]
                      |          |
Data Parallel 1:    [TP=4]  -> [TP=4]
```

- Tensor parallel: Within node (8 GPUs)
- Pipeline parallel: Across node groups
- Data parallel: Across pipeline replicas

## Best Practices

### Stage Balancing

1. **Profile each stage**: Measure forward and backward time
2. **Adjust layer assignment**: Move layers to balance
3. **Account for embedding**: First stage often heavier
4. **Consider memory**: Some stages may be memory-bound

```python
def profile_stages(model, input_shape, num_trials=10):
    times = []
    for stage in model.stages:
        stage_times = []
        x = torch.randn(input_shape).to(stage.device)
        for _ in range(num_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            out = stage(x)
            end.record()

            torch.cuda.synchronize()
            stage_times.append(start.elapsed_time(end))

        times.append(sum(stage_times) / num_trials)
    return times
```

### Debugging

**Verify activation shapes**:
```python
print(f"Stage {stage_id}: input={x.shape}, output={out.shape}")
```

**Check gradient flow**:
```python
for name, param in stage.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

**Monitor stage timing**:
```python
# Ensure no stage is significantly slower
assert max(stage_times) / min(stage_times) < 1.5
```
