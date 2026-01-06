# Distributed Training

## Summary

Distributed training enables training models across multiple GPUs and machines when a single accelerator cannot handle the workload. The core challenge is dividing computation and communication efficiently to achieve near-linear scaling. Different parallelism strategies address different bottlenecks: data parallelism for throughput, model parallelism for memory, tensor parallelism for large layers, and pipeline parallelism for deep networks.

Key points to remember:

- Data parallelism replicates the model; each device processes different data
- Model parallelism splits layers across devices; necessary when model exceeds GPU memory
- Tensor parallelism splits operations within layers; requires high-bandwidth interconnects
- Pipeline parallelism processes micro-batches through model stages; reduces idle time
- FSDP and ZeRO shard optimizer states to reduce memory per device
- All-reduce is the fundamental collective operation for gradient synchronization
- Ring-AllReduce provides bandwidth-optimal gradient aggregation
- Framework choice depends on scale: DDP for basic, DeepSpeed/FSDP for large models
- Communication backend matters: NCCL for GPU, Gloo for CPU, MPI for HPC environments

## Why Distribute Training

### Memory Constraints

Model parameters, gradients, optimizer states, and activations consume GPU memory:

| Component | Memory (FP32) | Memory (FP16) |
|-----------|---------------|---------------|
| Parameters | 4 bytes/param | 2 bytes/param |
| Gradients | 4 bytes/param | 2 bytes/param |
| Adam states | 8 bytes/param | 8 bytes/param |
| Activations | Depends on batch/sequence | Same |

A 7B parameter model requires approximately:
- Parameters: 28 GB (FP32) or 14 GB (FP16)
- Gradients: 28 GB (FP32) or 14 GB (FP16)
- Adam states: 56 GB
- Total (without activations): 112 GB minimum

This exceeds single GPU capacity, requiring distribution.

### Throughput Requirements

Large-scale training requires processing massive datasets:

| Dataset | Tokens | Time at 1 GPU | Time at 1000 GPUs |
|---------|--------|---------------|-------------------|
| GPT-3 training | 300B | Years | Weeks |
| LLaMA training | 1.4T | Years | Months |

Parallelism converts impractical training times into feasible projects.

### Fault Tolerance

Long training runs encounter hardware failures. Distributed systems can:
- Checkpoint across nodes
- Recover from node failures
- Continue training with reduced capacity

## Parallelism Strategies

### Data Parallelism

Each device holds a complete model copy, processes different data batches.

**Process**:
1. Broadcast model to all devices
2. Each device processes its batch (forward + backward)
3. All-reduce gradients across devices
4. Each device updates its model copy

**Scaling behavior**:
- Near-linear speedup with device count
- Effective batch size = per-device batch x device count
- Communication cost grows with model size

**When to use**: Model fits in single GPU memory. Most common starting point.

### Model Parallelism

Split model layers across devices sequentially.

**Process**:
1. Assign layer ranges to devices
2. Forward: pass activations between devices
3. Backward: pass gradients back through devices
4. Each device updates its layers

**Problem**: Pipeline bubble (GPU idle time)

```
Without pipelining:
GPU 0: [Forward] [  Idle  ] [Backward]
GPU 1: [  Idle  ] [Forward] [  Idle  ] [Backward]
```

**When to use**: Very deep models, when combined with pipeline parallelism.

### Tensor Parallelism

Split individual operations across devices.

**Example - Linear layer Y = XW**:
```
Standard: Y = X @ W

Tensor parallel (column):
W = [W1 | W2]  (Split columns)
Y1 = X @ W1   (GPU 0)
Y2 = X @ W2   (GPU 1)
Y = [Y1 | Y2]  (Concatenate)
```

**Characteristics**:
- Requires communication within each layer
- High bandwidth interconnect essential (NVLink)
- Typically limited to single-node (8 GPUs)

**When to use**: Large hidden dimensions, high-bandwidth interconnects available.

### Pipeline Parallelism

Process multiple micro-batches through model stages to hide bubble.

**GPipe schedule**:
```
GPU 0: F0 F1 F2 F3 -- -- -- -- B3 B2 B1 B0
GPU 1: -- F0 F1 F2 F3 -- -- -- -- B3 B2 B1
GPU 2: -- -- F0 F1 F2 F3 -- -- -- -- B3 B2
GPU 3: -- -- -- F0 F1 F2 F3 B3 B2 B1 B0 --
```

**1F1B schedule** (one forward, one backward):
```
GPU 0: F0 F1 F2 F3 B0 B1 B2 B3
GPU 1: -- F0 F1 F2 B0 F3 B1 B2 B3
GPU 2: -- -- F0 F1 B0 F2 B1 F3 B2 B3
GPU 3: -- -- -- F0 B0 F1 B1 F2 B2 F3 B3
```

1F1B reduces memory by limiting in-flight micro-batches.

**When to use**: Deep models, combined with data/tensor parallelism.

### Hybrid Parallelism (3D Parallelism)

Combine all strategies for maximum scale:

```
Data Parallel: Between nodes
Tensor Parallel: Within node (NVLink)
Pipeline Parallel: Across node groups
```

Example configuration for 64 GPUs across 8 nodes:
- Tensor parallel: 8 (within each node)
- Pipeline parallel: 4 (across 4 node groups)
- Data parallel: 2 (duplicate the pipeline)

## Memory Optimization

### ZeRO (Zero Redundancy Optimizer)

Eliminate redundant storage across data-parallel ranks:

**ZeRO Stage 1**: Partition optimizer states
- Each rank stores 1/N of optimizer states
- Memory reduction: approximately 4x for Adam

**ZeRO Stage 2**: Partition gradients
- Each rank stores 1/N of gradients
- Additional memory reduction

**ZeRO Stage 3**: Partition parameters
- Each rank stores 1/N of parameters
- Gather parameters when needed
- Maximum memory efficiency

**Trade-off**: Higher stages increase communication but reduce memory.

### FSDP (Fully Sharded Data Parallel)

PyTorch native implementation of ZeRO-3:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),    # Optional
)
```

**Sharding strategies**:
- FULL_SHARD: ZeRO-3, maximum memory savings
- SHARD_GRAD_OP: ZeRO-2, less communication
- NO_SHARD: Standard DDP

### Gradient Accumulation

Simulate larger batch sizes without memory increase:

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = micro_batch x accumulation_steps x num_devices

## Communication Patterns

### All-Reduce

Aggregate values across all devices, distribute result to all:

```
Before:    GPU0: [1]   GPU1: [2]   GPU2: [3]   GPU3: [4]
Operation: SUM
After:     GPU0: [10]  GPU1: [10]  GPU2: [10]  GPU3: [10]
```

Used for: Gradient synchronization in data parallelism

### All-Gather

Gather values from all devices, distribute complete set to all:

```
Before:    GPU0: [A]   GPU1: [B]   GPU2: [C]   GPU3: [D]
After:     GPU0: [ABCD] GPU1: [ABCD] GPU2: [ABCD] GPU3: [ABCD]
```

Used for: Parameter gathering in FSDP/ZeRO-3

### Reduce-Scatter

Reduce values, scatter different chunks to each device:

```
Before:    GPU0: [1,2]  GPU1: [3,4]  GPU2: [5,6]  GPU3: [7,8]
Operation: SUM over corresponding positions
After:     GPU0: [16]   GPU1: [20]   GPU2: [24]  GPU3: [28]
```

Used for: Gradient reduction in FSDP (each rank gets its gradient shard)

### Point-to-Point

Direct communication between specific devices:

- Send/Recv: Explicit transfers
- Used in pipeline parallelism for activation passing

## Framework Comparison

| Framework | Parallelism Support | Ease of Use | Scale |
|-----------|---------------------|-------------|-------|
| PyTorch DDP | Data | Simple | 100s GPUs |
| PyTorch FSDP | Data + Sharding | Moderate | 1000s GPUs |
| DeepSpeed | All + ZeRO | Moderate | 1000s GPUs |
| Megatron-LM | All (3D) | Complex | 10000s GPUs |
| Horovod | Data | Simple | 100s GPUs |
| Ray Train | Data + Actor-based | Simple | 1000s GPUs |
| ColossalAI | All + Auto | Simple | 1000s GPUs |

### Selection Guidance

**Small to medium models (fits in GPU)**:
- Start with PyTorch DDP
- Simple, well-documented, minimal overhead

**Large models (needs sharding)**:
- DeepSpeed ZeRO or PyTorch FSDP
- Both provide similar capabilities
- DeepSpeed: More features, separate library
- FSDP: Native PyTorch, growing ecosystem

**Very large models (needs all parallelism types)**:
- Megatron-LM for maximum control
- ColossalAI for easier configuration
- Often combined with DeepSpeed

**Multi-cloud or heterogeneous**:
- Ray Train for flexibility
- Horovod for portability

## Scaling Efficiency

### Measuring Efficiency

**Strong scaling**: Fixed total work, more devices
- Ideal: Linear speedup
- Reality: Communication overhead limits scaling

**Weak scaling**: Fixed work per device, more devices
- Ideal: Constant time
- Reality: More achievable than strong scaling

**Metrics**:
- Throughput: Samples/second or tokens/second
- Scaling efficiency: Actual speedup / ideal speedup
- GPU utilization: Percentage of compute capacity used

### Efficiency Factors

**Computation-to-communication ratio**:
- Higher is better for scaling
- Large models scale better (more compute per communication)

**Network bandwidth**:
- InfiniBand: 200-400 Gbps, good for multi-node
- NVLink: 600+ GB/s, good for intra-node
- Ethernet: 10-100 Gbps, limits scaling

**Batch size**:
- Larger batches amortize communication
- But very large batches may hurt convergence

## Common Issues

### Deadlocks

All devices must participate in collectives:
```python
# Wrong: conditional collective
if rank == 0:
    dist.all_reduce(tensor)  # Deadlock: other ranks waiting

# Right: all ranks participate
dist.all_reduce(tensor)
if rank == 0:
    print(tensor)
```

### Memory Imbalance

Different ranks may have different memory usage:
- First/last pipeline stages often differ
- Embedding layers often on first rank

Solution: Monitor per-rank memory, balance layer assignment.

### Gradient Desynchronization

Gradients must stay synchronized across ranks:
```python
# Verify gradients match
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.MAX)
    max_grad = param.grad.clone()
    dist.all_reduce(param.grad, op=dist.ReduceOp.MIN)
    min_grad = param.grad
    assert torch.allclose(max_grad, min_grad)
```

## Further Reading

### Concepts
Core distributed training patterns:
- [Data Parallelism](concepts/data-parallelism/ReadMe.md): Replicating models
- [Model Parallelism](concepts/model-parallelism/ReadMe.md): Splitting layers
- [Tensor Parallelism](concepts/tensor-parallelism/ReadMe.md): Splitting operations
- [Pipeline Parallelism](concepts/pipeline-parallelism/ReadMe.md): Micro-batch pipelining
- [FSDP](concepts/fsdp/ReadMe.md): Fully Sharded Data Parallel
- [ZeRO Optimization](concepts/zero-optimization/ReadMe.md): Memory-efficient training
- [Gradient Accumulation](concepts/gradient-accumulation/ReadMe.md): Simulating large batches
- [All-Reduce](concepts/all-reduce/ReadMe.md): Gradient aggregation
- [Ring-AllReduce](concepts/ring-allreduce/ReadMe.md): Bandwidth-optimal reduction

### Frameworks
Distributed training frameworks:
- [PyTorch Distributed](frameworks/pytorch-distributed/ReadMe.md): Native PyTorch
- [DeepSpeed](frameworks/deepspeed/ReadMe.md): Microsoft optimization library
- [Megatron-LM](frameworks/megatron-lm/ReadMe.md): NVIDIA large model training
- [Horovod](frameworks/horovod/ReadMe.md): Uber distributed training
- [Ray Train](frameworks/ray-train/ReadMe.md): Distributed ML on Ray
- [ColossalAI](frameworks/colossalai/ReadMe.md): Easy large model training

### Communication
Communication backends and protocols:
- [NCCL](communication/nccl/ReadMe.md): NVIDIA Collective Communications
- [Gloo](communication/gloo/ReadMe.md): Facebook collective library
- [MPI](communication/mpi/ReadMe.md): Message Passing Interface
