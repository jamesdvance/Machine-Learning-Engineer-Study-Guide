# Distributed Training Frameworks

## Summary

Distributed training frameworks provide abstractions and optimizations for scaling model training across multiple GPUs and machines. These frameworks handle the complexity of gradient synchronization, memory optimization, and parallelism strategies, allowing practitioners to focus on model development rather than distributed systems engineering.

Key points to remember:

- PyTorch Distributed (DDP/FSDP) provides native PyTorch support
- DeepSpeed offers ZeRO optimizations and extensive features for large model training
- Megatron-LM enables 3D parallelism for maximum-scale LLM training
- Horovod provides a simple, framework-agnostic approach
- Ray Train integrates with the Ray ecosystem for flexible distributed computing
- ColossalAI simplifies large model training with automatic parallelism
- Framework choice depends on scale, complexity, and ecosystem requirements
- Most frameworks can be combined (e.g., Megatron-LM + DeepSpeed)

## Framework Comparison

### Overview

| Framework | Primary Strength | Scale | Complexity |
|-----------|-----------------|-------|------------|
| PyTorch DDP | Simple data parallelism | 100s GPUs | Low |
| PyTorch FSDP | Native sharding | 1000s GPUs | Moderate |
| DeepSpeed | ZeRO + extensive features | 1000s GPUs | Moderate |
| Megatron-LM | 3D parallelism | 10000s GPUs | High |
| Horovod | Framework-agnostic | 100s GPUs | Low |
| Ray Train | Flexible scheduling | 1000s GPUs | Moderate |
| ColossalAI | Automatic parallelism | 1000s GPUs | Low |

### Feature Matrix

| Feature | DDP | FSDP | DeepSpeed | Megatron | Horovod | Ray Train | ColossalAI |
|---------|-----|------|-----------|----------|---------|-----------|------------|
| Data parallel | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Tensor parallel | No | No | Limited | Yes | No | No | Yes |
| Pipeline parallel | No | No | Yes | Yes | No | No | Yes |
| ZeRO Stage 1 | No | Partial | Yes | No | No | No | Yes |
| ZeRO Stage 2 | No | Yes | Yes | No | No | No | Yes |
| ZeRO Stage 3 | No | Yes | Yes | No | No | No | Yes |
| CPU offload | No | Yes | Yes | No | No | No | Yes |
| NVMe offload | No | No | Yes | No | No | No | No |
| Mixed precision | Manual | Yes | Yes | Yes | Manual | Manual | Yes |
| Activation checkpointing | Manual | Yes | Yes | Yes | Manual | Manual | Yes |

## Selection Guide

### By Model Size

**Small models (< 1B params, fits in GPU)**:
- Use PyTorch DDP
- Simple, well-tested, minimal overhead
- Add Horovod if need multi-framework support

**Medium models (1-10B params)**:
- Use FSDP or DeepSpeed ZeRO-2/3
- Both provide similar memory efficiency
- FSDP: Native PyTorch integration
- DeepSpeed: More features and documentation

**Large models (10-100B params)**:
- DeepSpeed ZeRO-3 with offloading
- Or Megatron-LM for maximum efficiency
- Consider ColossalAI for easier setup

**Very large models (100B+ params)**:
- Megatron-LM with 3D parallelism
- Often combined with DeepSpeed ZeRO
- Requires significant infrastructure expertise

### By Team Expertise

**Beginner**:
- Start with Hugging Face Trainer + DeepSpeed
- Or PyTorch Lightning with built-in distributed support
- Minimal code changes required

**Intermediate**:
- PyTorch FSDP or DeepSpeed directly
- More control over training loop
- Good documentation available

**Advanced**:
- Megatron-LM for custom architectures
- Custom combinations of frameworks
- Low-level optimization opportunities

### By Infrastructure

**Cloud (AWS, GCP, Azure)**:
- All frameworks work well
- Ray Train good for dynamic scaling
- Consider managed services (SageMaker, Vertex AI)

**HPC cluster**:
- Horovod or native PyTorch
- MPI integration often needed
- Existing job schedulers (Slurm)

**Heterogeneous hardware**:
- Ray Train for flexibility
- ColossalAI for automatic adaptation
- May need custom solutions

## Common Patterns

### Gradual Migration

Start simple and add complexity as needed:

```
1. Single GPU training (baseline)
   |
2. DDP for multi-GPU
   |
3. Add gradient accumulation for larger effective batch
   |
4. Add mixed precision for memory/speed
   |
5. Migrate to FSDP or DeepSpeed for larger models
   |
6. Add tensor/pipeline parallelism for very large models
```

### Combining Frameworks

Frameworks can be combined:

**Megatron-LM + DeepSpeed**:
- Megatron for tensor/pipeline parallelism
- DeepSpeed ZeRO for optimizer state sharding
- Common for large LLM training

**FSDP + Custom Tensor Parallel**:
- FSDP handles sharding
- Custom tensor parallel within nodes
- Good for research settings

**Ray Train + Any Framework**:
- Ray for job orchestration
- Any training framework for actual training
- Good for dynamic environments

## Ecosystem Considerations

### PyTorch Ecosystem

FSDP integrates well with:
- PyTorch Lightning
- Hugging Face Transformers
- TorchVision, TorchText
- Native PyTorch tools (profiler, etc.)

### Hugging Face Ecosystem

Good support for:
- DeepSpeed (via Trainer)
- FSDP (via Accelerate)
- Custom integration possible

### NVIDIA Ecosystem

Megatron-LM integrates with:
- NVIDIA NeMo framework
- Triton Inference Server
- TensorRT-LLM

## Debugging and Profiling

### Common Tools

**PyTorch Profiler**:
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    model(input)
print(prof.key_averages().table())
```

**NVIDIA Nsight**:
```bash
nsys profile python train.py
```

**DeepSpeed Flops Profiler**:
```python
from deepspeed.profiling.flops_profiler import FlopsProfiler
profiler = FlopsProfiler(model)
```

### Distributed-Specific Debugging

**NCCL Debug**:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**Hang detection**:
```python
# Set timeout for collectives
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
```

## Migration Paths

### From DDP to FSDP

```python
# Before: DDP
model = DDP(model, device_ids=[rank])

# After: FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=my_policy)
```

### From DDP to DeepSpeed

```python
# Before: DDP
model = DDP(model, device_ids=[rank])
optimizer = Adam(model.parameters())

# After: DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="ds_config.json"
)
```

### From DeepSpeed to Megatron

Requires significant code changes:
- Model must use Megatron parallelism primitives
- Training loop changes to Megatron style
- Often start with Megatron from scratch

## Further Reading

Detailed coverage of each framework:
- [PyTorch Distributed](pytorch-distributed/ReadMe.md): Native PyTorch DDP and FSDP
- [DeepSpeed](deepspeed/ReadMe.md): Microsoft's optimization library
- [Megatron-LM](megatron-lm/ReadMe.md): NVIDIA's large model training
- [Horovod](horovod/ReadMe.md): Uber's distributed training framework
- [Ray Train](ray-train/ReadMe.md): Distributed training on Ray
- [ColossalAI](colossalai/ReadMe.md): Easy large model training
