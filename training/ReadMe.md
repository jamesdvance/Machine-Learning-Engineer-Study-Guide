# Training

## Summary

Training machine learning models at scale requires understanding distributed systems, hardware optimization, and operational best practices. Modern training infrastructure has evolved from single-GPU scripts to sophisticated distributed systems spanning thousands of accelerators. The core challenge is achieving efficient hardware utilization while maintaining model convergence and training stability.

Key points to remember:

- Distribution strategy depends on model size: data parallelism for models that fit in memory, model/tensor/pipeline parallelism when they do not
- Mixed precision training (FP16/BF16) nearly doubles throughput with minimal accuracy impact when used correctly
- Gradient checkpointing trades compute for memory, enabling larger batch sizes or model sizes
- Checkpointing strategy directly impacts recovery time and storage costs
- Training frameworks abstract distributed complexity but understanding primitives helps debug issues
- Cost optimization requires balancing GPU utilization, spot instance interruptions, and convergence speed
- Hyperparameter tuning can improve final model quality more than architecture changes
- Observability is essential: training runs without proper logging are debugging nightmares

## The Training Landscape

### Scale Matters

Training requirements vary dramatically:

| Model Type | Typical Scale | Training Time | Key Challenges |
|------------|---------------|---------------|----------------|
| Small classifier | 1 GPU, hours | Minutes-hours | Overfitting |
| Medium vision model | 1-8 GPUs, days | Hours-days | Learning rate tuning |
| Large language model | 100s-1000s GPUs, weeks | Days-weeks | Distributed coordination |
| Frontier LLM | 10,000+ GPUs, months | Weeks-months | Hardware failures, cost |

The techniques in this section apply across scales, but their importance shifts. A single-GPU training run rarely needs pipeline parallelism. A thousand-GPU run cannot function without it.

### Hardware Considerations

Modern ML training runs on specialized accelerators:

**NVIDIA GPUs**: Dominant in ML training. H100 and A100 are current standards.
- H100: 80GB HBM3, 3.35 TB/s bandwidth, FP8 support
- A100: 40/80GB HBM2e, 2 TB/s bandwidth
- Consumer cards (RTX 4090): Capable but limited memory and no NVLink

**TPUs**: Google's custom accelerators, available through GCP.
- TPU v4: 32GB HBM, optimized for JAX/TensorFlow
- Strong for large-scale training when using Google's ecosystem

**AMD GPUs**: Emerging alternative with MI300X.
- Competitive specs but ecosystem still maturing
- ROCm stack improving but lags CUDA

**Memory hierarchy matters**: GPU HBM bandwidth often bottlenecks training. Understanding when you are compute-bound versus memory-bound shapes optimization strategy.

### Training Phases

Large model training typically involves distinct phases:

**Pretraining**: Learning general representations from large datasets.
- Highest compute cost
- Data parallelism with large batches
- Focus on throughput and stability

**Fine-tuning**: Adapting to specific tasks.
- Much smaller compute requirements
- Often uses parameter-efficient methods (LoRA, adapters)
- Focus on preventing catastrophic forgetting

**Alignment**: Making models follow instructions and human preferences.
- RLHF, DPO, or similar methods
- Requires reward model or preference data
- Computational cost varies by method

## Distribution Strategies

### Data Parallelism

The simplest distribution strategy: replicate the model across devices, split data batches.

```
GPU 0: Full model copy, batch 0
GPU 1: Full model copy, batch 1
GPU 2: Full model copy, batch 2
...
All-reduce gradients after backward pass
```

**When to use**: Model fits in single GPU memory. Most common case.

**Variants**:
- Distributed Data Parallel (DDP): Synchronous gradient averaging
- Fully Sharded Data Parallel (FSDP): Shards optimizer states and optionally weights
- ZeRO: Memory optimization stages (1/2/3) with increasing memory savings

### Model Parallelism

Split model layers across devices when model does not fit in memory.

```
GPU 0: Layers 0-10
GPU 1: Layers 11-20
GPU 2: Layers 21-30
```

**Problem**: GPUs idle while waiting for activations from previous device (pipeline bubble).

**Solution**: Pipeline parallelism with micro-batching.

### Tensor Parallelism

Split individual tensor operations across devices.

```
Matrix multiplication: A @ B
GPU 0: A @ B[:, :half]
GPU 1: A @ B[:, half:]
Concatenate results
```

**When to use**: Very large layers that benefit from parallel computation. Common in LLM training with large hidden dimensions.

**Trade-off**: Requires high-bandwidth interconnects (NVLink, InfiniBand) due to frequent communication.

### Pipeline Parallelism

Divide model into stages, process multiple micro-batches simultaneously.

```
Time ->
GPU 0: [B0] [B1] [B2] [B3] [  ] [  ] [B0'] [B1'] [B2'] [B3']
GPU 1: [  ] [B0] [B1] [B2] [B3] [  ] [  ]  [B0'] [B1'] [B2']
GPU 2: [  ] [  ] [B0] [B1] [B2] [B3] [  ]  [  ]  [B0'] [B1']
```

**Schedules**:
- GPipe: Simple but large bubble
- 1F1B: Interleaved forward/backward reduces bubble
- Interleaved 1F1B: Further reduces bubble with more stages per device

### Hybrid Approaches

Large-scale training combines strategies:

```
                    Data Parallel (across nodes)
                    /                           \
        Node 0                                   Node 1
    (Tensor Parallel)                       (Tensor Parallel)
    /      |      \                         /      |      \
  GPU0   GPU1   GPU2                      GPU0   GPU1   GPU2

  Pipeline stages distributed across tensor-parallel groups
```

The Megatron-LM paper demonstrated effective 3D parallelism combining all three approaches.

## Performance Optimization

### Mixed Precision Training

Using lower precision (FP16/BF16) for most operations while maintaining FP32 for critical computations.

**Benefits**:
- Nearly 2x memory reduction
- Faster tensor core operations
- Enables larger batch sizes

**Implementation**:
```python
# PyTorch automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP16 vs BF16**:
- FP16: Higher precision, narrower range, needs loss scaling
- BF16: Lower precision, same range as FP32, no loss scaling needed
- BF16 preferred on modern hardware (A100, H100) for stability

### Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass.

**Without checkpointing**: Store all activations (O(n) memory for n layers)
**With checkpointing**: Store checkpoints, recompute between them (O(sqrt(n)) memory)

**Usage**:
```python
# PyTorch
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

**Trade-off**: Approximately 33% compute overhead for significant memory savings.

### Flash Attention

Memory-efficient attention that avoids materializing the full attention matrix.

**Standard attention**: O(n^2) memory for sequence length n
**Flash attention**: O(n) memory by computing attention in tiles

**Impact**: Enables much longer sequences without memory explosion. Essential for long-context models.

### Compile Optimizations

Just-in-time compilation for faster execution:

**torch.compile** (PyTorch 2.0+):
```python
model = torch.compile(model)  # Default mode
model = torch.compile(model, mode="reduce-overhead")  # Lower latency
model = torch.compile(model, mode="max-autotune")  # Best throughput
```

**XLA** (TensorFlow/JAX):
```python
# JAX
@jax.jit
def train_step(params, batch):
    ...
```

Compilation can provide 20-50% speedup but adds initial compilation overhead.

## Training Frameworks

Several frameworks simplify distributed training:

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| PyTorch Distributed | Native, flexible | Custom training loops |
| PyTorch Lightning | Structured, less boilerplate | Research, prototyping |
| Hugging Face Accelerate | Simple API, HF integration | Transformer fine-tuning |
| DeepSpeed | ZeRO optimizations, large models | LLM training |
| Megatron-LM | 3D parallelism | Very large LLMs |
| JAX/Flax | Functional, TPU-native | Google Cloud, research |
| ColossalAI | Easy large model support | LLM fine-tuning |

### Framework Selection

**Starting out**: PyTorch Lightning or Hugging Face Accelerate. They handle common patterns and reduce boilerplate.

**Large models**: DeepSpeed for ZeRO optimizations. Megatron-LM for maximum scale with 3D parallelism.

**Research**: JAX/Flax for functional style and easy parallelization. Pure PyTorch for maximum control.

**Production**: Often custom training loops with specific optimizations. Frameworks are starting points.

## Observability

### Why It Matters

Training runs fail in subtle ways:
- Gradients explode or vanish without crashing
- Loss plateaus due to learning rate issues
- Memory leaks accumulate over hours
- One node silently underperforms

Without observability, these issues waste GPU hours and delay projects.

### Key Metrics

**Training metrics**:
- Loss (train and validation)
- Learning rate
- Gradient norm
- Throughput (samples/second, tokens/second)

**System metrics**:
- GPU utilization
- Memory usage (allocated vs reserved)
- Network bandwidth (for distributed)
- Temperature and throttling

**Debugging metrics**:
- Per-layer gradient statistics
- Activation magnitudes
- Parameter update ratios

### Checkpointing Strategy

Checkpoints serve two purposes: recovery from failures and saving model versions.

**Frequency trade-offs**:
- Too frequent: Storage costs, I/O overhead
- Too infrequent: Lost progress on failure

**Best practices**:
- Checkpoint based on steps, not time
- Keep last N checkpoints plus periodic archives
- Use async checkpointing to avoid blocking training
- Verify checkpoint integrity

## Cost Optimization

Training costs grow quickly at scale. Key optimization levers:

### Spot/Preemptible Instances

Cloud providers offer 60-90% discounts for interruptible instances.

**Requirements**:
- Robust checkpointing
- Fast recovery mechanism
- Graceful shutdown handling

**Not suitable for**: Short training runs, time-sensitive deadlines.

### Right-Sizing

Match GPU to workload:

| Workload | Recommended |
|----------|-------------|
| Fine-tuning small models | A10G, T4 |
| Training medium models | A100 40GB |
| Training large models | A100 80GB, H100 |
| LLM pretraining | H100 with NVLink |

Overprovisioning wastes money. Underprovisioning wastes time to failures.

### Utilization Optimization

GPU utilization below 80% suggests optimization opportunities:
- Increase batch size
- Reduce data loading overhead
- Profile and fix bottlenecks
- Use gradient accumulation

### Early Stopping and Learning Rate Scheduling

Do not train longer than necessary:
- Monitor validation metrics
- Stop when improvement plateaus
- Use learning rate scheduling to converge faster

## Hyperparameter Optimization

### What to Tune

High-impact hyperparameters:
1. Learning rate (and schedule)
2. Batch size
3. Weight decay
4. Warmup steps
5. Architecture choices (layers, dimensions)

Lower-impact but sometimes important:
- Dropout rate
- Optimizer betas
- Gradient clipping threshold

### Tuning Strategies

**Grid search**: Exhaustive but expensive. Only for 1-2 parameters.

**Random search**: More efficient than grid for most cases. Good baseline.

**Bayesian optimization**: Models the objective function. Good for expensive evaluations.

**Population-based training**: Evolves hyperparameters during training. Good for long runs.

### Practical Approach

1. Start with known-good hyperparameters from papers or similar work
2. Do a coarse random search over learning rate and batch size
3. Refine with Bayesian optimization if compute budget allows
4. Final runs with best hyperparameters and full training budget

## Common Pitfalls

### Training Instabilities

**Loss spikes**: Often caused by bad data batches or learning rate too high.
- Solution: Gradient clipping, lower learning rate, data filtering

**NaN losses**: Numerical overflow, usually in FP16.
- Solution: Enable loss scaling, switch to BF16, check for division by zero

**Gradient explosion**: Common in RNNs and deep networks.
- Solution: Gradient clipping, better initialization, skip connections

### Scaling Issues

**Learning rate scaling**: When increasing batch size, may need to adjust learning rate.
- Linear scaling: Double batch size, double learning rate (up to a point)
- Square root scaling: More conservative, often works better

**Batch size too large**: Can hurt generalization.
- Solution: Use gradient accumulation to simulate large batches while maintaining smaller effective batches

### Distributed Training Issues

**Straggler nodes**: One slow node bottlenecks all-reduce.
- Solution: Monitor per-node throughput, replace underperforming nodes

**Gradient synchronization bugs**: Incorrect all-reduce can cause divergence.
- Solution: Verify gradient statistics match across ranks

**Memory fragmentation**: Long runs can fragment GPU memory.
- Solution: Periodic memory cleanup, reserve memory upfront

## Further Reading

### Distributed Training
Understanding parallelism strategies and communication patterns:
- [Concepts](distributed-training/concepts/ReadMe.md): Core parallelism patterns
- [Frameworks](distributed-training/frameworks/ReadMe.md): DeepSpeed, Megatron-LM, Ray Train
- [Communication](distributed-training/communication/ReadMe.md): NCCL, Gloo, MPI

### Performance
Optimizing training throughput and memory usage:
- [Mixed Precision Training](performance/mixed-precision-training/ReadMe.md): FP16, BF16, FP8
- [Gradient Checkpointing](performance/gradient-checkpointing/ReadMe.md): Trading compute for memory
- [Memory Optimization](performance/memory-optimization/ReadMe.md): Flash Attention, offloading
- [Compile Optimizations](performance/compile-optimizations/ReadMe.md): torch.compile, XLA

### Frameworks
Tools for simplifying distributed training:
- [PyTorch Lightning](frameworks/pytorch-lightning/ReadMe.md): Structured training
- [Hugging Face Accelerate](frameworks/hugging-face-accelerate/ReadMe.md): Simple distributed API
- [JAX/Flax](frameworks/jax-flax/ReadMe.md): Functional ML framework

### Observability
Monitoring and debugging training runs:
- [Checkpointing](observability/checkpointing/ReadMe.md): Saving and recovering state
- [Logging](observability/logging/ReadMe.md): TensorBoard, W&B, MLflow
- [Debugging](observability/debugging/ReadMe.md): Gradient monitoring, NaN detection

### Cost Optimization
Reducing training expenses:
- [Spot Instances](cost-optimization/spot-preemptible-instances/ReadMe.md): Using interruptible compute
- [Right-Sizing](cost-optimization/right-sizing-gpu-selection/ReadMe.md): Matching GPU to workload
- [Learning Rate Scheduling](cost-optimization/learning-rate-scheduling/ReadMe.md): Converging faster

### Hyperparameter Optimization
Finding optimal training configurations:
- [Bayesian Optimization](hyperparameter-optimization/bayesian-optimization/ReadMe.md): Efficient search
- [Optuna](hyperparameter-optimization/optuna/ReadMe.md): Popular tuning framework
- [Ray Tune](hyperparameter-optimization/ray-tune/ReadMe.md): Distributed HPO
