# Data Parallelism

## Summary

Data parallelism is the most common distributed training strategy, where the model is replicated across multiple devices and each replica processes a different subset of the training data. After computing gradients locally, devices synchronize through an all-reduce operation to ensure all replicas maintain identical parameters. This approach is straightforward to implement and scales well when the model fits in GPU memory.

Key points to remember:

- Each device holds a complete copy of the model
- Data batches are split across devices, increasing effective batch size
- Gradients are synchronized via all-reduce after each backward pass
- Communication cost is proportional to model size, not data size
- PyTorch DDP is the standard implementation with good performance
- Effective batch size scales linearly with device count
- May require learning rate adjustment when scaling to many devices
- Works best when model fits comfortably in GPU memory

## How Data Parallelism Works

### Basic Algorithm

```
1. Initialize model on all N devices (broadcast from rank 0)
2. For each training step:
   a. Each device i receives batch_i (different data)
   b. Forward pass: output_i = model(batch_i)
   c. Loss computation: loss_i = criterion(output_i, target_i)
   d. Backward pass: loss_i.backward() computes gradients
   e. All-reduce: average gradients across all devices
   f. Optimizer step: update parameters (identical on all devices)
```

### Visual Representation

```
         Data Batch
    /    /    |    \    \
  GPU0  GPU1  GPU2  GPU3  GPU4
   |     |     |     |     |
  Model Model Model Model Model
   |     |     |     |     |
  Grad  Grad  Grad  Grad  Grad
   \     \     |    /     /
        All-Reduce (Average)
   /     /     |    \     \
  GPU0  GPU1  GPU2  GPU3  GPU4
   |     |     |     |     |
 Update Update Update Update Update
```

### Why It Works

Mathematically equivalent to training with a larger batch:

```
Single device:
  gradient = (1/B) * sum(grad_i for i in batch)

Data parallel (N devices):
  gradient = (1/N) * sum(local_gradient_j for j in devices)
           = (1/N) * sum((1/B) * sum(grad_i for i in local_batch_j))
           = (1/(N*B)) * sum(all gradients)
```

The final gradient is the same as computing over all data at once.

## PyTorch DDP Implementation

### Basic Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

    cleanup()
```

### Launching DDP Training

```python
import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
```

Or using torchrun:
```bash
torchrun --nproc_per_node=4 train.py
```

### DDP Internals

DDP performs several optimizations:

**Gradient bucketing**: Groups small gradients into larger buckets for more efficient all-reduce.

```python
# DDP creates buckets automatically, but you can configure
model = DDP(model, bucket_cap_mb=25)  # 25MB buckets
```

**Overlapping communication**: Starts all-reduce for computed gradients while backward pass continues.

```python
# This happens automatically when using hooks
# Backward computation and communication overlap
```

**Gradient synchronization hooks**: Registers hooks on backward pass to trigger communication.

## Distributed Data Sampling

### DistributedSampler

Ensures each rank sees different data:

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,
    drop_last=True  # Ensures equal batches across ranks
)

# Must set epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        ...
```

### Manual Data Splitting

For custom data loading:

```python
def get_rank_data(dataset, rank, world_size):
    total_size = len(dataset)
    per_rank = total_size // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank
    return dataset[start_idx:end_idx]
```

## Scaling Considerations

### Learning Rate Scaling

When increasing batch size via more GPUs, adjust learning rate:

```python
# Linear scaling rule
base_lr = 0.001
base_batch_size = 32
effective_batch_size = batch_size * world_size
scaled_lr = base_lr * (effective_batch_size / base_batch_size)

# With warmup for stability
warmup_steps = 1000
for step in range(warmup_steps):
    current_lr = scaled_lr * (step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
```

### Batch Size Considerations

| GPUs | Per-GPU Batch | Effective Batch | Notes |
|------|---------------|-----------------|-------|
| 1 | 32 | 32 | Baseline |
| 8 | 32 | 256 | May need LR adjustment |
| 64 | 32 | 2048 | Likely needs warmup |
| 512 | 32 | 16384 | Near scaling limits |

### Communication Overhead

All-reduce cost per step:
```
Time = 2 * model_size * (world_size - 1) / (world_size * bandwidth)
```

For 1B parameters (4GB in FP32), 100 Gbps network:
```
Time approximately equals 2 * 4GB / 12.5GB/s = 0.64 seconds
```

This limits scaling efficiency as GPU count increases.

## Common Patterns

### Gradient Clipping with DDP

```python
# Clip after backward, before optimizer step
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Mixed Precision with DDP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = DDP(model, device_ids=[rank])

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Evaluation with DDP

```python
def evaluate(model, dataloader, rank, world_size):
    model.eval()
    total_loss = torch.tensor(0.0, device=rank)
    total_samples = torch.tensor(0, device=rank)

    with torch.no_grad():
        for batch in dataloader:
            loss = model(batch)
            total_loss += loss * len(batch)
            total_samples += len(batch)

    # Aggregate across ranks
    dist.all_reduce(total_loss)
    dist.all_reduce(total_samples)

    return (total_loss / total_samples).item()
```

### Checkpointing with DDP

```python
def save_checkpoint(model, optimizer, epoch, rank):
    if rank == 0:  # Only save from rank 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Note: .module
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_{epoch}.pt')
    dist.barrier()  # Ensure all ranks wait

def load_checkpoint(model, optimizer, path, rank):
    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(path, map_location=map_location)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

## Performance Optimization

### Finding Unused Parameters

DDP needs to know about unused parameters to avoid hangs:

```python
model = DDP(model, device_ids=[rank], find_unused_parameters=True)
```

Only use when necessary as it adds overhead.

### Static Graph Optimization

For models with fixed computation graphs:

```python
model = DDP(model, device_ids=[rank], static_graph=True)
```

Enables additional optimizations for repeated execution patterns.

### Gradient Compression

Reduce communication volume:

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

model = DDP(model, device_ids=[rank])
model.register_comm_hook(
    state=None,
    hook=default_hooks.fp16_compress_hook
)
```

## Comparison with Alternatives

| Aspect | DDP | DataParallel | Horovod |
|--------|-----|--------------|---------|
| Multi-node | Yes | No | Yes |
| Performance | Best | Poor | Good |
| Setup complexity | Moderate | Simple | Moderate |
| Framework support | PyTorch | PyTorch | Multi-framework |

**DataParallel** (deprecated approach):
- Single-process, multi-GPU
- GIL bottleneck limits performance
- Avoid for new code

**Horovod**:
- Framework-agnostic
- Slightly different API
- Good for existing Horovod deployments

## Debugging Tips

### Verify Gradient Synchronization

```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_sum = param.grad.sum()
        dist.all_reduce(grad_sum)
        if rank == 0:
            print(f"{name}: grad_sum = {grad_sum.item()}")
```

### Check Data Distribution

```python
# Each rank should see different data
for batch_idx, batch in enumerate(dataloader):
    data_hash = hash(batch.data.cpu().numpy().tobytes())
    print(f"Rank {rank}, Batch {batch_idx}: hash = {data_hash}")
```

### Monitor All-Reduce Timing

```python
import time

loss.backward()

start = time.time()
# DDP handles all-reduce automatically
optimizer.step()  # Synchronization happens here
dist.barrier()
end = time.time()

if rank == 0:
    print(f"Sync time: {end - start:.3f}s")
```
