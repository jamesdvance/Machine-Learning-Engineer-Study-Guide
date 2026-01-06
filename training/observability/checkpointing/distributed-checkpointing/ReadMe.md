# Distributed Checkpointing

## Summary

Distributed checkpointing handles saving and loading model state across multiple GPUs and nodes. The complexity arises from sharded state (FSDP, ZeRO), coordination between processes, and efficient I/O. Proper distributed checkpointing ensures fault tolerance without creating bottlenecks or inconsistencies.

Key points to remember:

- Only rank 0 should save for DDP (replicated state)
- FSDP/ZeRO require gathering sharded state
- Use barriers for synchronization before/after I/O
- Parallel checkpointing for large models
- Handle optimizer state sharding
- Consider checkpoint size vs I/O bandwidth
- Test distributed resumption before long runs
- Use appropriate checkpoint format (full vs sharded)

## DDP Checkpointing

### Basic DDP Checkpoint

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# Only save on rank 0 (state is replicated)
if dist.get_rank() == 0:
    checkpoint = {
        'model_state_dict': model.module.state_dict(),  # .module to unwrap
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, 'checkpoint.pt')

# Synchronize all processes
dist.barrier()
```

### Loading in DDP

```python
# All ranks load the same checkpoint
map_location = {'cuda:0': f'cuda:{local_rank}'}
checkpoint = torch.load('checkpoint.pt', map_location=map_location)

# Load into unwrapped model before DDP wrapping
model.load_state_dict(checkpoint['model_state_dict'])

# Wrap with DDP
model = DDP(model, device_ids=[local_rank])

# All ranks load optimizer
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## FSDP Checkpointing

### Full State Dict (Gathered)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Configure to gather full state on rank 0
full_state_config = FullStateDictConfig(
    offload_to_cpu=True,
    rank0_only=True
)

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    full_state_config
):
    state_dict = model.state_dict()

    if dist.get_rank() == 0:
        torch.save(state_dict, 'checkpoint.pt')

dist.barrier()
```

### Sharded State Dict (Parallel I/O)

```python
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.checkpoint import save, load

# Configure sharded saving
sharded_config = ShardedStateDictConfig(offload_to_cpu=True)

with FSDP.state_dict_type(
    model,
    StateDictType.SHARDED_STATE_DICT,
    sharded_config
):
    state_dict = {"model": model.state_dict()}

    # Each rank saves its shard
    save(
        state_dict,
        checkpoint_id="checkpoint_dir",
    )
```

### Loading FSDP Checkpoint

```python
from torch.distributed.checkpoint import load

# Load sharded checkpoint
with FSDP.state_dict_type(
    model,
    StateDictType.SHARDED_STATE_DICT,
):
    state_dict = {"model": model.state_dict()}

    load(
        state_dict,
        checkpoint_id="checkpoint_dir",
    )

    model.load_state_dict(state_dict["model"])
```

## DeepSpeed Checkpointing

### ZeRO Stage 1-2

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# Save checkpoint (each rank saves its shard)
model_engine.save_checkpoint('checkpoints/', tag='step_1000')

# Load checkpoint
model_engine.load_checkpoint('checkpoints/', tag='step_1000')
```

### ZeRO Stage 3

```python
# Stage 3: Parameters are also sharded

# Save with parameter gathering for inference
model_engine.save_checkpoint(
    'checkpoints/',
    tag='step_1000',
    client_state={'epoch': epoch}
)

# Or save fp32 consolidated checkpoint
model_engine.save_16bit_model('model_fp16.pt')
```

### DeepSpeed Universal Checkpoint

```python
# Convert between different ZeRO stages
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

# Load ZeRO-3 checkpoint
state_dict = clone_tensors_for_torch_save(model_engine.module.state_dict())
torch.save(state_dict, 'consolidated.pt')
```

## Optimizer State Handling

### DDP Optimizer

```python
# Optimizer state is replicated in DDP
if dist.get_rank() == 0:
    torch.save(optimizer.state_dict(), 'optimizer.pt')

# All ranks load the same state
optimizer.load_state_dict(torch.load('optimizer.pt'))
```

### FSDP Optimizer

```python
from torch.distributed.fsdp import FullOptimStateDictConfig

optim_config = FullOptimStateDictConfig(
    offload_to_cpu=True,
    rank0_only=True
)

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    optim_state_dict_config=optim_config
):
    optim_state = FSDP.optim_state_dict(model, optimizer)

    if dist.get_rank() == 0:
        torch.save(optim_state, 'optim.pt')

# Loading
optim_state = torch.load('optim.pt')
optim_state = FSDP.optim_state_dict_to_load(
    model, optimizer, optim_state
)
optimizer.load_state_dict(optim_state)
```

## Parallel I/O

### Distributed Checkpoint API

```python
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import save

# Save in parallel
state_dict = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

save(state_dict, checkpoint_id="checkpoint_dir")

# Load in parallel
load(state_dict, checkpoint_id="checkpoint_dir")
model.load_state_dict(state_dict["model"])
```

### Custom Parallel Saving

```python
import os

def parallel_save(state_dict, base_path, rank, world_size):
    """Each rank saves a portion of the state dict."""
    keys = list(state_dict.keys())
    keys_per_rank = len(keys) // world_size

    start_idx = rank * keys_per_rank
    end_idx = start_idx + keys_per_rank if rank < world_size - 1 else len(keys)

    shard = {k: state_dict[k] for k in keys[start_idx:end_idx]}

    torch.save(shard, f"{base_path}/shard_{rank}.pt")
    dist.barrier()
```

## Synchronization

### Barrier Usage

```python
# Before saving: ensure all ranks have same state
dist.barrier()

if dist.get_rank() == 0:
    torch.save(checkpoint, 'checkpoint.pt')

# After saving: ensure file is written before others read
dist.barrier()
```

### Broadcast Checkpoint

```python
# Sometimes useful to broadcast checkpoint from rank 0
import io

if dist.get_rank() == 0:
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    data = buffer.getvalue()
    size = torch.tensor([len(data)], device='cuda')
else:
    size = torch.tensor([0], device='cuda')

dist.broadcast(size, src=0)

if dist.get_rank() != 0:
    data = bytes(size.item())

data_tensor = torch.ByteTensor(list(data)).cuda()
dist.broadcast(data_tensor, src=0)

if dist.get_rank() != 0:
    buffer = io.BytesIO(data_tensor.cpu().numpy().tobytes())
    checkpoint = torch.load(buffer)
```

## Best Practices

### 1. Verify Before Long Runs

```python
def verify_distributed_checkpoint():
    """Test checkpoint save/load before long training."""
    # Save
    save_distributed_checkpoint(model, optimizer, 'test_ckpt')

    # Create fresh model and optimizer
    test_model = create_model()
    test_optimizer = create_optimizer(test_model)

    # Load
    load_distributed_checkpoint(test_model, test_optimizer, 'test_ckpt')

    # Verify
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(),
        test_model.named_parameters()
    ):
        assert torch.equal(p1, p2), f"Mismatch in {n1}"

    print("Checkpoint verification passed!")
```

### 2. Handle Interruption

```python
import signal

def save_on_interrupt(signum, frame):
    """Emergency save on SIGTERM."""
    print(f"Rank {dist.get_rank()}: Received signal, saving checkpoint...")
    save_distributed_checkpoint(model, optimizer, 'emergency_ckpt')
    dist.barrier()
    exit(0)

signal.signal(signal.SIGTERM, save_on_interrupt)
```

### 3. Checkpoint Metadata

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),

    # Distributed training metadata
    'world_size': dist.get_world_size(),
    'rank': dist.get_rank(),

    # Training state
    'epoch': epoch,
    'global_step': global_step,

    # Configuration
    'config': config,
}
```

### 4. Async Checkpointing

```python
import threading

def async_save(state_dict, path):
    """Save checkpoint in background thread."""
    # Clone tensors to CPU (required for async)
    cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}

    def save_fn():
        torch.save(cpu_state, path)

    thread = threading.Thread(target=save_fn)
    thread.start()
    return thread

# Usage
save_thread = async_save(model.state_dict(), 'checkpoint.pt')
# Continue training...
save_thread.join()  # Wait before next checkpoint
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Deadlock | Missing barrier | Add barrier after save |
| Corrupt checkpoint | Incomplete write | Use atomic saves |
| OOM on save | Full state too large | Use sharded checkpointing |
| Slow save | Sequential I/O | Use parallel checkpointing |
| State mismatch | Different world sizes | Handle resharding |
