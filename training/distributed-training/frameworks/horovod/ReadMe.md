# Horovod

## Summary

Horovod is Uber's distributed deep learning framework that provides a simple, framework-agnostic approach to data-parallel training. Originally developed to scale TensorFlow training, Horovod now supports PyTorch, Keras, and MXNet with minimal code changes. Its design philosophy emphasizes simplicity: add a few lines of code to make single-GPU training distributed.

Key points to remember:

- Framework-agnostic: works with TensorFlow, PyTorch, Keras, MXNet
- Simple API: minimal code changes for distributed training
- Uses ring-allreduce for efficient gradient synchronization
- Good MPI integration for HPC environments
- Horovodrun launcher simplifies cluster deployment
- Elastic training support for fault tolerance
- Tensor Fusion for communication optimization
- Established and stable, though less feature-rich than alternatives

## Installation

```bash
# Basic installation
pip install horovod

# With specific framework support
HOROVOD_WITH_PYTORCH=1 pip install horovod

# With NCCL support
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

# Full installation
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
```

Verify installation:
```bash
horovodrun --check-build
```

## PyTorch Usage

### Basic Training Script

```python
import torch
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
torch.cuda.set_device(hvd.local_rank())

# Create model and move to GPU
model = MyModel().cuda()

# Scale learning rate by number of workers
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001 * hvd.size()
)

# Wrap optimizer with Horovod
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters()
)

# Broadcast initial parameters from rank 0
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Create data loader with distributed sampler
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank()
)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Saving Checkpoints

```python
# Only save on rank 0
if hvd.rank() == 0:
    torch.save(model.state_dict(), "model.pt")
```

### Metric Aggregation

```python
# Average metric across all workers
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# Usage
train_loss = metric_average(loss.item(), 'train_loss')
```

## TensorFlow Usage

### Keras Integration

```python
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# Initialize Horovod
hvd.init()

# Pin GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Create model
model = create_model()

# Scale learning rate
optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())

# Wrap optimizer
optimizer = hvd.DistributedOptimizer(optimizer)

# Compile model
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Broadcast initial variables
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Only save checkpoints on rank 0
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('model.h5'))

# Train
model.fit(
    dataset,
    epochs=10,
    callbacks=callbacks,
    steps_per_epoch=len(dataset) // hvd.size()
)
```

### TensorFlow 2 with Custom Training

```python
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

# Create model and optimizer
model = create_model()
optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())

@tf.function
def train_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    # Wrap tape for allreduce
    tape = hvd.DistributedGradientTape(tape)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Broadcast on first batch
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return loss

for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(dataset):
        loss = train_step(images, labels, batch_idx == 0)
```

## Launching

### horovodrun

```bash
# Single node, multiple GPUs
horovodrun -np 4 python train.py

# Multiple nodes
horovodrun -np 8 -H server1:4,server2:4 python train.py

# With specific network interface
horovodrun -np 4 --network-interface eth0 python train.py
```

### MPI

```bash
# Using mpirun
mpirun -np 4 python train.py

# Multiple nodes with hostfile
mpirun -np 8 --hostfile hosts python train.py
```

### Slurm

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

srun python train.py
```

## Configuration Options

### DistributedOptimizer Parameters

```python
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=hvd.Compression.fp16,  # Gradient compression
    backward_passes_per_step=1,         # For gradient accumulation
    op=hvd.Average,                     # Reduction operation
    gradient_predivide_factor=1.0       # Pre-scaling
)
```

### Tensor Fusion

Batch small tensors for efficient communication:

```python
# Environment variables
os.environ['HOROVOD_FUSION_THRESHOLD'] = '67108864'  # 64MB
os.environ['HOROVOD_CYCLE_TIME'] = '5'  # 5ms

# Or via horovodrun
horovodrun -np 4 \
    --fusion-threshold-mb 64 \
    --cycle-time-ms 5 \
    python train.py
```

### Hierarchical Allreduce

For multi-node with different network speeds:

```bash
horovodrun -np 8 \
    --hierarchical-allreduce \
    python train.py
```

First reduces within nodes (NVLink), then across nodes (InfiniBand).

## Elastic Training

### Enable Elastic Horovod

```python
import horovod.torch as hvd
from horovod.torch.elastic import run

def train(state):
    # Training code
    for epoch in range(state.epoch, num_epochs):
        for batch in dataloader:
            # ... training step ...
            state.commit()  # Mark progress

        state.epoch = epoch + 1

# Run with elastic wrapper
state = hvd.elastic.State(model, optimizer, epoch=0)
hvd.elastic.run(train, state)
```

### Launch Elastic Training

```bash
horovodrun -np 4 --min-np 2 --max-np 8 \
    --host-discovery-script discover_hosts.sh \
    python train.py
```

Features:
- Workers can join/leave during training
- Automatic state synchronization
- Recovery from failures

## Mixed Precision

### PyTorch with AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Compression

```python
# FP16 compression
optimizer = hvd.DistributedOptimizer(
    optimizer,
    compression=hvd.Compression.fp16
)
```

## Timeline Profiling

### Enable Timeline

```python
# Environment variable
os.environ['HOROVOD_TIMELINE'] = '/path/to/timeline.json'

# Or via horovodrun
horovodrun -np 4 --timeline-filename /path/to/timeline.json python train.py
```

### View Timeline

Open in Chrome: `chrome://tracing/`

Shows:
- Communication operations
- Computation operations
- Overlap patterns
- Bottlenecks

## Performance Tuning

### NCCL Parameters

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

### Horovod Parameters

```bash
# Increase fusion threshold for large models
export HOROVOD_FUSION_THRESHOLD=134217728

# Reduce cycle time for smaller models
export HOROVOD_CYCLE_TIME=1

# Enable NCCL operations
export HOROVOD_GPU_OPERATIONS=NCCL
```

### Best Practices

1. **Use NCCL backend** for GPU training
2. **Tune fusion threshold** based on model size
3. **Enable hierarchical allreduce** for multi-node
4. **Use gradient compression** for bandwidth-limited networks
5. **Profile with timeline** to identify bottlenecks

## Comparison with Alternatives

| Aspect | Horovod | PyTorch DDP | DeepSpeed |
|--------|---------|-------------|-----------|
| Simplicity | High | Moderate | Moderate |
| Framework support | Multi | PyTorch only | PyTorch focus |
| Features | Basic | Native | Extensive |
| Memory optimization | None | FSDP | ZeRO |
| Active development | Maintenance | Active | Active |

### When to Use Horovod

- Multi-framework environment
- Existing MPI infrastructure
- Simple data parallelism needs
- HPC cluster deployment
- Stable, proven solution needed

### When to Consider Alternatives

- Need memory optimization (ZeRO/FSDP)
- PyTorch-only environment
- Want latest features
- Need tensor/pipeline parallelism
