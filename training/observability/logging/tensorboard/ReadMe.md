# TensorBoard

## Summary

TensorBoard is a visualization toolkit originally developed for TensorFlow but now widely used with PyTorch and other frameworks. It provides interactive dashboards for visualizing training metrics, model graphs, embeddings, and more. TensorBoard is lightweight, runs locally, and integrates directly with PyTorch via torch.utils.tensorboard.

Key points to remember:

- Built-in PyTorch support via SummaryWriter
- Visualize scalars, images, histograms, graphs
- Runs locally or on TensorBoard.dev
- Low overhead logging
- Compare multiple runs side-by-side
- Profile training performance
- No account required for local use
- Supports custom plugins

## Basic Setup

### Installation

```bash
pip install tensorboard

# For PyTorch integration
pip install torch  # SummaryWriter included
```

### Basic Usage

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log scalar
for step in range(1000):
    loss = train_step()
    writer.add_scalar('Loss/train', loss, step)

# Close writer
writer.close()
```

### Launch TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir runs

# Access at http://localhost:6006
```

## Logging Metrics

### Scalars

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Single scalar
writer.add_scalar('loss', loss, global_step)

# Multiple scalars (same plot)
writer.add_scalars('loss', {
    'train': train_loss,
    'val': val_loss
}, global_step)

# Grouped scalars (different plots)
writer.add_scalar('Loss/train', train_loss, step)
writer.add_scalar('Loss/val', val_loss, step)
writer.add_scalar('Accuracy/train', train_acc, step)
writer.add_scalar('Accuracy/val', val_acc, step)
```

### Histograms

```python
# Log weight distributions
for name, param in model.named_parameters():
    writer.add_histogram(f'weights/{name}', param, step)
    if param.grad is not None:
        writer.add_histogram(f'gradients/{name}', param.grad, step)
```

### Images

```python
import torchvision.utils as vutils

# Single image
writer.add_image('sample', image_tensor, step)

# Grid of images
img_grid = vutils.make_grid(images, normalize=True)
writer.add_image('batch', img_grid, step)

# With different formats
# CHW format (default)
writer.add_image('image', img_chw, step)
# HWC format
writer.add_image('image', img_hwc, step, dataformats='HWC')
```

### Text

```python
# Log text
writer.add_text('prediction', f'Input: {input_text}\nOutput: {output_text}', step)

# Markdown supported
writer.add_text('summary', '**Bold** and *italic* text', step)
```

## Model Visualization

### Computation Graph

```python
# Log model graph
dummy_input = torch.randn(1, 3, 224, 224)
writer.add_graph(model, dummy_input)
```

### Embeddings

```python
# Log embeddings for visualization
import torch

# Features and labels
features = model.get_embeddings(data)  # [N, D]
labels = data.labels  # [N]
images = data.images  # [N, C, H, W] optional

writer.add_embedding(
    features,
    metadata=labels,
    label_img=images,
    global_step=step,
    tag='embeddings'
)
```

## Training Loop Example

```python
from torch.utils.tensorboard import SummaryWriter
import torch

# Initialize
writer = SummaryWriter(f'runs/{experiment_name}')

# Log hyperparameters
writer.add_hparams(
    hparam_dict={'lr': lr, 'batch_size': batch_size},
    metric_dict={}  # Will be updated at end
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for step, batch in enumerate(train_loader):
        loss = train_step(batch)
        epoch_loss += loss.item()

        # Log every N steps
        if step % log_interval == 0:
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('Loss/train_step', loss, global_step)

    # Log epoch metrics
    avg_train_loss = epoch_loss / len(train_loader)
    val_loss, val_acc = validate(model)

    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # Log histograms periodically
    if epoch % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

# Log final hyperparameters with results
writer.add_hparams(
    hparam_dict={'lr': lr, 'batch_size': batch_size},
    metric_dict={'final_val_loss': val_loss, 'final_val_acc': val_acc}
)

writer.close()
```

## Comparing Runs

### Multiple Experiments

```python
# Run 1
writer1 = SummaryWriter('runs/exp_lr_0.001')
# ... training ...

# Run 2
writer2 = SummaryWriter('runs/exp_lr_0.0001')
# ... training ...

# TensorBoard shows both in same view
```

### Organized Naming

```python
from datetime import datetime

# Include timestamp and config
run_name = f'{datetime.now():%Y%m%d_%H%M%S}_lr{lr}_bs{batch_size}'
writer = SummaryWriter(f'runs/{run_name}')
```

## Profiling

### PyTorch Profiler Integration

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Profile training
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=tensorboard_trace_handler('./runs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(batch)
        prof.step()

# View in TensorBoard with --logdir runs/profiler
```

## Custom Scalars

### Layout Configuration

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import custom_scalars

# Define custom layout
layout = {
    'Loss': {
        'combined': ['Multiline', ['Loss/train', 'Loss/val']],
    },
    'Accuracy': {
        'combined': ['Multiline', ['Accuracy/train', 'Accuracy/val']],
    }
}

writer = SummaryWriter()
writer.add_custom_scalars(layout)
```

## TensorBoard.dev

### Upload to Cloud

```bash
# Upload logs to TensorBoard.dev
tensorboard dev upload --logdir runs

# Share the generated URL with team
# https://tensorboard.dev/experiment/xxxxx
```

### Benefits

- Shareable links
- Persistent storage
- No local server needed
- Collaboration features

## Distributed Training

### Logging on Rank 0

```python
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# Only create writer on rank 0
writer = None
if not dist.is_initialized() or dist.get_rank() == 0:
    writer = SummaryWriter()

# Log only on rank 0
if writer is not None:
    writer.add_scalar('loss', loss, step)
```

## Integration with Frameworks

### PyTorch Lightning

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('logs/', name='my_model')
trainer = pl.Trainer(logger=logger)
```

### Hugging Face

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    report_to='tensorboard',
    logging_steps=100,
)
```

## Best Practices

1. **Use consistent naming**: Group related metrics with prefixes
2. **Log sparingly**: Don't log every step for histograms
3. **Flush periodically**: `writer.flush()` ensures data is written
4. **Close writer**: Always call `writer.close()` at end
5. **Use run names**: Include config info in run directory
6. **Profile early**: Use profiler to identify bottlenecks
7. **Clean old logs**: Manage disk space

## Troubleshooting

### Common Issues

```python
# Port already in use
tensorboard --logdir runs --port 6007

# Can't see updates
writer.flush()  # Force write to disk

# Too many events
# Reduce logging frequency or use add_scalar less often

# Memory issues with histograms
# Log histograms less frequently
if step % 1000 == 0:
    writer.add_histogram(...)
```

### Debug Mode

```python
# Verbose logging
import logging
logging.getLogger('tensorboard').setLevel(logging.DEBUG)
```
