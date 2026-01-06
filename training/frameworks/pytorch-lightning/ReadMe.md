# PyTorch Lightning

## Summary

PyTorch Lightning is a lightweight PyTorch wrapper that organizes code into reusable components while automating training boilerplate. It separates research code (model, loss, optimization) from engineering code (distributed training, mixed precision, logging), enabling cleaner code and easier scaling. Lightning provides structure without sacrificing PyTorch flexibility.

Key points to remember:

- Organizes PyTorch code into LightningModule and Trainer
- Automates distributed training, mixed precision, checkpointing
- Preserves full PyTorch flexibility within training_step
- Built-in logging to TensorBoard, W&B, MLflow
- Callbacks for customization without modifying training loop
- Supports DDP, FSDP, DeepSpeed out of the box
- LightningDataModule for reproducible data handling
- ~40+ Trainer flags for common features

## Core Components

### LightningModule

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Saves lr to checkpoint
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
```

### Trainer

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

trainer = Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=4,
    strategy='ddp',
    precision='16-mixed',
    callbacks=[
        ModelCheckpoint(monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5)
    ],
    logger=True,  # Default TensorBoard
    gradient_clip_val=1.0,
    accumulate_grad_batches=4
)

trainer.fit(model, train_loader, val_loader)
```

### LightningDataModule

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, data_dir='./data'):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self):
        # Download (called once, single process)
        MNIST(self.data_dir, download=True)

    def setup(self, stage=None):
        # Split, transform (called on each process)
        if stage == 'fit' or stage is None:
            full = MNIST(self.data_dir, train=True, transform=ToTensor())
            self.train_data, self.val_data = random_split(full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test_data = MNIST(self.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# Usage
datamodule = MNISTDataModule(batch_size=64)
trainer.fit(model, datamodule=datamodule)
```

## Distributed Training

### Data Parallel (DDP)

```python
trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy='ddp'
)
```

### Fully Sharded Data Parallel (FSDP)

```python
from pytorch_lightning.strategies import FSDPStrategy

strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",
    cpu_offload=False
)

trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy=strategy,
    precision='16-mixed'
)
```

### DeepSpeed

```python
from pytorch_lightning.strategies import DeepSpeedStrategy

strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=True
)

trainer = Trainer(
    accelerator='gpu',
    devices=8,
    strategy=strategy,
    precision='16-mixed'
)
```

### Multi-Node Training

```python
trainer = Trainer(
    accelerator='gpu',
    devices=8,
    num_nodes=4,
    strategy='ddp'
)

# Launch with:
# srun python train.py
# or
# torchrun --nnodes=4 --nproc_per_node=8 train.py
```

## Mixed Precision

### Automatic Mixed Precision

```python
# FP16 with loss scaling
trainer = Trainer(precision='16-mixed')

# BF16 (no loss scaling needed)
trainer = Trainer(precision='bf16-mixed')

# FP32 (default)
trainer = Trainer(precision='32-true')
```

### With FSDP

```python
trainer = Trainer(
    strategy='fsdp',
    precision='16-mixed'  # or 'bf16-mixed'
)
```

## Logging

### Built-in Loggers

```python
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger

# TensorBoard
logger = TensorBoardLogger('logs/', name='my_model')

# Weights & Biases
logger = WandbLogger(project='my_project', name='run_1')

# MLflow
logger = MLFlowLogger(experiment_name='my_experiment')

trainer = Trainer(logger=logger)
```

### Logging Metrics

```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # Simple logging
        self.log('train_loss', loss)

        # With options
        self.log('train_loss', loss,
            prog_bar=True,      # Show in progress bar
            on_step=True,       # Log at each step
            on_epoch=True,      # Log epoch average
            sync_dist=True      # Sync across devices
        )

        # Log multiple metrics
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc
        })

        return loss
```

## Callbacks

### Built-in Callbacks

```python
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    GradientAccumulationScheduler
)

callbacks = [
    # Save best models
    ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    ),

    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    ),

    # Log learning rate
    LearningRateMonitor(logging_interval='step'),

    # Nice progress bar
    RichProgressBar(),

    # Dynamic gradient accumulation
    GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
]

trainer = Trainer(callbacks=callbacks)
```

### Custom Callbacks

```python
from pytorch_lightning.callbacks import Callback

class CustomCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_train_epoch_end(self, trainer, pl_module):
        # Access metrics
        metrics = trainer.callback_metrics
        print(f"Epoch {trainer.current_epoch}: loss={metrics.get('train_loss', 0):.4f}")

    def on_validation_end(self, trainer, pl_module):
        # Custom logic after validation
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log custom metrics
        if batch_idx % 100 == 0:
            pl_module.log('custom_metric', compute_metric())

trainer = Trainer(callbacks=[CustomCallback()])
```

## Checkpointing

### Automatic Checkpointing

```python
# Saves automatically
trainer = Trainer(
    default_root_dir='./checkpoints',
    enable_checkpointing=True
)

# Custom checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='best-model',
    monitor='val_loss',
    save_top_k=1,
    save_last=True
)
```

### Resume Training

```python
# Resume from checkpoint
trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule, ckpt_path='path/to/checkpoint.ckpt')

# Load for inference
model = LitModel.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()
```

### Save/Load Custom State

```python
class LitModel(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint):
        checkpoint['custom_state'] = self.custom_state

    def on_load_checkpoint(self, checkpoint):
        self.custom_state = checkpoint['custom_state']
```

## Advanced Patterns

### Multiple Optimizers (GANs)

```python
class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # Manual optimization

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Train discriminator
        opt_d.zero_grad()
        d_loss = self.discriminator_loss(batch)
        self.manual_backward(d_loss)
        opt_d.step()

        # Train generator
        opt_g.zero_grad()
        g_loss = self.generator_loss(batch)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({'g_loss': g_loss, 'd_loss': d_loss})

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        return [opt_g, opt_d], []
```

### Gradient Accumulation

```python
# Fixed accumulation
trainer = Trainer(accumulate_grad_batches=8)

# Dynamic accumulation
from pytorch_lightning.callbacks import GradientAccumulationScheduler

scheduler = GradientAccumulationScheduler(
    scheduling={0: 8, 4: 4, 8: 1}  # epoch: accumulation_steps
)
trainer = Trainer(callbacks=[scheduler])
```

### Learning Rate Finder

```python
from pytorch_lightning.tuner import Tuner

trainer = Trainer()
tuner = Tuner(trainer)

# Find optimal learning rate
lr_finder = tuner.lr_find(model, datamodule)
suggested_lr = lr_finder.suggestion()

# Plot
lr_finder.plot(suggest=True)
plt.savefig('lr_finder.png')

# Update model
model.hparams.learning_rate = suggested_lr
```

### Batch Size Finder

```python
tuner = Tuner(trainer)
tuner.scale_batch_size(model, datamodule, mode='power')
```

## Testing and Inference

### Testing

```python
# Test with best checkpoint
trainer.test(model, datamodule, ckpt_path='best')

# Test with specific checkpoint
trainer.test(model, datamodule, ckpt_path='path/to/checkpoint.ckpt')
```

### Prediction

```python
# Batch prediction
predictions = trainer.predict(model, datamodule)

# With custom predict_step
class LitModel(pl.LightningModule):
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
```

### Export for Production

```python
# Export to TorchScript
script = model.to_torchscript()
torch.jit.save(script, 'model.pt')

# Export to ONNX
model.to_onnx('model.onnx', input_sample=torch.randn(1, 784))
```

## CLI Integration

### LightningCLI

```python
from pytorch_lightning.cli import LightningCLI

def cli_main():
    cli = LightningCLI(LitModel, MNISTDataModule)

if __name__ == '__main__':
    cli_main()
```

```bash
# Run with CLI
python train.py fit --model.learning_rate=1e-4 --data.batch_size=64
python train.py fit --config config.yaml
python train.py test --ckpt_path=best
```

## Best Practices

1. **Use LightningDataModule**: Encapsulate data logic
2. **Save hyperparameters**: `self.save_hyperparameters()`
3. **Use callbacks**: Keep LightningModule focused
4. **Log with sync_dist**: For distributed metrics
5. **Set deterministic mode**: For reproducibility
6. **Profile training**: Use profiler='simple' or 'advanced'
7. **Check gradients**: Use gradient_clip_val
8. **Monitor GPU memory**: Use precision='16-mixed'

## Debugging

```python
# Fast dev run (1 batch)
trainer = Trainer(fast_dev_run=True)

# Limit batches
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)

# Overfit on 1 batch
trainer = Trainer(overfit_batches=1)

# Detect anomalies
trainer = Trainer(detect_anomaly=True)

# Profile
trainer = Trainer(profiler='simple')  # or 'advanced'
```
