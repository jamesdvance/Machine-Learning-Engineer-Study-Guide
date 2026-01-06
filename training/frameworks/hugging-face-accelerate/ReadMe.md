# Hugging Face Accelerate

## Summary

Hugging Face Accelerate is a minimal library that enables the same PyTorch code to run on any distributed configuration with just a few lines of changes. It handles device placement, distributed training, mixed precision, and gradient accumulation while keeping your training loop explicit. Accelerate is ideal for teams wanting distributed training without a heavy framework.

Key points to remember:

- Minimal code changes to enable distributed training
- Works with existing PyTorch training loops
- Handles DDP, FSDP, DeepSpeed, TPU automatically
- Built-in mixed precision support (FP16, BF16, FP8)
- Gradient accumulation handled correctly across devices
- Integrates with Hugging Face Transformers and Datasets
- Configuration via YAML or CLI
- Launch scripts for all distributed backends

## Basic Usage

### Minimal Integration

```python
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Prepare model, optimizer, dataloader
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# Training loop (almost unchanged)
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(batch['input_ids'], labels=batch['labels'])
    loss = outputs.loss

    # Use accelerator for backward
    accelerator.backward(loss)
    optimizer.step()
```

### Before and After

```python
# Before: Single GPU
model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    batch = {k: v.cuda() for k, v in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# After: Multi-GPU ready
from accelerate import Accelerator
accelerator = Accelerator()

model = Model()
optimizer = torch.optim.Adam(model.parameters())
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    loss = model(**batch).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## Configuration

### CLI Configuration

```bash
# Interactive configuration
accelerate config

# Answer prompts for:
# - Number of GPUs
# - Distributed type (DDP, FSDP, DeepSpeed)
# - Mixed precision
# - etc.

# Creates ~/.cache/huggingface/accelerate/default_config.yaml
```

### Configuration File

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Programmatic Configuration

```python
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,
    broadcast_buffers=False
)

accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=4,
    kwargs_handlers=[ddp_kwargs]
)
```

## Launching Training

### Launch Commands

```bash
# Single machine, multiple GPUs
accelerate launch --num_processes 4 train.py

# With config file
accelerate launch --config_file accelerate_config.yaml train.py

# Multi-node
accelerate launch \
    --num_machines 2 \
    --num_processes 8 \
    --machine_rank 0 \
    --main_process_ip 10.0.0.1 \
    --main_process_port 29500 \
    train.py

# DeepSpeed
accelerate launch --use_deepspeed --num_processes 8 train.py
```

### Environment-Based Launch

```python
# Script checks environment automatically
from accelerate import Accelerator

accelerator = Accelerator()
# Uses ACCELERATE_* environment variables
```

## Mixed Precision

### Enabling Mixed Precision

```python
accelerator = Accelerator(mixed_precision='bf16')  # or 'fp16', 'fp8'

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    # Autocast handled automatically
    loss = model(**batch).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### Manual Autocast

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='bf16')

for batch in dataloader:
    with accelerator.autocast():
        loss = model(**batch).loss
    accelerator.backward(loss)
```

### FP8 Training (H100+)

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='fp8')
```

## Gradient Accumulation

### Automatic Handling

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    # Accumulates gradients correctly across devices
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

### Manual Control

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    accelerator.backward(loss)

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## FSDP Integration

### Basic FSDP

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=False
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

### FSDP with Transformer Wrapping

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Auto-wrap transformer layers
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

## DeepSpeed Integration

### Configuration

```python
from accelerate import Accelerator, DeepSpeedPlugin

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=4,
    offload_optimizer_device='cpu'
)

accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
```

### DeepSpeed Config File

```json
{
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

```python
deepspeed_plugin = DeepSpeedPlugin(
    hf_ds_config='deepspeed_config.json'
)
```

## Logging and Tracking

### Distributed-Safe Logging

```python
# Only log on main process
if accelerator.is_main_process:
    print(f"Loss: {loss.item()}")
    wandb.log({'loss': loss.item()})

# Or use accelerator.print
accelerator.print(f"Loss: {loss.item()}")  # Only prints on main
```

### Tracking Integration

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with=['wandb', 'tensorboard'])
accelerator.init_trackers(
    project_name='my_project',
    config={'learning_rate': lr}
)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    accelerator.log({'loss': loss, 'step': step})

accelerator.end_training()
```

## Checkpointing

### Saving State

```python
# Save everything needed to resume
accelerator.save_state(output_dir='checkpoint')

# Saves:
# - Model state dict
# - Optimizer state dict
# - Scheduler state dict
# - Random states
# - Custom registered objects
```

### Loading State

```python
# Resume training
accelerator.load_state('checkpoint')

# Training continues from where it left off
```

### Saving Models

```python
# Wait for all processes
accelerator.wait_for_everyone()

# Get unwrapped model
unwrapped_model = accelerator.unwrap_model(model)

# Save on main process only
if accelerator.is_main_process:
    torch.save(unwrapped_model.state_dict(), 'model.pt')

# Or use accelerator.save
accelerator.save(unwrapped_model.state_dict(), 'model.pt')
```

### HuggingFace Model Saving

```python
# For Transformers models
unwrapped_model = accelerator.unwrap_model(model)

if accelerator.is_main_process:
    unwrapped_model.save_pretrained('output_dir')
    tokenizer.save_pretrained('output_dir')
```

## Inference

### Distributed Inference

```python
from accelerate import Accelerator

accelerator = Accelerator()

model = Model.from_pretrained('model_path')
model = accelerator.prepare(model)

# Prepare dataloader
eval_dataloader = accelerator.prepare(eval_dataloader)

model.eval()
all_predictions = []

for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)

    # Gather from all processes
    all_predictions.extend(accelerator.gather(predictions).cpu().tolist())
```

### gather_for_metrics

```python
# Handles padding for uneven batches
predictions = accelerator.gather_for_metrics(predictions)
references = accelerator.gather_for_metrics(labels)

# Compute metrics
accuracy = (predictions == references).float().mean()
```

## Integration with Transformers

### Trainer Replacement

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='bf16')

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

### With PEFT (LoRA)

```python
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator

# Create PEFT model
peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(model, peft_config)

# Then use Accelerate as normal
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

## Advanced Features

### Gradient Clipping

```python
accelerator = Accelerator()

for batch in dataloader:
    loss = model(**batch).loss
    accelerator.backward(loss)

    # Clip gradients
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Or
    accelerator.clip_grad_value_(model.parameters(), clip_value=1.0)

    optimizer.step()
    optimizer.zero_grad()
```

### Custom Objects in Checkpoints

```python
# Register custom objects for checkpointing
accelerator.register_for_checkpointing(scheduler)
accelerator.register_for_checkpointing(custom_object)

# They'll be saved/loaded with save_state/load_state
```

### Synchronization

```python
# Wait for all processes
accelerator.wait_for_everyone()

# Free memory
accelerator.free_memory()

# Clear cache
accelerator.clear()
```

## TPU Support

### TPU Configuration

```python
from accelerate import Accelerator

# Automatically detects TPU
accelerator = Accelerator()

# Or explicitly
accelerator = Accelerator(cpu=False)
```

### TPU Launch

```bash
accelerate launch --tpu --tpu_zone us-central1-a train.py
```

## Best Practices

1. **Prepare everything**: Always use `accelerator.prepare()`
2. **Use accumulate context**: For gradient accumulation
3. **Check is_main_process**: For logging/saving
4. **Gather for metrics**: Use `gather_for_metrics` for eval
5. **Unwrap for saving**: Use `unwrap_model` before saving
6. **Wait before saving**: Use `wait_for_everyone()`
7. **Configure via CLI**: Use `accelerate config` for portability

## Debugging

```python
# Print configuration
accelerator.print(accelerator.state)

# Check distributed setup
print(f"Process {accelerator.process_index} of {accelerator.num_processes}")
print(f"Local process: {accelerator.local_process_index}")
print(f"Is main: {accelerator.is_main_process}")
print(f"Device: {accelerator.device}")
```

## Complete Training Script

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch

def main():
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=4,
        log_with='wandb'
    )

    # Initialize tracking
    accelerator.init_trackers('my_project')

    # Load model and data
    model = AutoModelForSequenceClassification.from_pretrained('bert-base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_loader = DataLoader(train_dataset, batch_size=16)

    # Prepare
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Training loop
    model.train()
    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                accelerator.log({'loss': loss.item(), 'step': step})

        # Save checkpoint
        accelerator.save_state(f'checkpoint-{epoch}')

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained('final_model')

    accelerator.end_training()

if __name__ == '__main__':
    main()
```
