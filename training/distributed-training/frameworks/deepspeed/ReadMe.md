# DeepSpeed

## Summary

DeepSpeed is Microsoft's deep learning optimization library that enables training of very large models through memory-efficient techniques and distributed training optimizations. Its core innovation is the ZeRO (Zero Redundancy Optimizer) family of optimizations that partition model states across data-parallel processes, dramatically reducing per-GPU memory requirements.

Key points to remember:

- ZeRO stages 1/2/3 progressively shard optimizer states, gradients, and parameters
- ZeRO-Offload extends memory to CPU and NVMe for extreme cases
- ZeRO-Infinity enables training trillion-parameter models
- Pipeline parallelism support for model parallelism
- Extensive configuration via JSON config file
- Good integration with Hugging Face Transformers
- Active development with regular new features
- DeepSpeed-Inference for optimized inference

## Installation

```bash
pip install deepspeed

# Or with specific features
pip install deepspeed[1bit-adam]
pip install deepspeed[sparse-attention]
```

Verify installation:
```bash
ds_report
```

## Basic Usage

### Training Script

```python
import deepspeed
import torch

model = MyModel()

# DeepSpeed handles optimizer creation from config
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### Configuration File

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,

    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    },

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

### Launching

```bash
deepspeed train.py --deepspeed_config ds_config.json

# Or with torchrun
torchrun --nproc_per_node=4 train.py --deepspeed_config ds_config.json
```

## ZeRO Optimization

### ZeRO Stage 1

Partition optimizer states:

```json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

Memory reduction: ~4x for Adam optimizer states.

### ZeRO Stage 2

Partition optimizer states and gradients:

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

Memory reduction: ~8x total.

### ZeRO Stage 3

Partition everything including parameters:

```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "sub_group_size": 1e12
    }
}
```

Memory reduction: Linear with GPU count.

### ZeRO-Offload (CPU)

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

Or with Stage 3:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### ZeRO-Infinity (NVMe)

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    }
}
```

## Mixed Precision

### FP16

```json
{
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

### BF16

```json
{
    "bf16": {
        "enabled": true
    }
}
```

BF16 is preferred on A100/H100 GPUs for stability.

## Activation Checkpointing

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "cpu_checkpointing": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

In code:
```python
from deepspeed.runtime.activation_checkpointing import checkpointing

checkpointing.configure(
    model,
    partition_activations=True,
    contiguous_checkpointing=True,
)
```

## Pipeline Parallelism

```python
from deepspeed.pipe import PipelineModule, LayerSpec

# Define layers as specs
layers = [
    LayerSpec(Embedding, vocab_size, hidden_size),
    *[LayerSpec(TransformerLayer, hidden_size) for _ in range(num_layers)],
    LayerSpec(OutputLayer, hidden_size, vocab_size)
]

# Create pipeline model
model = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=CrossEntropyLoss(),
    partition_method='uniform'
)

# Initialize with DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

Config:
```json
{
    "pipeline": {
        "pipe_partitioned": true,
        "grad_partitioned": true,
        "stages": "auto"
    }
}
```

## Checkpointing

### Save Checkpoint

```python
# Automatic saving
model_engine.save_checkpoint(
    save_dir="checkpoints",
    tag=f"step_{step}"
)

# Custom client state
client_state = {"epoch": epoch, "step": step}
model_engine.save_checkpoint("checkpoints", tag="latest", client_state=client_state)
```

### Load Checkpoint

```python
# Load latest
_, client_state = model_engine.load_checkpoint(
    load_dir="checkpoints",
    tag="latest"
)
epoch = client_state["epoch"]
step = client_state["step"]

# Load specific checkpoint
model_engine.load_checkpoint("checkpoints", tag="step_1000")
```

### Universal Checkpoint

For loading into different parallelism configurations:

```json
{
    "checkpoint": {
        "save_universal": true
    }
}
```

## Hugging Face Integration

### Trainer Integration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",
    per_device_train_batch_size=8,
    # Other arguments...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### Accelerate Integration

```python
from accelerate import Accelerator

accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    loss = model(batch)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## Optimized Optimizers

### 1-bit Adam

Communication-efficient Adam:

```json
{
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-4,
            "freeze_step": 400,
            "cuda_aware": false,
            "comm_backend_name": "nccl"
        }
    }
}
```

### LAMB

For large batch training:

```json
{
    "optimizer": {
        "type": "LAMB",
        "params": {
            "lr": 1e-3,
            "weight_decay": 0.01,
            "bias_correction": true
        }
    }
}
```

## Inference

### DeepSpeed-Inference

```python
import deepspeed

model = deepspeed.init_inference(
    model,
    mp_size=2,                    # Tensor parallelism
    dtype=torch.float16,
    replace_with_kernel_inject=True
)

output = model(input_ids)
```

### Inference Configuration

```python
model = deepspeed.init_inference(
    model,
    config={
        "tensor_parallel": {"tp_size": 4},
        "dtype": "fp16",
        "enable_cuda_graph": True,
        "max_out_tokens": 2048
    }
)
```

## Monitoring and Profiling

### TensorBoard Integration

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "./logs",
        "job_name": "train_run"
    }
}
```

### Wall Clock Breakdown

```json
{
    "wall_clock_breakdown": true
}
```

### Flops Profiler

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

prof = FlopsProfiler(model)
prof.start_profile()
output = model(input)
prof.stop_profile()
flops = prof.get_total_flops()
prof.print_model_profile()
```

### Communication Logging

```json
{
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false
    }
}
```

## Common Configurations

### 7B Model Training

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "cpu", "pin_memory": true},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5
    },

    "bf16": {"enabled": true},

    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true
    }
}
```

### 70B Model Training

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "nvme", "nvme_path": "/nvme"},
        "offload_param": {"device": "nvme", "nvme_path": "/nvme"},
        "overlap_comm": true,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },

    "bf16": {"enabled": true},

    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 4
    }
}
```

## Debugging

### Common Issues

**Out of memory**:
- Reduce micro_batch_size
- Enable offloading
- Enable activation checkpointing

**Slow training**:
- Disable offloading if possible
- Check overlap_comm is enabled
- Profile to find bottlenecks

**Convergence issues**:
- Check loss scaling settings
- Verify gradient accumulation
- Ensure learning rate is appropriate

### Debug Config

```json
{
    "dump_state": true,
    "verbose": true,
    "wall_clock_breakdown": true
}
```

### Environment Variables

```bash
export DEEPSPEED_VERBOSE=1
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```
