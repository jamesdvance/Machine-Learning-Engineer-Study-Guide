# ColossalAI

## Summary

ColossalAI is an open-source deep learning framework from HPC-AI Tech that simplifies large-scale distributed training. Its key innovation is making advanced parallelism techniques accessible through easy-to-use APIs, reducing the barrier to training large models. ColossalAI provides automated parallelism selection, memory optimization, and efficient implementations of various parallelism strategies.

Key points to remember:

- Simplifies large model training with user-friendly APIs
- Supports data, tensor, pipeline, and sequence parallelism
- Automatic parallelism selection for optimal configuration
- Gemini memory management for heterogeneous memory (GPU + CPU)
- Built-in support for popular models (LLaMA, BLOOM, etc.)
- Lower barrier to entry than Megatron-LM
- Good for teams without distributed systems expertise
- Active development with regular updates

## Installation

```bash
pip install colossalai

# With specific CUDA version
pip install colossalai --extra-index-url https://release.colossalai.org/cuda-11.8
```

## Basic Usage

### Simple Training Script

```python
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

# Initialize ColossalAI
colossalai.launch_from_torch()

# Create model, optimizer, dataloader
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

# Create booster with plugin
plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)

# Boost model, optimizer, dataloader
model, optimizer, _, dataloader, _ = booster.boost(
    model=model,
    optimizer=optimizer,
    dataloader=dataloader
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        booster.backward(loss, optimizer)
        optimizer.step()
```

### Launching

```bash
# Single node
colossalai run --nproc_per_node=4 train.py

# Multiple nodes
colossalai run --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr=master --master_port=29500 \
    train.py
```

## Plugins

### TorchDDP Plugin

Standard data parallelism:

```python
from colossalai.booster.plugin import TorchDDPPlugin

plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
```

### Low Level Zero Plugin

ZeRO optimization:

```python
from colossalai.booster.plugin import LowLevelZeroPlugin

# ZeRO Stage 1
plugin = LowLevelZeroPlugin(stage=1)

# ZeRO Stage 2
plugin = LowLevelZeroPlugin(stage=2)

# ZeRO Stage 2 with CPU offload
plugin = LowLevelZeroPlugin(
    stage=2,
    cpu_offload=True
)

booster = Booster(plugin=plugin)
```

### Gemini Plugin

Heterogeneous memory management:

```python
from colossalai.booster.plugin import GeminiPlugin

plugin = GeminiPlugin(
    placement_policy='auto',  # 'auto', 'cpu', 'cuda'
    precision='bf16',
    initial_scale=2**16
)

booster = Booster(plugin=plugin)
```

Gemini automatically manages memory between GPU and CPU.

### Hybrid Parallel Plugin

Combined parallelism:

```python
from colossalai.booster.plugin import HybridParallelPlugin

plugin = HybridParallelPlugin(
    tp_size=4,           # Tensor parallelism
    pp_size=2,           # Pipeline parallelism
    sp_size=1,           # Sequence parallelism
    zero_stage=1,        # ZeRO stage
    precision='bf16',
    enable_flash_attention=True
)

booster = Booster(plugin=plugin)
```

## Tensor Parallelism

### Automatic Model Conversion

```python
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer import ShardConfig

# ShardFormer automatically parallelizes model
shard_config = ShardConfig(
    tensor_parallel_mode='1d',
    enable_fused_normalization=True,
    enable_flash_attention=True
)

plugin = HybridParallelPlugin(
    tp_size=4,
    shard_config=shard_config
)
```

### Supported Models

ColossalAI automatically shards:
- LLaMA / LLaMA-2
- BLOOM
- GPT-2 / GPT-J / GPT-NeoX
- OPT
- BERT
- T5
- ViT
- And more

## Pipeline Parallelism

### Configuration

```python
plugin = HybridParallelPlugin(
    pp_size=4,
    num_microbatches=8,  # Number of micro-batches
    precision='bf16'
)

booster = Booster(plugin=plugin)
```

### Custom Pipeline Stages

```python
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule

schedule = OneForwardOneBackwardSchedule(
    num_microbatches=8,
    microbatch_size=4
)
```

## Sequence Parallelism

For long sequences:

```python
plugin = HybridParallelPlugin(
    tp_size=4,
    sp_size=2,  # Sequence parallelism within tensor parallel group
    enable_sequence_parallelism=True
)
```

## Memory Optimization

### Gemini Memory Manager

```python
from colossalai.booster.plugin import GeminiPlugin

plugin = GeminiPlugin(
    placement_policy='auto',      # Dynamic placement
    shard_param_frac=1.0,         # Shard all parameters
    offload_optim_frac=1.0,       # Offload optimizer to CPU
    offload_param_frac=0.0,       # Keep params on GPU
    precision='bf16',
    master_weights=True,
    initial_scale=2**16
)
```

### Activation Checkpointing

```python
from colossalai.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)

    def _forward(self, x):
        # Expensive computation
        return self.layers(x)
```

## Training LLaMA

### Complete Example

```python
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from transformers import LlamaForCausalLM, LlamaConfig

# Initialize
colossalai.launch_from_torch()

# Create model
config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32
)
model = LlamaForCausalLM(config)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Create plugin
plugin = HybridParallelPlugin(
    tp_size=4,
    pp_size=2,
    zero_stage=1,
    precision='bf16',
    enable_flash_attention=True
)

booster = Booster(plugin=plugin)

# Boost
model, optimizer, _, dataloader, _ = booster.boost(
    model=model,
    optimizer=optimizer,
    dataloader=dataloader
)

# Training
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    booster.backward(loss, optimizer)
    optimizer.step()
```

## Checkpointing

### Save Checkpoint

```python
booster.save_model(model, "checkpoint/model", shard=True)
booster.save_optimizer(optimizer, "checkpoint/optimizer", shard=True)
```

### Load Checkpoint

```python
booster.load_model(model, "checkpoint/model")
booster.load_optimizer(optimizer, "checkpoint/optimizer")
```

### Hugging Face Format

```python
from colossalai.utils import save_pretrained_from_model

# Save in HF format
save_pretrained_from_model(model, "output_model")
```

## Mixed Precision

### BF16 Training

```python
plugin = HybridParallelPlugin(
    precision='bf16',
    initial_scale=2**16,
    min_scale=1,
    growth_factor=2,
    backoff_factor=0.5,
    growth_interval=1000
)
```

### FP16 Training

```python
plugin = HybridParallelPlugin(
    precision='fp16',
    initial_scale=2**16
)
```

## Lazy Initialization

For models too large to initialize:

```python
from colossalai.lazy import LazyInitContext

with LazyInitContext():
    # Model not actually initialized yet
    model = LlamaForCausalLM(config)

# Initialization happens during boost
model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
```

## Command Line Interface

### Training

```bash
colossalai run --nproc_per_node=8 train.py \
    --tp 4 --pp 2 --zero 1 --precision bf16
```

### Inference

```bash
colossalai run --nproc_per_node=4 inference.py \
    --model_path /path/to/model --tp 4
```

## Performance Tuning

### Flash Attention

```python
plugin = HybridParallelPlugin(
    enable_flash_attention=True,
    enable_fused_normalization=True
)
```

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Communication Optimization

```python
plugin = HybridParallelPlugin(
    overlap_communication=True,
    communication_dtype=torch.float16
)
```

## Example Configurations

### 7B Model on 8 GPUs

```python
plugin = HybridParallelPlugin(
    tp_size=4,
    pp_size=1,
    zero_stage=2,
    precision='bf16'
)
```

### 70B Model on 64 GPUs

```python
plugin = HybridParallelPlugin(
    tp_size=8,
    pp_size=4,
    zero_stage=1,
    precision='bf16',
    enable_flash_attention=True
)
```

### Memory-Constrained Setup

```python
plugin = GeminiPlugin(
    placement_policy='auto',
    precision='bf16',
    offload_optim_frac=1.0
)
```

## Comparison

| Aspect | ColossalAI | DeepSpeed | Megatron-LM |
|--------|------------|-----------|-------------|
| Ease of use | High | Moderate | Low |
| Parallelism support | All | Data + ZeRO + Pipeline | All (3D) |
| Auto parallelism | Yes | No | No |
| Model support | Pre-built | Manual | Pre-built |
| Documentation | Good | Good | Limited |
| Community | Growing | Large | NVIDIA-focused |

### When to Use ColossalAI

- Want simplified large model training
- Need automatic parallelism configuration
- Training supported model architectures
- Limited distributed systems expertise
- Rapid prototyping of large models

### When to Consider Alternatives

- Need maximum performance (Megatron-LM)
- Existing DeepSpeed infrastructure
- Custom model architectures
- Production deployment focus
