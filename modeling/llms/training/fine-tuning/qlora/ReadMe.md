# QLoRA (Quantized Low-Rank Adaptation)

## Summary

QLoRA combines 4-bit quantization with LoRA to enable fine-tuning of large language models on consumer GPUs. The base model is loaded in 4-bit precision using NF4 (Normalized Float 4) quantization, dramatically reducing memory requirements, while LoRA adapters are trained in higher precision. This allows fine-tuning a 65B parameter model on a single 48GB GPU, or a 7B model on a 16GB GPU. QLoRA introduces NF4 (optimal for normally distributed weights) and double quantization (quantizing the quantization constants) to minimize quality loss while maximizing memory efficiency.

Key points to remember:

- 4-bit base model: NF4 quantization for base weights
- High-precision LoRA: Adapters trained in FP16/BF16
- Memory efficient: 70B model on single GPU
- Double quantization: Quantize the quantization scales
- Paged optimizers: Handle memory spikes
- Near full FT quality: Comparable to full precision fine-tuning
- Democratizing: Makes large model training accessible

## Core Components

### QLoRA Architecture

```
Standard Full Fine-Tuning:
Model (FP16) + Gradients (FP16) + Optimizer (FP32)
= 16 bytes per parameter × 7B = 112 GB

QLoRA:
Model (NF4)         = 0.5 bytes × 7B = 3.5 GB
LoRA (FP16)         = 2 bytes × 0.1% = 14 MB
LoRA Gradients      = 2 bytes × 0.1% = 14 MB
LoRA Optimizer      = 8 bytes × 0.1% = 56 MB
Activations         = ~2-4 GB
Total: ~8 GB (vs 112 GB)
```

### NF4 Quantization

```python
# NF4: Non-uniform 4-bit format optimized for normal distributions
# Neural network weights are approximately normal

NF4_LEVELS = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
]

# Computed from quantiles of N(0,1)
# More levels near zero where weights concentrate

def quantize_nf4(tensor, block_size=64):
    """Quantize to NF4 format."""
    # Reshape into blocks
    tensor = tensor.reshape(-1, block_size)

    # Per-block absmax scaling
    scales = tensor.abs().max(dim=1).values

    # Normalize to [-1, 1]
    normalized = tensor / scales.unsqueeze(1)

    # Find nearest NF4 level
    nf4_levels = torch.tensor(NF4_LEVELS)
    indices = torch.argmin(
        torch.abs(normalized.unsqueeze(-1) - nf4_levels),
        dim=-1
    )

    return indices, scales
```

### Double Quantization

```python
def double_quantize(weights, block_size=64, quant_block_size=256):
    """
    Quantize weights, then quantize the scales.

    Memory: 4 bits/weight + ~0.5 bits/weight for scales
    vs. 4 bits/weight + 2 bits/weight for FP16 scales
    """
    # First quantization: weights to NF4
    indices, scales = quantize_nf4(weights, block_size)

    # Scales are FP32, ~16 bits per 64 weights = 0.25 bits/weight overhead

    # Second quantization: scales to FP8
    scales_reshaped = scales.reshape(-1, quant_block_size // block_size)
    scale_scales = scales_reshaped.abs().max(dim=1).values
    scales_fp8 = quantize_fp8(scales_reshaped / scale_scales.unsqueeze(1))

    # Now: 4 bits/weight + 0.125 bits/weight for quantized scales
    return indices, scales_fp8, scale_scales
```

## Implementation

### With Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation in bf16
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Training Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qlora_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,  # or bf16=True
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    optim="paged_adamw_8bit",  # Paged optimizer
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
```

### Paged Optimizers

```python
# Paged optimizers move optimizer states to CPU when GPU memory is full
# Prevents OOM during memory spikes

from bitsandbytes.optim import PagedAdamW8bit

optimizer = PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)

# How it works:
# 1. During forward pass, optimizer states stay on CPU
# 2. During backward, load states to GPU as needed
# 3. Update parameters, move states back to CPU
# 4. Handles memory spikes from activations
```

## Memory Comparison

### Memory by Model Size

| Model | FP16 | QLoRA | Savings |
|-------|------|-------|---------|
| 7B | 14 GB | 6 GB | 2.3x |
| 13B | 26 GB | 10 GB | 2.6x |
| 33B | 66 GB | 18 GB | 3.7x |
| 65B | 130 GB | 33 GB | 3.9x |

### What Fits Where

```
RTX 3090 (24GB):
- QLoRA: Up to 33B models
- Full FT: Only 7B (barely)

RTX 4090 (24GB):
- QLoRA: Up to 33B models
- LoRA FP16: Up to 13B models

A100 40GB:
- QLoRA: 65B+ models
- LoRA FP16: Up to 33B models
- Full FT: Up to 7B models

A100 80GB:
- QLoRA: Any model
- Full FT: Up to 13B models
```

## Quality Analysis

### QLoRA vs Full Fine-Tuning

```
Benchmarks show QLoRA approaches full FT quality:

| Task | Full FT | QLoRA | Gap |
|------|---------|-------|-----|
| MMLU | 68.4% | 68.0% | -0.4% |
| HellaSwag | 85.3% | 84.9% | -0.4% |
| ARC | 89.2% | 88.7% | -0.5% |
| TruthfulQA | 53.1% | 52.4% | -0.7% |

Key findings:
- Gap narrows with larger models
- Quality preserved despite 4-bit base
- LoRA adapters sufficient for task adaptation
```

### Why Quality Is Preserved

```
Observations:
1. Pre-trained weights are already good
   - 4-bit captures the important information
   - Fine-tuning only needs small updates

2. LoRA adapters in full precision
   - Task-specific knowledge in FP16
   - Critical updates not quantized

3. NF4 is optimal for neural networks
   - Matches weight distribution
   - Minimal information loss
```

## Hyperparameters

### Recommended Settings

```python
qlora_config = {
    # LoRA settings
    "r": 64,  # Higher rank for QLoRA
    "lora_alpha": 16,
    "lora_dropout": 0.05,

    # Target all linear layers
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # Training
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "constant",

    # Optimizer
    "optim": "paged_adamw_8bit",
}
```

### Rank Selection for QLoRA

```
QLoRA typically uses higher rank than regular LoRA:
- Regular LoRA: r = 8-32
- QLoRA: r = 64-256

Reasoning:
- Base model is quantized (lower precision)
- LoRA must compensate
- Higher rank provides more capacity

Memory is still manageable:
- r=64 adds ~100MB for 7B model
- Tiny compared to base model savings
```

## Advanced Topics

### Gradient Checkpointing

```python
# Essential for very large models
model.gradient_checkpointing_enable()

# Trade compute for memory
# Recompute activations during backward instead of storing
# ~30% slower, but significant memory savings
```

### Merging QLoRA

```python
# After training, can merge LoRA into base model
# But: Base stays quantized, so merged model is also quantized

from peft import PeftModel

# Load base in higher precision
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,  # Not quantized
)

# Load and merge adapter
model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
model = model.merge_and_unload()

# Now have FP16 model with learned updates
# No quantization artifacts at inference
```

### Multi-Adapter Training

```python
# Train multiple adapters, switch between tasks
from peft import PeftModel

# Train adapter 1
model = get_peft_model(base_model, lora_config)
train(model, task1_data)
model.save_pretrained("adapter_task1")

# Train adapter 2 (fresh)
model = get_peft_model(base_model, lora_config)
train(model, task2_data)
model.save_pretrained("adapter_task2")

# At inference, load whichever needed
model = PeftModel.from_pretrained(base_model, "adapter_task1")
# or
model = PeftModel.from_pretrained(base_model, "adapter_task2")
```

## Common Issues

### Troubleshooting

```
Issue: OOM despite QLoRA
Solutions:
- Enable gradient checkpointing
- Reduce batch size
- Use gradient accumulation
- Reduce sequence length
- Check for memory leaks

Issue: Poor quality results
Solutions:
- Increase LoRA rank (r)
- Train longer
- Check data quality
- Verify proper tokenization

Issue: Slow training
Solutions:
- Use bf16 instead of fp16
- Enable flash attention
- Batch size optimization
- Check GPU utilization
```

### Best Practices

```python
# Always use prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# Set proper compute dtype
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Better than fp16
)

# Enable gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Use paged optimizer to handle memory spikes
optim="paged_adamw_8bit"
```

## Key Takeaways

1. **4-bit base + FP16 LoRA**: Best of both worlds.

2. **NF4 optimal**: Matches neural network weight distribution.

3. **Double quantization**: Further memory reduction.

4. **Consumer GPU training**: 70B models on 48GB GPU.

5. **Near full quality**: <1% gap from full fine-tuning.

6. **Higher LoRA rank**: Compensate for quantized base.

7. **Paged optimizers**: Handle memory spikes.
