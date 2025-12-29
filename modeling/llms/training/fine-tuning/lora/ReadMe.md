# LoRA (Low-Rank Adaptation)

## Summary

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into transformer layers. Instead of updating the full weight matrix W, LoRA learns two smaller matrices A and B such that the update is W + BA, where BA has low rank (typically 4-64). This reduces trainable parameters by 10,000x while achieving performance comparable to full fine-tuning. LoRA is the most widely adopted PEFT method due to its simplicity, effectiveness, and the ability to merge adapters back into the base model for zero-inference overhead.

Key points to remember:

- Low-rank decomposition: W_new = W + BA where rank(BA) << rank(W)
- 10,000x fewer parameters: Only train A and B matrices
- Comparable performance: Matches or approaches full fine-tuning
- Mergeable: Combine with base weights for no inference overhead
- Modular: Swap adapters for different tasks
- Memory efficient: Base model frozen, only small matrices need gradients
- Widely adopted: Standard for efficient LLM fine-tuning

## The Core Idea

### Low-Rank Update

```
Standard fine-tuning:
W' = W + ”W
- ”W has same dimensions as W
- Full gradient required for all parameters

LoRA insight:
W' = W + BA
- B  R^(d × r), A  R^(r × k)
- r << min(d, k) (typically r = 4-64)
- W stays frozen, only train A and B

Example (hidden_size=4096, r=16):
- W: 4096 × 4096 = 16.7M parameters
- B × A: (4096 × 16) + (16 × 4096) = 131K parameters
- Reduction: 128x fewer parameters
```

### Mathematical Formulation

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original frozen weights (not a parameter)
        self.weight = None  # Set from pre-trained

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Gaussian, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original forward
        result = F.linear(x, self.weight)

        # Add LoRA
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        result = result + self.scaling * lora_out

        return result

    def merge(self):
        """Merge LoRA into base weights for inference."""
        self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        # Can now discard lora_A and lora_B
```

## Implementation

### With PEFT Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

### Training

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Higher than full fine-tuning
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Save adapter weights
model.save_pretrained("./lora_adapter")
```

### Loading and Merging

```python
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Option 1: Keep separate for flexibility
output = model.generate(inputs)

# Option 2: Merge for faster inference
model = model.merge_and_unload()
# Now model is standard LLaMA with LoRA absorbed into weights
```

## Hyperparameters

### Rank (r)

```
Rank determines capacity:
- r = 1-4: Minimal adaptation, very efficient
- r = 8-16: Common default, good balance
- r = 32-64: Higher capacity, approaches full FT
- r = 128+: Diminishing returns

Guidelines:
- Simple tasks: r = 4-8
- Complex tasks: r = 16-32
- When unsure: Start with r = 16
```

### Alpha (lora_alpha)

```
Alpha controls scaling: scaling_factor = alpha / r

Common patterns:
- alpha = r: scaling = 1 (standard)
- alpha = 2r: scaling = 2 (more aggressive)
- alpha = r/2: scaling = 0.5 (more conservative)

Typical: alpha = 16-32 regardless of r
Higher alpha = larger effective learning rate for LoRA
```

### Target Modules

```python
# Common configurations for different architectures

# LLaMA/Mistral
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]  # + MLP

# GPT-2/GPT-Neo
target_modules = ["c_attn", "c_proj"]  # Attention
target_modules = ["c_attn", "c_proj", "c_fc"]  # + MLP

# General rule: More modules = more capacity = more parameters
```

### Learning Rate

```
LoRA uses higher learning rate than full fine-tuning:
- Full FT: 1e-6 to 2e-5
- LoRA: 1e-4 to 3e-4

Why higher?
- Fewer parameters to update
- Initialized near zero (small initial updates)
- Scaling factor moderates the effect
```

## Advanced Techniques

### LoRA+

```python
# LoRA+: Different learning rates for A and B matrices
# B matrix gets higher learning rate (16x default)

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "lora_B" in n],
        "lr": 2e-4 * 16  # Higher LR for B
    },
    {
        "params": [p for n, p in model.named_parameters() if "lora_A" in n],
        "lr": 2e-4  # Standard LR for A
    },
]
```

### DoRA (Weight-Decomposed LoRA)

```python
# DoRA: Decompose weight into magnitude and direction
# More expressive with similar parameter count

class DoRALayer(nn.Module):
    def __init__(self, base_layer, rank):
        super().__init__()
        self.base_layer = base_layer

        # Magnitude (scalar per output)
        self.magnitude = nn.Parameter(
            torch.norm(base_layer.weight, dim=1)
        )

        # Direction via LoRA
        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))

    def forward(self, x):
        # Combine base + LoRA direction
        combined = self.base_layer.weight + self.lora_B @ self.lora_A

        # Normalize direction, apply magnitude
        direction = combined / combined.norm(dim=1, keepdim=True)
        weight = self.magnitude.unsqueeze(1) * direction

        return F.linear(x, weight)
```

### LoRA Ensembling

```python
# Combine multiple LoRA adapters for multi-task

class MultiLoRAModel:
    def __init__(self, base_model, adapter_paths):
        self.base = base_model
        self.adapters = {
            name: load_adapter(path)
            for name, path in adapter_paths.items()
        }

    def forward(self, x, adapter_name):
        """Use specific adapter for forward pass."""
        # Temporarily apply adapter
        adapter = self.adapters[adapter_name]
        output = self.forward_with_adapter(x, adapter)
        return output

    def merge_adapters(self, weights):
        """Combine multiple adapters with weights."""
        merged = sum(
            w * adapter.get_weights()
            for w, adapter in zip(weights, self.adapters.values())
        )
        return merged
```

## Comparison with Other PEFT

| Method | Parameters | Performance | Inference | Complexity |
|--------|------------|-------------|-----------|------------|
| Full FT | 100% | Best | Same | Standard |
| LoRA | 0.1% | Very good | Mergeable | Simple |
| QLoRA | 0.1% | Very good | Slower | Moderate |
| Adapter | 0.5-1% | Good | Slower | Moderate |
| Prefix | 0.1% | Moderate | Slower | Simple |
| Prompt | 0.001% | Lower | Same | Simplest |

## Practical Tips

### When LoRA Works Well

```
Best scenarios:
- Task is related to pre-training domain
- Limited compute/memory
- Need to maintain multiple task adapters
- Quick iteration on tasks

Less ideal:
- Task very different from pre-training
- Extremely limited data (<100 examples)
- Maximum possible performance required
```

### Common Issues

```
Problem: Poor performance
Solutions:
- Increase rank (r)
- Add more target modules
- Increase alpha
- Train longer

Problem: Overfitting
Solutions:
- Increase lora_dropout
- Reduce rank
- Use weight decay
- Early stopping

Problem: Training instability
Solutions:
- Lower learning rate
- Reduce alpha/rank
- Gradient clipping
```

### Memory Calculation

```python
def lora_memory_estimate(model_params, rank, num_modules):
    """Estimate LoRA memory requirements."""
    # Base model (frozen, no gradients)
    base_memory = model_params * 2  # FP16

    # LoRA parameters
    lora_params = num_modules * 2 * rank * hidden_size  # A and B
    lora_memory = lora_params * (2 + 4 + 8)  # FP16 + grad + optimizer

    # Activations (still needed)
    activation_memory = estimate_activations()

    return base_memory + lora_memory + activation_memory

# Example: 7B model, r=16, 8 modules
# Base: 14 GB (frozen)
# LoRA: ~200 MB (with gradients)
# Total: ~16-20 GB (with activations)
```

## Key Takeaways

1. **Low-rank hypothesis**: Fine-tuning updates are low-rank in practice.

2. **Massive efficiency**: 10,000x fewer trainable parameters.

3. **Mergeable**: Zero inference overhead after merging.

4. **Modular**: Swap adapters for different tasks.

5. **Higher learning rate**: 10x higher than full fine-tuning.

6. **Rank matters**: r=16 is a good starting point.

7. **Target modules**: More modules = more capacity.
