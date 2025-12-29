# Prefix Tuning

## Summary

Prefix tuning prepends trainable continuous vectors (the "prefix") to the key and value representations in each transformer layer, while keeping all model parameters frozen. Unlike prompting with discrete tokens, these prefix vectors are learned in the continuous embedding space, giving them more expressive power. The prefix acts as virtual tokens that steer the model's behavior for a specific task. Prefix tuning is more parameter-efficient than full fine-tuning (typically 0.1-1% of parameters) and can maintain multiple task-specific prefixes with a single frozen model.

Key points to remember:

- Continuous prefix: Learned vectors prepended to K and V in attention
- All layers: Prefix applied at every transformer layer
- Frozen model: Only prefix parameters are trained
- ~0.1-1% parameters: Very parameter efficient
- Task-specific: Different prefix per task, same base model
- Continuous space: More expressive than discrete prompt tokens
- Historical importance: Precursor to modern PEFT methods

## The Core Idea

### Prefix Vectors

```
Standard Transformer Attention:
Q, K, V from input tokens
Attention(Q, K, V) = softmax(QK^T/d)V

Prefix Tuning:
Prepend learned prefix vectors P_k, P_v to K and V
K' = [P_k; K]  # Concatenate prefix to keys
V' = [P_v; V]  # Concatenate prefix to values
Attention(Q, K', V') = softmax(QK'^T/d)V'

Effect:
- All tokens attend to prefix
- Prefix "conditions" all attention computations
- Like virtual tokens that guide model behavior
```

### Layer-wise Prefixes

```python
# Prefix applied at EVERY transformer layer
# Each layer has its own prefix parameters

class PrefixModel:
    def __init__(self, model, prefix_length=20):
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads

        # Prefix parameters for all layers
        # Shape: (n_layers, 2, prefix_length, n_heads, head_dim)
        # 2 for key and value
        self.prefix = nn.Parameter(
            torch.randn(self.n_layers, 2, prefix_length,
                       self.n_heads, self.head_dim)
        )
```

## Implementation

### Basic Prefix Tuning

```python
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=20):
        super().__init__()
        self.prefix_length = prefix_length
        self.n_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # Prefix parameters
        self.prefix_embedding = nn.Embedding(prefix_length, config.hidden_size)

        # MLP to reparameterize (helps training stability)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size * 2, self.n_layers * 2 * config.hidden_size)
        )

    def get_prefix(self, batch_size):
        """Generate prefix key-value pairs for all layers."""
        # Get base embeddings
        prefix_tokens = torch.arange(self.prefix_length)
        prefix_embeds = self.prefix_embedding(prefix_tokens)

        # Transform through MLP
        prefix_kv = self.prefix_mlp(prefix_embeds)

        # Reshape: (prefix_len, n_layers * 2 * hidden)
        # -> (n_layers, 2, prefix_len, hidden)
        prefix_kv = prefix_kv.view(
            self.prefix_length,
            self.n_layers,
            2,
            self.hidden_size
        )
        prefix_kv = prefix_kv.permute(1, 2, 0, 3)

        # Expand for batch
        prefix_kv = prefix_kv.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        return prefix_kv  # (batch, n_layers, 2, prefix_len, hidden)
```

### With PEFT Library

```python
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, get_peft_model, TaskType

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Prefix tuning config
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # Prefix length
    encoder_hidden_size=768,  # Hidden size for prefix MLP
    prefix_projection=True,  # Use MLP reparameterization
)

# Apply prefix tuning
model = get_peft_model(model, config)

# Check parameters
model.print_trainable_parameters()
# trainable params: 983,040 || all params: 355,359,744 || trainable%: 0.2766
```

### Modifying Attention

```python
def prefix_attention_forward(
    self,
    hidden_states,
    attention_mask,
    prefix_key,
    prefix_value,
):
    """Modified attention forward with prefix."""
    batch_size, seq_len, _ = hidden_states.shape

    # Standard Q, K, V projection
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, self.n_heads, self.head_dim)
    key = key.view(batch_size, seq_len, self.n_heads, self.head_dim)
    value = value.view(batch_size, seq_len, self.n_heads, self.head_dim)

    # Prepend prefix to K and V
    prefix_len = prefix_key.shape[1]
    key = torch.cat([prefix_key, key], dim=1)
    value = torch.cat([prefix_value, value], dim=1)

    # Update attention mask for prefix
    prefix_mask = torch.ones(batch_size, prefix_len, device=hidden_states.device)
    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    # Standard attention computation
    attn_output = self.attention(query, key, value, attention_mask)

    return attn_output
```

## Hyperparameters

### Prefix Length

```
Prefix length trade-offs:

Short prefix (10-20 tokens):
- Fewer parameters
- Less capacity
- Good for simple tasks

Medium prefix (20-50 tokens):
- Common default
- Good balance

Long prefix (50-200 tokens):
- More capacity
- May hurt efficiency
- For complex tasks

Guidelines:
- Start with 20
- Increase if underperforming
- Decrease if overfitting
```

### Reparameterization

```python
# Direct optimization (can be unstable)
prefix = nn.Parameter(torch.randn(prefix_length, hidden_size))

# MLP reparameterization (recommended)
# Easier optimization landscape
class PrefixMLP(nn.Module):
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(prefix_length, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self):
        embeds = self.embedding(torch.arange(self.prefix_length))
        return self.mlp(embeds)
```

## Comparison with Related Methods

### Prefix Tuning vs Prompt Tuning

```
Prefix Tuning:
- Prepends to K, V in ALL layers
- More parameters (per layer)
- More expressive
- Affects attention computation directly

Prompt Tuning:
- Prepends to embeddings (input layer only)
- Fewer parameters
- Simpler
- Propagates through normal forward pass

Comparison:
| Aspect | Prefix Tuning | Prompt Tuning |
|--------|---------------|---------------|
| Where | All layers K,V | Input only |
| Params | ~0.5% | ~0.01% |
| Performance | Better | Slightly worse |
| Complexity | Moderate | Simple |
```

### Prefix Tuning vs LoRA

```
Prefix Tuning:
- Adds virtual tokens
- Modifies attention context
- Affects sequence length
- Historical (2021)

LoRA:
- Modifies weight matrices
- No extra tokens
- No inference overhead after merge
- Currently preferred

In practice:
- LoRA typically outperforms prefix tuning
- LoRA is more widely adopted
- Prefix tuning still useful for multi-task scenarios
```

## Applications

### Multi-Task Learning

```python
class MultiTaskPrefixModel:
    def __init__(self, base_model, task_prefixes):
        self.base_model = base_model
        self.prefixes = nn.ModuleDict({
            task: PrefixTuning(base_model.config)
            for task in task_prefixes
        })

    def forward(self, x, task):
        """Use task-specific prefix."""
        prefix_kv = self.prefixes[task].get_prefix(x.shape[0])
        return self.base_model(x, prefix_kv=prefix_kv)

# Single model, multiple prefixes
tasks = ["summarization", "translation", "qa"]
model = MultiTaskPrefixModel(base_model, tasks)

# Different behavior per task
summary = model(text, task="summarization")
translation = model(text, task="translation")
```

### Controllable Generation

```python
# Different prefixes for different styles/attributes
class StylePrefixModel:
    def __init__(self, base_model, styles):
        self.prefixes = {
            "formal": PrefixTuning(base_model.config),
            "casual": PrefixTuning(base_model.config),
            "technical": PrefixTuning(base_model.config),
        }

    def generate(self, prompt, style="formal"):
        prefix = self.prefixes[style].get_prefix(1)
        return self.base_model.generate(prompt, prefix_kv=prefix)
```

## Training Tips

### Optimization

```python
# Prefix tuning typically uses higher learning rate
training_args = TrainingArguments(
    learning_rate=5e-4,  # Higher than full FT
    weight_decay=0.01,
    num_train_epochs=5,
    warmup_ratio=0.1,
)

# Adam optimizer works well
optimizer = torch.optim.Adam(
    prefix_params,
    lr=5e-4,
    betas=(0.9, 0.999)
)
```

### Common Issues

```
Problem: Training instability
Solutions:
- Use reparameterization MLP
- Lower learning rate
- Gradient clipping

Problem: Poor performance
Solutions:
- Increase prefix length
- Ensure prefix applied to all layers
- Check attention mask includes prefix

Problem: Slow convergence
Solutions:
- Initialize prefix from actual tokens
- Use warmup schedule
- Increase prefix length
```

## Key Takeaways

1. **Continuous prefixes**: Learned vectors, not discrete tokens.

2. **All layers**: Prefix in every transformer layer's attention.

3. **Very efficient**: ~0.1-1% trainable parameters.

4. **Multi-task friendly**: Different prefix per task.

5. **Reparameterization helps**: MLP makes optimization easier.

6. **Historical importance**: Pioneered the PEFT paradigm.

7. **LoRA often preferred**: But prefix tuning still has niche uses.
