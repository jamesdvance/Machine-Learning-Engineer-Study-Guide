# Prompt Tuning

## Summary

Prompt tuning is the simplest parameter-efficient fine-tuning method, prepending learned continuous embeddings (soft prompts) to the input while keeping the entire model frozen. Unlike discrete prompts written in natural language, these learned vectors exist in continuous embedding space and are optimized through backpropagation. Prompt tuning requires only ~0.01% of full fine-tuning parameters (just the prompt embeddings) and becomes competitive with full fine-tuning as model size increases. While simpler than methods like LoRA, it's less commonly used today due to lower performance on smaller models.

Key points to remember:

- Soft prompts: Learned continuous embeddings prepended to input
- Frozen model: Only prompt embeddings are trainable
- Minimal parameters: ~0.01% of model (just a few vectors)
- Input layer only: Unlike prefix tuning, only at embedding level
- Scales with model size: Approaches full FT performance on very large models
- Task conditioning: Prompt steers model toward task behavior
- Historical: Early PEFT method, less common now

## The Core Idea

### Soft vs Hard Prompts

```
Hard prompt (discrete):
"Summarize the following text: [input]"
- Fixed tokens, no learning
- Requires prompt engineering

Soft prompt (continuous):
[P1][P2][P3][P4][P5] [input]
- Learned embedding vectors
- Optimized for task
- Not interpretable as words

Advantages of soft prompts:
- More expressive (continuous space)
- Optimized end-to-end
- No manual prompt engineering
- Compact task representation
```

### How It Works

```
Standard input:
Input: "The movie was great"
Embeddings: [E_the, E_movie, E_was, E_great]

With prompt tuning:
Input: "The movie was great"
Embeddings: [P1, P2, ..., Pn, E_the, E_movie, E_was, E_great]
            ^^^^^^^^^^^^^
            Learned soft prompt

Only P1...Pn are trainable
Everything else (model, embeddings) is frozen
```

## Implementation

### Basic Implementation

```python
import torch
import torch.nn as nn

class PromptTuning(nn.Module):
    def __init__(self, model, num_prompt_tokens=20):
        super().__init__()
        self.model = model
        self.num_prompt_tokens = num_prompt_tokens

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Learnable soft prompt
        embed_size = model.config.hidden_size
        self.soft_prompt = nn.Parameter(
            torch.randn(num_prompt_tokens, embed_size)
        )

        # Initialize from existing embeddings (optional)
        self.initialize_prompt()

    def initialize_prompt(self):
        """Initialize from random actual tokens."""
        # Sample random token IDs
        random_ids = torch.randint(
            0, self.model.config.vocab_size,
            (self.num_prompt_tokens,)
        )
        # Get their embeddings
        with torch.no_grad():
            init_embeds = self.model.get_input_embeddings()(random_ids)
        self.soft_prompt.data = init_embeds

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]

        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Prepend soft prompt
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.num_prompt_tokens,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Forward through model
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask
        )

        return outputs
```

### With PEFT Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

# Prompt tuning config
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init="RANDOM",  # or "TEXT"
    # prompt_tuning_init_text="Classify the sentiment:",  # if TEXT init
)

# Apply prompt tuning
model = get_peft_model(model, config)

# Check parameters
model.print_trainable_parameters()
# trainable params: 20,480 || all params: 559,234,048 || trainable%: 0.0037
```

### Text Initialization

```python
# Initialize soft prompt from meaningful text
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify whether this text is positive or negative:",
    tokenizer_name_or_path="bigscience/bloomz-560m",
)

# The text is tokenized and its embeddings become the initial soft prompt
# Then optimized through training
```

## Training

### Training Loop

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./prompt_tuning_model",
    num_train_epochs=10,  # Often needs more epochs
    per_device_train_batch_size=16,
    learning_rate=3e-2,  # High learning rate for few parameters
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save soft prompt
model.save_pretrained("./soft_prompt")
```

### Hyperparameters

```python
# Prompt tuning typically uses higher learning rates
# because there are very few parameters

hyperparameters = {
    "num_virtual_tokens": 20,  # 8-100 typically
    "learning_rate": 3e-2,  # Much higher than full FT
    "num_epochs": 10,  # More epochs often needed
    "batch_size": 16,
}

# Number of parameters:
# For hidden_size=4096, num_tokens=20:
# Parameters = 20 * 4096 = 81,920 (0.0001% of 7B model)
```

## Scaling Behavior

### Model Size Effects

```
Prompt tuning effectiveness increases with model size:

| Model Size | Full FT | Prompt Tuning | Gap |
|------------|---------|---------------|-----|
| 1B | 85.2% | 76.3% | -8.9% |
| 10B | 87.1% | 84.8% | -2.3% |
| 100B+ | 88.4% | 87.9% | -0.5% |

Why scaling helps:
- Larger models have more world knowledge
- Soft prompt can better access this knowledge
- Less need for parameter updates
```

### Prompt Length

```
More tokens = more capacity:

| Prompt Length | Parameters | Performance |
|---------------|------------|-------------|
| 5 tokens | ~20K | Lower |
| 20 tokens | ~80K | Moderate |
| 100 tokens | ~400K | Higher |

Trade-off:
- Longer prompts = more capacity
- But increases input length
- 20-50 tokens is common
```

## Comparison with Other Methods

### PEFT Method Comparison

| Method | Where | Trainable % | Performance |
|--------|-------|-------------|-------------|
| Full FT | All weights | 100% | Best |
| LoRA | Attention weights | 0.1% | Very good |
| Prefix | All layers K,V | 0.5% | Good |
| Prompt | Input embeddings | 0.01% | Moderate |

### When to Use Prompt Tuning

```
Prompt tuning works best:
- Very large models (100B+)
- Extremely limited compute
- Need very small task-specific storage
- Simple classification tasks

Less suitable:
- Small/medium models
- Complex generation tasks
- Maximum performance needed
- Tasks requiring significant adaptation
```

## Variants

### P-Tuning v2

```python
# P-Tuning v2: Apply prompts at every layer (like prefix tuning)
# Better performance than input-only prompt tuning

class PTuningV2(nn.Module):
    def __init__(self, model, num_prompt_tokens=20):
        super().__init__()
        self.model = model
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

        # Prompt at every layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompt_tokens, hidden_size))
            for _ in range(n_layers)
        ])
```

### Multitask Prompt Tuning

```python
# Different soft prompt per task
class MultitaskPromptTuning(nn.Module):
    def __init__(self, model, tasks, num_prompt_tokens=20):
        super().__init__()
        self.model = model
        embed_size = model.config.hidden_size

        self.task_prompts = nn.ParameterDict({
            task: nn.Parameter(torch.randn(num_prompt_tokens, embed_size))
            for task in tasks
        })

    def forward(self, input_ids, task):
        prompt = self.task_prompts[task]
        # ... rest of forward
```

## Practical Considerations

### Advantages

```
+ Minimal storage: Just prompt embeddings
+ No inference overhead: Same as original model
+ Easy to switch tasks: Just swap prompts
+ Simple to implement
+ Fast training (few parameters)
```

### Disadvantages

```
- Lower performance than LoRA/full FT
- Struggles with smaller models
- Less expressive than weight updates
- Sensitive to initialization
- May need many epochs
```

### When to Consider

```
Use prompt tuning when:
1. Model is very large (>100B parameters)
2. Storage is extremely constrained
3. Task is simple classification
4. Need quick prototyping

Otherwise, prefer:
- LoRA for most use cases
- Full fine-tuning for maximum performance
```

## Key Takeaways

1. **Simplest PEFT**: Just learn input embeddings.

2. **Minimal parameters**: ~0.01% of model size.

3. **Scales with model**: Better on larger models.

4. **Input layer only**: Unlike prefix tuning's all-layer approach.

5. **High learning rate**: Few parameters need larger updates.

6. **Text initialization helps**: Start from meaningful tokens.

7. **LoRA often better**: For most practical use cases.
