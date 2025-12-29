# Causal Language Modeling (CLM)

## Summary

Causal language modeling (also called autoregressive language modeling) trains models to predict the next token given all previous tokens. This is the pre-training objective for GPT, LLaMA, and most modern decoder-only LLMs. The model sees a sequence of tokens and learns to predict each position based only on prior context, enforced through causal masking in attention. This simple objective, trained on trillions of tokens, produces models capable of generation, understanding, and reasoning. CLM is the foundation of modern generative AI.

Key points to remember:

- Next token prediction: Given tokens 1...n, predict token n+1
- Causal attention: Each position only sees previous positions
- Decoder-only: No encoder, just autoregressive decoder
- Simple but powerful: Single objective scales to emergent abilities
- Massive data: Trained on trillions of tokens from web/books
- Foundation models: GPT, LLaMA, Claude, Mistral all use CLM
- Generation natural: Pre-training objective matches inference

## The Core Objective

### Next Token Prediction

```
Training objective:
Given sequence: "The cat sat on the"
Predict: "mat"

Mathematically:
L = -sum(log P(x_t | x_1, x_2, ..., x_{t-1}))

For each position t:
- See tokens 1 to t-1
- Predict probability of token t
- Minimize cross-entropy loss
```

### Causal Masking

```python
def causal_attention_mask(seq_len):
    """
    Create causal mask: position i can only attend to positions <= i

    Example for seq_len=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

    1 = can attend, 0 = cannot attend
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# In attention computation:
# attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
# After softmax, masked positions become 0
```

## Implementation

### Basic Training Loop

```python
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def causal_lm_training_step(model, batch, optimizer):
    """Single training step for causal LM."""
    input_ids = batch['input_ids']  # (batch, seq_len)

    # Shift for next token prediction
    # Input: tokens 0...n-1
    # Labels: tokens 1...n
    labels = input_ids[:, 1:].clone()
    inputs = input_ids[:, :-1]

    # Forward pass
    outputs = model(inputs)
    logits = outputs.logits  # (batch, seq_len-1, vocab_size)

    # Cross-entropy loss
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### With Hugging Face Trainer

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Data collator handles shifting labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir="./clm_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
```

## Data Preparation

### Tokenization

```python
def tokenize_for_clm(examples, tokenizer, max_length=2048):
    """Tokenize text for causal language modeling."""
    # Concatenate all texts
    concatenated = tokenizer(
        examples['text'],
        truncation=False,
        return_attention_mask=False,
    )

    # Chunk into fixed-length sequences
    all_tokens = []
    for tokens in concatenated['input_ids']:
        all_tokens.extend(tokens)

    # Create chunks
    chunks = []
    for i in range(0, len(all_tokens) - max_length, max_length):
        chunk = all_tokens[i:i + max_length]
        chunks.append(chunk)

    return {'input_ids': chunks}
```

### Data Sources for Pre-training

```
Common pre-training data:

Web text:
- Common Crawl (terabytes of web pages)
- C4 (cleaned Common Crawl)
- FineWeb (high-quality web data)

Books:
- Books3, BookCorpus
- Public domain books

Code:
- The Stack (open source code)
- GitHub public repositories

Scientific:
- arXiv papers
- PubMed abstracts

Conversation:
- Reddit, forums
- Dialogue datasets

Mixture:
- Most models use weighted mixture
- ~70% web, 15% books, 10% code, 5% other
```

## Training at Scale

### Pre-training Configuration

```python
# Typical pre-training hyperparameters for 7B model

pretraining_config = {
    # Model
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000,

    # Training
    "max_position_embeddings": 4096,
    "batch_size": 4_000_000,  # tokens per batch
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 2000,
    "total_steps": 1_000_000,
    "weight_decay": 0.1,

    # Optimizer
    "optimizer": "AdamW",
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "gradient_clipping": 1.0,
}
```

### Compute Requirements

```
Rough estimates for pre-training:

| Model | Parameters | Tokens | GPU Hours | Cost |
|-------|------------|--------|-----------|------|
| 1B | 1B | 20B | 2K | $5K |
| 7B | 7B | 1T | 100K | $500K |
| 13B | 13B | 2T | 400K | $2M |
| 70B | 70B | 2T | 2M | $10M |

Scaling law:
Compute (FLOPS) H 6 × Parameters × Tokens
```

### Distributed Training

```python
# DeepSpeed configuration for pre-training
deepspeed_config = {
    "train_batch_size": 4096,
    "gradient_accumulation_steps": 64,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"},
    },
    "gradient_clipping": 1.0,
}

# Or FSDP for PyTorch native
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.float16),
)
```

## Loss and Metrics

### Perplexity

```python
def compute_perplexity(model, dataset, tokenizer):
    """
    Perplexity = exp(average cross-entropy loss)
    Lower is better
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataset:
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss

            total_loss += loss.item() * batch['input_ids'].numel()
            total_tokens += batch['input_ids'].numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()

# Typical values:
# Random (untrained): ~vocab_size = 32000
# After pre-training: 5-20 on held-out data
```

### Training Curves

```
What to monitor during pre-training:

1. Loss curve
   - Should decrease smoothly
   - Spikes may indicate data issues

2. Learning rate schedule
   - Warmup then decay
   - Cosine or linear decay

3. Gradient norm
   - Should be stable
   - Spikes may need gradient clipping

4. Perplexity on validation
   - Check periodically
   - Monitor for overfitting
```

## Pre-training vs Fine-tuning

### Key Differences

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| Data | Trillions of tokens | Thousands to millions |
| Duration | Weeks to months | Hours to days |
| Learning rate | 1e-4 to 3e-4 | 1e-6 to 5e-5 |
| Objective | General language | Specific task |
| Compute | Massive | Modest |

### Continued Pre-training

```python
# Adapt pre-trained model to new domain
# Use CLM objective on domain data

def continued_pretraining(model, domain_data, epochs=1):
    """Continue pre-training on domain-specific data."""
    training_args = TrainingArguments(
        learning_rate=5e-5,  # Lower than initial pre-training
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=domain_data,
    )

    trainer.train()
```

## Emergent Abilities

### What CLM Learns

```
Pre-training produces:
1. Language fluency
   - Grammar, syntax, style

2. World knowledge
   - Facts, relationships, concepts

3. Reasoning patterns
   - Logic, math, causation

4. In-context learning
   - Follow patterns from examples

5. Instruction following
   - (Emerges at scale, enhanced with fine-tuning)

Scale matters:
- 1B: Basic fluency
- 7B: Good knowledge, some reasoning
- 70B+: Strong reasoning, emergent abilities
```

## Key Takeaways

1. **Simple objective**: Predict next token, but incredibly powerful at scale.

2. **Causal masking**: Each position only sees past, enabling generation.

3. **Massive data**: Trillions of tokens from diverse sources.

4. **Scaling matters**: More compute = better capabilities.

5. **Foundation**: CLM produces base models for fine-tuning.

6. **Generation natural**: Pre-training matches inference use.

7. **Emergent abilities**: Complex behaviors appear at scale.
