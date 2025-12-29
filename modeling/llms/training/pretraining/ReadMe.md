# LLM Pre-training

## Summary

Pre-training is the foundation of large language models, training on massive text corpora to learn language patterns, world knowledge, and reasoning abilities. The two main objectives are Causal Language Modeling (CLM, next-token prediction) for decoder models like GPT and LLaMA, and Masked Language Modeling (MLM, fill-in-the-blank) for encoder models like BERT. Pre-training requires enormous compute (thousands of GPU-hours to millions) and data (trillions of tokens), producing foundation models that can be fine-tuned for specific tasks. Most practitioners use pre-trained models rather than training from scratch.

Key points to remember:

- Foundation: Produces base models with general capabilities
- CLM vs MLM: Next-token (GPT) vs masked-token (BERT) prediction
- Massive scale: Trillions of tokens, thousands of GPUs
- Emergent abilities: Complex behaviors appear at scale
- Scaling laws: More compute = better performance
- Rarely done: Most use existing pre-trained models
- Continued pre-training: Adapt to new domains

## Training Objectives

### Comparison

| Aspect | CLM (GPT-style) | MLM (BERT-style) |
|--------|-----------------|------------------|
| Task | Predict next token | Predict masked tokens |
| Context | Left-to-right only | Bidirectional |
| Architecture | Decoder | Encoder |
| Best for | Generation | Understanding |
| Examples | GPT, LLaMA, Mistral | BERT, RoBERTa, DeBERTa |

### CLM: Next Token Prediction

```
Input:  "The cat sat on the"
Target: "mat"

Model sees: [The, cat, sat, on, the]
Predicts: P(mat | The, cat, sat, on, the)

Objective: Minimize cross-entropy over all positions
L = -sum(log P(x_t | x_1, ..., x_{t-1}))
```

### MLM: Masked Prediction

```
Input:  "The [MASK] sat on the mat"
Target: "cat"

Model sees: [The, [MASK], sat, on, the, mat]
Predicts: P(cat | context)

Typically 15% of tokens masked:
- 80% ’ [MASK]
- 10% ’ random token
- 10% ’ unchanged
```

## Pre-training Pipeline

### Data Preparation

```python
# Typical pre-training data mix
data_sources = {
    "web": 0.70,      # Common Crawl, C4
    "books": 0.15,    # Books, literature
    "code": 0.10,     # GitHub, StackOverflow
    "scientific": 0.03,  # arXiv, papers
    "other": 0.02,    # Wikipedia, etc.
}

# Data processing pipeline
# 1. Download and deduplicate
# 2. Quality filtering (length, language, etc.)
# 3. Tokenization
# 4. Chunking into sequences
# 5. Shuffling
```

### Training Configuration

```python
# Typical 7B model pre-training config
config = {
    # Model
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "vocab_size": 32000,
    "max_seq_length": 4096,

    # Training
    "batch_size_tokens": 4_000_000,
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "total_steps": 1_000_000,
    "weight_decay": 0.1,
    "gradient_clipping": 1.0,

    # Optimizer
    "optimizer": "AdamW",
    "beta1": 0.9,
    "beta2": 0.95,
}
```

### Compute Requirements

```
Rough estimates (using A100 GPUs):

| Model Size | Tokens | GPU Hours | Approximate Cost |
|------------|--------|-----------|------------------|
| 1B | 20B | 2,000 | $5K |
| 7B | 1T | 100,000 | $500K |
| 13B | 2T | 400,000 | $2M |
| 70B | 2T | 2,000,000 | $10M |

Scaling law (Chinchilla):
Optimal tokens H 20 × parameters
```

## Scaling Laws

### Compute-Optimal Training

```
Chinchilla scaling law:
- For given compute budget, balance model and data
- Optimal: 20 tokens per parameter
- Previous models (GPT-3) were undertrained

Example:
- 7B model ’ train on ~140B tokens (optimal)
- LLaMA trained on 1-2T tokens (overtrained for inference efficiency)
```

### Emergent Abilities

```
Capabilities that appear at scale:

Small (<1B):
- Basic language fluency
- Pattern matching

Medium (1-10B):
- Good knowledge
- Simple reasoning
- Instruction following

Large (10-100B):
- Complex reasoning
- Chain-of-thought
- In-context learning
- Code generation

Very Large (100B+):
- Advanced reasoning
- Complex tool use
- Emergent behaviors
```

## Continued Pre-training

### Domain Adaptation

```python
# Adapt pre-trained model to new domain
# Uses same objective (CLM) on domain data

from transformers import Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("base_model")

# Continued pre-training on domain data
training_args = TrainingArguments(
    learning_rate=5e-5,  # Lower than initial pre-training
    num_train_epochs=1,
    per_device_train_batch_size=32,
)

trainer = Trainer(
    model=model,
    train_dataset=domain_data,
)
trainer.train()
```

### Use Cases

```
When to continue pre-training:
- Specialized domain (medical, legal, code)
- New language or dialect
- Temporal adaptation (recent events)
- Private/proprietary data

Amount of data:
- Lightweight: 1-10B tokens
- Substantial: 10-100B tokens
- Full adaptation: 100B+ tokens
```

## Key Takeaways

1. **Foundation of LLMs**: Pre-training creates base capabilities.

2. **CLM for generation**: Decoder models predict next tokens.

3. **MLM for understanding**: Encoder models fill in blanks.

4. **Massive scale**: Trillions of tokens, enormous compute.

5. **Scaling laws**: More compute = predictable improvement.

6. **Most use existing**: Training from scratch is rare.

7. **Continued pre-training**: Cheaper domain adaptation.

## Further Reading

For detailed coverage of pre-training objectives, see:

- [Causal Language Modeling](causal-language-modeling/ReadMe.md) - Next-token prediction
- [Masked Language Modeling](masked-language-modeling/ReadMe.md) - Bidirectional fill-in-blank
