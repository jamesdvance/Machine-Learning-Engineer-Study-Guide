# Full Fine-Tuning

## Summary

Full fine-tuning updates all parameters of a pre-trained language model on task-specific data. This is the most direct adaptation approach: take a pre-trained model, continue training on your dataset with a lower learning rate. Full fine-tuning achieves the best task performance when data is sufficient, but requires substantial compute and memory since all parameters need gradients. For large models (7B+), this becomes prohibitively expensive for most practitioners, leading to the rise of parameter-efficient alternatives. However, full fine-tuning remains the gold standard for maximum performance.

Key points to remember:

- All parameters updated: Maximum capacity for adaptation
- Highest quality ceiling: Best possible task performance
- Resource intensive: Need memory for model + gradients + optimizer states
- Lower learning rate: Typically 10-100x smaller than pre-training
- Catastrophic forgetting: Risk of losing general capabilities
- Reference for comparison: Benchmark for PEFT methods
- Production viable: For organizations with sufficient compute

## The Process

### Basic Training Loop

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def full_fine_tune(model_name, train_data, output_dir):
    """Full fine-tuning on task-specific data."""
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,  # Lower than pre-training
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        save_strategy="epoch",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
```

### Memory Requirements

```
Memory per parameter (float32):
- Model weights: 4 bytes
- Gradients: 4 bytes
- Optimizer states (Adam): 8 bytes (momentum + variance)
Total: 16 bytes per parameter

For 7B model:
- Weights: 28 GB
- Gradients: 28 GB
- Optimizer: 56 GB
- Total: 112 GB (minimum)
- Plus activations: ~150-200 GB

Solutions:
- Mixed precision (FP16/BF16): ~60% reduction
- Gradient checkpointing: Trade compute for memory
- DeepSpeed ZeRO: Shard across GPUs
- CPU offloading: Slower but works
```

## Optimization Strategies

### Memory Optimization

```python
from transformers import TrainingArguments

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
training_args = TrainingArguments(
    ...,
    fp16=True,  # or bf16=True for newer GPUs
    gradient_checkpointing=True,
)

# Gradient accumulation for effective larger batch
training_args = TrainingArguments(
    ...,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
)
```

### DeepSpeed ZeRO

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,  # or 3 for largest models
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    },
    "fp16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}

# Training with DeepSpeed
training_args = TrainingArguments(
    ...,
    deepspeed="deepspeed_config.json",
)
```

### FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
    ),
)

# Or via Accelerate
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    fsdp_plugin=fsdp_plugin
)
```

## Hyperparameters

### Learning Rate

```python
# Key insight: Much lower than pre-training
# Pre-training: 1e-4 to 3e-4
# Fine-tuning: 1e-6 to 5e-5

learning_rate_guidelines = {
    "conservative": 1e-6,   # Minimal change, preserve capabilities
    "moderate": 2e-5,       # Common default
    "aggressive": 5e-5,     # More adaptation, more forgetting risk
}

# Learning rate scheduling
training_args = TrainingArguments(
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # or "linear"
    warmup_ratio=0.1,  # 10% of steps for warmup
)
```

### Batch Size and Epochs

```
Batch size:
- Larger batches = more stable gradients
- But: memory limited
- Use gradient accumulation to simulate larger batches

Epochs:
- Usually 1-5 epochs sufficient
- Watch for overfitting
- Monitor validation loss

Data size guidelines:
- < 1K examples: Risk of overfitting, consider PEFT
- 1K-10K: Standard fine-tuning works
- 10K-100K: Excellent for full fine-tuning
- > 100K: Consider continued pre-training approach
```

## Data Preparation

### Format for Causal LM

```python
def format_instruction_data(example):
    """Format for instruction tuning."""
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['response']}"
    }

def format_chat_data(example):
    """Format for chat fine-tuning."""
    messages = example['messages']
    formatted = ""
    for msg in messages:
        if msg['role'] == 'user':
            formatted += f"User: {msg['content']}\n"
        else:
            formatted += f"Assistant: {msg['content']}\n"
    return {"text": formatted}
```

### Tokenization

```python
def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize with proper padding and truncation."""
    def tokenize(example):
        outputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs

    return dataset.map(tokenize, remove_columns=dataset.column_names)
```

## Catastrophic Forgetting

### The Problem

```
Catastrophic forgetting:
- Model loses general capabilities while learning specific task
- Especially problematic with small datasets
- Can forget how to perform other tasks

Signs:
- Great task performance, poor general ability
- Degraded performance on benchmarks
- Less coherent general responses
```

### Mitigation Strategies

```python
# 1. Mix in general data
combined_dataset = concatenate_datasets([
    task_specific_data,  # 70%
    general_instruction_data,  # 30%
])

# 2. Lower learning rate
training_args = TrainingArguments(
    learning_rate=1e-6,  # Very conservative
)

# 3. Regularization
training_args = TrainingArguments(
    weight_decay=0.1,  # Higher than usual
)

# 4. Early stopping
trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 5. Use PEFT instead (LoRA, etc.)
# Only update small portion of parameters
```

## Multi-GPU Training

### Data Parallel

```python
# Simple multi-GPU with DataParallel
model = torch.nn.DataParallel(model)

# Better: DistributedDataParallel
# torchrun --nproc_per_node=4 train.py
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

### Model Parallel

```python
# Pipeline parallelism for very large models
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatic device assignment
    max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB"}
)
```

## Evaluation

### Monitoring Training

```python
training_args = TrainingArguments(
    ...,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Custom metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Custom evaluation logic
    return {"accuracy": accuracy}
```

### Comparing to PEFT

```
Benchmark comparison (7B model on task):

| Method | Task Accuracy | General Benchmark | Memory | Time |
|--------|--------------|-------------------|--------|------|
| Full FT | 92.3% | 68.1% | 120 GB | 10h |
| LoRA (r=16) | 91.5% | 71.2% | 24 GB | 3h |
| QLoRA | 90.8% | 70.9% | 16 GB | 4h |

Observations:
- Full FT: Best task performance, some forgetting
- PEFT: Close performance, better efficiency
- Trade-off depends on priorities
```

## When to Use Full Fine-Tuning

### Good Use Cases

```
Choose full fine-tuning when:
- Have sufficient compute (>4x A100 or equivalent)
- Need absolute best task performance
- Have large, high-quality dataset (>10K examples)
- Task is significantly different from pre-training
- Can afford training time and cost
- Maintaining separate model per task is acceptable
```

### Alternatives to Consider

```
Consider alternatives when:
- Limited compute ’ LoRA/QLoRA
- Small dataset ’ Few-shot or PEFT
- Need multi-task single model ’ Adapters
- Rapid iteration needed ’ Prompt tuning
- Memory constrained ’ QLoRA
```

## Key Takeaways

1. **Maximum capacity**: All parameters updated for best task fit.

2. **Resource intensive**: Requires substantial GPU memory.

3. **Lower learning rate**: 10-100x smaller than pre-training.

4. **Forgetting risk**: Monitor general capabilities.

5. **Gold standard**: Benchmark for PEFT comparison.

6. **Use optimization**: DeepSpeed/FSDP for large models.

7. **Data quality matters**: More than quantity for fine-tuning.
