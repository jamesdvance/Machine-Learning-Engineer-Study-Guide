# Masked Language Modeling (MLM)

## Summary

Masked Language Modeling randomly masks tokens in the input and trains the model to predict them using bidirectional context. Unlike causal language modeling which only sees past tokens, MLM sees the entire sequence except the masked positions, making it a "fill-in-the-blank" task. This is the pre-training objective for BERT and encoder models. MLM produces powerful representations for understanding tasks (classification, NER, QA) but is less suited for generation since it doesn't model left-to-right token probabilities.

Key points to remember:

- Fill-in-the-blank: Predict masked tokens from context
- Bidirectional: Model sees both left and right context
- Encoder architecture: Used for understanding, not generation
- BERT family: BERT, RoBERTa, DeBERTa, ALBERT
- 15% masking: Standard is to mask 15% of tokens
- 80/10/10 rule: 80% mask token, 10% random, 10% unchanged
- Representations: Produces embeddings for downstream tasks

## The Core Objective

### Masked Token Prediction

```
Input:  "The cat sat on the [MASK]"
Target: "mat"

Training objective:
L = -log P(x_masked | x_unmasked)

Key insight:
- Model sees entire context
- Must predict only masked positions
- Learns rich contextual representations
```

### Masking Strategy

```python
def apply_mlm_masking(input_ids, tokenizer, mask_prob=0.15):
    """
    Apply BERT-style masking.

    80% of masked tokens ’ [MASK] token
    10% of masked tokens ’ random token
    10% of masked tokens ’ unchanged

    This prevents model from only learning to predict [MASK]
    """
    labels = input_ids.clone()

    # Create masking probability matrix
    probability_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask special tokens
    special_tokens_mask = get_special_tokens_mask(input_ids, tokenizer)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Randomly select masked positions
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only compute loss on masked positions
    labels[~masked_indices] = -100  # Cross-entropy ignores -100

    # 80% of time, replace with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8)
    ).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of time, replace with random token
    indices_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.5)  # 0.5 * 0.2 = 0.1
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), input_ids.shape)
    input_ids[indices_random] = random_words[indices_random]

    # 10% of time, keep original (indices not replaced or random)

    return input_ids, labels
```

## Implementation

### Basic Training

```python
from transformers import BertForMaskedLM, BertTokenizer

def mlm_training_step(model, batch, optimizer):
    """Single training step for masked language modeling."""
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Apply masking
    masked_input_ids, labels = apply_mlm_masking(input_ids, tokenizer)

    # Forward pass
    outputs = model(
        input_ids=masked_input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### With Hugging Face

```python
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Load model
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Data collator handles masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Masked LM
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./mlm_model",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
```

## Architecture Differences

### MLM vs CLM Attention

```
Causal LM (GPT-style):
Position 4 can attend to: [1, 2, 3, 4]
Uses causal mask (triangular)

Masked LM (BERT-style):
Position 4 can attend to: [1, 2, 3, 4, 5, 6, 7, ...]
Uses full attention (no mask, except padding)

Implication:
- MLM: Richer context for each position
- CLM: Can generate left-to-right
```

### Encoder Architecture

```python
# BERT-style encoder
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)  # Bidirectional attention

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)

        # Full bidirectional attention
        # No causal mask needed
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=attention_mask
        )

        return encoder_outputs.last_hidden_state
```

## Variants

### RoBERTa Improvements

```python
# RoBERTa modifications to BERT:

# 1. Dynamic masking (different mask each epoch)
def dynamic_masking(input_ids, tokenizer):
    # Re-mask during training, not pre-compute
    return apply_mlm_masking(input_ids, tokenizer)

# 2. Larger batches
batch_size = 8192  # vs BERT's 256

# 3. Remove NSP (Next Sentence Prediction)
# Just MLM, no sentence pair task

# 4. Longer training
num_steps = 500000  # vs BERT's 100000

# 5. More data
# Trained on 160GB vs BERT's 16GB
```

### Whole Word Masking

```python
def whole_word_masking(input_ids, tokenizer, mask_prob=0.15):
    """
    Mask entire words, not just subword tokens.

    Standard: "playing" ’ "play", "##ing" ’ mask either
    WWM: "playing" ’ "play", "##ing" ’ mask both or neither
    """
    word_ids = tokenizer.get_word_ids()  # Map tokens to words

    # Group tokens by word
    word_to_tokens = defaultdict(list)
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            word_to_tokens[word_id].append(idx)

    # Decide which words to mask
    masked_words = set()
    for word_id in word_to_tokens:
        if random.random() < mask_prob:
            masked_words.add(word_id)

    # Mask all tokens in masked words
    labels = input_ids.clone()
    for word_id, token_indices in word_to_tokens.items():
        if word_id in masked_words:
            for idx in token_indices:
                # Apply 80/10/10 rule
                labels[idx] = input_ids[idx]
                input_ids[idx] = tokenizer.mask_token_id
        else:
            for idx in token_indices:
                labels[idx] = -100  # Don't compute loss

    return input_ids, labels
```

### Span Masking (SpanBERT)

```python
def span_masking(input_ids, tokenizer, mean_span_length=3.8):
    """
    Mask contiguous spans instead of random tokens.
    Spans follow geometric distribution.
    """
    labels = input_ids.clone()
    seq_len = len(input_ids)
    target_mask_count = int(seq_len * 0.15)

    masked_count = 0
    pos = 0

    while masked_count < target_mask_count and pos < seq_len:
        # Sample span length from geometric distribution
        span_length = np.random.geometric(1 / mean_span_length)
        span_length = min(span_length, seq_len - pos, target_mask_count - masked_count)

        # Mask the span
        for i in range(span_length):
            input_ids[pos + i] = tokenizer.mask_token_id

        masked_count += span_length
        pos += span_length + 1  # Skip ahead

    labels[input_ids != tokenizer.mask_token_id] = -100

    return input_ids, labels
```

## Use Cases

### Understanding Tasks

```python
# MLM pre-training ’ Fine-tune for understanding

# 1. Text Classification
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 2. Named Entity Recognition
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=9  # B-PER, I-PER, B-ORG, etc.
)

# 3. Question Answering
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained(
    "bert-base-uncased"
)
```

### Representations

```python
def get_bert_embeddings(model, tokenizer, text):
    """
    Extract embeddings from pre-trained BERT.
    Useful for similarity, clustering, retrieval.
    """
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Options for sentence representation:
    # 1. [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # 2. Mean pooling
    mean_embedding = outputs.last_hidden_state.mean(dim=1)

    # 3. Specific layer
    layer_embedding = outputs.hidden_states[7]  # Layer 7

    return cls_embedding
```

## MLM vs CLM Trade-offs

| Aspect | MLM (BERT) | CLM (GPT) |
|--------|------------|-----------|
| Context | Bidirectional | Left-to-right only |
| Task | Fill-in-blank | Next token |
| Architecture | Encoder | Decoder |
| Generation | Poor | Excellent |
| Understanding | Excellent | Good |
| Representations | Rich, contextual | Good for generation |
| Fine-tuning | Classification, NER, QA | Instruction, chat |

## Key Takeaways

1. **Bidirectional context**: Model sees full sequence for each prediction.

2. **15% masking**: Standard rate, 80/10/10 token replacement.

3. **Encoder models**: BERT family, excellent for understanding.

4. **Not for generation**: Doesn't model sequential probabilities.

5. **Rich representations**: [CLS] token or pooled embeddings.

6. **Variants improve**: RoBERTa, SpanBERT, Whole Word Masking.

7. **Complement to CLM**: Choose based on task (understanding vs generation).
