# T5 Architecture

## Summary

T5 (Text-to-Text Transfer Transformer) is Google's encoder-decoder model that frames all NLP tasks as text-to-text problems. Rather than having task-specific architectures, T5 converts every task into generating text from text: classification becomes generating a class label, summarization becomes generating a summary, translation becomes generating the translated text. This unified approach simplifies multi-task learning and transfer, making T5 highly versatile. Pre-trained with span corruption, T5 variants range from 60M to 11B parameters.

Key points to remember:

- Text-to-text paradigm: All tasks framed as text input to text output
- Span corruption pre-training: Mask contiguous spans, predict original
- Relative position embeddings: Better length generalization than absolute
- Task prefixes: "translate:", "summarize:", "question:" guide behavior
- Unified architecture: Same model handles classification, generation, NLU
- Variants: T5-small (60M) to T5-11B; Flan-T5 (instruction-tuned)
- Encoder-decoder: Bidirectional encoder + autoregressive decoder

## The Text-to-Text Framework

### Unifying All Tasks

```
Traditional Approach:
Classification: Input -> Classifier Head -> Label
Summarization:  Input -> Seq2Seq Model -> Summary
Translation:    Input -> Seq2Seq Model -> Translation
QA:             Input -> Span Extraction -> Answer

T5 Approach (Everything is text-to-text):
Classification: "classify: [text]" -> "positive"
Summarization:  "summarize: [article]" -> "[summary]"
Translation:    "translate English to German: [text]" -> "[German]"
QA:             "question: [q] context: [c]" -> "[answer]"
```

### Task Prefixes

| Task | Prefix | Input | Output |
|------|--------|-------|--------|
| Translation | "translate English to German:" | English text | German text |
| Summarization | "summarize:" | Article | Summary |
| Sentiment | "classify:" or "sentiment:" | Review | "positive"/"negative" |
| QA | "question: ... context: ..." | Question + passage | Answer |
| Grammar | "correct grammar:" | Bad sentence | Good sentence |

### Example Usage

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Translation
input_text = "translate English to German: The house is beautiful."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))  # "Das Haus ist schön."

# Summarization
article = "summarize: " + long_article
inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
summary = model.generate(**inputs, max_length=150)

# Classification
input_text = "classify: This movie was terrible."
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=10)
print(tokenizer.decode(output[0]))  # "negative"
```

## Architecture

### Structure

```
Input: "translate English to German: Hello world"
                    |
                    v
        +--------------------+
        |      Encoder       |
        | - Self-Attention   |  <- Bidirectional
        | - Feed-Forward     |
        | - Layer Norm       |
        | x N layers         |
        +--------------------+
                    |
                    v
           Encoder Output
                    |
        +-----------+------------+
        |                        |
        v                        |
+--------------------+           |
|      Decoder       |           |
| - Causal Self-Attn |  <- Masked        |
| - Cross-Attention -+-----------+  <- Attends to encoder
| - Feed-Forward     |
| x N layers         |
+--------------------+
        |
        v
Output: "Hallo Welt"
```

### Key Dimensions

| Model | Encoder Layers | Decoder Layers | d_model | d_ff | Heads | Params |
|-------|----------------|----------------|---------|------|-------|--------|
| T5-small | 6 | 6 | 512 | 2048 | 8 | 60M |
| T5-base | 12 | 12 | 768 | 3072 | 12 | 220M |
| T5-large | 24 | 24 | 1024 | 4096 | 16 | 770M |
| T5-3B | 24 | 24 | 1024 | 16384 | 32 | 3B |
| T5-11B | 24 | 24 | 1024 | 65536 | 128 | 11B |

## Pre-training: Span Corruption

### Objective

Replace random contiguous spans with sentinel tokens, predict original:

```python
def span_corruption(text, noise_density=0.15, mean_span_length=3):
    """T5's span corruption objective."""
    tokens = tokenize(text)
    n_tokens = len(tokens)
    n_noise_tokens = int(n_tokens * noise_density)
    n_spans = max(1, int(n_noise_tokens / mean_span_length))

    # Sample span starts
    span_starts = sorted(random.sample(range(n_tokens - mean_span_length), n_spans))

    # Create corrupted input and target
    input_tokens = []
    target_tokens = []
    sentinel_id = 0
    prev_end = 0

    for start in span_starts:
        # Add tokens before span
        input_tokens.extend(tokens[prev_end:start])

        # Add sentinel to input
        input_tokens.append(f"<extra_id_{sentinel_id}>")

        # Sample span length
        span_length = max(1, np.random.poisson(mean_span_length))
        end = min(start + span_length, n_tokens)

        # Add sentinel + original span to target
        target_tokens.append(f"<extra_id_{sentinel_id}>")
        target_tokens.extend(tokens[start:end])

        sentinel_id += 1
        prev_end = end

    # Add remaining tokens
    input_tokens.extend(tokens[prev_end:])
    target_tokens.append("</s>")  # End of sequence

    return input_tokens, target_tokens
```

### Example

```
Original: "The quick brown fox jumps over the lazy dog"

Corrupted Input:  "The <extra_id_0> fox <extra_id_1> the lazy dog"
Target:           "<extra_id_0> quick brown <extra_id_1> jumps over </s>"
```

## Implementation

### Relative Position Embeddings

T5 uses simplified relative position bias instead of absolute embeddings:

```python
class T5RelativePositionBias(nn.Module):
    def __init__(self, n_heads, n_buckets=32, max_distance=128):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    def _relative_position_bucket(self, relative_position, bidirectional=True):
        """Map relative position to bucket."""
        # For bidirectional (encoder): use both positive and negative buckets
        # For unidirectional (decoder): only negative buckets

        n_buckets = self.n_buckets
        max_exact = n_buckets // 2

        if bidirectional:
            n_buckets //= 2
            # Separate buckets for positive/negative
            relative_buckets = (relative_position > 0).long() * n_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Only attend to past (negative positions)
            relative_position = -torch.clamp(relative_position, max=0)
            relative_buckets = torch.zeros_like(relative_position)

        # Half buckets for exact positions
        is_small = relative_position < max_exact

        # Half buckets for log-scaled positions
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (n_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.clamp(
            relative_position_if_large, max=n_buckets - 1
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )

        return relative_buckets

    def forward(self, query_length, key_length, bidirectional=True):
        """Compute position bias for attention."""
        # Create position grids
        q_pos = torch.arange(query_length).unsqueeze(1)
        k_pos = torch.arange(key_length).unsqueeze(0)

        # Relative positions
        relative_position = k_pos - q_pos

        # Map to buckets
        buckets = self._relative_position_bucket(relative_position, bidirectional)

        # Look up bias values
        bias = self.relative_attention_bias(buckets)  # (q_len, k_len, n_heads)
        bias = bias.permute(2, 0, 1)  # (n_heads, q_len, k_len)

        return bias
```

### T5 Attention

```python
class T5Attention(nn.Module):
    def __init__(self, config, is_decoder=False, has_relative_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)

        if has_relative_bias:
            self.relative_bias = T5RelativePositionBias(
                config.n_heads,
                config.relative_attention_num_buckets
            )
        else:
            self.relative_bias = None

    def forward(self, hidden_states, key_value_states=None,
                position_bias=None, attention_mask=None):
        """
        Self-attention if key_value_states is None.
        Cross-attention if key_value_states provided.
        """
        B, T, _ = hidden_states.shape

        # Query always from hidden states
        q = self.q(hidden_states)

        # Key/value from hidden_states (self-attn) or key_value_states (cross-attn)
        if key_value_states is None:
            k = self.k(hidden_states)
            v = self.v(hidden_states)
            is_cross_attention = False
        else:
            k = self.k(key_value_states)
            v = self.v(key_value_states)
            is_cross_attention = True

        # Reshape for multi-head
        q = q.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Add position bias
        if position_bias is None and self.relative_bias is not None:
            position_bias = self.relative_bias(
                q.size(2), k.size(2),
                bidirectional=not self.is_decoder or is_cross_attention
            )

        if position_bias is not None:
            scores += position_bias

        # Apply attention mask
        if attention_mask is not None:
            scores += attention_mask

        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o(output)

        return output, position_bias
```

### T5 Block

```python
class T5Block(nn.Module):
    def __init__(self, config, is_decoder=False, has_relative_bias=False):
        super().__init__()
        self.is_decoder = is_decoder

        # Self-attention
        self.self_attn = T5Attention(
            config,
            is_decoder=is_decoder,
            has_relative_bias=has_relative_bias
        )
        self.self_attn_norm = T5LayerNorm(config.d_model)

        # Cross-attention (decoder only)
        if is_decoder:
            self.cross_attn = T5Attention(config, is_decoder=True)
            self.cross_attn_norm = T5LayerNorm(config.d_model)

        # Feed-forward
        self.ff = T5FeedForward(config)
        self.ff_norm = T5LayerNorm(config.d_model)

    def forward(self, hidden_states, encoder_hidden_states=None,
                position_bias=None, encoder_position_bias=None,
                attention_mask=None, encoder_attention_mask=None):

        # Self-attention
        normed = self.self_attn_norm(hidden_states)
        attn_output, position_bias = self.self_attn(
            normed, position_bias=position_bias, attention_mask=attention_mask
        )
        hidden_states = hidden_states + attn_output

        # Cross-attention (decoder)
        if self.is_decoder and encoder_hidden_states is not None:
            normed = self.cross_attn_norm(hidden_states)
            cross_output, encoder_position_bias = self.cross_attn(
                normed,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_position_bias,
                attention_mask=encoder_attention_mask
            )
            hidden_states = hidden_states + cross_output

        # Feed-forward
        normed = self.ff_norm(hidden_states)
        ff_output = self.ff(normed)
        hidden_states = hidden_states + ff_output

        return hidden_states, position_bias, encoder_position_bias


class T5LayerNorm(nn.Module):
    """T5 uses RMSNorm-style normalization without bias."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

## T5 Variants

### Standard T5 Family

```python
# Standard T5 models
T5_CONFIGS = {
    "t5-small": {"d_model": 512, "d_ff": 2048, "n_layers": 6, "n_heads": 8},
    "t5-base": {"d_model": 768, "d_ff": 3072, "n_layers": 12, "n_heads": 12},
    "t5-large": {"d_model": 1024, "d_ff": 4096, "n_layers": 24, "n_heads": 16},
    "t5-3b": {"d_model": 1024, "d_ff": 16384, "n_layers": 24, "n_heads": 32},
    "t5-11b": {"d_model": 1024, "d_ff": 65536, "n_layers": 24, "n_heads": 128},
}
```

### Flan-T5 (Instruction-Tuned)

```python
from transformers import T5ForConditionalGeneration

# Flan-T5 is T5 fine-tuned on 1800+ tasks with instructions
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# Better at following instructions
input_text = "Translate to French: I love programming."
# Or more complex instructions
input_text = "Given the passage below, answer the question.\nPassage: ...\nQuestion: ..."
```

### T5 1.1 (Improved Architecture)

```python
# T5 1.1 improvements:
# - GEGLU activation instead of ReLU
# - No parameter sharing between embedding and output
# - Pre-norm (like GPT-2) instead of post-norm

model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")
```

### mT5 (Multilingual)

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")

# Trained on 101 languages
input_text = "translate English to Japanese: Hello world"
```

## Fine-tuning

### Multi-task Training

```python
from datasets import concatenate_datasets

# Combine multiple datasets with task prefixes
train_datasets = []

# Add summarization data
summarization_data = load_dataset("xsum")["train"]
summarization_data = summarization_data.map(
    lambda x: {"input": f"summarize: {x['document']}",
               "target": x['summary']}
)
train_datasets.append(summarization_data)

# Add translation data
translation_data = load_dataset("wmt14", "de-en")["train"]
translation_data = translation_data.map(
    lambda x: {"input": f"translate German to English: {x['de']}",
               "target": x['en']}
)
train_datasets.append(translation_data)

# Combine
combined = concatenate_datasets(train_datasets)
```

### Single-Task Fine-tuning

```python
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer

training_args = TrainingArguments(
    output_dir="./t5-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_steps=500,
    fp16=True,
    predict_with_generate=True,  # For seq2seq
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Comparison

### T5 vs BART

| Aspect | T5 | BART |
|--------|-----|------|
| Pre-training | Span corruption | Multiple noise types |
| Position | Relative bias | Learned absolute |
| Task format | Text-to-text with prefix | Natural |
| Vocabulary | SentencePiece (32K) | BPE (50K) |
| Normalization | Pre-norm (RMSNorm-style) | Post-norm |

### T5 vs Decoder-Only

| Aspect | T5 | GPT/LLaMA |
|--------|-----|-----------|
| Architecture | Encoder-decoder | Decoder-only |
| Input processing | Bidirectional | Causal |
| Best for | Structured generation | Open-ended generation |
| Fine-tuning | Often needed | In-context learning |

## Key Takeaways

1. **Text-to-text unifies NLP**: One framework handles all tasks.

2. **Span corruption works well**: Better than single-token masking for pre-training.

3. **Relative position is flexible**: Generalizes better to unseen lengths.

4. **Task prefixes guide behavior**: Simple way to control multi-task models.

5. **Scale improves quality**: T5-11B significantly outperforms smaller variants.

6. **Flan-T5 for instruction following**: Fine-tuned version better at following prompts.

7. **Good for structured tasks**: Summarization, translation, QA where input-output structure matters.
