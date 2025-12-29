# BART Architecture

## Summary

BART (Bidirectional and Auto-Regressive Transformers) is an encoder-decoder model that combines BERT-style bidirectional encoding with GPT-style autoregressive decoding. Developed by Facebook AI, BART is pre-trained by corrupting text with various noise functions and learning to reconstruct the original. This denoising objective makes BART particularly effective for sequence-to-sequence tasks like summarization, translation, and question answering, where understanding the input and generating coherent output are both critical.

Key points to remember:

- Encoder-decoder architecture: Bidirectional encoder + autoregressive decoder
- Denoising pre-training: Learn to reconstruct corrupted text
- Corruption strategies: Token masking, deletion, infilling, sentence permutation
- Encoder: BERT-like bidirectional self-attention
- Decoder: GPT-like causal self-attention + cross-attention to encoder
- Best for: Summarization, translation, QA, text generation from context
- Variants: BART-base (140M), BART-large (400M), mBART (multilingual)

## Architecture Overview

### Structure

```
Input: "[corrupted text]"
            |
            v
+-------------------+
|     Encoder       |
| (Bidirectional)   |  <- Full attention to all input tokens
| - Self-Attention  |
| - FFN             |
| x N layers        |
+-------------------+
            |
            v
    Encoder Representations
            |
            +------------+
            |            |
            v            |
+-------------------+    |
|     Decoder       |    |
| (Autoregressive)  |    |
| - Causal Attn     |    |  <- Attend to previous output tokens
| - Cross Attn -----+----+  <- Attend to encoder output
| - FFN             |
| x N layers        |
+-------------------+
            |
            v
Output: "[original text]"
```

### Key Dimensions

| Model | Encoder Layers | Decoder Layers | d_model | Heads | d_ff | Params |
|-------|---------------|----------------|---------|-------|------|--------|
| BART-base | 6 | 6 | 768 | 12 | 3072 | 140M |
| BART-large | 12 | 12 | 1024 | 16 | 4096 | 400M |

## Pre-training Corruption Strategies

### Overview

BART is trained by corrupting text and learning to reconstruct it:

```python
def bart_pretrain_objective(model, original_text):
    """BART denoising objective."""
    # Corrupt the text
    corrupted = apply_noise(original_text)

    # Encode corrupted text
    encoder_output = model.encoder(corrupted)

    # Decode to original
    logits = model.decoder(original_text, encoder_output)

    # Loss is cross-entropy against original
    loss = cross_entropy(logits, original_text)
    return loss
```

### Corruption Types

| Strategy | Description | Example |
|----------|-------------|---------|
| Token Masking | Replace tokens with [MASK] | "The cat sat" -> "The [MASK] sat" |
| Token Deletion | Remove tokens entirely | "The cat sat" -> "The sat" |
| Text Infilling | Replace spans with single [MASK] | "The cat sat on mat" -> "The [MASK] mat" |
| Sentence Permutation | Shuffle sentence order | "A. B. C." -> "B. C. A." |
| Document Rotation | Rotate document | "ABCDE" -> "CDEAB" |

### Text Infilling (Most Effective)

```python
def text_infilling(tokens, mean_span_length=3, mask_ratio=0.3):
    """Replace spans with single [MASK] tokens."""
    result = []
    i = 0
    n_tokens = len(tokens)

    while i < n_tokens:
        if random.random() < mask_ratio:
            # Sample span length from Poisson distribution
            span_length = max(1, np.random.poisson(mean_span_length))
            span_length = min(span_length, n_tokens - i)

            # Replace entire span with single [MASK]
            result.append(MASK_TOKEN)
            i += span_length
        else:
            result.append(tokens[i])
            i += 1

    return result
```

## Implementation

### Encoder Block

```python
class BARTEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        # Self-attention with residual
        residual = x
        x, _ = self.self_attn(
            x, x, x,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        x = self.dropout(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # Feed-forward with residual
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class BARTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position, config.d_model)
        self.layers = nn.ModuleList([
            BARTEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        x = self.layernorm_embedding(x)

        # Transpose for attention: (seq, batch, dim)
        x = x.transpose(0, 1)

        # Encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x.transpose(0, 1)  # Back to (batch, seq, dim)
```

### Decoder Block

```python
class BARTDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)

        # Cross-attention to encoder
        self.encoder_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)

        # Feed-forward
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, causal_mask, encoder_mask=None):
        # Causal self-attention
        residual = x
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = self.dropout(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # Cross-attention to encoder output
        residual = x
        x, _ = self.encoder_attn(
            x,                      # Query from decoder
            encoder_output,         # Key from encoder
            encoder_output,         # Value from encoder
            key_padding_mask=encoder_mask,
            need_weights=False
        )
        x = self.dropout(x)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        # Feed-forward
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class BARTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position, config.d_model)
        self.layers = nn.ModuleList([
            BARTDecoderLayer(config) for _ in range(config.decoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, input_ids, encoder_output, encoder_mask=None):
        # Embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        x = self.layernorm_embedding(x)

        # Causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()

        x = x.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_mask)

        return x.transpose(0, 1)
```

### Complete BART Model

```python
class BART(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BARTEncoder(config)
        self.decoder = BARTDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Share embeddings between encoder, decoder, and output
        self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight
        self.lm_head.weight = self.encoder.embed_tokens.weight

    def forward(self, encoder_input_ids, decoder_input_ids,
                encoder_mask=None, decoder_mask=None):
        # Encode source
        encoder_output = self.encoder(encoder_input_ids, encoder_mask)

        # Decode target
        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_output,
            encoder_mask
        )

        # Project to vocabulary
        logits = self.lm_head(decoder_output)

        return logits

    def generate(self, encoder_input_ids, max_length=100, **kwargs):
        """Autoregressive generation."""
        encoder_output = self.encoder(encoder_input_ids)

        # Start with BOS token
        decoder_input = torch.tensor([[BOS_TOKEN]], device=encoder_input_ids.device)

        for _ in range(max_length):
            logits = self.decoder(decoder_input, encoder_output)
            next_token_logits = logits[:, -1, :]

            # Sample or argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if next_token == EOS_TOKEN:
                break

        return decoder_input
```

## Fine-tuning for Tasks

### Summarization

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Summarize
article = "Long article text here..."
inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### Question Answering

```python
# BART for extractive/abstractive QA
question = "What is the capital of France?"
context = "Paris is the capital and largest city of France."

input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt")
answer_ids = model.generate(inputs["input_ids"], max_length=50)
answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
```

### Text Classification (Encoder Only)

```python
from transformers import BartForSequenceClassification

model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli"
)

# Zero-shot classification
premise = "The movie was absolutely fantastic."
hypothesis = "This is a positive review."
inputs = tokenizer(premise, hypothesis, return_tensors="pt")
outputs = model(**inputs)
# outputs.logits: [contradiction, neutral, entailment]
```

## Comparison to Other Models

### BART vs BERT

| Aspect | BERT | BART |
|--------|------|------|
| Architecture | Encoder-only | Encoder-decoder |
| Pre-training | Masked LM + NSP | Denoising (multiple corruptions) |
| Generation | Requires special setup | Native autoregressive |
| Best for | Understanding tasks | Generation + understanding |

### BART vs T5

| Aspect | BART | T5 |
|--------|------|-----|
| Pre-training | Denoising | Span corruption (similar) |
| Task format | Natural | Text-to-text |
| Relative position | Learned absolute | Learned relative |
| Vocabulary | BPE (50K) | SentencePiece (32K) |

### BART vs GPT

| Aspect | BART | GPT |
|--------|------|-----|
| Architecture | Encoder-decoder | Decoder-only |
| Input processing | Bidirectional | Causal |
| Best for | Conditional generation | Open-ended generation |
| Context | Separate input/output | Unified |

## Variants

### mBART (Multilingual BART)

Pre-trained on 25 languages for multilingual NMT:

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Translate English to French
tokenizer.src_lang = "en_XX"
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
generated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
)
translation = tokenizer.decode(generated[0], skip_special_tokens=True)
```

### DistilBART

Smaller, distilled version:

```python
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
# 12 encoder layers, 6 decoder layers (vs 12-12 for full BART-large)
```

## Key Takeaways

1. **Best of both worlds**: Bidirectional encoding + autoregressive decoding.

2. **Denoising is powerful**: Learning to reconstruct corrupted text teaches robust representations.

3. **Text infilling key**: Multiple corruptions, especially span infilling, outperform simple masking.

4. **Natural for seq2seq**: Architecture matches summarization, translation, QA naturally.

5. **Flexible fine-tuning**: Can use encoder-only (classification), decoder-only, or full model.

6. **Strong summarization**: BART-CNN is a standard baseline for summarization.

7. **Multilingual capable**: mBART enables cross-lingual transfer and translation.
