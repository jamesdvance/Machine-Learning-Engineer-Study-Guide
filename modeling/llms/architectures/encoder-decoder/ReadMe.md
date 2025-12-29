# Encoder-Decoder Transformer Architectures

## Summary

Encoder-decoder transformers follow the original transformer architecture with separate encoder and decoder stacks. The encoder processes the input bidirectionally (attending to all positions), while the decoder generates output autoregressively (attending to previous outputs plus the encoder's representation via cross-attention). This separation makes encoder-decoder models particularly effective for sequence-to-sequence tasks where understanding the input fully before generating is beneficial. BART and T5 are the dominant models in this family.

Key points to remember:

- Two components: Bidirectional encoder + autoregressive decoder
- Cross-attention: Decoder attends to encoder output at each layer
- Best for: Structured generation (summarization, translation, QA)
- BART: Denoising pre-training with multiple corruption strategies
- T5: Text-to-text framework with span corruption pre-training
- Less common now: Decoder-only models dominate for general LLM tasks
- Still valuable: When explicit input-output separation benefits the task

## Architecture Overview

### Structure

```
              Encoder                           Decoder
         (Bidirectional)                   (Autoregressive)

Input -> [Embed] -> [Pos]                 Output -> [Embed] -> [Pos]
              |                                          |
              v                                          v
    +------------------+                      +------------------+
    | Self-Attention   |                      | Masked Self-Attn |
    | (full attention) |                      | (causal mask)    |
    +------------------+                      +------------------+
              |                                          |
              v                                          v
    +------------------+                      +------------------+
    | Feed-Forward     |                      | Cross-Attention  |<--+
    +------------------+                      | (Q from decoder  |   |
              |                               |  K,V from enc)   |   |
              v                               +------------------+   |
    +------------------+                               |             |
    |    x N layers    |                               v             |
    +------------------+                      +------------------+   |
              |                               | Feed-Forward     |   |
              |                               +------------------+   |
              |                                        |             |
              |                                        v             |
              |                               +------------------+   |
              |                               |    x N layers    |   |
              |                               +------------------+   |
              |                                        |             |
              +----------------------------------------+             |
                        (encoder output feeds cross-attention)       |
```

### Component Roles

| Component | Function |
|-----------|----------|
| Encoder self-attention | Full bidirectional understanding of input |
| Decoder self-attention | Causal generation (attend to past outputs) |
| Cross-attention | Decoder queries encoder representations |
| Encoder FFN | Non-linear transformation of input features |
| Decoder FFN | Non-linear transformation of output features |

## Encoder-Decoder vs Decoder-Only

### When to Use Each

| Scenario | Better Choice | Reason |
|----------|---------------|--------|
| Summarization | Encoder-decoder | Fully process article before generating |
| Translation | Encoder-decoder | Understand full source sentence |
| Open generation | Decoder-only | No clear input-output separation |
| Dialogue | Decoder-only | Unified context + response |
| Code completion | Decoder-only | Left context determines right |
| In-context learning | Decoder-only | Examples + query in one sequence |

### Attention Patterns

```
Encoder (Bidirectional):
Position 1 sees: 1, 2, 3, 4, 5
Position 2 sees: 1, 2, 3, 4, 5
Position 3 sees: 1, 2, 3, 4, 5
...

Decoder Self-Attention (Causal):
Position 1 sees: 1
Position 2 sees: 1, 2
Position 3 sees: 1, 2, 3
...

Decoder Cross-Attention:
All decoder positions see all encoder positions
```

## Cross-Attention Mechanism

### Implementation

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Queries come from decoder hidden state
        self.wq = nn.Linear(d_model, d_model)
        # Keys and values come from encoder output
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        decoder_hidden: (batch, decoder_seq, d_model)
        encoder_output: (batch, encoder_seq, d_model)
        """
        B, T_dec, D = decoder_hidden.shape
        _, T_enc, _ = encoder_output.shape

        # Query from decoder
        q = self.wq(decoder_hidden).view(B, T_dec, self.n_heads, self.d_head)

        # Key, value from encoder
        k = self.wk(encoder_output).view(B, T_enc, self.n_heads, self.d_head)
        v = self.wv(encoder_output).view(B, T_enc, self.n_heads, self.d_head)

        # Transpose for attention: (B, heads, seq, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores: decoder queries attend to encoder keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply encoder padding mask if provided
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, T_dec, D)
        return self.wo(output)
```

### Cross-Attention in Training vs Inference

```python
class EncoderDecoderModel:
    def forward(self, encoder_input, decoder_input):
        """Training: parallel forward pass."""
        # Encode once
        encoder_output = self.encoder(encoder_input)

        # Decode with teacher forcing (all targets at once)
        decoder_output = self.decoder(decoder_input, encoder_output)

        return decoder_output

    def generate(self, encoder_input, max_length):
        """Inference: autoregressive generation."""
        # Encode once (can cache this)
        encoder_output = self.encoder(encoder_input)

        # Generate token by token
        generated = [BOS_TOKEN]
        for _ in range(max_length):
            decoder_input = torch.tensor([generated])
            output = self.decoder(decoder_input, encoder_output)
            next_token = output[:, -1].argmax()
            generated.append(next_token.item())

            if next_token == EOS_TOKEN:
                break

        return generated
```

## Pre-training Objectives

### BART Denoising

```python
# BART uses multiple noise types:
noise_types = [
    "token_masking",      # Replace tokens with [MASK]
    "token_deletion",     # Remove tokens entirely
    "text_infilling",     # Replace spans with single [MASK]
    "sentence_permute",   # Shuffle sentences
    "document_rotation",  # Rotate document start
]

def bart_pretrain(text):
    corrupted = apply_random_noise(text, noise_types)
    encoder_output = encoder(corrupted)
    reconstruction = decoder(text, encoder_output)
    loss = cross_entropy(reconstruction, text)
    return loss
```

### T5 Span Corruption

```python
def t5_pretrain(text):
    # Replace random spans with sentinel tokens
    corrupted, targets = span_corrupt(text)
    # Input: "The <X> fox <Y> dog"
    # Target: "<X> quick brown <Y> jumps over the lazy </s>"

    encoder_output = encoder(corrupted)
    predictions = decoder(targets, encoder_output)
    loss = cross_entropy(predictions, targets)
    return loss
```

## Model Comparison

### BART vs T5

| Aspect | BART | T5 |
|--------|------|-----|
| Pre-training | Multiple noise types | Span corruption only |
| Position encoding | Learned absolute | Relative bias |
| Task format | Natural inputs | Text-to-text with prefixes |
| Normalization | Post-norm LayerNorm | Pre-norm RMSNorm-style |
| Best for | Summarization | Multi-task, instruction following |

### Encoder-Decoder vs Decoder-Only Comparison

| Metric | Encoder-Decoder | Decoder-Only |
|--------|-----------------|--------------|
| Compute per token | Higher (two stacks) | Lower |
| Memory efficiency | Cache encoder output | KV-cache grows |
| Input processing | Full bidirectional | Causal only |
| Generation quality | Better for conditioned gen | Better for open gen |
| In-context learning | Harder | Natural |
| Scaling | Less popular at scale | Dominant paradigm |

## Practical Usage

### Loading Models

```python
from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoTokenizer
)

# BART
bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# T5
t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
```

### Generation

```python
# Summarization with BART
article = "Long article text..."
inputs = bart_tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = bart.generate(
    inputs["input_ids"],
    max_length=150,
    num_beams=4,
    length_penalty=2.0
)
summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Translation with T5
text = "translate English to French: Hello, how are you?"
inputs = t5_tokenizer(text, return_tensors="pt")
output_ids = t5.generate(inputs["input_ids"], max_length=50)
translation = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### Fine-tuning

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
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

## Design Considerations

### When to Choose Encoder-Decoder

- Clear input-output separation
- Need to fully understand input before generating
- Summarization, translation, structured QA
- Smaller model with good conditioned generation

### When to Choose Decoder-Only

- General-purpose language modeling
- In-context learning / few-shot
- Dialogue and open-ended generation
- Scaling to very large models
- Unified training objective

## Further Reading

For detailed coverage of specific encoder-decoder models, see:

- [BART](bart/ReadMe.md) - Denoising pre-training for sequence-to-sequence
- [T5](t5/ReadMe.md) - Text-to-text unified framework
