# Transformer Models for Machine Translation

## Summary

Transformer-based machine translation replaced recurrent architectures with self-attention mechanisms, enabling full parallelization during training and better modeling of long-range dependencies. The encoder-decoder transformer architecture, introduced in "Attention is All You Need" (2017), remains the foundation of state-of-the-art translation systems. Modern developments include multilingual models, efficient variants, and scaling to hundreds of language pairs.

Key points to remember:

- Self-attention: Each position attends to all positions in the sequence
- Parallelization: No sequential dependency enables faster training
- Positional encoding: Injects sequence order information
- Multi-head attention: Multiple attention patterns learned in parallel
- Layer normalization: Critical for training stability
- Subword tokenization: BPE, SentencePiece handle open vocabulary
- Multilingual: Single model can translate many language pairs
- Pre-training: mBART, mT5 leverage large-scale unlabeled data

## Transformer Architecture

### Core Components

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        return self.W_o(context), attention


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Encoder and Decoder

```python
class TransformerEncoderLayer(nn.Module):
    """Single encoder layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with cross-attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention to encoder
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class TransformerMT(nn.Module):
    """Complete transformer for machine translation."""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.output_projection(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        return self.decode(tgt, encoder_output, src_mask, tgt_mask)
```

### Mask Creation

```python
def create_masks(src, tgt, pad_idx):
    """Create source and target masks."""
    # Source mask: hide padding
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target mask: hide padding and future tokens
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len), diagonal=1
    ).bool().to(tgt.device)
    tgt_causal_mask = ~tgt_causal_mask.unsqueeze(0).unsqueeze(0)

    tgt_mask = tgt_pad_mask & tgt_causal_mask

    return src_mask, tgt_mask
```

## Subword Tokenization

### Byte-Pair Encoding (BPE)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(texts, vocab_size=32000, save_path="tokenizer.json"):
    """Train BPE tokenizer."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)

    return tokenizer

# Usage
tokenizer = train_bpe_tokenizer(corpus)
encoded = tokenizer.encode("Hello world!")
print(encoded.tokens)  # ['Hell', 'o', 'world', '!']
```

### SentencePiece

```python
import sentencepiece as spm

def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    """Train SentencePiece model."""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

# Usage
sp = spm.SentencePieceProcessor()
sp.load('model.model')
tokens = sp.encode_as_pieces("Hello world!")
ids = sp.encode_as_ids("Hello world!")
```

### Shared Vocabulary

```python
class SharedEmbedding(nn.Module):
    """Share embeddings between source, target, and output."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    def project(self, x):
        """Project hidden states to vocabulary."""
        return torch.matmul(x, self.embedding.weight.t())
```

## Training

### Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: [batch, seq_len, vocab_size]
        # target: [batch, seq_len]

        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (self.vocab_size - 2))
            smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_labels[:, self.padding_idx] = 0

            # Mask padding
            mask = (target != self.padding_idx).float()

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss
```

### Learning Rate Schedule

```python
class TransformerLRScheduler:
    """Warm-up then inverse square root decay."""

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
```

### Training Loop

```python
def train_transformer(model, train_loader, val_loader, epochs, device):
    """Train transformer MT model."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    scheduler = TransformerLRScheduler(optimizer, model.d_model)
    criterion = LabelSmoothingLoss(model.tgt_vocab_size, pad_idx=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1], pad_idx=0)

            # Forward pass
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)

            # Compute loss
            loss = criterion(output, tgt[:, 1:])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
```

## Multilingual Translation

### mBART Architecture

```python
class mBARTForTranslation:
    """Multilingual BART for many-to-many translation."""

    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt"):
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=128
        )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Usage
translator = mBARTForTranslation()
translation = translator.translate(
    "Hello, how are you?",
    src_lang="en_XX",
    tgt_lang="de_DE"
)
```

### Language Tags

```python
class MultilingualMT(nn.Module):
    """Transformer with language tags."""

    def __init__(self, vocab_size, num_languages, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lang_embedding = nn.Embedding(num_languages, d_model)
        # ... rest of transformer

    def forward(self, src, tgt, src_lang_id, tgt_lang_id):
        # Add language embedding to source
        src_emb = self.token_embedding(src) + self.lang_embedding(src_lang_id)

        # Prepend target language tag
        tgt_emb = self.token_embedding(tgt) + self.lang_embedding(tgt_lang_id)

        # ... continue with transformer
```

## Efficient Transformers

### Knowledge Distillation

```python
class DistillationTrainer:
    """Distill large model to smaller one."""

    def __init__(self, teacher, student, temperature=2.0, alpha=0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def distill_loss(self, src, tgt, src_mask, tgt_mask):
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(src, tgt[:, :-1], src_mask, tgt_mask)
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)

        # Student predictions
        student_logits = self.student(src, tgt[:, :-1], src_mask, tgt_mask)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence loss
        distill_loss = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs),
            dim=-1
        ).mean()

        # Hard label loss
        hard_loss = nn.CrossEntropyLoss(ignore_index=0)(
            student_logits.reshape(-1, student_logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss
```

### Quantization

```python
import torch.quantization as quant

def quantize_model(model, calibration_data):
    """Quantize model for faster inference."""
    model.eval()

    # Prepare for quantization
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)

    # Calibrate
    for batch in calibration_data:
        model(batch['src'], batch['tgt'])

    # Convert to quantized model
    quant.convert(model, inplace=True)

    return model
```

## Decoding

### Autoregressive Generation

```python
def translate(model, src, src_mask, max_len, sos_idx, eos_idx, device):
    """Autoregressive translation."""
    model.eval()

    with torch.no_grad():
        # Encode source
        encoder_output = model.encode(src, src_mask)

        # Initialize target with SOS
        tgt = torch.tensor([[sos_idx]]).to(device)

        for _ in range(max_len):
            tgt_mask = create_causal_mask(tgt.size(1)).to(device)

            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

    return tgt[0].tolist()
```

### Speculative Decoding

```python
class SpeculativeDecoder:
    """Use small model to draft, large model to verify."""

    def __init__(self, large_model, small_model, k=4):
        self.large_model = large_model
        self.small_model = small_model
        self.k = k  # Number of tokens to draft

    def decode(self, src, src_mask, max_len, sos_idx, eos_idx):
        tgt = [sos_idx]

        while len(tgt) < max_len and tgt[-1] != eos_idx:
            # Draft k tokens with small model
            draft = self._draft(src, src_mask, tgt)

            # Verify with large model
            verified = self._verify(src, src_mask, tgt, draft)

            tgt.extend(verified)

        return tgt

    def _draft(self, src, src_mask, prefix):
        """Generate k draft tokens."""
        draft_tokens = []
        current = prefix.copy()

        for _ in range(self.k):
            output = self.small_model(src, torch.tensor([current]), src_mask)
            next_token = output[0, -1].argmax().item()
            draft_tokens.append(next_token)
            current.append(next_token)

        return draft_tokens

    def _verify(self, src, src_mask, prefix, draft):
        """Verify draft tokens with large model."""
        full_seq = prefix + draft
        output = self.large_model(src, torch.tensor([full_seq]), src_mask)

        verified = []
        for i, draft_token in enumerate(draft):
            pos = len(prefix) + i - 1
            if output[0, pos].argmax().item() == draft_token:
                verified.append(draft_token)
            else:
                # Accept up to this point, resample this position
                verified.append(output[0, pos].argmax().item())
                break

        return verified
```

## Evaluation

### BLEU and COMET

```python
from sacrebleu import corpus_bleu

def evaluate_translation(predictions, references):
    """Comprehensive MT evaluation."""

    # BLEU score
    bleu = corpus_bleu(predictions, [references])

    # chrF score (character-level F-score)
    from sacrebleu import corpus_chrf
    chrf = corpus_chrf(predictions, [references])

    return {
        'bleu': bleu.score,
        'chrf': chrf.score
    }

# COMET (neural metric)
def evaluate_with_comet(sources, predictions, references):
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]

    output = model.predict(data, batch_size=8)
    return output.system_score
```

## Production Deployment

### Batch Translation

```python
class BatchTranslator:
    """Efficient batch translation."""

    def __init__(self, model, tokenizer, device, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def translate_batch(self, texts, max_length=128):
        self.model.eval()
        translations = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4
                )

            # Decode
            batch_translations = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            translations.extend(batch_translations)

        return translations
```

## Benchmark Datasets

| Dataset | Languages | Size | Domain |
|---------|-----------|------|--------|
| WMT | Many pairs | Millions | News |
| OPUS | 100+ languages | Varies | Mixed |
| Europarl | EU languages | 2M sentences | Legal |
| ParaCrawl | Many pairs | Billions | Web |
| FLORES | 200 languages | 2K sentences | Evaluation |

## Best Practices

### Hyperparameters

| Parameter | Base | Large |
|-----------|------|-------|
| d_model | 512 | 1024 |
| num_heads | 8 | 16 |
| num_layers | 6 | 12 |
| d_ff | 2048 | 4096 |
| dropout | 0.1 | 0.1 |
| warmup_steps | 4000 | 8000 |

### Tips

1. Use shared BPE vocabulary for similar languages
2. Apply dropout to attention weights and residual connections
3. Initialize with Xavier/Glorot for linear layers
4. Use gradient checkpointing for large models
5. Monitor attention entropy for training health
6. Apply length penalty during beam search (0.6-1.0)
