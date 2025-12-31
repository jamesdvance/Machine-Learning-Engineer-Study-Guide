# Sequence-to-Sequence Models for Machine Translation

## Summary

Sequence-to-sequence (Seq2Seq) models revolutionized machine translation by enabling end-to-end learning from source to target language. The encoder-decoder architecture processes the entire source sentence into a fixed representation, which the decoder then converts into the target language. Attention mechanisms, introduced to address the information bottleneck, allow the decoder to focus on relevant source positions during generation.

Key points to remember:

- Encoder-decoder: Encoder compresses source, decoder generates target
- Information bottleneck: Fixed-size representation limits long sequences
- Attention mechanism: Allows decoder to attend to all encoder states
- Teacher forcing: Training uses ground truth tokens as decoder input
- Beam search: Decoding strategy explores multiple translation hypotheses
- Exposure bias: Discrepancy between training (teacher forcing) and inference
- Bidirectional encoder: Captures context from both directions
- Historical importance: Foundational architecture that led to transformers

## Architecture

### Basic Encoder-Decoder

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """LSTM encoder for source sequence."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))

        # Pack for variable length
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )

        outputs, (hidden, cell) = self.lstm(packed)

        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Combine bidirectional states
        # hidden: [num_layers * 2, batch, hidden_dim]
        hidden = self._combine_bidir(hidden)
        cell = self._combine_bidir(cell)

        return outputs, hidden, cell

    def _combine_bidir(self, state):
        """Combine forward and backward states."""
        # state: [num_layers * 2, batch, hidden_dim]
        num_layers = state.size(0) // 2
        batch = state.size(1)
        hidden_dim = state.size(2)

        # Reshape and concatenate
        state = state.view(num_layers, 2, batch, hidden_dim)
        state = torch.cat([state[:, 0], state[:, 1]], dim=-1)

        return state


class Decoder(nn.Module):
    """LSTM decoder with attention."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Hidden dim is doubled due to bidirectional encoder
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,  # Input: embedding + context
            hidden_dim * 2,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attention = Attention(hidden_dim * 2)
        self.output_projection = nn.Linear(hidden_dim * 4, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_outputs, hidden, cell, src_mask):
        # trg: [batch, 1] (single token for step-by-step decoding)
        embedded = self.dropout(self.embedding(trg))

        # Compute attention
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, src_mask
        )

        # Combine embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Combine output and context for prediction
        combined = torch.cat([output.squeeze(1), context], dim=-1)
        prediction = self.output_projection(combined)

        return prediction, hidden, cell, attention_weights
```

### Attention Mechanism

```python
class Attention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask=None):
        # query: [batch, hidden_dim] (decoder state)
        # keys: [batch, src_len, hidden_dim] (encoder outputs)

        # Project query and keys
        query_proj = self.W_query(query).unsqueeze(1)  # [batch, 1, hidden_dim]
        keys_proj = self.W_key(keys)  # [batch, src_len, hidden_dim]

        # Compute attention scores
        scores = self.v(torch.tanh(query_proj + keys_proj))  # [batch, src_len, 1]
        scores = scores.squeeze(-1)  # [batch, src_len]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """Luong-style multiplicative attention."""

    def __init__(self, hidden_dim, method='dot'):
        super().__init__()
        self.method = method

        if method == 'general':
            self.W = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.W = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask=None):
        if self.method == 'dot':
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        elif self.method == 'general':
            scores = torch.bmm(query.unsqueeze(1), self.W(keys).transpose(1, 2))
        elif self.method == 'concat':
            query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)
            concat = torch.cat([query_expanded, keys], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).transpose(1, 2)

        scores = scores.squeeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context, attention_weights
```

### Complete Seq2Seq Model

```python
class Seq2Seq(nn.Module):
    """Complete sequence-to-sequence model."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode source
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # Create source mask
        src_mask = (src != 0).float()

        # First decoder input is <sos> token
        decoder_input = trg[:, 0:1]

        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(
                decoder_input, encoder_outputs, hidden, cell, src_mask
            )

            outputs[:, t] = prediction

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing:
                decoder_input = trg[:, t:t+1]
            else:
                decoder_input = prediction.argmax(dim=-1, keepdim=True)

        return outputs
```

## Training

### Loss and Optimization

```python
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src, src_lengths, trg = batch

        optimizer.zero_grad()

        # Forward pass
        output = model(src, src_lengths, trg)

        # Reshape for loss computation
        output = output[:, 1:].reshape(-1, output.size(-1))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Training setup
model = Seq2Seq(encoder, decoder, device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Learning rate scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)
```

### Teacher Forcing Schedule

```python
class TeacherForcingScheduler:
    """Gradually reduce teacher forcing ratio."""

    def __init__(self, initial_ratio=1.0, min_ratio=0.0, decay_rate=0.05):
        self.ratio = initial_ratio
        self.min_ratio = min_ratio
        self.decay_rate = decay_rate

    def step(self):
        self.ratio = max(self.min_ratio, self.ratio - self.decay_rate)
        return self.ratio

# Usage
tf_scheduler = TeacherForcingScheduler()
for epoch in range(num_epochs):
    tf_ratio = tf_scheduler.step()
    train_epoch(model, dataloader, optimizer, criterion, tf_ratio=tf_ratio)
```

## Decoding Strategies

### Greedy Decoding

```python
def greedy_decode(model, src, src_lengths, max_len, sos_idx, eos_idx):
    """Decode by selecting highest probability token at each step."""
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
        src_mask = (src != 0).float()

        decoder_input = torch.tensor([[sos_idx]]).to(src.device)
        output_tokens = [sos_idx]

        for _ in range(max_len):
            prediction, hidden, cell, _ = model.decoder(
                decoder_input, encoder_outputs, hidden, cell, src_mask
            )

            next_token = prediction.argmax(dim=-1).item()
            output_tokens.append(next_token)

            if next_token == eos_idx:
                break

            decoder_input = torch.tensor([[next_token]]).to(src.device)

    return output_tokens
```

### Beam Search

```python
class BeamSearchDecoder:
    """Beam search for improved translation quality."""

    def __init__(self, model, beam_size=5, max_len=100, length_penalty=0.6):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty

    def decode(self, src, src_lengths, sos_idx, eos_idx):
        self.model.eval()

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
            src_mask = (src != 0).float()

            # Initialize beams
            # Each beam: (score, tokens, hidden, cell)
            beams = [(0.0, [sos_idx], hidden, cell)]

            for step in range(self.max_len):
                all_candidates = []

                for score, tokens, hidden, cell in beams:
                    if tokens[-1] == eos_idx:
                        all_candidates.append((score, tokens, hidden, cell))
                        continue

                    decoder_input = torch.tensor([[tokens[-1]]]).to(src.device)
                    prediction, new_hidden, new_cell, _ = self.model.decoder(
                        decoder_input, encoder_outputs, hidden, cell, src_mask
                    )

                    log_probs = torch.log_softmax(prediction, dim=-1)
                    top_probs, top_indices = log_probs.topk(self.beam_size)

                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        new_score = score + prob.item()
                        new_tokens = tokens + [idx.item()]
                        all_candidates.append((new_score, new_tokens, new_hidden, new_cell))

                # Select top beams with length normalization
                all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** self.length_penalty), reverse=True)
                beams = all_candidates[:self.beam_size]

                # Check if all beams ended
                if all(beam[1][-1] == eos_idx for beam in beams):
                    break

            # Return best beam
            best_beam = max(beams, key=lambda x: x[0] / (len(x[1]) ** self.length_penalty))
            return best_beam[1]
```

### Sampling Strategies

```python
def sample_decode(model, src, src_lengths, max_len, sos_idx, eos_idx,
                  temperature=1.0, top_k=0, top_p=0.9):
    """Decode with sampling for diversity."""
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
        src_mask = (src != 0).float()

        decoder_input = torch.tensor([[sos_idx]]).to(src.device)
        output_tokens = [sos_idx]

        for _ in range(max_len):
            prediction, hidden, cell, _ = model.decoder(
                decoder_input, encoder_outputs, hidden, cell, src_mask
            )

            # Apply temperature
            logits = prediction / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            output_tokens.append(next_token)

            if next_token == eos_idx:
                break

            decoder_input = torch.tensor([[next_token]]).to(src.device)

    return output_tokens
```

## Evaluation

### BLEU Score

```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

def compute_bleu(predictions, references):
    """
    Compute corpus-level BLEU score.

    Args:
        predictions: List of predicted translations (tokenized)
        references: List of reference translations (list of tokenized refs per prediction)
    """
    # Format for corpus_bleu: references should be list of lists
    if not isinstance(references[0][0], list):
        references = [[ref] for ref in references]

    bleu_score = corpus_bleu(references, predictions)
    return bleu_score

def compute_sentence_bleu(prediction, reference):
    """Compute sentence-level BLEU with smoothing."""
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference], prediction, smoothing_function=smoothing)
```

### Other Metrics

```python
def compute_meteor(predictions, references):
    """METEOR score considers synonyms and paraphrases."""
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)

def compute_ter(prediction, reference):
    """Translation Edit Rate."""
    import editdistance
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    edit_dist = editdistance.eval(pred_tokens, ref_tokens)
    ter = edit_dist / len(ref_tokens) if ref_tokens else 0

    return ter
```

## Challenges and Solutions

### Information Bottleneck

```python
# Problem: Fixed-size encoder state limits information
# Solution 1: Attention (allows access to all encoder states)
# Solution 2: Increase hidden dimension
# Solution 3: Use multiple layers with residual connections

class ResidualLSTM(nn.Module):
    """LSTM with residual connections."""

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LSTM(input_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
            for i in range(num_layers)
        ])
        self.projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = self.projection(x) if i == 0 and self.projection else x
            x, _ = layer(x)
            x = x + residual
        return x
```

### Exposure Bias

```python
class ScheduledSampling:
    """
    Address exposure bias by mixing teacher forcing with model predictions.
    """

    def __init__(self, initial_ratio=1.0, decay_steps=10000, min_ratio=0.1):
        self.initial_ratio = initial_ratio
        self.decay_steps = decay_steps
        self.min_ratio = min_ratio
        self.step_count = 0

    def get_ratio(self):
        ratio = self.initial_ratio - (self.initial_ratio - self.min_ratio) * (
            self.step_count / self.decay_steps
        )
        return max(ratio, self.min_ratio)

    def step(self):
        self.step_count += 1
```

### Long Sequences

```python
class ChunkedEncoder(nn.Module):
    """Process long sequences in chunks."""

    def __init__(self, base_encoder, chunk_size=512, overlap=50):
        super().__init__()
        self.encoder = base_encoder
        self.chunk_size = chunk_size
        self.overlap = overlap

    def forward(self, src, src_lengths):
        if src.size(1) <= self.chunk_size:
            return self.encoder(src, src_lengths)

        # Process in chunks
        chunks = []
        for i in range(0, src.size(1), self.chunk_size - self.overlap):
            chunk = src[:, i:i + self.chunk_size]
            chunk_len = torch.clamp(src_lengths - i, min=1, max=self.chunk_size)
            chunk_out, _, _ = self.encoder(chunk, chunk_len)
            chunks.append(chunk_out)

        # Combine chunks (averaging overlapping regions)
        return self._combine_chunks(chunks)
```

## Historical Context

### Evolution of Seq2Seq

```
2014: Sutskever et al. - Basic Encoder-Decoder with LSTM
         Reversed source sequence for better gradient flow

2015: Bahdanau et al. - Attention Mechanism
         Addressed information bottleneck

2015: Luong et al. - Attention Variants
         Dot product, general, concat attention

2016: Google Neural MT - Production-scale Seq2Seq
         Residual connections, wordpiece tokenization

2017: Transformer - "Attention is All You Need"
         Self-attention replaced recurrence
```

### Comparison with Transformers

| Aspect | Seq2Seq (LSTM) | Transformer |
|--------|---------------|-------------|
| Parallelization | Sequential | Fully parallel |
| Long-range deps | Difficult | Native |
| Training speed | Slower | Faster |
| Memory | Lower | Higher (attention) |
| Interpretability | Attention patterns | Attention patterns |
| State-of-the-art | Historical | Current |

## Best Practices

### Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Hidden dim | 256-1024 | Match encoder/decoder |
| Layers | 2-6 | More for complex tasks |
| Dropout | 0.2-0.5 | Higher for more data |
| Learning rate | 1e-4 to 1e-3 | With scheduler |
| Beam size | 4-10 | Larger isn't always better |
| Length penalty | 0.6-1.0 | Tune per language pair |

### Tips

1. Reverse source sequence (helps gradient flow in vanilla seq2seq)
2. Use bidirectional encoder for attention-based models
3. Initialize embeddings with pre-trained vectors
4. Apply dropout consistently (embedding, LSTM, attention)
5. Use gradient clipping (typically 1.0-5.0)
6. Monitor attention patterns for debugging
