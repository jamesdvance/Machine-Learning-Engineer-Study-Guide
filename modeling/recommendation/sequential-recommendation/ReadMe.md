# Sequential Recommendation

## Summary

Sequential recommendation models capture the temporal dynamics and order dependencies in user behavior to predict the next item a user will interact with. Unlike traditional collaborative filtering which treats user history as an unordered set, sequential models explicitly model the sequence of interactions, recognizing that what a user does next often depends on what they did recently. Modern approaches use transformer architectures to capture complex sequential patterns at scale.

Key points to remember:

- Models order and temporal patterns in user interaction sequences
- Predicts next item based on recent interaction history
- Session-based: models anonymous sessions (e.g., e-commerce browsing)
- Sequence-based: models long-term user history
- Markov chains: simple first-order dependencies
- RNNs (GRU4Rec): capture longer dependencies
- Transformers (SASRec, BERT4Rec): state-of-the-art via self-attention
- Causal masking: only attend to past items (autoregressive)
- Bidirectional masking: BERT-style training with masked items
- Compared to CF: captures dynamics, needs sequence data, more complex

## Problem Formulation

### Next-Item Prediction

```
Given: User's interaction sequence [i1, i2, i3, ..., it]
Predict: Next item i(t+1)

Example:
User browsed: [iPhone case, AirPods, MacBook charger, iPad]
Predict: What will they view/buy next?
```

### Session vs User Sequences

**Session-based:**
- Anonymous users (no user ID)
- Short sequences (single session)
- Cold start for every session
- Focus on immediate intent

**User-based:**
- Identified users with history
- Long sequences (months/years)
- Can leverage long-term preferences
- Balance recency with history

## Classical Approaches

### Markov Chains

First-order Markov: next item depends only on current item:

```python
from collections import defaultdict
import numpy as np

class MarkovChainRecommender:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.item_counts = defaultdict(int)

    def fit(self, sequences):
        """
        sequences: list of item sequences
        """
        for seq in sequences:
            for i in range(len(seq) - 1):
                current = seq[i]
                next_item = seq[i + 1]
                self.transitions[current][next_item] += 1
                self.item_counts[current] += 1

    def predict_next(self, current_item, n=10):
        """Predict next items given current item."""
        if current_item not in self.transitions:
            return []

        candidates = self.transitions[current_item]
        total = self.item_counts[current_item]

        # Sort by transition probability
        probs = [(item, count / total) for item, count in candidates.items()]
        probs.sort(key=lambda x: x[1], reverse=True)

        return probs[:n]

    def predict_from_sequence(self, sequence, n=10):
        """Use last item in sequence."""
        if not sequence:
            return []
        return self.predict_next(sequence[-1], n)
```

### Higher-Order Markov

Consider last k items:

```python
class HigherOrderMarkov:
    def __init__(self, order=3):
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(int))

    def fit(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - 1):
                # Use last 'order' items as context
                start = max(0, i - self.order + 1)
                context = tuple(seq[start:i + 1])
                next_item = seq[i + 1]
                self.transitions[context][next_item] += 1

    def predict(self, sequence, n=10):
        # Try decreasing order until match found
        for k in range(min(self.order, len(sequence)), 0, -1):
            context = tuple(sequence[-k:])
            if context in self.transitions:
                candidates = self.transitions[context]
                sorted_items = sorted(candidates.items(),
                                     key=lambda x: x[1], reverse=True)
                return sorted_items[:n]
        return []
```

## RNN-Based Models

### GRU4Rec

The pioneering deep learning approach for session-based recommendation:

```python
import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, n_items, embedding_dim=64, hidden_dim=128, n_layers=1):
        super().__init__()

        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        )
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, sequences, lengths):
        """
        sequences: [batch_size, max_seq_len] item IDs
        lengths: [batch_size] actual sequence lengths
        """
        # Embed items
        x = self.item_embedding(sequences)  # [batch, seq_len, emb_dim]

        # Pack for variable length
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # GRU forward
        _, hidden = self.gru(packed)  # hidden: [n_layers, batch, hidden_dim]

        # Use last layer's hidden state
        output = self.output(hidden[-1])  # [batch, n_items]

        return output

    def predict_next(self, sequence):
        """Predict next item for a single sequence."""
        self.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor([sequence])
            length = torch.tensor([len(sequence)])
            logits = self.forward(seq_tensor, length)
            return torch.softmax(logits, dim=-1).squeeze()
```

### Training GRU4Rec

```python
def train_gru4rec(model, train_loader, optimizer, n_epochs=10):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            sequences = batch['sequence']      # [batch, seq_len]
            targets = batch['target']          # [batch] next item IDs
            lengths = batch['length']          # [batch] sequence lengths

            optimizer.zero_grad()

            logits = model(sequences, lengths)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
```

### BPR Loss for GRU4Rec

Original paper uses BPR (Bayesian Personalized Ranking):

```python
def bpr_loss(positive_scores, negative_scores):
    """
    positive_scores: scores for ground truth next items
    negative_scores: scores for sampled negative items
    """
    return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))

def train_step_bpr(model, sequences, lengths, positive_items, negative_items):
    logits = model(sequences, lengths)  # [batch, n_items]

    pos_scores = logits.gather(1, positive_items.unsqueeze(1)).squeeze()
    neg_scores = logits.gather(1, negative_items.unsqueeze(1)).squeeze()

    return bpr_loss(pos_scores, neg_scores)
```

## Transformer-Based Models

### SASRec (Self-Attentive Sequential Recommendation)

Causal (left-to-right) attention for next-item prediction:

```python
class SASRec(nn.Module):
    def __init__(self, n_items, max_seq_len, embedding_dim=64,
                 n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()

        self.n_items = n_items
        self.max_seq_len = max_seq_len

        # Embeddings
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            SASRecBlock(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        sequences: [batch_size, seq_len] item IDs (0 = padding)
        """
        seq_len = sequences.shape[1]

        # Create causal mask
        mask = self._create_causal_mask(seq_len, sequences.device)

        # Padding mask
        padding_mask = (sequences == 0)

        # Embed items + positions
        positions = torch.arange(seq_len, device=sequences.device)
        x = self.item_embedding(sequences) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        x = self.layer_norm(x)

        return x

    def _create_causal_mask(self, seq_len, device):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def predict(self, sequences):
        """Predict next item scores."""
        x = self.forward(sequences)  # [batch, seq_len, dim]

        # Use last position's representation
        last_hidden = x[:, -1, :]  # [batch, dim]

        # Score all items
        item_embeddings = self.item_embedding.weight[1:]  # Exclude padding
        scores = torch.matmul(last_hidden, item_embeddings.T)

        return scores


class SASRecBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, padding_mask):
        # Self-attention with causal masking
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
```

### BERT4Rec

Bidirectional training with masked item prediction:

```python
class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_seq_len, embedding_dim=64,
                 n_heads=2, n_layers=2, dropout=0.2, mask_prob=0.2):
        super().__init__()

        self.n_items = n_items
        self.mask_token = n_items + 1  # Special mask token
        self.mask_prob = mask_prob

        # Embeddings (include mask token)
        self.item_embedding = nn.Embedding(n_items + 2, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.output = nn.Linear(embedding_dim, n_items + 1)  # Predict items (not mask)

    def forward(self, sequences, padding_mask=None):
        seq_len = sequences.shape[1]

        # Embed items + positions
        positions = torch.arange(seq_len, device=sequences.device)
        x = self.item_embedding(sequences) + self.position_embedding(positions)

        # Transformer (bidirectional - no causal mask)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Predict items
        logits = self.output(x)

        return logits

    def mask_sequence(self, sequences):
        """Apply random masking for training."""
        masked = sequences.clone()
        labels = torch.full_like(sequences, -100)  # Ignore non-masked

        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                if sequences[i, j] == 0:  # Skip padding
                    continue
                if torch.rand(1) < self.mask_prob:
                    labels[i, j] = sequences[i, j]
                    masked[i, j] = self.mask_token

        return masked, labels


def train_bert4rec(model, train_loader, optimizer, n_epochs=10):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            sequences = batch['sequence']
            padding_mask = (sequences == 0)

            # Mask sequences
            masked_seq, labels = model.mask_sequence(sequences)

            optimizer.zero_grad()

            logits = model(masked_seq, padding_mask)

            # Flatten for loss
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
```

## Data Preparation

### Sequence Dataset

```python
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, max_len=50, mode='train'):
        """
        sequences: list of item ID sequences per user
        max_len: maximum sequence length
        mode: 'train' or 'test'
        """
        self.max_len = max_len
        self.mode = mode
        self.data = []

        for seq in sequences:
            if len(seq) < 2:
                continue

            if mode == 'train':
                # Create training examples from subsequences
                for i in range(1, len(seq)):
                    input_seq = seq[max(0, i - max_len):i]
                    target = seq[i]
                    self.data.append((input_seq, target))
            else:
                # Test: predict last item from history
                input_seq = seq[:-1][-max_len:]
                target = seq[-1]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]

        # Pad sequence
        padded = [0] * (self.max_len - len(input_seq)) + input_seq

        return {
            'sequence': torch.tensor(padded),
            'target': torch.tensor(target),
            'length': torch.tensor(len(input_seq))
        }
```

### Session Data Processing

```python
def create_sessions(interactions, session_gap_minutes=30):
    """
    Split user interactions into sessions based on time gaps.

    interactions: DataFrame with columns [user_id, item_id, timestamp]
    """
    sessions = []

    for user_id, group in interactions.groupby('user_id'):
        group = group.sort_values('timestamp')

        session = []
        last_time = None

        for _, row in group.iterrows():
            if last_time is not None:
                gap = (row['timestamp'] - last_time).total_seconds() / 60
                if gap > session_gap_minutes:
                    if len(session) >= 2:
                        sessions.append(session)
                    session = []

            session.append(row['item_id'])
            last_time = row['timestamp']

        if len(session) >= 2:
            sessions.append(session)

    return sessions
```

## Evaluation

### Metrics

```python
def evaluate_sequential(model, test_data, k_values=[5, 10, 20]):
    """
    test_data: list of (input_sequence, target_item)
    """
    model.eval()
    metrics = {f'hr@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics['mrr'] = []

    with torch.no_grad():
        for input_seq, target in test_data:
            # Get predictions
            scores = model.predict_next(input_seq)

            # Rank items
            ranked = torch.argsort(scores, descending=True)

            # Find target rank
            target_rank = (ranked == target).nonzero(as_tuple=True)[0]
            if len(target_rank) > 0:
                rank = target_rank[0].item() + 1
            else:
                rank = len(ranked) + 1

            # MRR
            metrics['mrr'].append(1.0 / rank)

            # HR and NDCG at k
            for k in k_values:
                if rank <= k:
                    metrics[f'hr@{k}'].append(1.0)
                    metrics[f'ndcg@{k}'].append(1.0 / np.log2(rank + 1))
                else:
                    metrics[f'hr@{k}'].append(0.0)
                    metrics[f'ndcg@{k}'].append(0.0)

    return {k: np.mean(v) for k, v in metrics.items()}
```

## Comparison of Approaches

| Model | Temporal | Long-Range | Complexity | Best For |
|-------|----------|------------|------------|----------|
| Markov Chain | Yes | No (1st order) | O(1) | Simple patterns |
| GRU4Rec | Yes | Moderate | O(n) | Sessions |
| SASRec | Yes | Long | O(n^2) | User sequences |
| BERT4Rec | Bidirectional | Long | O(n^2) | Rich context |

## When to Use Sequential Models

Sequential recommendation is well-suited for:
- Session-based scenarios (e-commerce, streaming)
- When order matters (browsing paths, playlists)
- Short-term intent prediction
- Next-item prediction tasks

Consider alternatives when:
- Order doesn't matter (static preferences)
- Very long histories (use sampling/summarization)
- Real-time latency critical (simpler models)
- No temporal patterns in data

## Further Reading

For specific implementations:
- [Transformers4Rec](transformers4rec/ReadMe.md) - Production transformer library
- [SASRec](sasrec/ReadMe.md) - Self-attentive sequential recommendation
