# SASRec (Self-Attentive Sequential Recommendation)

## Summary

SASRec applies the Transformer's self-attention mechanism to sequential recommendation, using causal (left-to-right) masking to predict the next item based on a user's interaction history. By replacing RNNs with self-attention, SASRec captures long-range dependencies more effectively and enables parallel training. The model has become a foundational architecture for sequential recommendation, consistently outperforming RNN-based approaches while being simpler to implement.

Key points to remember:

- Transformer encoder with causal masking for next-item prediction
- Self-attention captures item-item relationships in sequence
- Learnable position embeddings encode sequence order
- Causal mask ensures each position only attends to previous items
- Shared item embeddings for input and output (tied weights)
- Binary cross-entropy loss with negative sampling
- Outperforms GRU4Rec and Caser on standard benchmarks
- O(n^2) attention complexity limits very long sequences
- Compared to BERT4Rec: unidirectional, simpler, better for generation

## Architecture

### Model Structure

```
Input Sequence: [item1, item2, item3, item4, item5]
                   |      |      |      |      |
                   v      v      v      v      v
            +------------------------------------------+
            |        Item Embeddings (E)               |
            +------------------------------------------+
                   |      |      |      |      |
                   +      +      +      +      +
            +------------------------------------------+
            |      Position Embeddings (P)             |
            +------------------------------------------+
                   |      |      |      |      |
                   v      v      v      v      v
            +------------------------------------------+
            |    Self-Attention Block 1                |
            |    (with causal mask)                    |
            +------------------------------------------+
                   |      |      |      |      |
                   v      v      v      v      v
            +------------------------------------------+
            |    Self-Attention Block 2                |
            +------------------------------------------+
                   |      |      |      |      |
                   v      v      v      v      v
            [h1]  [h2]  [h3]  [h4]  [h5]
                                          |
                                          v
                                    Predict item6
```

### Causal Attention Mask

```
Attention mask for sequence length 5:

     i1  i2  i3  i4  i5
i1 [  1   0   0   0   0 ]  <- i1 sees only itself
i2 [  1   1   0   0   0 ]  <- i2 sees i1, i2
i3 [  1   1   1   0   0 ]  <- i3 sees i1, i2, i3
i4 [  1   1   1   1   0 ]  <- i4 sees i1-i4
i5 [  1   1   1   1   1 ]  <- i5 sees all

0 = masked (cannot attend)
1 = visible (can attend)
```

## Implementation

### Complete SASRec Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SASRec(nn.Module):
    def __init__(
        self,
        n_items,
        max_seq_len=50,
        embedding_dim=64,
        n_heads=2,
        n_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Item embedding (padding_idx=0)
        self.item_embedding = nn.Embedding(
            n_items + 1,  # +1 for padding
            embedding_dim,
            padding_idx=0
        )

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.attention_layers = nn.ModuleList([
            SASRecLayer(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sequences):
        """
        sequences: [batch_size, seq_len] - item IDs (0 = padding)

        Returns:
            hidden: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device

        # Create masks
        attention_mask = self._create_attention_mask(seq_len, device)
        padding_mask = (sequences == 0)  # [batch, seq_len]

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.item_embedding(sequences) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        # Apply padding mask to embeddings
        x = x * (~padding_mask).unsqueeze(-1).float()

        # Transformer layers
        for layer in self.attention_layers:
            x = layer(x, attention_mask, padding_mask)

        x = self.final_layer_norm(x)

        return x

    def _create_attention_mask(self, seq_len, device):
        """Create causal attention mask."""
        # Upper triangular = future positions to mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        return mask

    def predict(self, sequences, candidates=None):
        """
        Predict scores for next items.

        sequences: [batch_size, seq_len]
        candidates: [batch_size, n_candidates] or None for all items

        Returns:
            scores: [batch_size, n_items] or [batch_size, n_candidates]
        """
        hidden = self.forward(sequences)

        # Use last non-padding position
        # Find last non-zero position for each sequence
        seq_lens = (sequences != 0).sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lens]  # [batch, dim]

        if candidates is None:
            # Score all items (excluding padding token)
            item_embeddings = self.item_embedding.weight[1:]  # [n_items, dim]
            scores = torch.matmul(last_hidden, item_embeddings.T)
        else:
            # Score only candidates
            candidate_embeddings = self.item_embedding(candidates)
            scores = torch.bmm(
                candidate_embeddings,
                last_hidden.unsqueeze(-1)
            ).squeeze(-1)

        return scores


class SASRecLayer(nn.Module):
    """Single SASRec transformer layer."""

    def __init__(self, embedding_dim, n_heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask, padding_mask):
        """
        x: [batch, seq_len, dim]
        attention_mask: [seq_len, seq_len] causal mask
        padding_mask: [batch, seq_len] padding positions
        """
        # Self-attention with residual
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x
```

## Training

### Binary Cross-Entropy with Negative Sampling

```python
class SASRecTrainer:
    def __init__(self, model, n_items, lr=0.001, n_negatives=1):
        self.model = model
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            sequences = batch['sequence']      # [batch, seq_len]
            positives = batch['positive']      # [batch, seq_len] - next items
            padding_mask = (sequences == 0)

            self.optimizer.zero_grad()

            # Forward pass
            hidden = self.model(sequences)  # [batch, seq_len, dim]

            # Sample negatives
            negatives = self._sample_negatives(
                positives, self.n_negatives
            )  # [batch, seq_len, n_neg]

            # Compute loss
            loss = self._compute_loss(hidden, positives, negatives, padding_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _sample_negatives(self, positives, n_negatives):
        """Sample random negative items."""
        batch_size, seq_len = positives.shape

        negatives = torch.randint(
            1, self.n_items + 1,  # Exclude padding (0)
            (batch_size, seq_len, n_negatives),
            device=positives.device
        )

        return negatives

    def _compute_loss(self, hidden, positives, negatives, padding_mask):
        """
        Binary cross-entropy loss.

        hidden: [batch, seq_len, dim]
        positives: [batch, seq_len]
        negatives: [batch, seq_len, n_neg]
        """
        # Positive scores
        pos_embeddings = self.model.item_embedding(positives)
        pos_scores = (hidden * pos_embeddings).sum(dim=-1)  # [batch, seq_len]

        # Negative scores
        neg_embeddings = self.model.item_embedding(negatives)
        neg_scores = torch.bmm(
            neg_embeddings.view(-1, negatives.shape[-1], hidden.shape[-1]),
            hidden.view(-1, hidden.shape[-1], 1)
        ).squeeze(-1).view(hidden.shape[0], hidden.shape[1], -1)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.ones_like(pos_scores),
            reduction='none'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.zeros_like(neg_scores),
            reduction='none'
        ).mean(dim=-1)

        # Mask padding positions
        mask = ~padding_mask
        loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum()

        return loss
```

### Data Preparation

```python
class SASRecDataset(torch.utils.data.Dataset):
    def __init__(self, user_sequences, max_len=50):
        """
        user_sequences: dict mapping user_id to list of item IDs
        """
        self.max_len = max_len
        self.data = []

        for user_id, items in user_sequences.items():
            if len(items) < 2:
                continue

            # Input: all items except last
            # Target: shifted by 1 (next item at each position)
            input_seq = items[:-1]
            target_seq = items[1:]

            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]

        # Truncate to max_len
        input_seq = input_seq[-self.max_len:]
        target_seq = target_seq[-self.max_len:]

        # Pad from left
        pad_len = self.max_len - len(input_seq)
        input_padded = [0] * pad_len + input_seq
        target_padded = [0] * pad_len + target_seq

        return {
            'sequence': torch.tensor(input_padded),
            'positive': torch.tensor(target_padded)
        }
```

## Inference

### Next-Item Prediction

```python
def predict_next_items(model, sequence, k=10, exclude_seen=True):
    """
    Predict top-k next items for a sequence.

    sequence: list of item IDs
    k: number of items to return
    exclude_seen: whether to exclude items in sequence
    """
    model.eval()

    with torch.no_grad():
        # Prepare input
        seq_tensor = torch.tensor([sequence])
        scores = model.predict(seq_tensor)  # [1, n_items]
        scores = scores.squeeze(0)

        # Exclude seen items
        if exclude_seen:
            for item in sequence:
                if item > 0:  # Not padding
                    scores[item - 1] = float('-inf')  # -1 because no padding in scores

        # Get top-k
        top_scores, top_indices = torch.topk(scores, k)

        # Convert to item IDs (add 1 for padding offset)
        top_items = (top_indices + 1).tolist()
        top_scores = top_scores.tolist()

        return list(zip(top_items, top_scores))
```

### Batch Inference

```python
def batch_predict(model, sequences, k=10):
    """
    Batch prediction for multiple sequences.
    """
    model.eval()

    # Pad sequences to same length
    max_len = max(len(s) for s in sequences)
    padded = []
    for seq in sequences:
        padded.append([0] * (max_len - len(seq)) + seq)

    with torch.no_grad():
        seq_tensor = torch.tensor(padded)
        scores = model.predict(seq_tensor)  # [batch, n_items]

        # Get top-k for each
        all_results = []
        for i, seq in enumerate(sequences):
            seq_scores = scores[i].clone()

            # Exclude seen
            for item in seq:
                if item > 0:
                    seq_scores[item - 1] = float('-inf')

            top_scores, top_indices = torch.topk(seq_scores, k)
            results = [(idx.item() + 1, score.item())
                      for idx, score in zip(top_indices, top_scores)]
            all_results.append(results)

        return all_results
```

## Hyperparameters

### Recommended Settings

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| embedding_dim | 32-128 | 64 often sufficient |
| n_layers | 1-3 | 2 usually best |
| n_heads | 1-4 | 2 common |
| dropout | 0.1-0.5 | 0.2-0.3 typical |
| max_seq_len | 20-200 | Depends on domain |
| learning_rate | 1e-4 to 1e-3 | With Adam |
| batch_size | 64-256 | Larger for in-batch negatives |

### Tuning Tips

```python
# Hyperparameter search
param_grid = {
    'embedding_dim': [32, 64, 128],
    'n_layers': [1, 2, 3],
    'n_heads': [1, 2, 4],
    'dropout': [0.1, 0.2, 0.3]
}

best_ndcg = 0
for params in product_dict(param_grid):
    model = SASRec(n_items, **params)
    trainer = SASRecTrainer(model, n_items)

    for epoch in range(20):
        trainer.train_epoch(train_loader)

    metrics = evaluate(model, test_data)
    if metrics['ndcg@10'] > best_ndcg:
        best_ndcg = metrics['ndcg@10']
        best_params = params
```

## Evaluation

```python
def evaluate_sasrec(model, test_data, k_values=[5, 10, 20]):
    """
    test_data: list of (input_sequence, target_item)
    """
    model.eval()
    results = {f'HR@{k}': [] for k in k_values}
    results.update({f'NDCG@{k}': [] for k in k_values})
    results['MRR'] = []

    with torch.no_grad():
        for input_seq, target in test_data:
            # Predict
            seq_tensor = torch.tensor([input_seq])
            scores = model.predict(seq_tensor).squeeze()

            # Rank
            ranked_items = torch.argsort(scores, descending=True) + 1

            # Find target rank
            target_rank = (ranked_items == target).nonzero()
            if len(target_rank) > 0:
                rank = target_rank[0].item() + 1
            else:
                rank = len(ranked_items) + 1

            # Compute metrics
            results['MRR'].append(1.0 / rank)

            for k in k_values:
                hit = 1.0 if rank <= k else 0.0
                ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0

                results[f'HR@{k}'].append(hit)
                results[f'NDCG@{k}'].append(ndcg)

    return {k: sum(v) / len(v) for k, v in results.items()}
```

## Comparison with Alternatives

| Model | Attention | Training | Long-Range | Speed |
|-------|-----------|----------|------------|-------|
| GRU4Rec | None | Sequential | Moderate | Fast |
| SASRec | Causal | Parallel | Good | Medium |
| BERT4Rec | Bidirectional | Masked | Good | Medium |
| Transformers4Rec | Flexible | Various | Excellent | Varies |

## When to Use SASRec

SASRec is well-suited for:
- Next-item prediction tasks
- Moderate sequence lengths (10-100 items)
- When long-range dependencies matter
- Autoregressive generation scenarios

Consider alternatives when:
- Very short sequences (Markov, GRU)
- Very long sequences (efficient transformers)
- Bidirectional context helpful (BERT4Rec)
- Production deployment (Transformers4Rec)
