# Neural Collaborative Filtering (NCF)

## Summary

Neural Collaborative Filtering replaces the linear dot product in matrix factorization with neural networks, enabling the model to learn non-linear user-item interactions. The seminal NCF paper introduced Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) components, which can be combined into a unified NeuMF architecture. This approach bridges traditional collaborative filtering with deep learning, offering improved expressiveness while maintaining the embedding-based paradigm.

Key points to remember:

- Replaces dot product with neural network for non-linear interactions
- GMF: Element-wise product of embeddings (generalized MF)
- MLP: Concatenated embeddings through hidden layers
- NeuMF: Combines GMF and MLP with separate embedding spaces
- Learns interaction function from data rather than assuming linearity
- Implicit feedback focused (binary classification)
- Negative sampling required for training
- Pre-training GMF/MLP separately improves NeuMF performance
- Compared to MF: more expressive, but slower and needs more data
- Compared to deep models: simpler architecture, good baseline

## Architecture

### From Matrix Factorization to NCF

Matrix Factorization assumes linear interaction:

```
score(u, i) = u_u^T * v_i = sum(u_uk * v_ik)
```

NCF generalizes this with a neural network:

```
score(u, i) = f(u_u, v_i | theta)
```

Where f is a neural network with parameters theta.

### NCF Framework

```
Input Layer:
  User ID (one-hot) --> User Embedding
  Item ID (one-hot) --> Item Embedding

Neural CF Layers:
  [User Embedding, Item Embedding] --> Hidden Layers --> Output

Output Layer:
  Predicted score (0-1 for implicit feedback)
```

### Generalized Matrix Factorization (GMF)

Element-wise product with learnable output weights:

```
GMF(u, i) = sigmoid(h^T * (p_u * q_i))

where:
- p_u: user embedding
- q_i: item embedding
- *: element-wise product
- h: output weights
```

```python
class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.output = nn.Linear(n_factors, 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Element-wise product
        interaction = user_emb * item_emb

        # Output layer
        score = torch.sigmoid(self.output(interaction))
        return score.squeeze()
```

### Multi-Layer Perceptron (MLP)

Concatenated embeddings through hidden layers:

```
MLP(u, i) = sigmoid(W_L * (... ReLU(W_2 * ReLU(W_1 * [p_u; q_i]))))
```

```python
class MLP(nn.Module):
    def __init__(self, n_users, n_items, n_factors, layers=[64, 32, 16]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        # MLP layers
        mlp_layers = []
        input_size = n_factors * 2  # Concatenated embeddings

        for layer_size in layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Concatenate embeddings
        concat = torch.cat([user_emb, item_emb], dim=-1)

        # MLP forward
        hidden = self.mlp(concat)
        score = torch.sigmoid(self.output(hidden))
        return score.squeeze()
```

### Neural Matrix Factorization (NeuMF)

Combines GMF and MLP with separate embedding spaces:

```
NeuMF(u, i) = sigmoid(h^T * [GMF_output; MLP_output])
```

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_factors=32, mlp_factors=32,
                 mlp_layers=[64, 32, 16]):
        super().__init__()

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, gmf_factors)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_factors)

        # MLP embeddings (separate from GMF)
        self.mlp_user_emb = nn.Embedding(n_users, mlp_factors)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_factors)

        # MLP layers
        mlp_modules = []
        input_size = mlp_factors * 2

        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layer
        self.output = nn.Linear(gmf_factors + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP path
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        score = torch.sigmoid(self.output(concat))
        return score.squeeze()
```

## Training

### Negative Sampling

For implicit feedback, sample negative items (not interacted):

```python
class NCFDataset(torch.utils.data.Dataset):
    def __init__(self, interactions, n_items, n_negatives=4):
        """
        interactions: list of (user_id, item_id) positive pairs
        n_negatives: negative samples per positive
        """
        self.interactions = interactions
        self.n_items = n_items
        self.n_negatives = n_negatives

        # Build set of positive items per user
        self.user_positives = defaultdict(set)
        for user, item in interactions:
            self.user_positives[user].add(item)

    def __len__(self):
        return len(self.interactions) * (1 + self.n_negatives)

    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        pos_idx = idx // (1 + self.n_negatives)
        is_positive = (idx % (1 + self.n_negatives)) == 0

        user, pos_item = self.interactions[pos_idx]

        if is_positive:
            return user, pos_item, 1.0
        else:
            # Sample negative item
            neg_item = np.random.randint(self.n_items)
            while neg_item in self.user_positives[user]:
                neg_item = np.random.randint(self.n_items)
            return user, neg_item, 0.0
```

### Training Loop

```python
def train_ncf(model, train_loader, optimizer, n_epochs=20):
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()

            pred = model(user, item)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Evaluate
        if (epoch + 1) % 5 == 0:
            hr, ndcg = evaluate(model, test_data)
            print(f"  HR@10 = {hr:.4f}, NDCG@10 = {ndcg:.4f}")
```

### Pre-training Strategy

NeuMF benefits from pre-training GMF and MLP separately:

```python
def pretrain_and_combine(train_data, n_users, n_items):
    # Pre-train GMF
    gmf = GMF(n_users, n_items, n_factors=32)
    train_model(gmf, train_data, epochs=10)

    # Pre-train MLP
    mlp = MLP(n_users, n_items, n_factors=32, layers=[64, 32, 16])
    train_model(mlp, train_data, epochs=10)

    # Initialize NeuMF with pre-trained weights
    neumf = NeuMF(n_users, n_items, gmf_factors=32, mlp_factors=32)

    # Copy GMF weights
    neumf.gmf_user_emb.weight.data = gmf.user_embedding.weight.data
    neumf.gmf_item_emb.weight.data = gmf.item_embedding.weight.data

    # Copy MLP weights
    neumf.mlp_user_emb.weight.data = mlp.user_embedding.weight.data
    neumf.mlp_item_emb.weight.data = mlp.item_embedding.weight.data

    for (name, param), (_, pretrained) in zip(
        neumf.mlp.named_parameters(), mlp.mlp.named_parameters()
    ):
        param.data = pretrained.data

    # Fine-tune NeuMF with smaller learning rate
    optimizer = torch.optim.Adam(neumf.parameters(), lr=0.0001)
    train_model(neumf, train_data, epochs=10, optimizer=optimizer)

    return neumf
```

### Loss Functions

**Binary Cross-Entropy (standard):**

```python
criterion = nn.BCELoss()
loss = criterion(predictions, labels)
```

**BPR Loss (pairwise ranking):**

```python
def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalized Ranking loss."""
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

# Training with BPR
pos_scores = model(users, pos_items)
neg_scores = model(users, neg_items)
loss = bpr_loss(pos_scores, neg_scores)
```

**Margin Loss:**

```python
def margin_loss(pos_scores, neg_scores, margin=0.5):
    """Hinge loss with margin."""
    return torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0))
```

## Evaluation

### Ranking Metrics

```python
def evaluate(model, test_data, k=10):
    """
    test_data: dict mapping user_id to list of (item_id, timestamp)
    """
    model.eval()
    hits = []
    ndcgs = []

    with torch.no_grad():
        for user_id, test_items in test_data.items():
            # Get ground truth
            true_item = test_items[0]  # Item to predict

            # Score all items for this user
            all_items = torch.arange(n_items).to(device)
            user_tensor = torch.full((n_items,), user_id).to(device)

            scores = model(user_tensor, all_items)

            # Get top-k items
            _, top_k = torch.topk(scores, k)
            top_k = top_k.cpu().numpy()

            # Hit Rate
            hit = 1 if true_item in top_k else 0
            hits.append(hit)

            # NDCG
            if true_item in top_k:
                rank = np.where(top_k == true_item)[0][0]
                ndcg = 1 / np.log2(rank + 2)
            else:
                ndcg = 0
            ndcgs.append(ndcg)

    return np.mean(hits), np.mean(ndcgs)
```

### Leave-One-Out Evaluation

Standard protocol for implicit feedback:

```python
def leave_one_out_split(interactions):
    """
    For each user, hold out the last interaction for testing.
    """
    # Sort by timestamp
    user_items = defaultdict(list)
    for user, item, timestamp in interactions:
        user_items[user].append((item, timestamp))

    train = []
    test = {}

    for user, items in user_items.items():
        items.sort(key=lambda x: x[1])  # Sort by time

        # Last item for test
        test[user] = items[-1][0]

        # Rest for training
        for item, _ in items[:-1]:
            train.append((user, item))

    return train, test
```

## Practical Implementation

### Using PyTorch Lightning

```python
import pytorch_lightning as pl

class NCFModule(pl.LightningModule):
    def __init__(self, n_users, n_items, config):
        super().__init__()
        self.model = NeuMF(
            n_users, n_items,
            gmf_factors=config['gmf_factors'],
            mlp_factors=config['mlp_factors'],
            mlp_layers=config['mlp_layers']
        )
        self.criterion = nn.BCELoss()
        self.config = config

    def forward(self, user, item):
        return self.model(user, item)

    def training_step(self, batch, batch_idx):
        user, item, label = batch
        pred = self(user, item)
        loss = self.criterion(pred, label.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user, item, label = batch
        pred = self(user, item)
        loss = self.criterion(pred, label.float())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

# Training
trainer = pl.Trainer(max_epochs=20, gpus=1)
trainer.fit(model, train_loader, val_loader)
```

### Hyperparameter Configuration

```python
config = {
    # Embedding dimensions
    'gmf_factors': 32,
    'mlp_factors': 32,

    # MLP architecture
    'mlp_layers': [64, 32, 16],
    'dropout': 0.2,

    # Training
    'lr': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'n_negatives': 4,
    'epochs': 20,
}
```

### Inference Optimization

For production, pre-compute user embeddings:

```python
class NCFInference:
    def __init__(self, model, n_items):
        self.model = model
        self.n_items = n_items

        # Pre-compute item embeddings
        with torch.no_grad():
            items = torch.arange(n_items)
            self.gmf_items = model.gmf_item_emb(items)
            self.mlp_items = model.mlp_item_emb(items)

    def recommend(self, user_id, k=10, exclude_items=None):
        with torch.no_grad():
            # Get user embeddings
            user = torch.tensor([user_id])
            gmf_user = self.model.gmf_user_emb(user)
            mlp_user = self.model.mlp_user_emb(user)

            # GMF scores (vectorized)
            gmf_output = gmf_user * self.gmf_items

            # MLP scores (batch)
            mlp_input = torch.cat([
                mlp_user.expand(self.n_items, -1),
                self.mlp_items
            ], dim=-1)
            mlp_output = self.model.mlp(mlp_input)

            # Combine
            concat = torch.cat([gmf_output, mlp_output], dim=-1)
            scores = torch.sigmoid(self.model.output(concat)).squeeze()

            # Exclude seen items
            if exclude_items:
                scores[list(exclude_items)] = -float('inf')

            # Top-k
            _, top_k = torch.topk(scores, k)
            return top_k.numpy()
```

## Extensions

### NCF with Side Features

Incorporate user/item features:

```python
class NCFWithFeatures(nn.Module):
    def __init__(self, n_users, n_items, n_factors,
                 user_feature_dim, item_feature_dim):
        super().__init__()

        # ID embeddings
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        # Feature encoders
        self.user_encoder = nn.Linear(user_feature_dim, n_factors)
        self.item_encoder = nn.Linear(item_feature_dim, n_factors)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_factors * 4, 64),  # ID + features for both
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids, user_features, item_features):
        # ID embeddings
        user_id_emb = self.user_emb(user_ids)
        item_id_emb = self.item_emb(item_ids)

        # Feature embeddings
        user_feat_emb = self.user_encoder(user_features)
        item_feat_emb = self.item_encoder(item_features)

        # Combine
        user_repr = torch.cat([user_id_emb, user_feat_emb], dim=-1)
        item_repr = torch.cat([item_id_emb, item_feat_emb], dim=-1)

        concat = torch.cat([user_repr, item_repr], dim=-1)
        score = torch.sigmoid(self.mlp(concat))
        return score.squeeze()
```

### Attention-based NCF

Add attention to weight embedding dimensions:

```python
class AttentionNCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(n_factors * 2, n_factors),
            nn.ReLU(),
            nn.Linear(n_factors, n_factors),
            nn.Softmax(dim=-1)
        )

        self.output = nn.Linear(n_factors, 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)

        # Compute attention weights
        concat = torch.cat([user_emb, item_emb], dim=-1)
        attention_weights = self.attention(concat)

        # Weighted element-wise product
        interaction = attention_weights * (user_emb * item_emb)

        score = torch.sigmoid(self.output(interaction))
        return score.squeeze()
```

### Variational NCF

Add uncertainty estimation:

```python
class VariationalNCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()

        # Mean and log-variance for user embeddings
        self.user_mu = nn.Embedding(n_users, n_factors)
        self.user_logvar = nn.Embedding(n_users, n_factors)

        self.item_mu = nn.Embedding(n_items, n_factors)
        self.item_logvar = nn.Embedding(n_items, n_factors)

        self.output = nn.Linear(n_factors, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, user_ids, item_ids):
        # Sample user embedding
        user_mu = self.user_mu(user_ids)
        user_logvar = self.user_logvar(user_ids)
        user_emb = self.reparameterize(user_mu, user_logvar)

        # Sample item embedding
        item_mu = self.item_mu(item_ids)
        item_logvar = self.item_logvar(item_ids)
        item_emb = self.reparameterize(item_mu, item_logvar)

        interaction = user_emb * item_emb
        score = torch.sigmoid(self.output(interaction))

        # KL divergence for regularization
        kl_user = -0.5 * torch.sum(1 + user_logvar - user_mu.pow(2) - user_logvar.exp())
        kl_item = -0.5 * torch.sum(1 + item_logvar - item_mu.pow(2) - item_logvar.exp())

        return score.squeeze(), kl_user + kl_item
```

## Comparison with Alternatives

| Method | Pros | Cons |
|--------|------|------|
| NCF | Non-linear, flexible | Slower than MF |
| Matrix Factorization | Fast, simple | Linear only |
| Wide & Deep | Features + embeddings | More complex |
| Autoencoders | Reconstruction-based | Different paradigm |
| Graph Neural Networks | Structure-aware | Need graph |

## Hyperparameter Tuning

### Key Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| gmf_factors | 8-64 | Higher for complex data |
| mlp_factors | 8-64 | Often same as GMF |
| mlp_layers | [64,32,16] to [128,64,32] | Deeper for more data |
| dropout | 0.0-0.5 | Higher for regularization |
| n_negatives | 1-10 | More for sparse data |
| learning_rate | 1e-4 to 1e-2 | Lower for pre-trained |

### Grid Search Example

```python
from itertools import product

param_grid = {
    'gmf_factors': [16, 32, 64],
    'mlp_factors': [16, 32, 64],
    'mlp_layers': [[64, 32], [128, 64, 32]],
    'lr': [0.001, 0.0001],
    'n_negatives': [4, 8]
}

best_hr = 0
for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    model = train_ncf(config, train_data)
    hr, ndcg = evaluate(model, test_data)

    if hr > best_hr:
        best_hr = hr
        best_config = config
```

## When to Use NCF

NCF is well-suited for:
- Implicit feedback recommendation
- When MF performance plateaus
- Moderate-scale systems (millions of interactions)
- Research baselines and ablation studies
- When non-linear interactions matter

Consider alternatives when:
- Very large scale (use simpler MF)
- Rich features available (use Wide & Deep)
- Sequential patterns matter (use sequential models)
- Graph structure important (use GNNs)
- Need real-time updates (MF with SGD easier)
