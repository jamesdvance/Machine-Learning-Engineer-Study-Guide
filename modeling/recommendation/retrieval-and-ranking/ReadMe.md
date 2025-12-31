# Retrieval and Ranking

## Summary

Retrieval and ranking is the two-stage paradigm that powers most production recommendation and search systems at scale. The retrieval stage (also called candidate generation) quickly narrows millions of items to hundreds using approximate methods, while the ranking stage applies more expensive models to score and order the final candidates. This architecture balances the competing demands of coverage (seeing all potentially relevant items) and precision (accurately scoring the best items) within strict latency budgets.

Key points to remember:

- Two-stage design: fast retrieval (O(1) or O(log n)) + expensive ranking
- Retrieval uses approximate nearest neighbors on embeddings
- Ranking uses feature-rich models (cross-features, user history)
- Latency budget typically: retrieval <10ms, ranking <50ms
- Retrieval optimizes recall@k, ranking optimizes NDCG/precision
- Common pattern: retrieve 100-1000 candidates, rank top 10-50
- Multiple retrieval sources often combined (embedding ANN, popularity, rules)
- Ranking models: gradient boosting, neural rankers, cross-attention
- Re-ranking stage sometimes added for diversity/business rules

## The Two-Stage Architecture

```
                     Latency Budget: ~100ms total

+------------------+      +-------------------+      +-----------------+
|   Item Corpus    |      |   Retrieval       |      |   Ranking       |
|   (millions)     | ---> |   (fast, approx)  | ---> |   (slow, exact) |
+------------------+      +-------------------+      +-----------------+
                               |                           |
                               v                           v
                          ~500 candidates            ~20 results
                          <10ms latency              <50ms latency

                     Optional: Re-ranking for diversity/business rules
```

## Why Two Stages?

### The Scale Problem

Scoring every item with a complex model is computationally infeasible:

```python
# If ranking takes 1ms per item...
items = 10_000_000  # 10M items
ranking_time_per_item = 0.001  # 1ms
total_time = items * ranking_time_per_item  # 10,000 seconds!

# With two-stage approach:
retrieval_time = 0.005  # 5ms for ANN search
candidates = 500
ranking_time = candidates * ranking_time_per_item  # 0.5 seconds
# Still too slow! Need batched ranking...

# Batched GPU ranking:
batch_ranking_time = 0.030  # 30ms for 500 items in one batch
total = retrieval_time + batch_ranking_time  # 35ms - acceptable!
```

### Retrieval vs. Ranking Trade-offs

| Aspect | Retrieval | Ranking |
|--------|-----------|---------|
| Speed | O(1) to O(log n) | O(k) where k is candidate count |
| Model complexity | Lightweight (dot product) | Heavy (cross-features, sequences) |
| Accuracy | Approximate | Precise |
| Features | User/item embeddings only | All features including interactions |
| Objective | Maximize recall@k | Maximize precision/NDCG |
| Updates | Periodic (hours/days) | Can be real-time |

## Retrieval Strategies

### Embedding-Based Retrieval

```python
import numpy as np
import faiss

class EmbeddingRetriever:
    """Retrieve candidates using approximate nearest neighbor search."""

    def __init__(self, embedding_dim: int, n_items: int):
        self.embedding_dim = embedding_dim
        # IVF index for fast approximate search
        quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product
        self.index = faiss.IndexIVFFlat(
            quantizer, embedding_dim,
            min(int(np.sqrt(n_items)), 1000),  # n_clusters
            faiss.METRIC_INNER_PRODUCT
        )
        self.item_ids = None

    def build_index(self, item_embeddings: np.ndarray, item_ids: np.ndarray):
        """Build the ANN index from item embeddings."""
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(item_embeddings)

        # Train the index
        self.index.train(item_embeddings)
        self.index.add(item_embeddings)
        self.item_ids = item_ids

        # Set search parameters
        self.index.nprobe = 10  # Clusters to search

    def retrieve(self, user_embedding: np.ndarray, k: int = 100) -> list:
        """Retrieve top-k candidates for a user."""
        # Normalize query
        user_embedding = user_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(user_embedding)

        # Search
        scores, indices = self.index.search(user_embedding, k)

        return [
            {'item_id': self.item_ids[idx], 'score': score}
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0  # FAISS returns -1 for missing results
        ]
```

### Multi-Source Retrieval

Production systems typically combine multiple retrieval sources:

```python
from typing import List, Dict, Set
from dataclasses import dataclass
import heapq

@dataclass
class Candidate:
    item_id: int
    score: float
    source: str

class MultiSourceRetriever:
    """Combine candidates from multiple retrieval sources."""

    def __init__(self):
        self.retrievers = {}
        self.weights = {}

    def add_retriever(self, name: str, retriever, weight: float = 1.0):
        self.retrievers[name] = retriever
        self.weights[name] = weight

    def retrieve(self, user_context: dict, k: int = 500) -> List[Candidate]:
        """Retrieve and merge candidates from all sources."""
        all_candidates = {}

        for name, retriever in self.retrievers.items():
            # Each retriever returns its own candidates
            source_candidates = retriever.retrieve(user_context, k=k)

            for candidate in source_candidates:
                item_id = candidate['item_id']
                weighted_score = candidate['score'] * self.weights[name]

                if item_id in all_candidates:
                    # Combine scores from multiple sources
                    all_candidates[item_id].score += weighted_score
                else:
                    all_candidates[item_id] = Candidate(
                        item_id=item_id,
                        score=weighted_score,
                        source=name
                    )

        # Return top-k by combined score
        return heapq.nlargest(k, all_candidates.values(), key=lambda x: x.score)


# Example sources:
# 1. Embedding similarity (collaborative)
# 2. Content-based embedding similarity
# 3. Popular items (global or segment-specific)
# 4. Recently trending
# 5. Rule-based (same category, same brand)
# 6. Previously interacted (for re-engagement)
```

### Filtering During Retrieval

```python
class FilteredRetriever:
    """Retrieval with business rule filtering."""

    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.filters = []

    def add_filter(self, filter_fn):
        """Add a filter function: (item_id, user_context) -> bool."""
        self.filters.append(filter_fn)

    def retrieve(self, user_context: dict, k: int = 100) -> list:
        # Over-fetch to account for filtering
        over_fetch_factor = 3
        candidates = self.base_retriever.retrieve(
            user_context,
            k=k * over_fetch_factor
        )

        # Apply filters
        filtered = []
        for candidate in candidates:
            if all(f(candidate['item_id'], user_context) for f in self.filters):
                filtered.append(candidate)
                if len(filtered) >= k:
                    break

        return filtered

# Common filters:
# - Already purchased/seen
# - Out of stock
# - Geographic restrictions
# - Age/content restrictions
# - Business rules (excluded brands, etc.)
```

## Ranking Models

### Feature Engineering for Ranking

```python
from typing import Dict, List, Any
import numpy as np

class RankingFeatureExtractor:
    """Extract features for ranking candidates."""

    def extract_features(
        self,
        user: Dict,
        item: Dict,
        context: Dict,
        retrieval_score: float
    ) -> Dict[str, float]:
        """Extract features for a user-item pair."""

        features = {}

        # Retrieval score (useful signal!)
        features['retrieval_score'] = retrieval_score

        # User features
        features['user_age_days'] = user.get('account_age_days', 0)
        features['user_total_purchases'] = user.get('purchase_count', 0)
        features['user_avg_order_value'] = user.get('avg_order_value', 0)

        # Item features
        features['item_price'] = item.get('price', 0)
        features['item_popularity'] = item.get('popularity_score', 0)
        features['item_age_days'] = item.get('days_since_added', 0)
        features['item_avg_rating'] = item.get('avg_rating', 0)
        features['item_review_count'] = item.get('review_count', 0)

        # Cross features (user-item interactions)
        features['price_vs_avg'] = (
            item.get('price', 0) / max(user.get('avg_order_value', 1), 1)
        )

        # Category match with user preferences
        user_category_affinities = user.get('category_affinities', {})
        item_category = item.get('category', '')
        features['category_affinity'] = user_category_affinities.get(
            item_category, 0
        )

        # Temporal features
        features['hour_of_day'] = context.get('hour', 12)
        features['day_of_week'] = context.get('day_of_week', 0)
        features['is_weekend'] = 1 if context.get('day_of_week', 0) >= 5 else 0

        # Recency features
        last_interaction = user.get('last_interaction_with_item', {}).get(
            item.get('id'), None
        )
        if last_interaction:
            features['days_since_interaction'] = (
                context.get('timestamp', 0) - last_interaction
            ) / 86400
        else:
            features['days_since_interaction'] = -1  # Never interacted

        return features
```

### Gradient Boosted Ranker

```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import numpy as np

class GBMRanker:
    """XGBoost-based learning-to-rank model."""

    def __init__(self, params: dict = None):
        self.params = params or {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # Fast histogram-based
        }
        self.model = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,  # Query/user groups
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        groups_val: np.ndarray = None,
    ):
        """Train the ranking model."""
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)

        evals = [(dtrain, 'train')]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(groups_val)
            evals.append((dval, 'val'))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score candidates."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
```

### Neural Ranker

```python
import torch
import torch.nn as nn

class NeuralRanker(nn.Module):
    """Deep neural network for ranking."""

    def __init__(
        self,
        user_embedding_dim: int,
        item_embedding_dim: int,
        numeric_features: int,
        hidden_dims: list = [256, 128, 64],
    ):
        super().__init__()

        # Feature transformation layers
        self.user_transform = nn.Linear(user_embedding_dim, hidden_dims[0] // 2)
        self.item_transform = nn.Linear(item_embedding_dim, hidden_dims[0] // 2)
        self.numeric_transform = nn.Linear(numeric_features, hidden_dims[0])

        # Cross-feature interaction
        self.cross_layer = nn.Bilinear(
            hidden_dims[0] // 2,  # User features
            hidden_dims[0] // 2,  # Item features
            hidden_dims[0]
        )

        # Deep layers
        layers = []
        input_dim = hidden_dims[0] * 3  # user + item + numeric + cross
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            input_dim = hidden_dim

        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        # Transform embeddings
        user_repr = torch.relu(self.user_transform(user_embedding))
        item_repr = torch.relu(self.item_transform(item_embedding))
        numeric_repr = torch.relu(self.numeric_transform(numeric_features))

        # Cross-feature interaction
        cross_repr = self.cross_layer(user_repr, item_repr)

        # Concatenate all representations
        combined = torch.cat([user_repr, item_repr, numeric_repr, cross_repr], dim=1)

        # Deep layers
        deep_out = self.deep(combined)

        return self.output(deep_out).squeeze(-1)


class ListwiseLoss(nn.Module):
    """Listwise ranking loss (softmax cross-entropy)."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,  # (batch, num_candidates)
        labels: torch.Tensor,  # (batch, num_candidates) - relevance grades
    ) -> torch.Tensor:
        # Softmax over candidates
        log_probs = torch.log_softmax(scores / self.temperature, dim=-1)

        # Weight by relevance
        label_probs = torch.softmax(labels / self.temperature, dim=-1)

        # Cross-entropy
        loss = -torch.sum(label_probs * log_probs, dim=-1)
        return loss.mean()
```

### Cross-Encoder Ranker (for text-heavy domains)

```python
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class CrossEncoderRanker(nn.Module):
    """BERT-based cross-encoder for ranking."""

    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, query: str, documents: list) -> torch.Tensor:
        """Score query against multiple documents."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Encode
        outputs = self.encoder(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Score
        scores = self.classifier(cls_embeddings).squeeze(-1)
        return scores
```

## Re-Ranking Stage

An optional third stage for diversity, business rules, or real-time signals:

```python
from typing import List, Tuple
import numpy as np

class ReRanker:
    """Post-ranking adjustments for diversity and business rules."""

    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight

    def maximal_marginal_relevance(
        self,
        candidates: List[dict],
        item_embeddings: np.ndarray,
        k: int = 10,
    ) -> List[dict]:
        """
        MMR re-ranking for diversity.
        Balances relevance with diversity from already selected items.
        """
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = None
            best_score = float('-inf')

            for idx in remaining:
                # Relevance score from ranker
                relevance = candidates[idx]['rank_score']

                # Similarity to already selected items
                if selected_embeddings:
                    similarities = np.dot(
                        np.array(selected_embeddings),
                        item_embeddings[idx]
                    )
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0

                # MMR score
                mmr_score = (
                    (1 - self.diversity_weight) * relevance
                    - self.diversity_weight * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Select best item
            selected.append(candidates[best_idx])
            selected_embeddings.append(item_embeddings[best_idx])
            remaining.remove(best_idx)

        return selected

    def apply_business_rules(
        self,
        candidates: List[dict],
        rules: List[callable],
    ) -> List[dict]:
        """Apply business rule adjustments."""
        for candidate in candidates:
            for rule in rules:
                candidate['rank_score'] = rule(candidate)

        return sorted(candidates, key=lambda x: x['rank_score'], reverse=True)


# Example business rules:
def boost_sponsored(candidate):
    """Boost sponsored items."""
    if candidate.get('is_sponsored'):
        return candidate['rank_score'] * 1.5
    return candidate['rank_score']

def penalize_recently_shown(candidate):
    """Penalize items shown recently to this user."""
    hours_since_shown = candidate.get('hours_since_shown', float('inf'))
    if hours_since_shown < 24:
        decay = 0.5 + 0.5 * (hours_since_shown / 24)
        return candidate['rank_score'] * decay
    return candidate['rank_score']
```

## Full Pipeline Example

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

@dataclass
class RecommendationResult:
    items: List[Dict]
    retrieval_time_ms: float
    ranking_time_ms: float
    reranking_time_ms: float

class RecommendationPipeline:
    """End-to-end recommendation pipeline."""

    def __init__(
        self,
        retriever: MultiSourceRetriever,
        ranker: GBMRanker,
        reranker: Optional[ReRanker] = None,
        feature_extractor: RankingFeatureExtractor = None,
    ):
        self.retriever = retriever
        self.ranker = ranker
        self.reranker = reranker
        self.feature_extractor = feature_extractor or RankingFeatureExtractor()

    def recommend(
        self,
        user: Dict,
        context: Dict,
        n_retrieve: int = 500,
        n_rank: int = 50,
        n_return: int = 20,
    ) -> RecommendationResult:
        """Generate recommendations for a user."""

        # Stage 1: Retrieval
        start = time.time()
        candidates = self.retriever.retrieve(
            {'user': user, 'context': context},
            k=n_retrieve
        )
        retrieval_time = (time.time() - start) * 1000

        # Stage 2: Ranking
        start = time.time()

        # Extract features for all candidates
        features = []
        for candidate in candidates:
            item = self._get_item(candidate.item_id)
            feat = self.feature_extractor.extract_features(
                user, item, context, candidate.score
            )
            features.append(list(feat.values()))

        # Score with ranker
        import numpy as np
        feature_matrix = np.array(features)
        scores = self.ranker.predict(feature_matrix)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.rank_score = score

        candidates.sort(key=lambda x: x.rank_score, reverse=True)
        candidates = candidates[:n_rank]
        ranking_time = (time.time() - start) * 1000

        # Stage 3: Re-ranking (optional)
        start = time.time()
        if self.reranker:
            embeddings = np.array([
                self._get_item_embedding(c.item_id) for c in candidates
            ])
            candidates = self.reranker.maximal_marginal_relevance(
                [{'item_id': c.item_id, 'rank_score': c.rank_score}
                 for c in candidates],
                embeddings,
                k=n_return
            )
        else:
            candidates = candidates[:n_return]
        reranking_time = (time.time() - start) * 1000

        return RecommendationResult(
            items=[self._get_item(c['item_id'] if isinstance(c, dict) else c.item_id)
                   for c in candidates],
            retrieval_time_ms=retrieval_time,
            ranking_time_ms=ranking_time,
            reranking_time_ms=reranking_time,
        )

    def _get_item(self, item_id: int) -> Dict:
        """Fetch item details (would typically query a database/cache)."""
        # Implementation depends on your data storage
        pass

    def _get_item_embedding(self, item_id: int) -> np.ndarray:
        """Fetch item embedding for diversity calculation."""
        # Implementation depends on your embedding storage
        pass
```

## Evaluation Metrics

```python
import numpy as np
from typing import List

def recall_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
    """Recall@k: fraction of relevant items retrieved in top-k."""
    predictions_at_k = set(predictions[:k])
    relevant = set(ground_truth)
    return len(predictions_at_k & relevant) / len(relevant) if relevant else 0

def ndcg_at_k(predictions: List[int], relevances: List[float], k: int) -> float:
    """NDCG@k: normalized discounted cumulative gain."""
    def dcg(relevances: List[float]) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    # Get relevances for predictions
    pred_relevances = [relevances.get(p, 0) for p in predictions[:k]]

    # Ideal ranking
    ideal_relevances = sorted(relevances.values(), reverse=True)[:k]

    dcg_score = dcg(pred_relevances)
    idcg_score = dcg(ideal_relevances)

    return dcg_score / idcg_score if idcg_score > 0 else 0

def mrr(predictions: List[int], ground_truth: List[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    relevant = set(ground_truth)
    for i, pred in enumerate(predictions):
        if pred in relevant:
            return 1.0 / (i + 1)
    return 0.0

# Evaluate retrieval stage (optimize for recall)
# Evaluate ranking stage (optimize for NDCG, precision)
```

## Sub-Topics

- [Approximate Nearest Neighbors](approximate-nearest-neighbors/ReadMe.md): Algorithms for fast similarity search
- [FAISS](faiss/ReadMe.md): Facebook AI Similarity Search library
- [ScaNN](scann/ReadMe.md): Google's Scalable Nearest Neighbors

## When to Use Two-Stage Retrieval and Ranking

This architecture is appropriate when:
- Catalog size exceeds ~10K items
- Latency requirements are <100ms
- Ranking model is too expensive for full catalog
- Multiple retrieval signals need combining
- Diversity or business rules need post-processing

Consider simpler approaches when:
- Small catalog (<10K items)
- Relaxed latency requirements
- Simple recommendation logic suffices
- Real-time personalization not critical
