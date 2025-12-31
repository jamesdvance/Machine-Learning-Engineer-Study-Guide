# Hybrid Recommendation Methods

## Summary

Hybrid recommendation systems combine multiple recommendation approaches to leverage the strengths of each while mitigating their weaknesses. The most common hybridization combines collaborative filtering (CF) with content-based filtering (CBF), but hybrids can also incorporate knowledge-based, demographic, or context-aware methods. By combining signals, hybrids typically outperform any single approach and better handle challenges like cold start and data sparsity.

Key points to remember:

- Combines multiple recommendation paradigms (CF, CBF, knowledge-based)
- Seven hybridization strategies: weighted, switching, mixed, cascade, feature combination, feature augmentation, meta-level
- Addresses cold start by using content when collaborative data is sparse
- Netflix Prize winner was an ensemble of 100+ algorithms
- Modern systems often use implicit hybridization through unified embeddings
- Weighted hybrids simplest to implement; meta-level most sophisticated
- Feature augmentation: use one method's output as input to another
- Compared to single methods: better accuracy, more robust, more complex

## Hybridization Strategies

### 1. Weighted Hybrid

Combine scores from multiple recommenders:

```python
class WeightedHybrid:
    def __init__(self, recommenders, weights=None):
        """
        recommenders: list of (name, recommender) tuples
        weights: dict mapping name to weight (default: equal weights)
        """
        self.recommenders = recommenders

        if weights is None:
            self.weights = {name: 1.0 / len(recommenders)
                           for name, _ in recommenders}
        else:
            self.weights = weights
            # Normalize
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def recommend(self, user_id, n=10, **kwargs):
        """Combine scores from all recommenders."""
        all_scores = {}

        for name, rec in self.recommenders:
            recs = rec.recommend(user_id, n=n * 2, **kwargs)
            weight = self.weights[name]

            for item_id, score in recs:
                if item_id not in all_scores:
                    all_scores[item_id] = 0
                all_scores[item_id] += weight * score

        # Sort by combined score
        sorted_items = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_items[:n]

# Usage
hybrid = WeightedHybrid([
    ('collaborative', cf_model),
    ('content', cb_model),
    ('popularity', pop_model)
], weights={'collaborative': 0.5, 'content': 0.3, 'popularity': 0.2})
```

**Dynamic Weighting:**

```python
class DynamicWeightedHybrid:
    def __init__(self, recommenders):
        self.recommenders = recommenders

    def get_weights(self, user_id):
        """Adjust weights based on user context."""
        user_history = get_user_history(user_id)

        if len(user_history) < 5:
            # Cold start: favor content-based
            return {'collaborative': 0.2, 'content': 0.6, 'popularity': 0.2}
        elif len(user_history) < 20:
            # Warming up: balanced
            return {'collaborative': 0.4, 'content': 0.4, 'popularity': 0.2}
        else:
            # Established user: favor collaborative
            return {'collaborative': 0.6, 'content': 0.3, 'popularity': 0.1}

    def recommend(self, user_id, n=10):
        weights = self.get_weights(user_id)
        # ... rest same as weighted hybrid
```

### 2. Switching Hybrid

Select which recommender to use based on context:

```python
class SwitchingHybrid:
    def __init__(self, primary, fallback, switch_condition):
        """
        primary: main recommender
        fallback: backup recommender
        switch_condition: function(user_id) -> bool (True = use primary)
        """
        self.primary = primary
        self.fallback = fallback
        self.switch_condition = switch_condition

    def recommend(self, user_id, n=10):
        if self.switch_condition(user_id):
            return self.primary.recommend(user_id, n)
        else:
            return self.fallback.recommend(user_id, n)

# Example: switch based on user history
def has_enough_history(user_id, min_interactions=5):
    return len(get_user_history(user_id)) >= min_interactions

hybrid = SwitchingHybrid(
    primary=cf_model,
    fallback=content_model,
    switch_condition=has_enough_history
)
```

### 3. Mixed Hybrid

Present recommendations from multiple sources together:

```python
class MixedHybrid:
    def __init__(self, recommenders, mix_ratio=None):
        """
        recommenders: list of (name, recommender) tuples
        mix_ratio: dict mapping name to number of items to include
        """
        self.recommenders = recommenders
        self.mix_ratio = mix_ratio or {name: 1 for name, _ in recommenders}

    def recommend(self, user_id, n=10):
        """Interleave recommendations from multiple sources."""
        results = []
        seen = set()

        # Get recommendations from each
        rec_lists = {}
        for name, rec in self.recommenders:
            rec_lists[name] = rec.recommend(user_id, n=n)

        # Interleave according to ratio
        indices = {name: 0 for name, _ in self.recommenders}

        while len(results) < n:
            for name, _ in self.recommenders:
                for _ in range(self.mix_ratio[name]):
                    if len(results) >= n:
                        break

                    # Get next unseen item from this recommender
                    while indices[name] < len(rec_lists[name]):
                        item_id, score = rec_lists[name][indices[name]]
                        indices[name] += 1

                        if item_id not in seen:
                            results.append((item_id, score, name))
                            seen.add(item_id)
                            break

        return results[:n]
```

### 4. Cascade Hybrid

Use one recommender to refine another's output:

```python
class CascadeHybrid:
    def __init__(self, coarse_ranker, fine_ranker, candidate_size=100):
        """
        coarse_ranker: fast, generates candidates
        fine_ranker: slow, accurate, refines ranking
        """
        self.coarse = coarse_ranker
        self.fine = fine_ranker
        self.candidate_size = candidate_size

    def recommend(self, user_id, n=10):
        # Stage 1: Get candidates
        candidates = self.coarse.recommend(user_id, n=self.candidate_size)
        candidate_items = [item for item, _ in candidates]

        # Stage 2: Re-rank candidates
        refined = self.fine.score_items(user_id, candidate_items)

        # Sort by refined scores
        refined.sort(key=lambda x: x[1], reverse=True)
        return refined[:n]

# Example: popularity -> collaborative
cascade = CascadeHybrid(
    coarse_ranker=popularity_model,  # Fast
    fine_ranker=neural_model,        # Slow but accurate
    candidate_size=100
)
```

### 5. Feature Combination

Combine features from different sources into single model:

```python
class FeatureCombinationHybrid:
    def __init__(self, n_users, n_items, user_features, item_features,
                 embedding_dim=32):
        """
        user_features: array of user feature vectors
        item_features: array of item feature vectors
        """
        self.model = self._build_model(
            n_users, n_items,
            user_features.shape[1],
            item_features.shape[1],
            embedding_dim
        )
        self.user_features = user_features
        self.item_features = item_features

    def _build_model(self, n_users, n_items, user_feat_dim, item_feat_dim, emb_dim):
        import torch.nn as nn

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Collaborative embeddings
                self.user_emb = nn.Embedding(n_users, emb_dim)
                self.item_emb = nn.Embedding(n_items, emb_dim)

                # Content feature encoders
                self.user_encoder = nn.Linear(user_feat_dim, emb_dim)
                self.item_encoder = nn.Linear(item_feat_dim, emb_dim)

                # Combination layers
                self.fc = nn.Sequential(
                    nn.Linear(emb_dim * 4, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, user_ids, item_ids, user_feats, item_feats):
                # Collaborative part
                user_cf = self.user_emb(user_ids)
                item_cf = self.item_emb(item_ids)

                # Content part
                user_cb = self.user_encoder(user_feats)
                item_cb = self.item_encoder(item_feats)

                # Combine all features
                combined = torch.cat([user_cf, item_cf, user_cb, item_cb], dim=-1)

                return self.fc(combined).squeeze()

        return HybridModel()
```

### 6. Feature Augmentation

Use one method's output as features for another:

```python
class FeatureAugmentationHybrid:
    def __init__(self, cf_model, final_model):
        """
        cf_model: Collaborative filtering model
        final_model: Model that uses CF predictions as features
        """
        self.cf_model = cf_model
        self.final_model = final_model

    def prepare_features(self, user_id, items, content_features):
        """Augment content features with CF predictions."""
        augmented = []

        for item_id, content_feat in zip(items, content_features):
            # Get CF prediction as additional feature
            cf_score = self.cf_model.predict(user_id, item_id)

            # Concatenate
            augmented_feat = np.concatenate([content_feat, [cf_score]])
            augmented.append(augmented_feat)

        return np.array(augmented)

    def recommend(self, user_id, candidate_items, content_features, n=10):
        # Augment features with CF scores
        aug_features = self.prepare_features(
            user_id, candidate_items, content_features
        )

        # Score with final model
        scores = self.final_model.predict(aug_features)

        # Rank
        ranked = sorted(zip(candidate_items, scores),
                       key=lambda x: x[1], reverse=True)
        return ranked[:n]
```

### 7. Meta-Level Hybrid

Use one model's output to train another:

```python
class MetaLevelHybrid:
    def __init__(self, base_model, meta_model):
        """
        base_model: Generates user representations (e.g., content-based)
        meta_model: Uses those representations for collaborative learning
        """
        self.base = base_model
        self.meta = meta_model

    def fit(self, interactions, item_features):
        # Step 1: Build user profiles using content model
        user_profiles = {}
        for user_id, items in group_by_user(interactions):
            item_feats = [item_features[i] for i in items]
            user_profiles[user_id] = self.base.build_profile(item_feats)

        # Step 2: Train meta model using profiles as user features
        self.meta.fit(interactions, user_features=user_profiles)

    def recommend(self, user_id, n=10):
        return self.meta.recommend(user_id, n)
```

## Practical Implementation

### LightFM: Hybrid Factorization

LightFM is a popular library for hybrid matrix factorization:

```python
from lightfm import LightFM
from lightfm.data import Dataset

# Prepare data
dataset = Dataset()
dataset.fit(users, items)
dataset.fit_partial(users, user_features=user_features)
dataset.fit_partial(items, item_features=item_features)

# Build interaction matrix
interactions, weights = dataset.build_interactions(
    [(u, i, r) for u, i, r in ratings]
)

# Build feature matrices
user_features_matrix = dataset.build_user_features(
    [(u, feats) for u, feats in user_feature_dict.items()]
)
item_features_matrix = dataset.build_item_features(
    [(i, feats) for i, feats in item_feature_dict.items()]
)

# Train hybrid model
model = LightFM(
    no_components=64,
    loss='warp',  # Ranking loss
    learning_rate=0.05
)

model.fit(
    interactions,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=20,
    num_threads=4
)

# Get recommendations
user_id = 0
n_items = interactions.shape[1]
scores = model.predict(user_id, np.arange(n_items),
                       user_features=user_features_matrix,
                       item_features=item_features_matrix)
top_items = np.argsort(-scores)[:10]
```

### Deep Hybrid with TensorFlow

```python
import tensorflow as tf

class DeepHybridRecommender(tf.keras.Model):
    def __init__(self, n_users, n_items, user_feat_dim, item_feat_dim,
                 embedding_dim=64):
        super().__init__()

        # Collaborative components
        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(n_items, embedding_dim)

        # Content encoders
        self.user_content_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])

        self.item_content_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])

        # Fusion layer
        self.fusion = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        user_ids, item_ids, user_features, item_features = inputs

        # Collaborative embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Content embeddings
        user_content = self.user_content_encoder(user_features)
        item_content = self.item_content_encoder(item_features)

        # Combine user representations
        user_combined = tf.concat([user_emb, user_content], axis=-1)

        # Combine item representations
        item_combined = tf.concat([item_emb, item_content], axis=-1)

        # Interaction
        interaction = tf.concat([
            user_combined,
            item_combined,
            user_combined * item_combined  # Element-wise product
        ], axis=-1)

        return self.fusion(interaction)
```

## Balancing Cold Start

### User Cold Start

```python
class ColdStartAwareHybrid:
    def __init__(self, cf_model, content_model, transition_threshold=10):
        self.cf = cf_model
        self.content = content_model
        self.threshold = transition_threshold

    def recommend(self, user_id, user_features=None, n=10):
        history_size = len(get_user_history(user_id))

        if history_size == 0:
            # Complete cold start: use content only or popular items
            if user_features is not None:
                return self.content.recommend_from_features(user_features, n)
            else:
                return self.get_popular_items(n)

        elif history_size < self.threshold:
            # Partial cold start: weighted combination
            weight = history_size / self.threshold

            cf_recs = dict(self.cf.recommend(user_id, n * 2))
            cb_recs = dict(self.content.recommend(user_id, n * 2))

            combined = {}
            for item in set(cf_recs.keys()) | set(cb_recs.keys()):
                cf_score = cf_recs.get(item, 0)
                cb_score = cb_recs.get(item, 0)
                combined[item] = weight * cf_score + (1 - weight) * cb_score

            return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]

        else:
            # Established user: primarily collaborative
            return self.cf.recommend(user_id, n)
```

### Item Cold Start

```python
def handle_item_cold_start(item_id, item_features, similar_items_model, cf_model):
    """
    For new items with no interactions, use content similarity
    to find similar established items.
    """
    # Find similar items with interaction history
    similar_items = similar_items_model.get_similar(item_features, n=10)

    # Aggregate CF predictions for similar items
    aggregated_users = {}
    for sim_item, similarity in similar_items:
        # Get users who liked this similar item
        users = get_users_who_liked(sim_item)
        for user_id in users:
            if user_id not in aggregated_users:
                aggregated_users[user_id] = 0
            aggregated_users[user_id] += similarity

    # Return users most likely to like the new item
    return sorted(aggregated_users.items(), key=lambda x: x[1], reverse=True)
```

## Evaluation

### A/B Testing Hybrids

```python
def evaluate_hybrid_variants(variants, test_users, metrics=['hr@10', 'ndcg@10']):
    """Compare hybrid configurations."""
    results = {}

    for name, hybrid in variants.items():
        scores = {m: [] for m in metrics}

        for user_id in test_users:
            true_items = get_test_items(user_id)
            recs = hybrid.recommend(user_id, n=10)
            rec_items = [item for item, _ in recs]

            if 'hr@10' in metrics:
                hit = any(item in rec_items for item in true_items)
                scores['hr@10'].append(float(hit))

            if 'ndcg@10' in metrics:
                ndcg = compute_ndcg(rec_items, true_items)
                scores['ndcg@10'].append(ndcg)

        results[name] = {m: np.mean(v) for m, v in scores.items()}

    return results

# Example
variants = {
    'cf_only': cf_model,
    'weighted_0.5': WeightedHybrid([cf, cb], {'cf': 0.5, 'cb': 0.5}),
    'weighted_0.7': WeightedHybrid([cf, cb], {'cf': 0.7, 'cb': 0.3}),
    'cascade': CascadeHybrid(popularity, cf),
}

results = evaluate_hybrid_variants(variants, test_users)
```

### Learning Optimal Weights

```python
from scipy.optimize import minimize

def learn_hybrid_weights(recommenders, validation_data):
    """Learn optimal weights via optimization."""

    def objective(weights):
        weights = weights / weights.sum()  # Normalize
        hybrid = WeightedHybrid(recommenders, dict(zip(names, weights)))

        # Evaluate on validation set
        score = 0
        for user_id, true_item in validation_data:
            recs = hybrid.recommend(user_id, n=10)
            if true_item in [item for item, _ in recs]:
                score += 1

        return -score  # Minimize negative (maximize hits)

    names = [name for name, _ in recommenders]
    n = len(recommenders)

    # Initial weights
    x0 = np.ones(n) / n

    # Constraints: weights sum to 1, all >= 0
    constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    optimal_weights = result.x / result.x.sum()
    return dict(zip(names, optimal_weights))
```

## Production Considerations

### Online vs Offline Components

```python
class ProductionHybrid:
    def __init__(self):
        # Offline-computed models (updated daily)
        self.cf_model = load_model('cf_model.pkl')
        self.content_index = load_index('content_index.faiss')

        # Online components
        self.user_cache = TTLCache(maxsize=10000, ttl=3600)
        self.real_time_features = RealTimeFeatureStore()

    def recommend(self, user_id, context, n=10):
        # Get user's recent activity (real-time)
        recent_items = self.real_time_features.get_recent_items(user_id)

        # Get CF recommendations (offline)
        cf_recs = self.cf_model.recommend(user_id, n=n * 2)

        # Get content recommendations based on recent items (online)
        if recent_items:
            content_recs = self.content_index.search(recent_items[-1], n * 2)
        else:
            content_recs = []

        # Combine with context-aware weighting
        return self._combine(cf_recs, content_recs, context)
```

## Comparison of Strategies

| Strategy | Complexity | Cold Start | Explainability | Best For |
|----------|------------|------------|----------------|----------|
| Weighted | Low | Moderate | Low | Simple combination |
| Switching | Low | Good | High | Clear use cases |
| Mixed | Low | Moderate | High | Diversity |
| Cascade | Medium | Moderate | Medium | Two-stage retrieval |
| Feature Combination | High | Good | Low | Deep learning |
| Feature Augmentation | Medium | Good | Medium | Incremental improvement |
| Meta-Level | High | Good | Low | Complex relationships |

## When to Use Hybrid Methods

Hybrid methods are well-suited for:
- Production systems needing robustness
- Handling cold start effectively
- Combining multiple data sources
- Improving over single-method baselines
- Systems requiring both accuracy and coverage

Consider simpler approaches when:
- Limited engineering resources
- Single data source dominates
- Rapid iteration needed
- Interpretability critical
