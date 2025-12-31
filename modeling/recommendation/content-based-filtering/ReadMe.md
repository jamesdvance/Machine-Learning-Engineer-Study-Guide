# Content-Based Filtering

## Summary

Content-Based Filtering recommends items similar to those a user has liked in the past, based on item features or content. Unlike collaborative filtering which relies on user-item interaction patterns, content-based methods analyze item attributes (text descriptions, categories, metadata) to build user preference profiles. This approach excels at handling the cold-start problem for new items and provides explainable recommendations.

Key points to remember:

- Recommends items based on feature similarity to user's past preferences
- No need for other users' data (works with single user)
- Handles new item cold start well (just needs item features)
- User profile built from features of liked items
- TF-IDF common for text features; embeddings for modern approaches
- Explainable: "Recommended because you liked X with similar features"
- Limited serendipity (recommends more of the same)
- Struggles with new user cold start (no preference history)
- Compared to CF: better cold start, less diverse, no collaborative signal

## Core Concepts

### Item Representation

Items represented as feature vectors:

```python
# Example: Movie features
movie = {
    'title': 'The Matrix',
    'genres': ['Action', 'Sci-Fi'],
    'director': 'Wachowski',
    'actors': ['Keanu Reeves', 'Laurence Fishburne'],
    'year': 1999,
    'description': 'A computer hacker learns about the true nature of reality...',
    'keywords': ['virtual reality', 'dystopia', 'AI', 'chosen one']
}

# Convert to feature vector
# Option 1: One-hot encoding for categorical
# Option 2: TF-IDF for text
# Option 3: Embeddings for semantic similarity
```

### User Profile

Aggregate features of items user has liked:

```python
def build_user_profile(user_ratings, item_features):
    """
    Build user profile as weighted average of item features.

    user_ratings: dict of {item_id: rating}
    item_features: dict of {item_id: feature_vector}
    """
    profile = np.zeros(feature_dim)
    total_weight = 0

    for item_id, rating in user_ratings.items():
        if rating > threshold:  # Only positive ratings
            weight = rating  # Or binary
            profile += weight * item_features[item_id]
            total_weight += weight

    if total_weight > 0:
        profile /= total_weight

    return profile
```

### Recommendation

Score items by similarity to user profile:

```python
def recommend(user_profile, item_features, n=10, exclude_seen=None):
    """
    Recommend items most similar to user profile.
    """
    scores = []

    for item_id, features in item_features.items():
        if exclude_seen and item_id in exclude_seen:
            continue

        similarity = cosine_similarity(user_profile, features)
        scores.append((item_id, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]
```

## Feature Extraction

### Text Features with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFContentBased:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def fit(self, items):
        """
        items: list of dicts with 'id' and 'text' fields
        """
        texts = [item['text'] for item in items]
        self.item_ids = [item['id'] for item in items]

        # Fit TF-IDF
        self.item_vectors = self.vectorizer.fit_transform(texts)

        # Create lookup
        self.id_to_idx = {id: idx for idx, id in enumerate(self.item_ids)}

    def get_item_vector(self, item_id):
        idx = self.id_to_idx[item_id]
        return self.item_vectors[idx].toarray().flatten()

    def build_user_profile(self, liked_items):
        """Build profile from liked items."""
        vectors = []
        for item_id in liked_items:
            if item_id in self.id_to_idx:
                vectors.append(self.get_item_vector(item_id))

        if not vectors:
            return np.zeros(self.item_vectors.shape[1])

        return np.mean(vectors, axis=0)

    def recommend(self, user_profile, n=10, exclude=None):
        """Recommend items similar to profile."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Compute similarities to all items
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            self.item_vectors
        ).flatten()

        # Rank items
        ranked_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            item_id = self.item_ids[idx]
            if exclude and item_id in exclude:
                continue
            recommendations.append((item_id, similarities[idx]))
            if len(recommendations) >= n:
                break

        return recommendations
```

### Embedding-Based Features

Modern approach using pre-trained embeddings:

```python
from sentence_transformers import SentenceTransformer

class EmbeddingContentBased:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_items(self, items):
        """
        items: list of dicts with 'id' and 'text' fields
        """
        self.item_ids = [item['id'] for item in items]
        texts = [item['text'] for item in items]

        # Encode all items
        self.item_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.id_to_idx = {id: idx for idx, id in enumerate(self.item_ids)}

    def get_similar_items(self, item_id, n=10):
        """Find items similar to given item."""
        idx = self.id_to_idx[item_id]
        query_embedding = self.item_embeddings[idx]

        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.item_embeddings
        ).flatten()

        # Exclude self and sort
        similarities[idx] = -1
        top_indices = np.argsort(similarities)[::-1][:n]

        return [(self.item_ids[i], similarities[i]) for i in top_indices]
```

### Multi-Modal Features

Combine different feature types:

```python
class MultiModalContentBased:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.genre_encoder = None  # Will be fit

    def encode_items(self, items):
        """
        items: list of dicts with 'id', 'text', 'genres', 'year'
        """
        self.item_ids = [item['id'] for item in items]

        # Text embeddings
        texts = [item.get('text', '') for item in items]
        text_embeddings = self.text_encoder.encode(texts)

        # Genre one-hot encoding
        all_genres = set()
        for item in items:
            all_genres.update(item.get('genres', []))
        self.genre_list = sorted(all_genres)

        genre_vectors = np.zeros((len(items), len(self.genre_list)))
        for i, item in enumerate(items):
            for genre in item.get('genres', []):
                genre_vectors[i, self.genre_list.index(genre)] = 1

        # Year normalization
        years = np.array([item.get('year', 2000) for item in items])
        years_normalized = (years - years.mean()) / years.std()
        years_normalized = years_normalized.reshape(-1, 1)

        # Concatenate features (with weighting)
        self.item_embeddings = np.hstack([
            text_embeddings * 0.6,      # Text weight
            genre_vectors * 0.3,        # Genre weight
            years_normalized * 0.1      # Year weight
        ])

        # Normalize
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        self.item_embeddings = self.item_embeddings / (norms + 1e-8)

        self.id_to_idx = {id: idx for idx, id in enumerate(self.item_ids)}
```

## User Profile Variants

### Simple Average

```python
def simple_average_profile(liked_item_vectors):
    """Average of all liked items."""
    return np.mean(liked_item_vectors, axis=0)
```

### Weighted by Rating

```python
def rating_weighted_profile(item_vectors, ratings):
    """Weight by user rating."""
    weights = np.array(ratings)
    weighted_sum = np.sum(item_vectors * weights.reshape(-1, 1), axis=0)
    return weighted_sum / np.sum(weights)
```

### Weighted by Recency

```python
def recency_weighted_profile(item_vectors, timestamps, decay=0.1):
    """Exponential decay by recency."""
    now = max(timestamps)
    days_ago = [(now - t).days for t in timestamps]
    weights = np.exp(-decay * np.array(days_ago))

    weighted_sum = np.sum(item_vectors * weights.reshape(-1, 1), axis=0)
    return weighted_sum / np.sum(weights)
```

### TF-IDF Weighted Profile

```python
def tfidf_profile(liked_items, item_term_matrix, idf_weights):
    """
    Build profile using TF-IDF weighting across user's items.
    """
    # Term frequency across user's liked items
    user_tf = np.sum(item_term_matrix[liked_items], axis=0)

    # Apply IDF weighting
    profile = user_tf * idf_weights

    # Normalize
    return profile / (np.linalg.norm(profile) + 1e-8)
```

## Handling Feature Types

### Categorical Features

```python
from sklearn.preprocessing import MultiLabelBinarizer

def encode_categorical(items, field):
    """One-hot encode multi-valued categorical field."""
    mlb = MultiLabelBinarizer()
    values = [item.get(field, []) for item in items]
    return mlb.fit_transform(values), mlb.classes_
```

### Numerical Features

```python
from sklearn.preprocessing import StandardScaler

def encode_numerical(items, fields):
    """Normalize numerical features."""
    values = np.array([[item.get(f, 0) for f in fields] for item in items])
    scaler = StandardScaler()
    return scaler.fit_transform(values)
```

### Image Features

```python
import torch
from torchvision import models, transforms
from PIL import Image

class ImageFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(tensor)

        return features.squeeze().numpy()
```

## Practical Implementation

### Complete Content-Based Recommender

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ContentBasedRecommender:
    def __init__(self, item_features):
        """
        item_features: dict of {item_id: feature_vector}
        """
        self.item_ids = list(item_features.keys())
        self.item_matrix = np.array([item_features[id] for id in self.item_ids])
        self.id_to_idx = {id: idx for idx, id in enumerate(self.item_ids)}

        # Precompute item-item similarities
        self.item_similarities = cosine_similarity(self.item_matrix)

        # User profiles
        self.user_profiles = {}
        self.user_history = defaultdict(set)

    def update_user_profile(self, user_id, item_id, rating):
        """Update user profile with new interaction."""
        self.user_history[user_id].add(item_id)

        # Rebuild profile
        liked_items = [
            self.item_matrix[self.id_to_idx[iid]]
            for iid in self.user_history[user_id]
        ]

        if liked_items:
            self.user_profiles[user_id] = np.mean(liked_items, axis=0)

    def recommend_for_user(self, user_id, n=10):
        """Recommend items for user based on profile."""
        if user_id not in self.user_profiles:
            # Cold start: return popular items
            return self._popular_items(n)

        profile = self.user_profiles[user_id]
        seen = self.user_history[user_id]

        # Score all unseen items
        scores = cosine_similarity(
            profile.reshape(1, -1),
            self.item_matrix
        ).flatten()

        # Exclude seen items
        for item_id in seen:
            scores[self.id_to_idx[item_id]] = -1

        # Top-n
        top_indices = np.argsort(scores)[::-1][:n]
        return [(self.item_ids[idx], scores[idx]) for idx in top_indices]

    def get_similar_items(self, item_id, n=10):
        """Find items similar to given item."""
        if item_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[item_id]
        similarities = self.item_similarities[idx]

        # Exclude self
        similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:n]
        return [(self.item_ids[i], similarities[i]) for i in top_indices]

    def explain_recommendation(self, user_id, item_id):
        """Explain why item was recommended."""
        if user_id not in self.user_profiles:
            return "Recommended as popular item"

        # Find most similar items user has liked
        item_idx = self.id_to_idx[item_id]
        similarities = []

        for liked_id in self.user_history[user_id]:
            liked_idx = self.id_to_idx[liked_id]
            sim = self.item_similarities[item_idx, liked_idx]
            similarities.append((liked_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:3]

        return f"Similar to items you liked: {top_similar}"

    def _popular_items(self, n):
        """Return popular items for cold start."""
        # In practice, track item popularity
        return [(self.item_ids[i], 0) for i in range(min(n, len(self.item_ids)))]
```

## Evaluation

### Offline Metrics

```python
def evaluate_content_based(recommender, test_data, k=10):
    """
    test_data: list of (user_id, item_id) test interactions
    """
    hits = 0
    ndcg_sum = 0

    for user_id, true_item in test_data:
        recommendations = recommender.recommend_for_user(user_id, n=k)
        rec_items = [item_id for item_id, _ in recommendations]

        # Hit rate
        if true_item in rec_items:
            hits += 1

            # NDCG
            rank = rec_items.index(true_item)
            ndcg_sum += 1 / np.log2(rank + 2)

    n_users = len(test_data)
    return {
        'hit_rate': hits / n_users,
        'ndcg': ndcg_sum / n_users
    }
```

### Coverage and Diversity

```python
def catalog_coverage(recommendations, all_items):
    """Fraction of catalog recommended to any user."""
    recommended_items = set()
    for user_recs in recommendations.values():
        recommended_items.update([item for item, _ in user_recs])
    return len(recommended_items) / len(all_items)

def intra_list_diversity(recommendations, item_similarities):
    """Average dissimilarity within recommendation lists."""
    diversities = []
    for user_recs in recommendations.values():
        items = [item for item, _ in user_recs]
        if len(items) < 2:
            continue

        # Average pairwise dissimilarity
        dissim_sum = 0
        pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = item_similarities[items[i], items[j]]
                dissim_sum += (1 - sim)
                pairs += 1

        diversities.append(dissim_sum / pairs)

    return np.mean(diversities)
```

## Improving Diversity

### Maximal Marginal Relevance (MMR)

Balance relevance with diversity:

```python
def mmr_rerank(query_profile, item_vectors, item_ids, n=10, lambda_=0.5):
    """
    MMR: Select items that are relevant but diverse from already selected.

    lambda_: Trade-off (1=only relevance, 0=only diversity)
    """
    # Initial relevance scores
    relevance = cosine_similarity(
        query_profile.reshape(1, -1),
        item_vectors
    ).flatten()

    selected = []
    remaining = list(range(len(item_ids)))

    while len(selected) < n and remaining:
        mmr_scores = []

        for idx in remaining:
            rel_score = relevance[idx]

            # Max similarity to already selected
            if selected:
                selected_vecs = item_vectors[selected]
                sim_to_selected = cosine_similarity(
                    item_vectors[idx].reshape(1, -1),
                    selected_vecs
                ).max()
            else:
                sim_to_selected = 0

            mmr = lambda_ * rel_score - (1 - lambda_) * sim_to_selected
            mmr_scores.append((idx, mmr))

        # Select highest MMR
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = mmr_scores[0][0]

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [(item_ids[idx], relevance[idx]) for idx in selected]
```

## Comparison with Collaborative Filtering

| Aspect | Content-Based | Collaborative |
|--------|---------------|---------------|
| New items | Handles well | Cold start |
| New users | Cold start | Cold start |
| Serendipity | Low | Higher |
| Explainability | High | Lower |
| Feature engineering | Required | Not needed |
| Scalability | O(items) features | O(users x items) |
| Data needed | Item features | Interactions |

## Hybrid Approaches

Combine content-based with collaborative filtering:

```python
class HybridRecommender:
    def __init__(self, content_model, cf_model, alpha=0.5):
        self.content = content_model
        self.cf = cf_model
        self.alpha = alpha  # Weight for content-based

    def recommend(self, user_id, n=10):
        # Get scores from both models
        content_recs = dict(self.content.recommend(user_id, n=n*2))
        cf_recs = dict(self.cf.recommend(user_id, n=n*2))

        # Combine scores
        all_items = set(content_recs.keys()) | set(cf_recs.keys())
        combined = []

        for item in all_items:
            content_score = content_recs.get(item, 0)
            cf_score = cf_recs.get(item, 0)
            score = self.alpha * content_score + (1 - self.alpha) * cf_score
            combined.append((item, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:n]
```

## When to Use Content-Based Filtering

Content-based is well-suited for:
- New item recommendations (no cold start)
- Explainable recommendations required
- Rich item features available
- Privacy concerns (no need for other users' data)
- Specialized domains with clear feature semantics

Consider alternatives when:
- Limited item features
- Serendipity/discovery important
- Strong collaborative signals available
- User features matter (CF captures implicitly)
