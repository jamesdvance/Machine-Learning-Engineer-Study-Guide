# Extractive Summarization

## Summary

Extractive summarization creates summaries by selecting and concatenating the most important sentences from the source document. Unlike abstractive methods, extractive approaches guarantee that output text comes directly from the source, avoiding hallucination. The task involves scoring sentences by importance and selecting a subset that covers key information while minimizing redundancy.

Key points to remember:

- Selection-based: Chooses existing sentences rather than generating new text
- No hallucination: Output is verbatim from source document
- Sentence scoring: Rank sentences by importance using various signals
- Redundancy handling: MMR and other methods prevent repetitive selections
- Graph-based: TextRank and LexRank model document as sentence graph
- Neural approaches: BERT-based models learn sentence importance
- Trade-off: Less fluent than abstractive but more faithful
- Position bias: Lead sentences often most important (especially news)

## Core Approaches

### Sentence Scoring Methods

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class SentenceScorer:
    """Multiple methods for scoring sentence importance."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(sentences)

    def tfidf_score(self, sentence_idx):
        """Score based on average TF-IDF of terms."""
        row = self.tfidf_matrix[sentence_idx].toarray()[0]
        return np.mean(row[row > 0])

    def position_score(self, sentence_idx, decay=0.9):
        """Earlier sentences often more important."""
        return decay ** sentence_idx

    def length_score(self, sentence_idx, optimal_length=20):
        """Prefer sentences of moderate length."""
        words = len(self.sentences[sentence_idx].split())
        return 1 - abs(words - optimal_length) / optimal_length

    def keyword_score(self, sentence_idx, keywords):
        """Score based on keyword presence."""
        sentence_words = set(self.sentences[sentence_idx].lower().split())
        overlap = len(sentence_words & set(keywords))
        return overlap / len(keywords) if keywords else 0

    def combined_score(self, sentence_idx, weights=None):
        """Combine multiple scoring methods."""
        if weights is None:
            weights = {'tfidf': 0.4, 'position': 0.3, 'length': 0.3}

        score = 0
        score += weights.get('tfidf', 0) * self.tfidf_score(sentence_idx)
        score += weights.get('position', 0) * self.position_score(sentence_idx)
        score += weights.get('length', 0) * self.length_score(sentence_idx)
        return score
```

### TextRank Algorithm

Graph-based approach inspired by PageRank:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextRank:
    """TextRank for extractive summarization."""

    def __init__(self, damping=0.85, min_diff=1e-5, max_iter=100):
        self.damping = damping
        self.min_diff = min_diff
        self.max_iter = max_iter

    def summarize(self, sentences, embeddings, num_sentences=3):
        # Build similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Normalize to create transition matrix
        np.fill_diagonal(similarity_matrix, 0)
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / row_sums

        # Initialize scores
        n = len(sentences)
        scores = np.ones(n) / n

        # Power iteration
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = (1 - self.damping) / n + self.damping * transition_matrix.T @ scores

            if np.abs(scores - prev_scores).sum() < self.min_diff:
                break

        # Select top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        # Return in original order for coherence
        return sorted(ranked_indices)

# Usage
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["First sentence.", "Second sentence.", "Third sentence."]
embeddings = encoder.encode(sentences)

textrank = TextRank()
summary_indices = textrank.summarize(sentences, embeddings, num_sentences=2)
summary = [sentences[i] for i in summary_indices]
```

### LexRank

Similar to TextRank but with threshold-based graph construction:

```python
class LexRank:
    """LexRank with IDF-modified cosine similarity."""

    def __init__(self, threshold=0.1, damping=0.85):
        self.threshold = threshold
        self.damping = damping

    def summarize(self, sentences, num_sentences=3):
        # Compute IDF-weighted TF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Apply threshold to create graph
        adjacency_matrix = (similarity_matrix > self.threshold).astype(float)
        np.fill_diagonal(adjacency_matrix, 0)

        # Compute degree centrality as fallback
        degrees = adjacency_matrix.sum(axis=1)

        # Run PageRank on adjacency matrix
        n = len(sentences)
        scores = self._pagerank(adjacency_matrix, n)

        # Select top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        return sorted(ranked_indices)

    def _pagerank(self, adjacency, n, max_iter=100):
        scores = np.ones(n) / n
        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition = adjacency / row_sums

        for _ in range(max_iter):
            scores = (1 - self.damping) / n + self.damping * transition.T @ scores

        return scores
```

## Neural Extractive Summarization

### BERT-based Sentence Selection

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertSumExt(nn.Module):
    """BERT-based extractive summarization (BertSum style)."""

    def __init__(self, bert_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, sentence_positions):
        """
        Args:
            input_ids: Document tokens with [CLS] between sentences
            attention_mask: Attention mask
            sentence_positions: Positions of [CLS] tokens for each sentence
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Extract [CLS] representations for each sentence
        sentence_vectors = hidden_states[:, sentence_positions, :]

        # Classify each sentence
        logits = self.classifier(sentence_vectors).squeeze(-1)
        return logits

class BertSumExtractor:
    """Wrapper for inference."""

    def __init__(self, model_path):
        self.model = BertSumExt()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def summarize(self, sentences, num_sentences=3):
        # Prepare input with [CLS] between sentences
        tokens = ['[CLS]']
        sentence_positions = [0]

        for sent in sentences:
            sent_tokens = self.tokenizer.tokenize(sent)
            tokens.extend(sent_tokens)
            tokens.append('[CLS]')
            sentence_positions.append(len(tokens) - 1)

        # Remove last [CLS]
        tokens = tokens[:-1]
        sentence_positions = sentence_positions[:-1]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.ones_like(input_ids)

        # Get scores
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, sentence_positions)

        # Select top sentences
        scores = torch.sigmoid(logits[0]).numpy()
        top_indices = np.argsort(scores)[::-1][:num_sentences]

        return sorted(top_indices)
```

### Transformer with Inter-Sentence Attention

```python
class TransformerSummarizer(nn.Module):
    """Transformer layers on top of sentence embeddings."""

    def __init__(self, hidden_dim=768, num_heads=8, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, sentence_embeddings, padding_mask=None):
        """
        Args:
            sentence_embeddings: [batch, num_sentences, hidden_dim]
            padding_mask: [batch, num_sentences] True for padded positions
        """
        # Inter-sentence attention
        contextualized = self.transformer(
            sentence_embeddings,
            src_key_padding_mask=padding_mask
        )

        # Classify each sentence
        logits = self.classifier(contextualized).squeeze(-1)
        return logits
```

## Redundancy Handling

### Maximal Marginal Relevance (MMR)

```python
from sklearn.metrics.pairwise import cosine_similarity

class MMRSelector:
    """Select sentences balancing relevance and diversity."""

    def __init__(self, lambda_param=0.5):
        self.lambda_param = lambda_param

    def select(self, sentences, embeddings, query_embedding=None, num_sentences=3):
        """
        Select sentences using MMR.

        MMR = » * Sim(s, query) - (1-») * max(Sim(s, selected))
        """
        if query_embedding is None:
            # Use document centroid as query
            query_embedding = embeddings.mean(axis=0)

        selected_indices = []
        remaining_indices = list(range(len(sentences)))

        while len(selected_indices) < num_sentences and remaining_indices:
            best_score = float('-inf')
            best_idx = None

            for idx in remaining_indices:
                # Relevance to query
                relevance = cosine_similarity(
                    [embeddings[idx]], [query_embedding]
                )[0][0]

                # Similarity to already selected
                if selected_indices:
                    selected_embeddings = embeddings[selected_indices]
                    max_similarity = cosine_similarity(
                        [embeddings[idx]], selected_embeddings
                    ).max()
                else:
                    max_similarity = 0

                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return sorted(selected_indices)
```

### Trigram Blocking

```python
def trigram_blocking_select(sentences, scores, num_sentences=3):
    """Select sentences while blocking trigram overlap."""
    selected = []
    selected_trigrams = set()

    # Sort by score
    scored_sentences = sorted(
        enumerate(sentences),
        key=lambda x: scores[x[0]],
        reverse=True
    )

    for idx, sentence in scored_sentences:
        if len(selected) >= num_sentences:
            break

        # Get trigrams
        words = sentence.lower().split()
        trigrams = set(zip(words, words[1:], words[2:]))

        # Check overlap
        overlap = trigrams & selected_trigrams
        overlap_ratio = len(overlap) / max(len(trigrams), 1)

        if overlap_ratio < 0.3:  # Allow up to 30% overlap
            selected.append(idx)
            selected_trigrams.update(trigrams)

    return sorted(selected)
```

## Evaluation

### ROUGE Metrics

```python
from rouge_score import rouge_scorer

def evaluate_extractive_summary(system_summary, reference_summary):
    """Evaluate with ROUGE metrics."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference_summary, system_summary)

    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall
    }
```

### Oracle Extractive Summary

```python
def compute_oracle_summary(sentences, reference, num_sentences=3):
    """
    Find the combination of sentences that maximizes ROUGE.
    Used as upper bound for extractive methods.
    """
    from itertools import combinations

    best_score = 0
    best_combination = None

    for combo in combinations(range(len(sentences)), num_sentences):
        summary = ' '.join([sentences[i] for i in combo])
        score = evaluate_extractive_summary(summary, reference)['rouge2_f']

        if score > best_score:
            best_score = score
            best_combination = combo

    return list(best_combination), best_score
```

## Multi-Document Summarization

### Cluster-Based Approach

```python
from sklearn.cluster import KMeans

class MultiDocumentSummarizer:
    """Summarize multiple documents by clustering sentences."""

    def __init__(self, encoder):
        self.encoder = encoder

    def summarize(self, documents, num_sentences=5):
        # Collect all sentences
        all_sentences = []
        for doc in documents:
            all_sentences.extend(self._split_sentences(doc))

        # Encode sentences
        embeddings = self.encoder.encode(all_sentences)

        # Cluster sentences
        num_clusters = min(num_sentences, len(all_sentences))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Select sentence closest to each centroid
        selected = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]

            # Find closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(closest_idx)

        # Return sentences
        return [all_sentences[i] for i in sorted(selected)]
```

### Chronological Ordering

```python
def order_sentences_chronologically(sentences, document_positions):
    """
    Order selected sentences by their position in source documents.

    Args:
        sentences: List of selected sentences
        document_positions: List of (doc_idx, sent_idx) tuples
    """
    # Sort by document index, then sentence index
    ordered = sorted(
        zip(sentences, document_positions),
        key=lambda x: (x[1][0], x[1][1])
    )
    return [sent for sent, pos in ordered]
```

## Comparison: Extractive vs Abstractive

| Aspect | Extractive | Abstractive |
|--------|------------|-------------|
| Output | Verbatim sentences | Generated text |
| Hallucination | None | Possible |
| Fluency | Lower (sentence boundaries) | Higher |
| Compression | Limited | Higher |
| Training data | Sentence labels | Reference summaries |
| Speed | Generally faster | Slower |
| Faithfulness | Guaranteed | Requires verification |

## Benchmark Datasets

| Dataset | Domain | Documents | Summary Type |
|---------|--------|-----------|--------------|
| CNN/DailyMail | News | 300K | Highlights |
| NYT | News | 650K | Abstract |
| arXiv/PubMed | Scientific | 400K | Abstract |
| WikiHow | How-to | 230K | Summaries |
| Reddit TIFU | Social | 120K | TL;DR |
| Multi-News | News | 56K | Multi-document |

## Production Considerations

### Sentence Boundary Detection

```python
import spacy

class SentenceSplitter:
    """Robust sentence splitting for summarization."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split(self, text):
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            # Clean up sentence
            clean_sent = sent.text.strip()

            # Filter too short or too long
            word_count = len(clean_sent.split())
            if 5 <= word_count <= 100:
                sentences.append(clean_sent)

        return sentences
```

### Summary Length Control

```python
def select_sentences_by_length(sentences, scores, target_words=100):
    """Select sentences to meet target word count."""
    selected = []
    current_words = 0

    # Sort by score
    scored = sorted(enumerate(sentences), key=lambda x: scores[x[0]], reverse=True)

    for idx, sentence in scored:
        words = len(sentence.split())

        if current_words + words <= target_words * 1.1:  # 10% tolerance
            selected.append(idx)
            current_words += words

        if current_words >= target_words * 0.9:
            break

    return sorted(selected)
```

## Best Practices

### When to Use Extractive

| Scenario | Recommendation |
|----------|----------------|
| Legal documents | Extractive (preserve exact wording) |
| News highlights | Extractive or hybrid |
| Meeting notes | Extractive (capture decisions) |
| Scientific papers | Often abstractive better |
| Social media | Abstractive (compression needed) |
| High-stakes applications | Extractive (verifiable) |

### Quality Checklist

```python
def validate_extractive_summary(summary_sentences, source_sentences):
    """Validate extractive summary quality."""
    issues = []

    # Check sentences exist in source
    for sent in summary_sentences:
        if sent not in source_sentences:
            issues.append(f"Sentence not from source: {sent[:50]}...")

    # Check redundancy
    for i, s1 in enumerate(summary_sentences):
        for j, s2 in enumerate(summary_sentences[i+1:], i+1):
            if compute_similarity(s1, s2) > 0.8:
                issues.append(f"Redundant sentences: {i} and {j}")

    # Check coherence (adjacent sentences should be related)
    for i in range(len(summary_sentences) - 1):
        sim = compute_similarity(summary_sentences[i], summary_sentences[i+1])
        if sim < 0.1:
            issues.append(f"Low coherence between sentences {i} and {i+1}")

    return issues
```
