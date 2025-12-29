# Reranking for RAG

## Summary

Reranking is a second-stage retrieval step that improves the precision of retrieved documents by using a more sophisticated model to reorder initial results. While first-stage retrievers (sparse or dense) prioritize recall and speed over large corpora, rerankers trade speed for accuracy on a smaller candidate set. Cross-encoder rerankers that jointly encode query and document achieve the highest quality, while lighter approaches like ColBERT provide a balance between speed and accuracy. Reranking is essential for production RAG systems where retrieval quality directly impacts generation quality.

Key points to remember:

- Two-stage retrieval: Fast retrieval (top-k) followed by accurate reranking (top-n)
- Cross-encoders are most accurate: Joint query-document encoding captures interactions
- Trade-off: Rerankers are slower but more precise than retrievers
- Typical setup: Retrieve 100 candidates, rerank to top 5-10
- ColBERT: Token-level late interaction balances speed and quality
- LLM-based reranking: Use the LLM itself for relevance scoring
- Quality gains: 5-15% improvement in retrieval metrics common

## The Two-Stage Retrieval Pipeline

### Architecture

```
Query
   |
   v
+------------------+
| Stage 1:         |
| Fast Retrieval   |  <- BM25 or dense embeddings
| Top-k candidates |     k = 50-1000
+------------------+
   |
   v (k candidates)
+------------------+
| Stage 2:         |
| Reranking        |  <- Cross-encoder or ColBERT
| Top-n results    |     n = 3-10
+------------------+
   |
   v (n documents)
+------------------+
| LLM Generation   |
+------------------+
```

### Why Two Stages?

| Metric | First Stage | Second Stage |
|--------|-------------|--------------|
| Latency | ~10ms | ~100ms |
| Throughput | High | Low |
| Accuracy | Good | Excellent |
| Corpus size | Millions | Hundreds |

## Cross-Encoder Reranking

### Concept

Cross-encoders jointly encode query and document, allowing full attention between them:

```
Bi-encoder (retrieval):
  Query  -> Encoder -> embedding_q
  Doc    -> Encoder -> embedding_d
  Score = similarity(embedding_q, embedding_d)

Cross-encoder (reranking):
  [Query, SEP, Document] -> Encoder -> Score
  Full attention between query and document tokens
```

### Implementation

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        """Rerank documents for a query."""
        # Create query-document pairs
        pairs = [[query, doc['content']] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [
            {**doc, 'rerank_score': float(score)}
            for doc, score in scored_docs[:top_k]
        ]

# Usage
reranker = CrossEncoderReranker()
candidates = retriever.search(query, k=100)
reranked = reranker.rerank(query, candidates, top_k=5)
```

### Popular Cross-Encoder Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| ms-marco-MiniLM-L-6-v2 | Fast | Good | General |
| ms-marco-MiniLM-L-12-v2 | Medium | Better | Balanced |
| bge-reranker-base | Medium | Very Good | General |
| bge-reranker-large | Slow | Excellent | Max quality |

### Batched Reranking

```python
class BatchedCrossEncoder:
    def __init__(self, model_name, batch_size=32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank_batch(self, queries, documents_list, top_k=5):
        """Rerank multiple queries efficiently."""
        all_pairs = []
        pair_indices = []

        # Flatten all pairs
        for q_idx, (query, docs) in enumerate(zip(queries, documents_list)):
            for d_idx, doc in enumerate(docs):
                all_pairs.append([query, doc['content']])
                pair_indices.append((q_idx, d_idx))

        # Score in batches
        all_scores = []
        for i in range(0, len(all_pairs), self.batch_size):
            batch = all_pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
            all_scores.extend(scores)

        # Reconstruct per-query results
        results = [[] for _ in queries]
        for (q_idx, d_idx), score in zip(pair_indices, all_scores):
            results[q_idx].append((documents_list[q_idx][d_idx], score))

        # Sort and return top-k per query
        return [
            sorted(r, key=lambda x: x[1], reverse=True)[:top_k]
            for r in results
        ]
```

## ColBERT Reranking

### Late Interaction

ColBERT uses token-level matching with late interaction:

```python
class ColBERTReranker:
    """ColBERT-style late interaction reranking."""

    def __init__(self, model_name="colbert-ir/colbertv2.0"):
        from colbert import Searcher
        self.searcher = Searcher(model_name)

    def score(self, query_embeddings, doc_embeddings):
        """MaxSim scoring."""
        # For each query token, find max similarity to any doc token
        # query_embeddings: (q_len, dim)
        # doc_embeddings: (d_len, dim)

        similarities = torch.matmul(query_embeddings, doc_embeddings.T)
        # (q_len, d_len)

        # Max over document tokens for each query token
        max_sims = similarities.max(dim=1).values  # (q_len,)

        # Sum of max similarities
        return max_sims.sum()

    def rerank(self, query, documents, top_k=5):
        """Rerank using MaxSim."""
        query_embs = self.encode_query(query)

        scored = []
        for doc in documents:
            doc_embs = self.encode_doc(doc['content'])
            score = self.score(query_embs, doc_embs)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

### ColBERT Advantages

| Aspect | Cross-Encoder | ColBERT |
|--------|---------------|---------|
| Speed | Slower | Faster |
| Quality | Highest | Very high |
| Precomputation | None possible | Can cache doc embeddings |
| Scalability | Linear in docs | Better scaling |

## LLM-Based Reranking

### Zero-Shot with Prompting

```python
class LLMReranker:
    """Use LLM for reranking via prompting."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def rerank(self, query, documents, top_k=5):
        """Score documents using LLM."""
        scored = []

        for doc in documents:
            prompt = f"""Rate the relevance of this document to the query.
Query: {query}

Document: {doc['content'][:1000]}

Rate from 1-10 where 10 is highly relevant. Return only the number."""

            response = self.llm.generate(prompt, max_tokens=5)
            try:
                score = float(response.strip())
            except:
                score = 0

            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

### Listwise Reranking

```python
class ListwiseLLMReranker:
    """Rerank all documents at once."""

    def rerank(self, query, documents, top_k=5):
        # Create numbered list of document snippets
        doc_list = "\n".join([
            f"[{i+1}] {doc['content'][:200]}..."
            for i, doc in enumerate(documents[:20])  # Limit for context
        ])

        prompt = f"""Given the query and documents below, rank the documents by relevance.
Return the document numbers in order from most to least relevant.

Query: {query}

Documents:
{doc_list}

Return only the numbers separated by commas, e.g., "3, 1, 5, 2, 4"
"""

        response = self.llm.generate(prompt, max_tokens=100)

        # Parse ranking
        try:
            ranking = [int(x.strip()) - 1 for x in response.split(',')]
            ranked_docs = [documents[i] for i in ranking if i < len(documents)]
            return ranked_docs[:top_k]
        except:
            return documents[:top_k]
```

### RankGPT Pattern

```python
class RankGPT:
    """Permutation-based reranking with LLM."""

    def rerank(self, query, documents, window_size=20, step=10):
        """Sliding window bubble-sort style reranking."""
        docs = documents.copy()

        # Multiple passes with sliding window
        for start in range(0, len(docs) - window_size + 1, step):
            window = docs[start:start + window_size]

            # Get LLM to rank window
            ranked_window = self._rank_window(query, window)

            # Replace window with ranked version
            docs[start:start + window_size] = ranked_window

        return docs

    def _rank_window(self, query, window):
        """Rank a window of documents."""
        # Similar to listwise but for smaller window
        pass
```

## Hybrid Reranking

### Combining Signals

```python
class HybridReranker:
    """Combine multiple reranking signals."""

    def __init__(self, cross_encoder, bm25_weight=0.3, dense_weight=0.3, ce_weight=0.4):
        self.cross_encoder = cross_encoder
        self.weights = {
            'bm25': bm25_weight,
            'dense': dense_weight,
            'cross_encoder': ce_weight
        }

    def rerank(self, query, documents, top_k=5):
        """Combine multiple relevance signals."""
        # Get cross-encoder scores
        ce_scores = self._get_ce_scores(query, documents)

        # Normalize all scores to [0, 1]
        bm25_scores = self._normalize([d.get('bm25_score', 0) for d in documents])
        dense_scores = self._normalize([d.get('dense_score', 0) for d in documents])
        ce_scores = self._normalize(ce_scores)

        # Weighted combination
        final_scores = []
        for i, doc in enumerate(documents):
            score = (
                self.weights['bm25'] * bm25_scores[i] +
                self.weights['dense'] * dense_scores[i] +
                self.weights['cross_encoder'] * ce_scores[i]
            )
            final_scores.append((doc, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_scores[:top_k]]

    def _normalize(self, scores):
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
```

## Optimization Techniques

### Caching for Repeated Queries

```python
from functools import lru_cache

class CachedReranker:
    def __init__(self, reranker):
        self.reranker = reranker

    @lru_cache(maxsize=1000)
    def rerank_cached(self, query, doc_ids_tuple):
        """Cache results for repeated queries."""
        # Note: documents must be fetched by ID for caching to work
        documents = self.fetch_documents(doc_ids_tuple)
        return tuple(self.reranker.rerank(query, documents))
```

### Quantized Cross-Encoders

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load with quantization for faster inference
model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    torch_dtype=torch.float16  # FP16 for speed
)

# Or use ONNX for production
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    export=True
)
```

## Evaluation

### Metrics

```python
def evaluate_reranking(queries, ground_truth, retriever, reranker, k=10):
    """Evaluate reranking quality."""
    metrics = {'retrieval': {}, 'reranked': {}}

    for metric_dict, docs_fn in [
        (metrics['retrieval'], lambda q: retriever.search(q, k=100)[:k]),
        (metrics['reranked'], lambda q: reranker.rerank(q, retriever.search(q, k=100), k))
    ]:
        mrr_sum = 0
        ndcg_sum = 0
        recall_sum = 0

        for query, relevant in zip(queries, ground_truth):
            docs = docs_fn(query)
            doc_ids = [d['id'] for d in docs]

            # MRR
            for i, doc_id in enumerate(doc_ids):
                if doc_id in relevant:
                    mrr_sum += 1 / (i + 1)
                    break

            # Recall@k
            recall_sum += len(set(doc_ids) & set(relevant)) / len(relevant)

        n = len(queries)
        metric_dict['MRR@10'] = mrr_sum / n
        metric_dict['Recall@10'] = recall_sum / n

    return metrics
```

## Key Takeaways

1. **Two-stage is standard**: Retrieve broadly, rerank precisely.

2. **Cross-encoders are most accurate**: Full query-document attention captures nuance.

3. **Speed matters**: Choose model size based on latency budget.

4. **ColBERT balances**: Late interaction offers quality close to cross-encoders with better speed.

5. **LLMs can rerank**: Prompting-based reranking works but is expensive.

6. **Combine signals**: Hybrid approaches often outperform single methods.

7. **Measure impact**: Track retrieval metrics before and after reranking.
