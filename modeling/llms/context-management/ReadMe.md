# Context Management for LLMs

## Summary

Context management addresses the fundamental challenge of working within LLM context windows: how to fit relevant information into limited space and extend that space when needed. The two main approaches are long context techniques (scaling position embeddings, efficient attention) and retrieval-augmented generation (RAG). Long context methods extend what the model can process directly, while RAG selectively retrieves relevant information from larger corpora. Most production systems combine both: use RAG to select relevant content, then process it with models capable of handling reasonable context lengths.

Key points to remember:

- Context windows are finite: Even 128K tokens has limits
- Long context: RoPE scaling + sliding window attention extend processing
- RAG: Retrieve relevant chunks from large corpora
- Trade-offs: Long context for full understanding vs RAG for scale
- Combine approaches: RAG to select, long context to process
- Quality degrades: Both approaches have position and relevance challenges
- Production requires: Chunking, retrieval, reranking, monitoring

## The Context Challenge

### Why Context Matters

```
LLM Processing:
Input: [System Prompt] + [Context/Documents] + [User Query]
                          ^^^^^^^^^^^^^^^^
                          The challenge: What goes here?

Options:
1. Short context: Fast but limited information
2. Long context: More info but slower, quality concerns
3. RAG: Select relevant info from large corpus
4. Hybrid: Retrieve broadly, process with long context
```

### Context Window Evolution

| Model | Context Length | Year |
|-------|---------------|------|
| GPT-3 | 2K | 2020 |
| GPT-3.5 | 4K / 16K | 2023 |
| GPT-4 | 8K / 128K | 2023/2024 |
| Claude 2 | 100K | 2023 |
| Claude 3 | 200K | 2024 |
| Gemini 1.5 | 1M | 2024 |

## Approach Comparison

### Long Context vs RAG

| Aspect | Long Context | RAG |
|--------|--------------|-----|
| Information access | Everything in window | Selected chunks |
| Corpus size | Limited by window | Unlimited |
| Latency | Higher for long inputs | Retrieval + generation |
| Cost | More tokens processed | Fewer tokens, more infra |
| Quality | Position effects | Retrieval quality |
| Updates | Need to include | Update index only |

### When to Use Each

```
Use Long Context When:
- Need full document understanding
- Cross-references throughout document
- Document fits in context
- Real-time processing (no index)

Use RAG When:
- Large corpus (beyond context)
- Frequently updated content
- Need citations/provenance
- Cost-sensitive (process less)

Use Hybrid When:
- Large corpus but need context
- Complex reasoning over retrieved docs
- Production systems (common case)
```

## Long Context Techniques

### Position Encoding Extension

```python
# RoPE scaling for extended context
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    rope_scaling={
        "type": "yarn",    # Best quality
        "factor": 4.0      # 4x context extension
    }
)
```

### Efficient Attention

```python
# Sliding window for memory efficiency
# Mistral processes 32K+ with 4K window

# Information propagates through layers:
# Layer 1: See 4K tokens directly
# Layer N: Effective receptive field = 4K * N

# Memory: O(window) instead of O(sequence)
```

### Key Methods

| Method | Mechanism | Trade-off |
|--------|-----------|-----------|
| RoPE Scaling | Adjust position frequencies | Quality at extremes |
| Sliding Window | Local attention | Layer propagation |
| Paged Attention | Memory management | Infra complexity |
| Sparse Attention | Skip connections | Some positions less connected |

## RAG Pipeline

### Architecture

```
Documents -> [Chunking] -> [Embedding] -> Vector Store
                                              |
Query -> [Retrieval] <- -----------------------+
              |
              v
         [Reranking]
              |
              v
         [Augmentation] -> LLM -> Response
```

### Components

```python
class RAGSystem:
    def __init__(self):
        self.chunker = RecursiveChunker(chunk_size=500)
        self.retriever = HybridRetriever(sparse_weight=0.3)
        self.reranker = CrossEncoderReranker()
        self.llm = LLM()

    def ingest(self, documents):
        chunks = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))
        self.retriever.index(chunks)

    def query(self, question, k=5):
        # Retrieve candidates
        candidates = self.retriever.search(question, k=20)

        # Rerank for precision
        top_docs = self.reranker.rerank(question, candidates, k=k)

        # Generate with context
        context = format_context(top_docs)
        return self.llm.generate(question, context)
```

## Hybrid Approaches

### RAG + Long Context

```python
def hybrid_qa(query, corpus, model, max_context=16000):
    """Retrieve broadly, process with long context."""
    # Step 1: Retrieve more than we need
    candidates = retriever.search(query, k=50)

    # Step 2: Fit as much as possible in context
    context_docs = []
    total_tokens = 0

    for doc in candidates:
        doc_tokens = count_tokens(doc)
        if total_tokens + doc_tokens < max_context:
            context_docs.append(doc)
            total_tokens += doc_tokens
        else:
            break

    # Step 3: Process with long context model
    context = format_context(context_docs)
    return model.generate(query, context)
```

### Multi-Stage Retrieval

```python
def multi_stage_retrieval(query, retriever, reranker, k_retrieve=100, k_rerank=20, k_final=5):
    """Multiple refinement stages."""
    # Stage 1: Broad retrieval
    candidates = retriever.search(query, k=k_retrieve)

    # Stage 2: Rerank with cross-encoder
    reranked = reranker.rerank(query, candidates, k=k_rerank)

    # Stage 3: Diversity sampling
    diverse = select_diverse(reranked, k=k_final)

    return diverse
```

## Quality Considerations

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Lost in middle | Middle content ignored | Reorder important info |
| Retrieval miss | Answer not in retrieved | Improve retrieval, expand k |
| Context overflow | Truncation | Better chunking, prioritize |
| Irrelevant context | Noise in generation | Stricter reranking |

### Evaluation

```python
def evaluate_context_system(system, test_cases):
    """Evaluate context management quality."""
    metrics = {
        'answer_quality': [],
        'retrieval_recall': [],
        'context_utilization': []
    }

    for case in test_cases:
        result = system.query(case['query'])

        # Does answer match ground truth?
        metrics['answer_quality'].append(
            score_answer(result['answer'], case['expected'])
        )

        # Was relevant info retrieved?
        metrics['retrieval_recall'].append(
            recall_at_k(result['retrieved'], case['relevant_ids'])
        )

        # Was retrieved context used?
        metrics['context_utilization'].append(
            context_usage(result['answer'], result['retrieved'])
        )

    return {k: np.mean(v) for k, v in metrics.items()}
```

## Production Architecture

### System Design

```
User Query
     |
     v
[Query Understanding]  <- Classify, expand
     |
     v
[Retrieval Layer]
  - Vector Search (FAISS/Pinecone)
  - Sparse Search (Elasticsearch)
  - Hybrid Fusion
     |
     v
[Reranking Layer]
  - Cross-encoder
  - Diversity
     |
     v
[Context Assembly]
  - Truncation
  - Ordering
     |
     v
[Generation Layer]
  - Long context LLM
  - Streaming
     |
     v
[Post-processing]
  - Citation extraction
  - Verification
```

### Infrastructure Components

| Component | Options |
|-----------|---------|
| Vector Store | Pinecone, Weaviate, Qdrant, FAISS |
| Sparse Index | Elasticsearch, OpenSearch |
| Reranker | Cross-encoder models, Cohere |
| LLM | GPT-4, Claude, LLaMA |
| Orchestration | LangChain, LlamaIndex, custom |

## Key Takeaways

1. **Context is finite**: Even 1M tokens has limits; manage carefully.

2. **Long context for depth**: When you need full document understanding.

3. **RAG for breadth**: When corpus exceeds any context window.

4. **Combine approaches**: Retrieve to select, long context to process.

5. **Quality compounds**: Each stage (chunk, retrieve, rerank) matters.

6. **Test thoroughly**: Needle-in-haystack, retrieval metrics, end-to-end.

7. **Monitor in production**: Track retrieval quality, latency, costs.

## Further Reading

For detailed coverage of context management techniques, see:

- [Long Context](long-context/ReadMe.md) - RoPE scaling, sliding window attention
- [RAG](rag/ReadMe.md) - Retrieval-augmented generation pipeline
