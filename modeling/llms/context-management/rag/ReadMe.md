# Retrieval-Augmented Generation (RAG)

## Summary

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant documents from an external knowledge base and including them in the prompt context. This enables LLMs to access current information beyond their training cutoff, provide citations, and ground responses in specific sources. RAG systems involve three core components: chunking documents into retrievable units, retrieving relevant chunks for a query, and optionally reranking results for precision. Effective RAG requires careful attention to each component.

Key points to remember:

- RAG = Retrieval + Generation: Ground LLM responses in external knowledge
- Three components: Chunking, retrieval, reranking (optional)
- Chunking trade-offs: Size affects precision vs context preservation
- Retrieval methods: Sparse (BM25), dense (embeddings), hybrid (both)
- Reranking improves precision: Cross-encoders refine initial retrieval
- Quality compounds: Each stage affects final generation quality
- Production requires: Vector stores, caching, monitoring

## RAG Architecture

### Pipeline Overview

```
User Query
     |
     v
+-------------+
| Retrieval   |  <- Find relevant documents
+-------------+
     |
     v (Top-k documents)
+-------------+
| Reranking   |  <- Refine relevance (optional)
+-------------+
     |
     v (Top-n documents)
+-------------+
| Augment     |  <- Add documents to prompt
+-------------+
     |
     v
+-------------+
| Generate    |  <- LLM produces response
+-------------+
     |
     v
Response (with citations)
```

### Why RAG?

| Challenge | RAG Solution |
|-----------|--------------|
| Knowledge cutoff | Access current information |
| Hallucinations | Ground in sources |
| Domain specificity | Use specialized documents |
| Verifiability | Provide citations |
| Updates | Update index, not model |

## Core Components

### 1. Document Processing

```python
class RAGPipeline:
    def __init__(self, chunker, retriever, reranker=None, llm=None):
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def ingest(self, documents):
        """Process and index documents."""
        all_chunks = []

        for doc in documents:
            # Chunk document
            chunks = self.chunker.chunk(doc['content'])

            # Add metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{doc['id']}_chunk_{i}",
                    'content': chunk,
                    'source': doc['source'],
                    'metadata': doc.get('metadata', {})
                })

        # Index chunks
        self.retriever.index_documents(all_chunks)

        return len(all_chunks)
```

### 2. Retrieval

```python
    def retrieve(self, query, k=10):
        """Retrieve relevant chunks."""
        # Initial retrieval
        candidates = self.retriever.search(query, k=k * 2)

        # Optional reranking
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k=k)
        else:
            candidates = candidates[:k]

        return candidates
```

### 3. Generation

```python
    def generate(self, query, k=5):
        """Generate response with retrieved context."""
        # Retrieve
        chunks = self.retrieve(query, k=k)

        # Build prompt
        context = "\n\n".join([
            f"[{i+1}] {chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""Use the following sources to answer the question.
Cite sources using [1], [2], etc.

Sources:
{context}

Question: {query}

Answer:"""

        # Generate
        response = self.llm.generate(prompt)

        return {
            'answer': response,
            'sources': chunks
        }
```

## RAG Patterns

### Basic RAG

```python
def basic_rag(query, retriever, llm, k=5):
    """Simple retrieve-then-generate."""
    docs = retriever.search(query, k=k)
    context = format_context(docs)

    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.generate(prompt)
```

### Self-RAG (Reflective)

```python
def self_rag(query, retriever, llm, k=5):
    """Retrieve, generate, verify, regenerate if needed."""
    docs = retriever.search(query, k=k)
    response = generate_with_context(query, docs, llm)

    # Verify response
    verification_prompt = f"""
Does this response accurately reflect the sources?
Sources: {format_context(docs)}
Response: {response}
Answer Yes or No with explanation."""

    verification = llm.generate(verification_prompt)

    if "No" in verification:
        # Regenerate with more careful instructions
        response = generate_with_verification_feedback(
            query, docs, verification, llm
        )

    return response
```

### Corrective RAG (CRAG)

```python
def corrective_rag(query, retriever, llm, web_search, k=5):
    """Evaluate relevance and fall back if needed."""
    docs = retriever.search(query, k=k)

    # Evaluate document relevance
    relevance_scores = evaluate_relevance(query, docs, llm)

    if max(relevance_scores) < 0.5:
        # Poor retrieval, try web search
        web_docs = web_search(query)
        docs = docs + web_docs

    # Filter to relevant only
    relevant_docs = [d for d, s in zip(docs, relevance_scores) if s > 0.3]

    return generate_with_context(query, relevant_docs, llm)
```

### Agentic RAG

```python
def agentic_rag(query, retriever, llm, tools):
    """Multi-step retrieval with reasoning."""
    context = []
    max_steps = 5

    for step in range(max_steps):
        # Decide next action
        action = decide_action(query, context, llm)

        if action['type'] == 'retrieve':
            # Retrieve for specific sub-query
            docs = retriever.search(action['sub_query'])
            context.extend(docs)

        elif action['type'] == 'answer':
            # Ready to answer
            return generate_final_answer(query, context, llm)

        elif action['type'] == 'refine':
            # Need different search
            refined_query = action['refined_query']
            docs = retriever.search(refined_query)
            context.extend(docs)

    return generate_final_answer(query, context, llm)
```

## Prompt Patterns

### Basic Citation

```python
CITATION_PROMPT = """Use the following sources to answer the question.
Cite your sources using [1], [2], etc.

Sources:
{context}

Question: {question}

Provide a comprehensive answer with citations:"""
```

### Structured Output

```python
STRUCTURED_PROMPT = """Based on the sources below, provide a structured answer.

Sources:
{context}

Question: {question}

Format your response as:
SUMMARY: Brief answer
DETAILS: Detailed explanation with citations [1], [2]
CONFIDENCE: High/Medium/Low based on source quality
SOURCES_USED: List of source numbers used"""
```

### Chain-of-Thought with RAG

```python
COT_RAG_PROMPT = """I'll reason through this step by step using the provided sources.

Sources:
{context}

Question: {question}

Let me think through this:
1. First, I'll identify the key facts from the sources...
2. Then, I'll connect them to answer the question...
3. Finally, I'll synthesize my answer...

Step-by-step reasoning:"""
```

## Evaluation

### Retrieval Metrics

```python
def evaluate_retrieval(queries, ground_truth, retriever, k=10):
    """Evaluate retrieval quality."""
    metrics = {'recall': [], 'precision': [], 'mrr': []}

    for query, relevant_ids in zip(queries, ground_truth):
        retrieved = retriever.search(query, k=k)
        retrieved_ids = [r['id'] for r in retrieved]

        # Recall@k
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
        metrics['recall'].append(recall)

        # Precision@k
        precision = len(set(retrieved_ids) & set(relevant_ids)) / k
        metrics['precision'].append(precision)

        # MRR
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                metrics['mrr'].append(1 / (i + 1))
                break
        else:
            metrics['mrr'].append(0)

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

### End-to-End Evaluation

```python
def evaluate_rag(rag_pipeline, test_cases):
    """Evaluate full RAG pipeline."""
    results = {
        'answer_relevance': [],
        'faithfulness': [],
        'context_relevance': []
    }

    for case in test_cases:
        response = rag_pipeline.generate(case['query'])

        # Answer relevance: Does answer address the question?
        relevance = judge_relevance(
            case['query'], response['answer']
        )
        results['answer_relevance'].append(relevance)

        # Faithfulness: Is answer supported by sources?
        faithfulness = judge_faithfulness(
            response['answer'], response['sources']
        )
        results['faithfulness'].append(faithfulness)

        # Context relevance: Are retrieved docs relevant?
        context_rel = judge_context_relevance(
            case['query'], response['sources']
        )
        results['context_relevance'].append(context_rel)

    return {k: sum(v)/len(v) for k, v in results.items()}
```

## Production Considerations

### Caching

```python
class CachedRAG:
    def __init__(self, rag_pipeline, cache):
        self.rag = rag_pipeline
        self.cache = cache

    def generate(self, query, **kwargs):
        # Check cache
        cache_key = self._make_key(query, kwargs)
        cached = self.cache.get(cache_key)

        if cached:
            return cached

        # Generate and cache
        response = self.rag.generate(query, **kwargs)
        self.cache.set(cache_key, response, ttl=3600)

        return response
```

### Monitoring

```python
class MonitoredRAG:
    def generate(self, query, **kwargs):
        start = time.time()

        # Retrieve with timing
        retrieve_start = time.time()
        docs = self.retrieve(query)
        retrieve_time = time.time() - retrieve_start

        # Generate with timing
        generate_start = time.time()
        response = self._generate(query, docs)
        generate_time = time.time() - generate_start

        # Log metrics
        self.metrics.record({
            'query': query,
            'retrieve_time_ms': retrieve_time * 1000,
            'generate_time_ms': generate_time * 1000,
            'num_docs_retrieved': len(docs),
            'response_length': len(response['answer'])
        })

        return response
```

## Key Takeaways

1. **RAG grounds responses**: External knowledge reduces hallucinations.

2. **Quality compounds**: Chunking affects retrieval affects generation.

3. **Hybrid retrieval is robust**: Combine sparse and dense methods.

4. **Reranking improves precision**: Worth the latency for quality-critical apps.

5. **Evaluate holistically**: Measure retrieval and generation quality.

6. **Patterns matter**: Self-RAG, CRAG, agentic patterns improve quality.

7. **Production needs infrastructure**: Caching, monitoring, vector stores.

## Further Reading

For detailed coverage of RAG components, see:

- [Chunking Strategies](chunking-strategies/ReadMe.md) - Document splitting approaches
- [Retrieval Methods](retrieval-methods/ReadMe.md) - Sparse, dense, hybrid retrieval
- [Reranking](reranking/ReadMe.md) - Cross-encoders and refinement
