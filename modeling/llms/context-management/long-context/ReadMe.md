# Long Context for LLMs

## Summary

Long context techniques extend LLMs beyond their training context lengths, enabling processing of lengthy documents, codebases, and multi-turn conversations. The main approaches are RoPE scaling (modifying position embeddings to handle longer sequences) and sliding window attention (limiting each token's attention span for computational efficiency). These techniques, combined with inference optimizations like paged attention, have pushed practical context limits from 4K tokens to 128K+ while maintaining acceptable quality and reasonable compute costs.

Key points to remember:

- RoPE scaling: Modify rotation frequencies for extended positions
- Sliding window: Limit attention to local window, rely on layer propagation
- Trade-offs: Longer context typically means higher cost or reduced per-position quality
- Combine techniques: Scaling + efficient attention + paged memory
- Test quality: Use needle-in-haystack and retrieval benchmarks
- Consider alternatives: RAG may be better for very long documents
- Memory is the bottleneck: KV-cache optimization is critical

## The Long Context Challenge

### Why Context Length Matters

```
Short context (4K tokens):
- Simple Q&A
- Short document summarization
- Single code file

Medium context (16K-32K tokens):
- Multi-document reasoning
- Book chapters
- Codebase understanding

Long context (100K+ tokens):
- Full books
- Entire codebases
- Long conversation histories
- Complex multi-step reasoning
```

### Fundamental Limitations

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Position extrapolation | Models don't generalize to unseen positions | RoPE scaling |
| Quadratic attention | O(n^2) compute and memory | Sliding window, sparse attention |
| KV-cache growth | Memory grows linearly with length | Paged attention, rolling buffer |
| Quality degradation | Attention dilution over long sequences | Better architectures, fine-tuning |

## Approaches Overview

### Position Encoding Solutions

```
Problem: RoPE trained on positions 0-4095 fails at position 8000

Solutions:
1. Position Interpolation
   - Compress: position 8000 -> 4000
   - Simple but loses resolution

2. NTK-Aware Scaling
   - Adjust base frequency
   - Preserves local patterns better

3. YaRN
   - Frequency-specific scaling
   - Best quality, more complex
```

### Attention Efficiency Solutions

```
Problem: Full attention is O(n^2)

Solutions:
1. Sliding Window (Mistral)
   - Each token attends to window of w tokens
   - O(n*w) complexity
   - Info propagates through layers

2. Sparse Attention Patterns
   - Mix local + global tokens
   - Longformer, BigBird patterns

3. Linear Attention
   - Approximate attention in O(n)
   - Quality trade-offs
```

## Method Comparison

### Scaling Techniques

| Method | Quality | Complexity | Fine-tuning |
|--------|---------|------------|-------------|
| Position Interpolation | Good | Low | Recommended |
| NTK-Aware | Better | Low | Optional |
| Dynamic NTK | Good | Low | Optional |
| YaRN | Best | Medium | Recommended |

### Attention Techniques

| Method | Complexity | Memory | Quality Trade-off |
|--------|------------|--------|------------------|
| Full Attention | O(n^2) | O(n) | Baseline |
| Sliding Window | O(n*w) | O(w) | Minor (via layers) |
| Sparse Patterns | O(n*k) | O(k) | Varies |
| Linear Attention | O(n) | O(d) | Larger |

## Practical Implementation

### Extending LLaMA-style Models

```python
from transformers import AutoModelForCausalLM

# Method 1: RoPE scaling only
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    rope_scaling={
        "type": "yarn",  # or "linear", "dynamic"
        "factor": 4.0    # Extend 4K -> 16K
    },
    torch_dtype=torch.float16
)

# Method 2: Sliding window (Mistral has this built-in)
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
)
print(mistral.config.sliding_window)  # 4096
```

### Fine-tuning for Long Context

```python
from transformers import Trainer, TrainingArguments

# Key settings for long context fine-tuning
training_args = TrainingArguments(
    output_dir="./extended-context-model",
    per_device_train_batch_size=1,       # Small batch for memory
    gradient_accumulation_steps=16,      # Maintain effective batch size
    gradient_checkpointing=True,         # Essential for long sequences
    learning_rate=2e-5,
    max_steps=1000,                       # Brief fine-tuning usually sufficient
    fp16=True,
    dataloader_num_workers=4,
)

# Use data with varying context lengths
train_dataset = LongContextDataset(
    max_length=16384,
    include_short=True,  # Mix of lengths helps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### Memory Management

```python
# Paged attention (vLLM-style)
from vllm import LLM

llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    max_model_len=32768,  # Extended context
    gpu_memory_utilization=0.9,
)

# Inference with long context
output = llm.generate(
    long_document + "\n\nSummarize:",
    max_tokens=500
)
```

## Quality Evaluation

### Needle-in-Haystack Test

```python
def needle_test(model, tokenizer, lengths, depths):
    """Test retrieval at various positions."""
    results = {}

    needle = "The secret password is: elephant42"
    question = "What is the secret password?"

    for length in lengths:
        for depth in depths:
            # Create context with needle at specified depth
            context = create_haystack(length, needle, depth)
            prompt = f"{context}\n\n{question}"

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
            response = tokenizer.decode(outputs[0])

            # Check if needle was retrieved
            success = "elephant42" in response
            results[(length, depth)] = success

    return results

# Run test
results = needle_test(
    model, tokenizer,
    lengths=[4096, 8192, 16384, 32768],
    depths=[0.0, 0.25, 0.5, 0.75, 1.0]  # Position in context
)
```

### Common Quality Issues

| Issue | Symptom | Mitigation |
|-------|---------|------------|
| Lost-in-middle | Poor recall for middle content | Reorder important info |
| Attention dilution | Weak connections to distant content | Shorter context or RAG |
| Position artifacts | Better at certain positions | Varied training data |

## When to Use Long Context vs RAG

### Use Long Context When

- Document fits comfortably (< 100K tokens)
- Need full context for reasoning
- Multiple cross-document references
- Code understanding (need full file)

### Use RAG When

- Very large corpora
- Frequently updated content
- Need specific passages, not full context
- Cost/latency constraints

### Hybrid Approach

```python
def hybrid_retrieval(query, documents, model, max_context=16000):
    """Combine RAG with long context."""
    # Step 1: Retrieve potentially relevant docs
    candidates = retriever.search(query, k=10)

    # Step 2: If fits in context, use all
    total_tokens = sum(len(tokenize(d)) for d in candidates)

    if total_tokens < max_context:
        context = "\n\n".join(candidates)
    else:
        # Step 3: Otherwise, rerank and truncate
        ranked = reranker.rerank(query, candidates)
        context = ""
        for doc in ranked:
            if len(tokenize(context + doc)) < max_context:
                context += "\n\n" + doc
            else:
                break

    # Step 4: Generate with long context model
    return model.generate(f"{context}\n\nQuestion: {query}")
```

## Key Takeaways

1. **Position scaling enables extension**: RoPE scaling lets models handle unseen positions.

2. **Efficiency is critical**: Sliding window and paged attention make long context practical.

3. **Quality degrades gracefully**: Most tasks work well with properly scaled models.

4. **Fine-tuning helps**: Brief training on long data improves quality significantly.

5. **Test thoroughly**: Needle-in-haystack reveals position-specific weaknesses.

6. **Memory is the bottleneck**: Focus on KV-cache optimization.

7. **Consider RAG**: For very long documents, retrieval may be more effective than long context.

## Further Reading

For detailed coverage of specific techniques, see:

- [RoPE Scaling](rope-scaling/ReadMe.md) - Position interpolation, NTK, YaRN
- [Sliding Window Attention](sliding-window-attention/ReadMe.md) - Local attention with layer propagation
