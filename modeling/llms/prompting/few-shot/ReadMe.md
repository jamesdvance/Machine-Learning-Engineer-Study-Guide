# Few-Shot Prompting

## Summary

Few-shot prompting teaches LLMs to perform tasks by providing a small number of input-output examples in the prompt, rather than through gradient-based training. This in-context learning allows models to generalize to new inputs by pattern matching against the examples. The technique works because large language models can recognize patterns and apply them without parameter updates. Few-shot prompting is the foundation of prompt engineering, enabling task adaptation without fine-tuning, making it practical for rapid prototyping and deployment.

Key points to remember:

- Examples as instruction: Demonstrate task with 3-10 input-output pairs
- No training required: Model learns from context, not gradient updates
- Example quality matters: Clear, representative examples improve performance
- Order effects: Example sequence can significantly impact results
- Format consistency: Consistent structure helps model pattern-match
- Complementary to CoT: Combine with chain-of-thought for reasoning tasks
- Model-size dependent: Larger models are better few-shot learners

## The Core Concept

### Zero-Shot vs Few-Shot

```
Zero-Shot (no examples):
Classify the sentiment: "I love this product!"
Sentiment:

Few-Shot (with examples):
Classify the sentiment:
Text: "This is amazing!" ’ Positive
Text: "Terrible experience." ’ Negative
Text: "It was okay." ’ Neutral
Text: "I love this product!" ’

Key insight:
- Zero-shot relies on instruction understanding
- Few-shot relies on pattern recognition
- Few-shot often more reliable for specific formats
```

### How In-Context Learning Works

```
Theoretical explanations:
1. Pattern completion: Model treats examples as context, predicts next
2. Implicit fine-tuning: Attention heads implement learning algorithms
3. Task location: Examples help model find relevant pretraining knowledge
4. Format specification: Examples define expected output structure

Practical observation:
- Works better with more examples (up to a point)
- Quality matters more than quantity
- Diverse examples improve generalization
- Similar-to-test examples help most
```

## Constructing Few-Shot Prompts

### Basic Structure

```python
def build_few_shot_prompt(task_description, examples, query):
    """
    Standard few-shot prompt structure.
    """
    prompt = f"{task_description}\n\n"

    for example in examples:
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"

    prompt += f"Input: {query}\n"
    prompt += "Output:"

    return prompt

# Example: Sentiment classification
task = "Classify the sentiment of the text as Positive, Negative, or Neutral."
examples = [
    {"input": "Best purchase I've ever made!", "output": "Positive"},
    {"input": "Complete waste of money.", "output": "Negative"},
    {"input": "It works as expected.", "output": "Neutral"},
]
query = "I'm really happy with this!"

prompt = build_few_shot_prompt(task, examples, query)
```

### Format Variations

```python
# Style 1: Arrow notation
"""
Q: What is the capital of France?
A: Paris

Q: What is the capital of Japan?
A: Tokyo

Q: What is the capital of Brazil?
A:
"""

# Style 2: Labeled format
"""
Text: "Great product!"
Sentiment: Positive

Text: "Awful service."
Sentiment: Negative

Text: "Super helpful!"
Sentiment:
"""

# Style 3: JSON format
"""
{"text": "I love it", "label": "positive"}
{"text": "I hate it", "label": "negative"}
{"text": "Best ever", "label":
"""

# Style 4: Markdown/structured
"""
| Input | Output |
|-------|--------|
| cat | animal |
| car | vehicle |
| dog |
"""
```

## Example Selection Strategies

### Random Selection

```python
import random

def random_examples(example_pool, k=5):
    """Simple random sampling."""
    return random.sample(example_pool, min(k, len(example_pool)))
```

### Similarity-Based Selection

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_similarity_examples(query, example_pool, k=5):
    """
    Select examples most similar to the query.
    Often performs best for specific tasks.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode query and all examples
    query_embedding = model.encode(query)
    example_embeddings = model.encode([e['input'] for e in example_pool])

    # Compute similarities
    similarities = np.dot(example_embeddings, query_embedding)

    # Select top-k most similar
    top_indices = np.argsort(similarities)[-k:][::-1]

    return [example_pool[i] for i in top_indices]
```

### Diversity-Based Selection

```python
def diverse_examples(example_pool, k=5):
    """
    Select diverse examples to cover different patterns.
    Uses maximal marginal relevance.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([e['input'] for e in example_pool])

    selected = []
    selected_indices = []

    for _ in range(k):
        best_score = -float('inf')
        best_idx = None

        for i, emb in enumerate(embeddings):
            if i in selected_indices:
                continue

            # Diversity: distance from already selected
            if selected_indices:
                selected_embs = embeddings[selected_indices]
                max_sim = max(np.dot(emb, s) for s in selected_embs)
            else:
                max_sim = 0

            # Score: novelty (1 - max similarity to selected)
            score = 1 - max_sim

            if score > best_score:
                best_score = score
                best_idx = i

        selected_indices.append(best_idx)
        selected.append(example_pool[best_idx])

    return selected
```

### Label-Balanced Selection

```python
def balanced_examples(example_pool, k=6):
    """
    Ensure equal representation of each label.
    Important for classification tasks.
    """
    from collections import defaultdict

    by_label = defaultdict(list)
    for example in example_pool:
        by_label[example['output']].append(example)

    # Equal samples from each label
    k_per_label = k // len(by_label)
    selected = []

    for label, examples in by_label.items():
        selected.extend(random.sample(
            examples,
            min(k_per_label, len(examples))
        ))

    return selected
```

## Example Ordering

### Order Matters

```
Experiments show:
- Accuracy can vary 15-30% based on example order
- Recency bias: Later examples may have more influence
- Primacy effects: First examples set expectations

Best practices:
1. Put most relevant examples last (closer to query)
2. Ensure balanced label distribution throughout
3. Avoid clustering same labels together
4. Test multiple orderings if critical
```

### Ordering Strategies

```python
def order_by_similarity(examples, query, model):
    """Order from least to most similar (most similar last)."""
    embeddings = model.encode([e['input'] for e in examples] + [query])
    query_emb = embeddings[-1]
    example_embs = embeddings[:-1]

    similarities = [np.dot(e, query_emb) for e in example_embs]
    order = np.argsort(similarities)  # Ascending

    return [examples[i] for i in order]

def shuffle_with_label_spread(examples):
    """Spread labels evenly throughout examples."""
    from collections import defaultdict

    by_label = defaultdict(list)
    for e in examples:
        by_label[e['output']].append(e)

    # Interleave labels
    result = []
    while any(by_label.values()):
        for label in list(by_label.keys()):
            if by_label[label]:
                result.append(by_label[label].pop(0))

    return result
```

## Number of Examples

### How Many Examples?

```
Guidelines:
- 3-5 examples: Good starting point
- 5-10 examples: Often optimal
- 10+ examples: Diminishing returns, context limits

Factors affecting optimal count:
1. Task complexity: Complex tasks need more examples
2. Model size: Larger models need fewer examples
3. Context length: Must fit in window
4. Example length: Longer examples = fewer fit
5. Diversity needs: More categories = more examples

Empirical approach:
- Start with 3 examples
- Add more until performance plateaus
- Watch for context length limits
```

### Context Length Considerations

```python
def fit_examples_to_context(examples, query, max_tokens=4096, buffer=500):
    """
    Select maximum examples that fit in context.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    query_tokens = len(enc.encode(query))
    available = max_tokens - query_tokens - buffer

    selected = []
    used_tokens = 0

    for example in examples:
        example_text = f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        example_tokens = len(enc.encode(example_text))

        if used_tokens + example_tokens <= available:
            selected.append(example)
            used_tokens += example_tokens
        else:
            break

    return selected
```

## Task-Specific Patterns

### Classification

```python
classification_prompt = """Classify the movie review as Positive or Negative.

Review: "This film was a masterpiece. The acting was superb and the story was captivating."
Classification: Positive

Review: "Waste of time. Poor acting, boring plot, nothing redeeming about it."
Classification: Negative

Review: "Beautiful cinematography but the story felt rushed and incomplete."
Classification: Negative

Review: "A delightful experience from start to finish. Highly recommend!"
Classification: Positive

Review: "{query}"
Classification:"""
```

### Entity Extraction

```python
extraction_prompt = """Extract the person names and organizations from the text.

Text: "John Smith joined Microsoft in 2020."
Entities: Person: John Smith | Organization: Microsoft

Text: "CEO Tim Cook announced Apple's new product line."
Entities: Person: Tim Cook | Organization: Apple

Text: "Dr. Sarah Johnson from Stanford University published her research."
Entities: Person: Dr. Sarah Johnson | Organization: Stanford University

Text: "{query}"
Entities:"""
```

### Translation

```python
translation_prompt = """Translate English to French.

English: "Hello, how are you?"
French: "Bonjour, comment allez-vous?"

English: "The weather is nice today."
French: "Le temps est beau aujourd'hui."

English: "I would like to order a coffee."
French: "Je voudrais commander un café."

English: "{query}"
French:"""
```

### Code Generation

```python
code_prompt = """Write a Python function for the given description.

Description: Return the sum of two numbers
```python
def add(a, b):
    return a + b
```

Description: Return the larger of two numbers
```python
def max_of_two(a, b):
    return a if a > b else b
```

Description: Return True if a string is a palindrome
```python
def is_palindrome(s):
    return s == s[::-1]
```

Description: {query}
```python"""
```

## Combining with Other Techniques

### Few-Shot + Chain-of-Thought

```python
cot_few_shot = """Solve the math problem step by step.

Q: A store sells apples for $2 each. If I buy 3 apples and pay with $10, how much change do I get?
A: The cost of 3 apples is 3 × $2 = $6.
   I pay with $10.
   My change is $10 - $6 = $4.
   The answer is $4.

Q: A train travels at 60 mph. How far does it travel in 2.5 hours?
A: Distance = speed × time.
   Speed is 60 mph.
   Time is 2.5 hours.
   Distance = 60 × 2.5 = 150 miles.
   The answer is 150 miles.

Q: {query}
A:"""
```

### Few-Shot + System Prompts

```python
# OpenAI API with system message + few-shot
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that classifies text sentiment."
    },
    {
        "role": "user",
        "content": "Great product!"
    },
    {
        "role": "assistant",
        "content": "Positive"
    },
    {
        "role": "user",
        "content": "Terrible experience."
    },
    {
        "role": "assistant",
        "content": "Negative"
    },
    {
        "role": "user",
        "content": query
    }
]
```

## Practical Considerations

### Prompt Caching

```python
class FewShotPromptCache:
    """Cache expensive example retrieval."""

    def __init__(self, example_pool, embedding_model):
        self.examples = example_pool
        self.model = embedding_model
        # Pre-compute embeddings
        self.embeddings = self.model.encode(
            [e['input'] for e in example_pool]
        )

    def get_examples(self, query, k=5):
        query_emb = self.model.encode(query)
        similarities = np.dot(self.embeddings, query_emb)
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k]
```

### Evaluation and Iteration

```python
def evaluate_few_shot_config(
    model,
    test_set,
    example_pool,
    example_selector,
    k_values=[3, 5, 10]
):
    """Find optimal few-shot configuration."""
    results = {}

    for k in k_values:
        correct = 0
        for test in test_set:
            examples = example_selector(example_pool, test['input'], k)
            prompt = build_few_shot_prompt(examples, test['input'])
            prediction = model.generate(prompt)

            if prediction.strip() == test['expected']:
                correct += 1

        results[k] = correct / len(test_set)

    return results
```

## Common Pitfalls

### Failure Modes

```
1. Format mismatch
   - Examples use one format, model outputs another
   - Fix: Be explicit about output format in all examples

2. Distribution shift
   - Examples don't match test distribution
   - Fix: Use similarity-based selection

3. Label imbalance
   - Over-representation of one class
   - Fix: Balance examples across labels

4. Too few examples for task complexity
   - Model can't infer pattern
   - Fix: Add more examples, simplify task

5. Context overflow
   - Examples exceed context window
   - Fix: Shorter examples, dynamic selection
```

### Quality Checks

```python
def validate_examples(examples):
    """Check example quality."""
    issues = []

    # Check format consistency
    formats = set()
    for e in examples:
        format_signature = (len(e['input'].split()), len(e['output'].split()))
        formats.add(format_signature)

    if len(formats) > 2:
        issues.append("Inconsistent input/output lengths")

    # Check label distribution
    from collections import Counter
    labels = Counter(e['output'] for e in examples)
    min_count = min(labels.values())
    max_count = max(labels.values())

    if max_count > 2 * min_count:
        issues.append(f"Imbalanced labels: {labels}")

    # Check for duplicates
    inputs = [e['input'] for e in examples]
    if len(inputs) != len(set(inputs)):
        issues.append("Duplicate inputs in examples")

    return issues
```

## Key Takeaways

1. **Examples teach patterns**: Model learns task structure from input-output pairs.

2. **Quality over quantity**: 3-5 high-quality examples often beat 10+ poor ones.

3. **Selection matters**: Similar-to-query and diverse examples work best.

4. **Order affects output**: Last examples may have most influence.

5. **Format consistency**: Match structure across all examples.

6. **Balance labels**: Equal representation prevents bias.

7. **Context limits apply**: Fit examples within token budget.
