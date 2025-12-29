# Chain-of-Thought Prompting

## Summary

Chain-of-Thought (CoT) prompting dramatically improves LLM performance on reasoning tasks by instructing models to show their step-by-step reasoning before arriving at a final answer. Instead of directly outputting answers, the model generates intermediate reasoning steps, which helps it perform multi-step computations, catch errors, and solve problems requiring logical deduction. CoT can be elicited with few-shot examples showing reasoning chains, or zero-shot with phrases like "Let's think step by step." This technique is particularly effective for math, logic, and commonsense reasoning tasks.

Key points to remember:

- Explicit reasoning: Model shows intermediate steps before final answer
- Few-shot or zero-shot: Examples with chains, or simple prompts like "Let's think step by step"
- Math and logic: Largest gains on multi-step reasoning problems
- Model size matters: Primarily effective for large models (>~10B parameters)
- Self-consistency: Sample multiple chains, majority vote for robustness
- Verification possible: Inspect reasoning for debugging and trust
- Trade-off: More tokens generated, higher latency and cost

## The Core Idea

### Standard vs Chain-of-Thought

```
Standard Prompting:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11

Chain-of-Thought Prompting:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls.
   2 cans of 3 balls each = 2 × 3 = 6 balls.
   5 + 6 = 11 balls.
   The answer is 11.

Why CoT works:
1. Breaks complex problems into manageable steps
2. Allows model to use intermediate results
3. Reduces working memory burden
4. Makes reasoning verifiable
```

### Why Reasoning Steps Help

```
Without CoT (failure mode):
Q: A bat and ball cost $1.10. The bat costs $1 more than the ball.
   How much does the ball cost?
A: $0.10   Wrong! (Intuitive but incorrect)

With CoT (correct reasoning):
Q: A bat and ball cost $1.10. The bat costs $1 more than the ball.
   How much does the ball cost?
A: Let's call the ball's cost x.
   The bat costs x + $1.
   Total: x + (x + $1) = $1.10
   2x + $1 = $1.10
   2x = $0.10
   x = $0.05
   The ball costs $0.05.   Correct!
```

## Few-Shot Chain-of-Thought

### Constructing Examples

```python
def build_cot_prompt(question, examples):
    """
    Build few-shot CoT prompt with reasoning examples.
    """
    prompt = ""

    for example in examples:
        prompt += f"Q: {example['question']}\n"
        prompt += f"A: {example['reasoning']}\n"
        prompt += f"Therefore, the answer is {example['answer']}.\n\n"

    prompt += f"Q: {question}\n"
    prompt += "A: "

    return prompt

# Example usage
examples = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5"
    }
]

question = "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
prompt = build_cot_prompt(question, examples)
```

### Effective Example Design

```
Good CoT examples have:
1. Clear step-by-step logic
2. Explicit calculations shown
3. Natural language explanations
4. Consistent format across examples

Example structure:
- State the given information
- Identify what needs to be found
- Show each reasoning step
- Conclude with explicit answer

Avoid:
- Skipping intermediate steps
- Overly complex reasoning in examples
- Examples that are too similar to test questions
```

## Zero-Shot Chain-of-Thought

### The Magic Phrase

```python
def zero_shot_cot(question):
    """
    Zero-shot CoT: Just add 'Let's think step by step.'
    Surprisingly effective without any examples.
    """
    prompt = f"""Q: {question}

A: Let's think step by step."""

    return prompt

# Alternatives that work:
# - "Let's work through this step by step."
# - "Let's break this down."
# - "Think carefully and show your reasoning."
# - "Before answering, let's analyze this."
```

### Two-Stage Approach

```python
def two_stage_zero_shot_cot(model, question):
    """
    Stage 1: Generate reasoning
    Stage 2: Extract answer
    """
    # Stage 1: Elicit reasoning
    reasoning_prompt = f"""Q: {question}

A: Let's think step by step."""

    reasoning = model.generate(reasoning_prompt)

    # Stage 2: Extract final answer
    extraction_prompt = f"""Q: {question}

A: {reasoning}

Therefore, the answer (just the number or single word) is:"""

    answer = model.generate(extraction_prompt)

    return answer, reasoning
```

## Self-Consistency

### Majority Voting Over Chains

```python
import collections

def self_consistency(model, question, n_samples=5, temperature=0.7):
    """
    Sample multiple reasoning chains, take majority vote.
    More robust than single chain.
    """
    prompt = build_cot_prompt(question, examples)
    answers = []

    for _ in range(n_samples):
        # Generate with temperature for diversity
        response = model.generate(
            prompt,
            temperature=temperature,
            max_tokens=500
        )

        # Extract final answer
        answer = extract_answer(response)
        answers.append(answer)

    # Majority vote
    vote_counts = collections.Counter(answers)
    final_answer = vote_counts.most_common(1)[0][0]

    return final_answer, answers

# Example:
# Sample 1: "...so 5 + 6 = 11" ’ 11
# Sample 2: "...total is 11 balls" ’ 11
# Sample 3: "...gives us 12" ’ 12 (error)
# Sample 4: "...answer is 11" ’ 11
# Sample 5: "...11 tennis balls" ’ 11
# Majority: 11 (4/5)
```

### Weighted Voting

```python
def weighted_self_consistency(model, question, n_samples=5):
    """
    Weight votes by log probability of the answer.
    Higher confidence answers count more.
    """
    answers_with_scores = []

    for _ in range(n_samples):
        response, logprobs = model.generate_with_logprobs(
            prompt,
            temperature=0.7
        )

        answer = extract_answer(response)
        answer_logprob = get_answer_logprob(response, logprobs)

        answers_with_scores.append((answer, answer_logprob))

    # Aggregate by answer, sum log probabilities
    answer_scores = collections.defaultdict(float)
    for answer, score in answers_with_scores:
        answer_scores[answer] += score

    final_answer = max(answer_scores, key=answer_scores.get)
    return final_answer
```

## Benchmarks and Results

### Performance Improvements

| Benchmark | Standard | Zero-Shot CoT | Few-Shot CoT |
|-----------|----------|---------------|--------------|
| GSM8K (math) | 17.9% | 40.5% | 56.4% |
| SVAMP (math) | 62.5% | 73.8% | 79.0% |
| AddSub (arithmetic) | 77.0% | 85.3% | 91.3% |
| CommonsenseQA | 73.5% | 75.2% | 78.1% |
| StrategyQA | 65.4% | 73.5% | 75.8% |

### Model Size Effects

```
CoT effectiveness by model size (GSM8K):
- 1B parameters:  Minimal improvement
- 8B parameters:  Modest improvement
- 70B parameters: Large improvement
- 175B+ parameters: Dramatic improvement

Why size matters:
- Smaller models lack coherent multi-step reasoning
- CoT requires "emergent" abilities
- Generally effective for models > 10B parameters
```

## Advanced Techniques

### Least-to-Most Prompting

```python
def least_to_most(model, problem):
    """
    Decompose problem into subproblems, solve bottom-up.
    """
    # Stage 1: Decompose
    decompose_prompt = f"""Q: {problem}

To solve this problem, we need to first:"""

    subproblems = model.generate(decompose_prompt)

    # Stage 2: Solve each subproblem
    solutions = []
    context = ""

    for subproblem in parse_subproblems(subproblems):
        solve_prompt = f"""{context}

Subproblem: {subproblem}
Solution:"""

        solution = model.generate(solve_prompt)
        solutions.append(solution)
        context += f"\n{subproblem}: {solution}"

    # Stage 3: Combine for final answer
    final_prompt = f"""{context}

Given the above, the answer to "{problem}" is:"""

    return model.generate(final_prompt)
```

### Complexity-Based Selection

```python
def complexity_based_cot(model, question, examples, n_samples=5):
    """
    Select the most detailed reasoning chain.
    Longer chains often correlate with correctness.
    """
    chains = []

    for _ in range(n_samples):
        response = model.generate(
            build_cot_prompt(question, examples),
            temperature=0.7
        )
        chains.append(response)

    # Score by complexity (number of reasoning steps)
    def count_steps(chain):
        # Count sentences, equations, or explicit markers
        indicators = ['. ', '= ', 'Therefore', 'So', 'Thus']
        return sum(chain.count(ind) for ind in indicators)

    best_chain = max(chains, key=count_steps)
    return extract_answer(best_chain), best_chain
```

### Verification Chain

```python
def verified_cot(model, question, examples):
    """
    Generate reasoning, then verify it.
    """
    # Generate initial solution
    cot_prompt = build_cot_prompt(question, examples)
    solution = model.generate(cot_prompt)

    # Verify the solution
    verify_prompt = f"""Question: {question}

Proposed solution:
{solution}

Let's verify this solution step by step. Check each calculation:"""

    verification = model.generate(verify_prompt)

    # Check if verification found errors
    if "error" in verification.lower() or "incorrect" in verification.lower():
        # Re-solve with verification context
        retry_prompt = f"""{verify_prompt}

{verification}

Given these issues, the correct solution is:"""
        solution = model.generate(retry_prompt)

    return extract_answer(solution)
```

## Practical Implementation

### With OpenAI API

```python
import openai

def cot_with_openai(question, examples):
    prompt = build_cot_prompt(question, examples)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Solve problems step by step."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # Deterministic for single query
        max_tokens=500
    )

    return response.choices[0].message.content

def cot_self_consistency(question, examples, n=5):
    """Self-consistency with multiple API calls."""
    answers = []

    for _ in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": build_cot_prompt(question, examples)}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = extract_answer(response.choices[0].message.content)
        answers.append(answer)

    return collections.Counter(answers).most_common(1)[0][0]
```

### With LangChain

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

cot_template = """Solve the following problem step by step.

{examples}

Question: {question}

Step-by-step solution:"""

cot_prompt = PromptTemplate(
    input_variables=["examples", "question"],
    template=cot_template
)

chain = LLMChain(llm=llm, prompt=cot_prompt)
result = chain.run(examples=formatted_examples, question=question)
```

## Common Pitfalls

### When CoT Doesn't Help

```
CoT may not help or hurt when:
1. Task requires no reasoning (simple retrieval)
2. Model is too small (<10B typically)
3. Problem is outside model's knowledge
4. Reasoning examples are low quality
5. Task is perception-heavy (not language reasoning)

Signs CoT isn't working:
- Nonsensical intermediate steps
- Steps don't logically connect
- Final answer doesn't follow from reasoning
- Worse accuracy than standard prompting
```

### Hallucinated Reasoning

```
Problem: Model generates plausible-sounding but wrong reasoning

Example:
Q: What is 17 × 24?
A: Let's see... 17 × 24.
   17 × 20 = 340
   17 × 4 = 72   Wrong (should be 68)
   340 + 72 = 412   Wrong answer

Mitigations:
1. Self-consistency (voting reduces random errors)
2. Verification step
3. External tool use for calculations
4. Fine-tuned models for specific domains
```

## Key Takeaways

1. **Show your work**: Explicit reasoning steps improve accuracy.

2. **Zero-shot works**: "Let's think step by step" is surprisingly effective.

3. **Self-consistency helps**: Multiple samples + voting is more robust.

4. **Model size matters**: CoT is most effective for large models (>10B).

5. **Examples shape reasoning**: Quality of few-shot examples matters.

6. **Verification adds reliability**: Check reasoning before accepting answers.

7. **Trade-off exists**: More tokens, higher latency, but better accuracy.
