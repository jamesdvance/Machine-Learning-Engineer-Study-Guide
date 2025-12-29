# Zero-Shot Prompting

## Summary

Zero-shot prompting instructs LLMs to perform tasks without providing any examples, relying entirely on the model's pre-trained knowledge and the clarity of the task description. This is the simplest prompting approach: describe what you want, and the model generates a response. Zero-shot works because large language models have learned task patterns during pre-training and instruction tuning. While less reliable than few-shot for complex tasks, zero-shot is fast to iterate, requires no example curation, and often sufficient for straightforward tasks.

Key points to remember:

- No examples needed: Just describe the task clearly
- Relies on pre-training: Model uses learned patterns from training
- Instruction tuning helps: Chat/instruction models are better at zero-shot
- Fast to iterate: No example selection or formatting
- Task clarity matters: Ambiguous instructions lead to unpredictable results
- Baseline approach: Start here, add examples if needed
- System prompts enhance: Role and context setting improves results

## Basic Structure

### Simple Zero-Shot

```
Instruction-based:
Translate the following text to French: "Hello, how are you?"

Question-based:
What is the capital of Japan?

Task-based:
Summarize this article in three sentences: [article text]

Classification:
Is the following review positive or negative? "Great product, highly recommend!"
```

### Effective Patterns

```python
# Pattern 1: Direct instruction
prompt = "Classify the sentiment of this text as positive, negative, or neutral: {text}"

# Pattern 2: Task + format specification
prompt = """Analyze the following text and provide:
1. Main topic
2. Key entities mentioned
3. Overall tone

Text: {text}"""

# Pattern 3: Role-based
prompt = """You are an expert proofreader.
Review the following text for grammatical errors and list them:
{text}"""

# Pattern 4: Step-by-step instruction
prompt = """Extract all dates from the following text.
Format each date as YYYY-MM-DD.
List one date per line.
If no dates found, respond with "No dates found."

Text: {text}"""
```

## Instruction Design

### Clarity Principles

```
1. Be specific about the task
   Bad:  "Handle this text"
   Good: "Summarize this text in 2-3 sentences"

2. Specify output format
   Bad:  "List the countries"
   Good: "List the countries, one per line, in alphabetical order"

3. Define edge cases
   Bad:  "Extract emails"
   Good: "Extract all email addresses. If none found, return 'No emails found'"

4. Set constraints
   Bad:  "Describe the image"
   Good: "Describe the image in exactly 50 words, focusing on main subjects"
```

### Common Instruction Types

```python
# Classification
"Classify the following text into one of these categories: "
"Technology, Sports, Politics, Entertainment. "
"Only respond with the category name."

# Extraction
"Extract all person names from the text. "
"Return as a JSON array. "
"If no names found, return an empty array []."

# Generation
"Write a professional email responding to a customer complaint about delayed shipping. "
"Keep the tone apologetic but solution-focused. "
"Maximum 150 words."

# Transformation
"Rewrite the following text to be more concise. "
"Keep all key information but reduce length by 50%."

# Analysis
"Analyze this code for potential bugs. "
"For each issue found, explain the problem and suggest a fix."
```

## System Prompts and Roles

### Role-Based Zero-Shot

```python
# OpenAI API pattern
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that translates English to Spanish. "
                       "Only provide the translation, no explanations."
        },
        {
            "role": "user",
            "content": "The weather is beautiful today."
        }
    ]
)

# Effective roles
roles = {
    "translator": "You are a professional translator. Translate accurately while preserving tone.",
    "proofreader": "You are an expert proofreader. Identify all errors without rewriting.",
    "analyst": "You are a data analyst. Provide insights based on evidence from the data.",
    "tutor": "You are a patient tutor. Explain concepts clearly for beginners.",
    "critic": "You are a constructive critic. Provide balanced feedback with actionable suggestions."
}
```

### Context Setting

```python
# Domain context
prompt = """You are a medical professional reviewing patient notes.
Identify any concerning symptoms mentioned in the following text.
Use clinical terminology in your response.

Patient notes: {notes}"""

# Audience context
prompt = """Explain quantum computing to a high school student.
Use simple analogies and avoid technical jargon.
Keep the explanation under 200 words."""

# Constraint context
prompt = """You are writing for a legal document.
Use formal language and precise terminology.
Avoid contractions and colloquialisms.

Draft a privacy policy clause for: {topic}"""
```

## Output Control

### Format Specification

```python
# JSON output
prompt = """Extract information from the following text and return as JSON:
{
    "title": "...",
    "author": "...",
    "date": "...",
    "summary": "..."
}

Text: {text}"""

# Structured list
prompt = """Analyze the pros and cons of this proposal.

Format your response as:
PROS:
- [point 1]
- [point 2]

CONS:
- [point 1]
- [point 2]

RECOMMENDATION: [your recommendation]

Proposal: {proposal}"""

# Specific format
prompt = """Convert the following date to ISO 8601 format (YYYY-MM-DD).
Only output the formatted date, nothing else.

Date: {date}"""
```

### Controlling Length

```python
# Word count
prompt = "Summarize this article in exactly 50 words: {text}"

# Sentence count
prompt = "Explain this concept in 3 sentences: {concept}"

# Character limit (less reliable)
prompt = "Write a tweet (max 280 characters) about: {topic}"

# Implicit length
prompt = "Briefly describe..."  # Short
prompt = "Provide a comprehensive analysis of..."  # Long
```

## Zero-Shot Techniques

### Zero-Shot Chain-of-Thought

```python
# Magic phrase addition
prompt = f"""Q: {question}
A: Let's think step by step."""

# Variants that work
phrases = [
    "Let's think step by step.",
    "Let's work through this carefully.",
    "Let's break this down.",
    "Before answering, let me reason through this.",
    "Think step by step to find the answer."
]

# Two-stage approach
def zero_shot_cot(model, question):
    # Stage 1: Generate reasoning
    reasoning_prompt = f"Q: {question}\nA: Let's think step by step."
    reasoning = model.generate(reasoning_prompt)

    # Stage 2: Extract answer
    answer_prompt = f"""Q: {question}
A: {reasoning}

Therefore, the final answer is:"""
    answer = model.generate(answer_prompt)

    return answer
```

### Instruction Following Techniques

```python
# Explicit constraints
prompt = """IMPORTANT: Follow these rules exactly:
1. Respond only in JSON format
2. Include all required fields
3. If uncertain, use null

Task: {task}"""

# Negative constraints
prompt = """Summarize this text.
DO NOT include:
- Personal opinions
- Information not in the text
- More than 100 words

Text: {text}"""

# Format examples (still zero-shot for task)
prompt = """Convert dates to ISO format.
Input format examples: "Jan 15, 2024", "15/01/2024", "January 15th 2024"
Output format: YYYY-MM-DD

Convert: {date}"""
```

## Task Categories

### Classification Tasks

```python
# Binary classification
prompt = "Is this email spam? Reply with only 'spam' or 'not spam': {email}"

# Multi-class
prompt = """Classify this support ticket:
- billing
- technical
- general
- urgent

Reply with only the category name.
Ticket: {ticket}"""

# Multi-label
prompt = """What topics does this article cover? Select all that apply:
Technology, Health, Finance, Sports, Entertainment

Return topics as comma-separated list.
Article: {article}"""
```

### Generation Tasks

```python
# Creative writing
prompt = "Write a haiku about artificial intelligence."

# Technical writing
prompt = """Write a docstring for this Python function:
```python
{code}
```"""

# Completion
prompt = "Continue this story in the same style (2 paragraphs): {story_start}"
```

### Transformation Tasks

```python
# Style transfer
prompt = "Rewrite this text in a more formal tone: {text}"

# Simplification
prompt = "Explain this technical concept for a 10-year-old: {concept}"

# Translation
prompt = "Translate to French: {text}"

# Paraphrasing
prompt = "Rephrase this sentence 3 different ways: {sentence}"
```

## When Zero-Shot Works Best

### Ideal Conditions

```
Zero-shot excels when:
1. Task is common (model has seen many similar examples in training)
2. Instructions are unambiguous
3. Output format is straightforward
4. Model is instruction-tuned (GPT-4, Claude, etc.)
5. Task doesn't require domain-specific patterns

Examples of good zero-shot tasks:
- Simple translations
- Basic summarization
- Sentiment analysis
- Question answering (factual)
- Text generation (open-ended)
- Grammar correction
```

### When to Add Examples

```
Consider few-shot when:
1. Output format is unusual
2. Task is domain-specific
3. Consistent style is critical
4. Zero-shot accuracy is low
5. Edge cases need clarification

Signs zero-shot isn't working:
- Inconsistent output format
- Wrong interpretation of task
- Missing important aspects
- Too much irrelevant content
```

## Debugging Zero-Shot Prompts

### Common Issues

```python
# Issue: Inconsistent format
# Bad
prompt = "Analyze this data"
# Good
prompt = "Analyze this data. Structure your response with: Summary, Key Findings, Recommendations"

# Issue: Ambiguous task
# Bad
prompt = "Fix this text"
# Good
prompt = "Correct any grammatical and spelling errors in this text. Do not change the meaning or style."

# Issue: Over-generation
# Bad
prompt = "What is the capital of France?"
# Response: "The capital of France is Paris. Paris is known for..."
# Good
prompt = "What is the capital of France? Answer with just the city name."

# Issue: Under-specification
# Bad
prompt = "Summarize"
# Good
prompt = "Summarize the following article in 3 bullet points, focusing on the main arguments."
```

### Iterative Refinement

```python
def refine_prompt(initial_prompt, test_inputs, expected_outputs):
    """Iteratively improve zero-shot prompt."""
    prompt = initial_prompt

    for input_text, expected in zip(test_inputs, expected_outputs):
        response = model.generate(prompt.format(text=input_text))

        if response != expected:
            # Analyze failure and adjust
            if len(response) > len(expected) * 2:
                prompt += "\nBe concise."
            if format_differs(response, expected):
                prompt += f"\nFormat: {describe_format(expected)}"
            # ... other adjustments

    return prompt
```

## Practical Patterns

### Production-Ready Zero-Shot

```python
class ZeroShotClassifier:
    def __init__(self, model, categories):
        self.model = model
        self.categories = categories
        self.prompt_template = self._build_prompt()

    def _build_prompt(self):
        cats = ", ".join(self.categories)
        return f"""Classify the following text into exactly one category.
Categories: {cats}

Rules:
- Respond with only the category name
- Choose the single best match
- If unclear, choose the closest category

Text: {{text}}
Category:"""

    def classify(self, text):
        prompt = self.prompt_template.format(text=text)
        response = self.model.generate(prompt, max_tokens=20, temperature=0)

        # Validate response
        response = response.strip()
        if response in self.categories:
            return response

        # Fuzzy match if needed
        return self._fuzzy_match(response)
```

### Batch Processing

```python
def batch_zero_shot(model, prompts, max_concurrent=10):
    """Process multiple zero-shot prompts efficiently."""
    import asyncio

    async def process_batch(batch):
        tasks = [model.generate_async(p) for p in batch]
        return await asyncio.gather(*tasks)

    results = []
    for i in range(0, len(prompts), max_concurrent):
        batch = prompts[i:i + max_concurrent]
        batch_results = asyncio.run(process_batch(batch))
        results.extend(batch_results)

    return results
```

## Key Takeaways

1. **Clarity is everything**: Unambiguous instructions drive good results.

2. **Specify format**: Tell the model exactly how to structure output.

3. **Use system prompts**: Set role and context for better responses.

4. **Start simple**: Zero-shot first, add examples only if needed.

5. **Test edge cases**: Verify behavior on unusual inputs.

6. **Iterate on failures**: Refine prompts based on error patterns.

7. **Know the limits**: Switch to few-shot when zero-shot fails consistently.
