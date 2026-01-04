# Visual Question Answering (VQA)

## Summary

Visual Question Answering is the task of answering natural language questions about images. Given an image and a question, the model must understand both the visual content and the question semantics to produce a correct answer. VQA spans a spectrum from simple recognition questions ("What color is the car?") to complex reasoning ("What might happen if the person lets go of the balloon?").

Key points to remember:

- Requires joint understanding of visual content and natural language
- Question types range from recognition to counting, spatial reasoning, and inference
- Open-ended VQA generates free-form answers; multiple-choice VQA selects from options
- Classical approaches used separate vision and language encoders with fusion
- Modern VLMs handle VQA as conversation or instruction-following
- Evaluation challenges include answer normalization and handling valid alternatives
- Key benchmarks: VQAv2, GQA, TextVQA, OK-VQA, VizWiz

## Task Taxonomy

### By Answer Type

**Yes/No Questions**: Binary answers about image content.
- "Is there a dog in this image?"
- "Is the traffic light red?"

**Counting Questions**: Numerical answers.
- "How many people are in the photo?"
- "How many windows are on the building?"

**Attribute Questions**: Object properties.
- "What color is the car?"
- "What material is the table made of?"

**Location Questions**: Spatial information.
- "Where is the cat?"
- "What is to the left of the lamp?"

**Action Questions**: Activities and events.
- "What is the person doing?"
- "What sport is being played?"

**Reasoning Questions**: Inference beyond direct observation.
- "Why might the person be smiling?"
- "What season is it likely to be?"

### By Knowledge Requirements

**Visual-only**: Answer derivable from image alone.
- "What color is the sky?"
- "How many cars are visible?"

**Visual + Common Sense**: Requires everyday knowledge.
- "Is this weather good for a picnic?"
- "Would this be a safe place for a child to play?"

**Visual + External Knowledge (OK-VQA)**: Requires factual knowledge.
- "What year was this building constructed?"
- "What is the capital of the country shown in this flag?"

## Benchmarks

### VQAv2

The standard VQA benchmark:
- 265K images from COCO
- 1.1M questions
- 10 answers per question from different annotators
- Balanced to reduce language bias

**Evaluation**: Accuracy with soft scoring based on human agreement:
```
accuracy = min(human_count / 3, 1)
```
If 3+ annotators agree with the prediction, full credit.

### GQA (Compositional Reasoning)

Tests compositional and spatial reasoning:
- 22M questions generated from scene graphs
- Controlled question generation ensures answer balance
- Includes consistency and validity metrics

### TextVQA

Questions requiring reading text in images:
- "What does the sign say?"
- "What brand is shown on the product?"
- Requires OCR capability

### OK-VQA (Outside Knowledge)

Questions requiring external knowledge:
- "Who painted this style?"
- "What animal is this the national symbol of?"
- Tests knowledge integration

### VizWiz

Real questions from blind users:
- Photos taken by visually impaired individuals
- Often lower quality images
- Tests real-world robustness

## Approaches

### Classical Fusion Models

Early approaches fused CNN image features with RNN question features:

```
Image -> CNN -> [image features]
                      |
                   fusion -> classifier -> answer
                      |
Question -> LSTM -> [question features]
```

Fusion methods included element-wise multiplication, concatenation, and attention.

### Attention-Based Models

Attention mechanisms focus on relevant image regions:

```python
# Simplified attention
attention_weights = softmax(question_features @ image_features.T)
attended_features = attention_weights @ image_features
answer = classifier(attended_features, question_features)
```

**Bottom-Up Top-Down Attention** (2017): Used object detector features instead of grid features, achieving significant improvements.

### Transformer-Based Models

Modern approaches use vision-language transformers:

**LXMERT**: Separate encoders with cross-modal attention.

**ViLBERT**: Dual-stream transformer with co-attention.

**UNITER**: Single-stream unified transformer.

These models pretrain on image-text pairs then fine-tune on VQA.

### VLM-Based VQA

Modern VLMs treat VQA as conversation:

```python
# Using GPT-4V
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)
answer = response.choices[0].message.content
```

VLMs offer zero-shot VQA capability without task-specific training.

## Evaluation Challenges

### Answer Normalization

Answers can be expressed many ways:
- "Two" vs "2" vs "a couple"
- "Red car" vs "The car is red" vs "It's red"

**Standard normalization**:
```python
def normalize_answer(answer):
    answer = answer.lower()
    answer = remove_articles(answer)  # a, an, the
    answer = remove_punctuation(answer)
    answer = convert_numbers(answer)  # "two" -> "2"
    return answer.strip()
```

### Open-Ended vs Closed Vocabulary

**Closed vocabulary**: Treat as classification over common answers.
- Limited to training vocabulary
- Easier to evaluate
- May miss valid rare answers

**Open-ended**: Generate free-form answers.
- More flexible
- Harder to evaluate (semantic equivalence)
- Risk of verbose or incorrect responses

### Soft Accuracy

VQAv2 uses soft scoring to handle answer variation:

```python
def compute_accuracy(prediction, ground_truth_list):
    # ground_truth_list contains 10 human answers
    matching = sum(1 for gt in ground_truth_list if normalize(prediction) == normalize(gt))
    return min(matching / 3, 1.0)
```

This rewards predictions that match multiple annotators.

## Common Challenges

### Language Bias

Models can exploit language patterns without understanding images:
- "What color is the banana?" -> "Yellow" (correct 90%+ of time)
- "Is there a..." -> "Yes" (often correct)

**Mitigation**:
- VQAv2 balances answers
- Include adversarial examples
- Evaluate on counterfactual images

### Counting

VLMs struggle with accurate counting:
- Consistent undercounting of large groups
- Confusion with overlapping objects
- Errors increase with object count

**Mitigation**:
- Use object detection models for counting
- Break into sub-regions
- Fine-tune specifically for counting tasks

### Spatial Reasoning

Understanding relationships like "left of," "behind," "next to":
- Requires geometric understanding
- Perspective affects relationships
- Ambiguity in complex scenes

### Compositional Generalization

Handling novel combinations of concepts:
- Training: "red apple," "green car"
- Test: "red car," "green apple"

Models often memorize rather than compositionally understand.

## Implementation

### Basic VQA Pipeline

```python
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, question):
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]
```

### VLM-Based with Structured Output

```python
def vqa_structured(image_url, question, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""Answer the following question about this image.
Question: {question}

Provide your answer in this format:
Answer: [your concise answer]
Confidence: [high/medium/low]
Reasoning: [brief explanation]"""},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
    )
    return parse_response(response.choices[0].message.content)
```

### Batch VQA Evaluation

```python
def evaluate_vqa(dataset, model):
    correct = 0
    total = 0

    for item in dataset:
        prediction = model.answer(item['image'], item['question'])
        accuracy = compute_soft_accuracy(
            normalize(prediction),
            [normalize(a) for a in item['answers']]
        )
        correct += accuracy
        total += 1

    return correct / total
```

## Applications

### Accessibility

Answering questions from visually impaired users:

```python
# VizWiz-style interface
prompt = f"""A visually impaired user took this photo and asks: "{question}"
If you cannot answer due to image quality, explain why.
If you can answer, provide a helpful response."""
```

### Visual Search and Retrieval

Finding images matching specific criteria:

```python
# Filter images by visual properties
def filter_images(images, criterion):
    matching = []
    for img in images:
        answer = vqa_model.answer(img, f"Does this image show {criterion}?")
        if answer.lower() in ["yes", "true"]:
            matching.append(img)
    return matching
```

### Content Moderation

Detecting specific content types:

```python
questions = [
    "Does this image contain violence?",
    "Is there explicit content?",
    "Are there identifiable faces?",
]
flags = {q: vqa_model.answer(image, q) for q in questions}
```

### Data Annotation

Automated labeling of image datasets:

```python
def annotate_dataset(images, questions):
    annotations = []
    for img in images:
        answers = {q: vqa_model.answer(img, q) for q in questions}
        annotations.append(answers)
    return annotations
```

## Best Practices

### Prompt Design

For VLM-based VQA:

```python
# Encourage concise answers
prompt = "Answer this question about the image in 1-3 words: {question}"

# For counting
prompt = "Count the number of {object} in the image. Respond with only a number."

# For yes/no
prompt = "Answer yes or no: {question}"
```

### Handling Uncertainty

```python
prompt = """Look at this image and answer the question.
If you cannot determine the answer from the image, say "Cannot determine" rather than guessing.
Question: {question}"""
```

### Multi-Step Reasoning

For complex questions:

```python
prompt = """To answer this question, follow these steps:
1. First, identify all relevant objects in the image
2. Note their properties and relationships
3. Then answer the question based on your observations

Question: {question}
Show your reasoning, then provide the final answer."""
```

### Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| Standard VQA | ViLT, BLIP-2 fine-tuned |
| TextVQA | Document-focused VLMs |
| Complex reasoning | GPT-4V, Gemini |
| Low latency | Smaller fine-tuned models |
| Domain-specific | Fine-tuned on domain data |

## Comparison with Related Tasks

### VQA vs Image Captioning

| Aspect | VQA | Captioning |
|--------|-----|------------|
| Output | Answer to specific question | General description |
| Focus | Question-directed | Image-directed |
| Evaluation | Accuracy | BLEU, CIDEr |
| Interactivity | Interactive | One-shot |

### VQA vs Visual Reasoning

| Aspect | VQA | Visual Reasoning |
|--------|-----|------------------|
| Complexity | Variable | High |
| Inference | Sometimes required | Always required |
| Knowledge | Variable | Often external |
| Examples | "What color?" | "What would happen if...?" |
