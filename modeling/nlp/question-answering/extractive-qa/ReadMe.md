# Extractive Question Answering

## Summary

Extractive Question Answering (QA) identifies and extracts answer spans directly from a given context passage. Given a question and a context document, the model locates the exact substring that answers the question. This approach guarantees that answers are grounded in the source text, making it suitable for applications requiring verifiable, factual responses.

Key points to remember:

- Span extraction: Answer is a contiguous substring of the context
- Start and end prediction: Model predicts start and end token positions
- Reading comprehension: Question and context encoded together
- SQuAD benchmark: Standard evaluation with exact match (EM) and F1
- BERT-based models: Dominant approach with fine-tuning on QA datasets
- No-answer detection: Some questions have no valid answer in context
- Context length limits: Transformer models have max sequence length constraints
- Use cases: FAQ systems, document search, knowledge base queries

## Task Definition

### Input/Output Format

```
Context: "Albert Einstein was born in Ulm, Germany, on March 14, 1879.
         He developed the theory of relativity and won the Nobel Prize
         in Physics in 1921."

Question: "Where was Einstein born?"
Answer: "Ulm, Germany" (span at positions 28-39)

Question: "When did Einstein win the Nobel Prize?"
Answer: "1921" (span at positions 156-160)
```

### Answer Types

| Type | Example Question | Answer |
|------|-----------------|--------|
| Entity | "Who founded Microsoft?" | "Bill Gates" |
| Location | "Where is Apple headquartered?" | "Cupertino, California" |
| Date/Time | "When was Python released?" | "1991" |
| Number | "How many employees does Google have?" | "150,000" |
| Description | "What is machine learning?" | "a branch of artificial intelligence..." |

## Architecture

### BERT for Extractive QA

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # Start and end logits

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state

        # Predict start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return {
            'loss': loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        }
```

### Input Encoding

```
[CLS] question tokens [SEP] context tokens [SEP]

Token Type IDs:
  0   0 0 0 0 0 0 0  0   1 1 1 1 1 1 1 1 1  1

Example:
[CLS] Where was Einstein born ? [SEP] Albert Einstein was born in Ulm , Germany ... [SEP]
```

## Using Pre-trained Models

### Hugging Face Pipeline

```python
from transformers import pipeline

# Load extractive QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

context = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf
tropical rainforest in the Amazon biome that covers most of the Amazon
basin of South America. This basin encompasses 7,000,000 km2, of which
5,500,000 km2 are covered by the rainforest.
"""

question = "How large is the Amazon basin?"

result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.4f}")
print(f"Start: {result['start']}, End: {result['end']}")
```

### Batch Processing

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def batch_qa(questions, contexts, batch_size=8):
    results = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_contexts = contexts[i:i + batch_size]

        inputs = tokenizer(
            batch_questions,
            batch_contexts,
            padding=True,
            truncation="only_second",
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        for j in range(len(batch_questions)):
            start_idx = torch.argmax(outputs.start_logits[j])
            end_idx = torch.argmax(outputs.end_logits[j])

            answer_tokens = inputs.input_ids[j][start_idx:end_idx + 1]
            answer = tokenizer.decode(answer_tokens)

            results.append({
                'question': batch_questions[j],
                'answer': answer,
                'start': start_idx.item(),
                'end': end_idx.item()
            })

    return results
```

## Fine-tuning

### Data Preparation

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def prepare_train_features(examples):
    # Tokenize questions and contexts
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Map character positions to token positions
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            # No answer case
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token positions
            start_token = None
            end_token = None

            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx

            if start_token is None or end_token is None:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                tokenized["start_positions"].append(start_token)
                tokenized["end_positions"].append(end_token)

    return tokenized

# Load and process dataset
dataset = load_dataset("squad")
tokenized_dataset = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)
```

### Training Loop

```python
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

trainer.train()
```

## Handling Long Documents

### Sliding Window Approach

```python
def process_long_document(question, document, tokenizer, model, max_length=512, stride=128):
    """Process documents longer than max_length using sliding window."""

    # Tokenize with overlapping chunks
    inputs = tokenizer(
        question,
        document,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt"
    )

    # Get predictions for each chunk
    chunk_results = []

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    for i in range(len(inputs["input_ids"])):
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]

        # Find best span in this chunk
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        if end_idx >= start_idx:
            score = start_logits[start_idx] + end_logits[end_idx]
            chunk_results.append({
                'chunk_idx': i,
                'start': start_idx,
                'end': end_idx,
                'score': score.item(),
                'offset_mapping': inputs["offset_mapping"][i]
            })

    # Select best answer across chunks
    if chunk_results:
        best = max(chunk_results, key=lambda x: x['score'])
        # Map back to original document positions
        offsets = best['offset_mapping']
        start_char = offsets[best['start']][0]
        end_char = offsets[best['end']][1]
        answer = document[start_char:end_char]
        return answer

    return ""
```

## No-Answer Detection

### SQuAD 2.0 Style

```python
class QAModelWithNoAnswer(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size

        self.qa_outputs = nn.Linear(hidden_size, 2)  # Start, end
        self.has_answer = nn.Linear(hidden_size, 2)  # Has answer classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Span predictions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        # No-answer prediction
        has_answer_logits = self.has_answer(pooled_output)

        return {
            'start_logits': start_logits.squeeze(-1),
            'end_logits': end_logits.squeeze(-1),
            'has_answer_logits': has_answer_logits
        }

def predict_with_threshold(model, inputs, no_answer_threshold=0.5):
    outputs = model(**inputs)

    # Check if has answer
    has_answer_prob = torch.softmax(outputs['has_answer_logits'], dim=-1)[0, 1]

    if has_answer_prob < no_answer_threshold:
        return {"answer": "", "has_answer": False}

    # Extract answer span
    start_idx = torch.argmax(outputs['start_logits'])
    end_idx = torch.argmax(outputs['end_logits'])

    return {
        "answer": decode_span(inputs, start_idx, end_idx),
        "has_answer": True,
        "confidence": has_answer_prob.item()
    }
```

## Evaluation

### Metrics

```python
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction, ground_truth):
    """Check if normalized prediction matches any ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def evaluate_qa(predictions, ground_truths):
    """Evaluate QA predictions against ground truths."""
    em_total = 0
    f1_total = 0

    for pred, truths in zip(predictions, ground_truths):
        # Handle multiple valid answers
        em_total += max(exact_match_score(pred, t) for t in truths)
        f1_total += max(f1_score(pred, t) for t in truths)

    n = len(predictions)
    return {
        'exact_match': 100.0 * em_total / n,
        'f1': 100.0 * f1_total / n
    }
```

### SQuAD Evaluation

```python
from datasets import load_metric

squad_metric = load_metric("squad_v2")

def compute_squad_metrics(predictions, references):
    """
    predictions: List of {"id": str, "prediction_text": str, "no_answer_probability": float}
    references: List of {"id": str, "answers": {"text": List[str], "answer_start": List[int]}}
    """
    return squad_metric.compute(predictions=predictions, references=references)
```

## Common Benchmarks

| Dataset | Size | No-Answer | Domain |
|---------|------|-----------|--------|
| SQuAD 1.1 | 100K | No | Wikipedia |
| SQuAD 2.0 | 150K | Yes | Wikipedia |
| Natural Questions | 300K | Yes | Google Search |
| TriviaQA | 650K | No | Trivia |
| NewsQA | 100K | No | News |
| BioASQ | 3K | No | Biomedical |

## Best Practices

### Model Selection

| Scenario | Recommended Model |
|----------|------------------|
| General English | roberta-base-squad2 |
| High accuracy | bert-large-squad2 |
| Speed priority | distilbert-base-squad2 |
| Multilingual | xlm-roberta-large-squad2 |
| Domain-specific | Fine-tune on domain data |

### Answer Post-processing

```python
def postprocess_answer(answer, context):
    """Clean up extracted answer."""
    # Remove leading/trailing whitespace
    answer = answer.strip()

    # Remove partial words at boundaries
    if answer and answer[0].islower():
        # Might have cut off beginning of word
        start_idx = context.find(answer)
        if start_idx > 0 and context[start_idx - 1].isalnum():
            # Find word boundary
            while start_idx > 0 and context[start_idx - 1].isalnum():
                start_idx -= 1
            answer = context[start_idx:start_idx + len(answer)]

    # Remove incomplete sentences for long answers
    if len(answer) > 100 and '.' in answer:
        sentences = answer.split('.')
        if sentences[-1].strip() == '':
            answer = '.'.join(sentences[:-1]) + '.'

    return answer
```

### Confidence Calibration

```python
def calibrate_confidence(start_logits, end_logits, temperature=1.0):
    """Calibrate model confidence scores."""
    # Apply temperature scaling
    start_probs = torch.softmax(start_logits / temperature, dim=-1)
    end_probs = torch.softmax(end_logits / temperature, dim=-1)

    # Compute span probability
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    if end_idx >= start_idx:
        span_prob = start_probs[start_idx] * end_probs[end_idx]
    else:
        span_prob = 0.0

    return span_prob.item()
```

## Production Considerations

### Caching and Indexing

```python
import faiss
import numpy as np

class EfficientQA:
    def __init__(self, model, tokenizer, passages):
        self.model = model
        self.tokenizer = tokenizer
        self.passages = passages

        # Pre-encode passages
        self.passage_embeddings = self._encode_passages()
        self._build_index()

    def _encode_passages(self):
        embeddings = []
        for passage in self.passages:
            inputs = self.tokenizer(passage, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model.bert(**inputs)
            embedding = outputs.pooler_output.numpy()
            embeddings.append(embedding)
        return np.vstack(embeddings)

    def _build_index(self):
        dimension = self.passage_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.passage_embeddings)
        self.index.add(self.passage_embeddings)

    def answer(self, question, top_k=5):
        # Retrieve relevant passages
        query_embedding = self._encode_question(question)
        _, indices = self.index.search(query_embedding, top_k)

        # Run QA on top passages
        best_answer = None
        best_score = -float('inf')

        for idx in indices[0]:
            passage = self.passages[idx]
            answer, score = self._extract_answer(question, passage)
            if score > best_score:
                best_score = score
                best_answer = answer

        return best_answer
```
