# Intent Classification

## Summary

Intent classification is the task of determining the purpose or goal behind a user's text input. It is a core component of conversational AI systems, chatbots, virtual assistants, and customer service automation. The model maps user utterances to predefined intent categories, enabling systems to route queries and trigger appropriate responses or actions.

Key points to remember:

- Multi-class classification: Map text to one of many predefined intents
- Often paired with slot filling/entity extraction for complete understanding
- Training requires labeled examples for each intent category
- Out-of-scope detection: Critical for handling queries outside known intents
- Few-shot learning: Useful when limited examples exist per intent
- Production systems need confidence thresholds and fallback handling
- Domain-specific: Intents are defined per application (banking, e-commerce, support)

## Core Concepts

### Intent vs Entity

Intent classification works alongside entity extraction to understand user queries:

```
User: "Book a flight from New York to London tomorrow"

Intent: book_flight
Entities:
  - departure: "New York"
  - destination: "London"
  - date: "tomorrow"
```

Intent determines what action to take; entities provide the parameters.

### Intent Taxonomy Design

Designing the intent set is crucial:

```
Good intent design:
- Mutually exclusive categories when possible
- Balanced number of training examples
- Clear boundaries between intents
- Hierarchical structure for complex domains

Example taxonomy for banking:
- account
  - account.balance
  - account.statement
  - account.open
  - account.close
- transfer
  - transfer.internal
  - transfer.external
  - transfer.schedule
- support
  - support.complaint
  - support.feedback
  - support.hours
```

### Multi-label vs Multi-class

| Type | Description | Example |
|------|-------------|---------|
| Multi-class | One intent per utterance | "What's my balance?" -> account_balance |
| Multi-label | Multiple intents per utterance | "Check balance and pay my bill" -> [account_balance, pay_bill] |

Most production systems use multi-class but may need multi-label for complex queries.

## Implementation Approaches

### Traditional ML

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Build intent classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ('classifier', LinearSVC(C=1.0, class_weight='balanced'))
])

# Train
pipeline.fit(train_texts, train_intents)

# Evaluate with cross-validation
scores = cross_val_score(pipeline, train_texts, train_intents, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Predict
intent = pipeline.predict(["What's my account balance?"])[0]
```

### Transformer-Based Classification

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np

# Prepare data
train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
})

# Load model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(intent_labels)
)

# Tokenize
def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64  # Intent queries are typically short
    )

tokenized_train = train_dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./intent_model",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

trainer.train()
```

### Using Sentence Transformers

Embedding-based approach with nearest neighbor classification:

```python
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Encode training examples
model = SentenceTransformer('all-MiniLM-L6-v2')
train_embeddings = model.encode(train_texts)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(train_embeddings, train_intents)

# Predict new queries
def predict_intent(query):
    embedding = model.encode([query])
    intent = knn.predict(embedding)[0]
    probabilities = knn.predict_proba(embedding)[0]
    confidence = probabilities.max()
    return intent, confidence

intent, confidence = predict_intent("Show me my account balance")
print(f"Intent: {intent}, Confidence: {confidence:.3f}")
```

### Few-Shot Classification with SetFit

For limited training data:

```python
from setfit import SetFitModel, SetFitTrainer

# Load pre-trained model
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Create trainer with few examples per class
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    num_iterations=20,
    num_epochs=1,
    batch_size=16,
)

# Train
trainer.train()

# Predict
predictions = model.predict(["What's my balance?"])
```

### Zero-Shot Classification

When no training data exists for new intents:

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "I need to transfer money to my savings account"
candidate_labels = [
    "account_balance",
    "money_transfer",
    "account_opening",
    "complaint"
]

result = classifier(text, candidate_labels)
print(f"Intent: {result['labels'][0]}")
print(f"Confidence: {result['scores'][0]:.3f}")
```

## Out-of-Scope Detection

Critical for production systems to recognize queries outside known intents.

### Confidence Thresholding

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_with_oos_detection(text, model, tokenizer, threshold=0.7):
    """Predict intent with out-of-scope detection."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_class = probabilities.max(dim=-1)

    if confidence.item() < threshold:
        return "out_of_scope", confidence.item()

    return id2label[predicted_class.item()], confidence.item()
```

### Training an OOS Detector

Add out-of-scope examples to training:

```python
# Collect OOS examples
# - Random sentences from other domains
# - Nonsensical queries
# - Edge cases from production logs

oos_examples = [
    "The weather is nice today",
    "Who won the Super Bowl?",
    "asdfghjkl",
    "Can you write me a poem?",
]

# Add as separate class
train_texts.extend(oos_examples)
train_labels.extend(["out_of_scope"] * len(oos_examples))
```

### Contrastive OOS Detection

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ContrastiveOOSDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.intent_centroids = {}

    def fit(self, texts, intents):
        """Compute centroid for each intent."""
        embeddings = self.encoder.encode(texts)

        for intent in set(intents):
            mask = [i == intent for i in intents]
            intent_embeddings = embeddings[mask]
            self.intent_centroids[intent] = intent_embeddings.mean(axis=0)

    def predict(self, text, oos_threshold=0.3):
        """Predict intent or detect OOS."""
        embedding = self.encoder.encode([text])[0]

        # Find closest intent centroid
        best_intent = None
        best_similarity = -1

        for intent, centroid in self.intent_centroids.items():
            similarity = np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent

        if best_similarity < oos_threshold:
            return "out_of_scope", best_similarity

        return best_intent, best_similarity
```

## Slot Filling Integration

Intent classification paired with entity extraction:

```python
from transformers import pipeline

# Combined intent + entity model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

def understand_query(text, intent_model, ner_model):
    """Full query understanding with intent and slots."""
    # Get intent
    intent_result = intent_model(text)
    intent = intent_result["labels"][0]
    intent_confidence = intent_result["scores"][0]

    # Extract entities
    entities = ner_model(text)

    # Map to slots based on intent
    slots = {}
    if intent == "book_flight":
        for entity in entities:
            if entity["entity"] == "LOC":
                if "departure" not in slots:
                    slots["departure"] = entity["word"]
                else:
                    slots["destination"] = entity["word"]
            elif entity["entity"] == "DATE":
                slots["date"] = entity["word"]

    return {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "slots": slots,
        "raw_entities": entities
    }
```

## Hierarchical Intent Classification

For complex domains with many intents:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HierarchicalIntentClassifier:
    def __init__(self, coarse_model_path, fine_models_paths):
        self.coarse_model = AutoModelForSequenceClassification.from_pretrained(coarse_model_path)
        self.coarse_tokenizer = AutoTokenizer.from_pretrained(coarse_model_path)

        self.fine_models = {}
        self.fine_tokenizers = {}

        for category, path in fine_models_paths.items():
            self.fine_models[category] = AutoModelForSequenceClassification.from_pretrained(path)
            self.fine_tokenizers[category] = AutoTokenizer.from_pretrained(path)

    def predict(self, text):
        # First: coarse classification
        coarse_inputs = self.coarse_tokenizer(text, return_tensors="pt", truncation=True)
        coarse_outputs = self.coarse_model(**coarse_inputs)
        coarse_probs = torch.softmax(coarse_outputs.logits, dim=-1)
        coarse_intent = self.coarse_id2label[coarse_probs.argmax().item()]

        # Second: fine-grained classification within category
        fine_model = self.fine_models[coarse_intent]
        fine_tokenizer = self.fine_tokenizers[coarse_intent]

        fine_inputs = fine_tokenizer(text, return_tensors="pt", truncation=True)
        fine_outputs = fine_model(**fine_inputs)
        fine_probs = torch.softmax(fine_outputs.logits, dim=-1)
        fine_intent = self.fine_id2labels[coarse_intent][fine_probs.argmax().item()]

        return f"{coarse_intent}.{fine_intent}"

# Usage
# classifier.predict("What's my checking account balance?")
# -> "account.balance"
```

## Data Augmentation

Expand limited training data:

```python
import random

def augment_intent_data(texts, intents, augmentations_per_sample=2):
    """Augment training data for intent classification."""
    augmented_texts = []
    augmented_intents = []

    for text, intent in zip(texts, intents):
        augmented_texts.append(text)
        augmented_intents.append(intent)

        for _ in range(augmentations_per_sample):
            aug_text = random_augmentation(text)
            augmented_texts.append(aug_text)
            augmented_intents.append(intent)

    return augmented_texts, augmented_intents

def random_augmentation(text):
    """Apply random augmentation."""
    augmentations = [
        synonym_replacement,
        random_insertion,
        random_deletion,
        random_swap,
    ]
    aug_func = random.choice(augmentations)
    return aug_func(text)

# Using nlpaug library
import nlpaug.augmenter.word as naw

synonym_aug = naw.SynonymAug(aug_src='wordnet')
augmented = synonym_aug.augment("Book a flight to London")
```

### Paraphrase Generation

```python
from transformers import pipeline

paraphraser = pipeline(
    "text2text-generation",
    model="Vamsi/T5_Paraphrase_Paws"
)

def generate_paraphrases(text, num_paraphrases=3):
    """Generate paraphrases for data augmentation."""
    prompt = f"paraphrase: {text}"
    outputs = paraphraser(
        prompt,
        max_length=60,
        num_return_sequences=num_paraphrases,
        do_sample=True,
        temperature=0.7
    )
    return [out["generated_text"] for out in outputs]

# Example
paraphrases = generate_paraphrases("What is my account balance?")
# ["Show me my balance", "How much money do I have?", "Check my balance"]
```

## Evaluation

### Metrics

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Classification report
print(classification_report(y_true, y_pred, target_names=intent_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=intent_names, yticklabels=intent_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Intent Classification Confusion Matrix')
plt.show()

# Per-intent accuracy
for intent in intent_names:
    mask = [y == intent for y in y_true]
    intent_acc = accuracy_score(
        [y_true[i] for i in range(len(y_true)) if mask[i]],
        [y_pred[i] for i in range(len(y_pred)) if mask[i]]
    )
    print(f"{intent}: {intent_acc:.3f}")
```

### Common Datasets

| Dataset | Domain | Intents | Utterances | Notes |
|---------|--------|---------|------------|-------|
| ATIS | Airline travel | 21 | 5,871 | Classic benchmark |
| SNIPS | Voice assistant | 7 | 14,484 | Slot filling included |
| Banking77 | Banking | 77 | 13,083 | Fine-grained intents |
| CLINC150 | General assistant | 150 | 23,700 | Includes OOS examples |
| HWU64 | Home assistant | 64 | 25,716 | Diverse domains |

### CLINC150 OOS Evaluation

```python
from datasets import load_dataset

# Load CLINC150 with OOS examples
dataset = load_dataset("clinc_oos", "plus")

# Evaluate OOS detection
def evaluate_oos_detection(model, test_data, threshold):
    oos_correct = 0
    oos_total = 0
    in_scope_correct = 0
    in_scope_total = 0

    for example in test_data:
        pred_intent, confidence = model.predict(example["text"])
        true_intent = example["intent"]

        if true_intent == "oos":
            oos_total += 1
            if confidence < threshold or pred_intent == "oos":
                oos_correct += 1
        else:
            in_scope_total += 1
            if pred_intent == true_intent and confidence >= threshold:
                in_scope_correct += 1

    return {
        "oos_recall": oos_correct / oos_total,
        "in_scope_accuracy": in_scope_correct / in_scope_total
    }
```

## Production Best Practices

### Logging and Monitoring

```python
import logging
from datetime import datetime

class IntentClassifier:
    def __init__(self, model, threshold=0.7):
        self.model = model
        self.threshold = threshold
        self.logger = logging.getLogger("intent_classifier")

    def predict(self, text, session_id=None):
        start_time = datetime.now()

        intent, confidence = self._classify(text)

        # Log for monitoring and retraining
        self.logger.info({
            "session_id": session_id,
            "text": text,
            "predicted_intent": intent,
            "confidence": confidence,
            "below_threshold": confidence < self.threshold,
            "latency_ms": (datetime.now() - start_time).total_seconds() * 1000
        })

        if confidence < self.threshold:
            return {"intent": "low_confidence", "fallback": True, "confidence": confidence}

        return {"intent": intent, "fallback": False, "confidence": confidence}
```

### Active Learning

Use low-confidence predictions for labeling:

```python
def collect_for_labeling(predictions, threshold_low=0.5, threshold_high=0.7):
    """Collect uncertain predictions for human labeling."""
    for_labeling = []

    for pred in predictions:
        if threshold_low < pred["confidence"] < threshold_high:
            for_labeling.append({
                "text": pred["text"],
                "predicted_intent": pred["intent"],
                "confidence": pred["confidence"]
            })

    return for_labeling
```

### Fallback Strategies

```python
def handle_query(text, intent_model, fallback_threshold=0.6):
    """Handle query with fallback strategies."""
    intent, confidence = intent_model.predict(text)

    if confidence >= fallback_threshold:
        return execute_intent(intent, text)

    # Fallback 1: Ask for clarification
    if confidence >= 0.4:
        return {
            "action": "clarify",
            "message": f"Did you mean {intent}?",
            "options": get_similar_intents(intent)
        }

    # Fallback 2: Suggest common intents
    if confidence >= 0.2:
        return {
            "action": "suggest",
            "message": "I'm not sure what you need. Here are some things I can help with:",
            "options": get_popular_intents()
        }

    # Fallback 3: Hand off to human
    return {
        "action": "handoff",
        "message": "Let me connect you with a human agent."
    }
```

## Comparison with Sentiment Analysis

| Aspect | Intent Classification | Sentiment Analysis |
|--------|----------------------|-------------------|
| Output | Discrete action categories | Polarity (positive/negative/neutral) |
| Classes | Domain-specific, many classes | Usually 2-5 classes |
| Use case | Chatbots, virtual assistants | Reviews, social media |
| Training data | Requires labeled examples per intent | General sentiment datasets available |
| OOS handling | Critical | Less important |

Both are text classification tasks but serve different purposes. Intent classification often requires paired entity extraction, while sentiment analysis typically stands alone.
