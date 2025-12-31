# Sentiment Analysis

## Summary

Sentiment analysis is the task of automatically determining the emotional tone or opinion expressed in text. It classifies text as positive, negative, or neutral, and can extend to more fine-grained emotions or aspect-based opinions. Sentiment analysis powers product review analysis, social media monitoring, customer feedback systems, and brand reputation management.

Key points to remember:

- Task types: Binary (positive/negative), ternary (positive/neutral/negative), multi-class (fine-grained emotions)
- Approaches: Lexicon-based (rule-based), traditional ML (with features), deep learning (end-to-end)
- Transformer models (BERT, RoBERTa) achieve state-of-the-art on benchmark datasets
- Aspect-based sentiment analysis (ABSA) extracts sentiment for specific entities or features
- Domain matters: Sentiment expressions vary across domains (product reviews vs social media vs finance)
- Common challenges: Sarcasm, negation, implicit sentiment, domain adaptation
- Pre-trained sentiment models available on Hugging Face for quick deployment

## Task Variations

### Granularity Levels

| Type | Classes | Example |
|------|---------|---------|
| Binary | Positive, Negative | Review classification |
| Ternary | Positive, Neutral, Negative | Social media monitoring |
| 5-point scale | 1-5 stars | Rating prediction |
| Emotion | Joy, Anger, Fear, Sadness, etc. | Emotion detection |

### Aspect-Based Sentiment Analysis (ABSA)

Instead of overall sentiment, ABSA identifies sentiment toward specific aspects:

```
"The food was delicious but the service was slow."

Aspects:
- food: positive
- service: negative
```

Components:
1. Aspect extraction: Identify what is being discussed
2. Sentiment classification: Determine polarity toward each aspect

## Approaches

### Lexicon-Based Methods

Use sentiment dictionaries mapping words to polarity scores:

```python
# Simple lexicon-based sentiment
positive_words = {"good", "great", "excellent", "amazing", "love"}
negative_words = {"bad", "terrible", "awful", "hate", "poor"}

def lexicon_sentiment(text):
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"
```

Popular lexicons:
- VADER: Optimized for social media, handles emojis and slang
- SentiWordNet: WordNet synsets with sentiment scores
- AFINN: Simple word list with integer scores (-5 to +5)

### VADER (Valence Aware Dictionary for Sentiment Reasoning)

VADER handles social media conventions including punctuation, capitalization, and emojis:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "This movie was great!",
    "This movie was GREAT!!!",
    "This movie was not great.",
    "This movie was great :(",
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"{text}")
    print(f"  Compound: {scores['compound']:.3f}")
    print(f"  Pos: {scores['pos']:.3f}, Neg: {scores['neg']:.3f}, Neu: {scores['neu']:.3f}")
```

VADER returns:
- `compound`: Normalized score from -1 (most negative) to +1 (most positive)
- `pos`, `neg`, `neu`: Proportions that sum to 1

### Traditional Machine Learning

Features + classifier approach:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(train_texts, train_labels)

# Predict
predictions = pipeline.predict(test_texts)
probabilities = pipeline.predict_proba(test_texts)
```

Feature engineering options:
- Bag of words / TF-IDF
- N-grams (unigrams, bigrams)
- Part-of-speech tags
- Negation handling
- Sentiment lexicon features
- Word embeddings (averaged)

### Deep Learning

Neural network approaches that learn representations:

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))

        # Pack for variable length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)

        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        return self.fc(self.dropout(hidden))
```

### Transformer-Based Models

State-of-the-art approach using pre-trained language models:

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch

# Load pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # positive, neutral, negative
)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
```

## Using Pre-trained Models

### Hugging Face Pipeline

```python
from transformers import pipeline

# Load pre-trained sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "The weather is okay today.",
]

results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"{text}")
    print(f"  Label: {result['label']}, Score: {result['score']:.4f}")
```

### Popular Pre-trained Models

| Model | Domain | Labels | Notes |
|-------|--------|--------|-------|
| cardiffnlp/twitter-roberta-base-sentiment | Twitter | 3 classes | Optimized for social media |
| nlptown/bert-base-multilingual-uncased-sentiment | Reviews | 5 stars | Multilingual, product reviews |
| siebert/sentiment-roberta-large-english | General | Binary | High accuracy |
| finiteautomata/bertweet-base-sentiment-analysis | Twitter | 3 classes | BERTweet-based |

### Batch Processing

```python
from transformers import pipeline

# Configure for batch processing
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0  # Use GPU
)

# Process in batches
batch_size = 32
all_results = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    results = sentiment_pipeline(batch)
    all_results.extend(results)
```

## Domain Adaptation

Sentiment expressions vary by domain:

```
Product reviews: "This vacuum sucks!" -> positive (it works well)
General context: "This movie sucks!" -> negative

Finance: "The stock dropped" -> negative
Sports: "He dropped the ball" -> negative (different meaning)
```

### Fine-tuning for Domain

```python
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load domain-specific data
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Start from general sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Fine-tune on domain data
training_args = TrainingArguments(
    output_dir="./finance_sentiment",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

trainer.train()
```

## Handling Challenges

### Negation

Negation flips sentiment polarity:

```python
def handle_negation(tokens):
    """Mark tokens following negation words."""
    negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                      "nowhere", "hardly", "barely", "scarcely", "n't"}

    negated = []
    in_negation = False

    for token in tokens:
        if token.lower() in negation_words:
            in_negation = True
        elif token in ".!?":
            in_negation = False

        if in_negation and token.lower() not in negation_words:
            negated.append(f"NOT_{token}")
        else:
            negated.append(token)

    return negated

# "not good" becomes ["not", "NOT_good"]
```

### Sarcasm Detection

Sarcasm inverts the literal meaning. Approaches:
- Contextual features (hashtags, emojis)
- Incongruity detection (positive words + negative emoji)
- Pre-trained sarcasm detection models

```python
# Sarcasm detection as preprocessing
from transformers import pipeline

sarcasm_detector = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-irony"
)

def sentiment_with_sarcasm(text):
    sarcasm_result = sarcasm_detector(text)[0]

    if sarcasm_result["label"] == "irony" and sarcasm_result["score"] > 0.8:
        # Flip sentiment for sarcastic text
        sentiment = get_sentiment(text)
        return invert_sentiment(sentiment)
    else:
        return get_sentiment(text)
```

### Multi-lingual Sentiment

```python
from transformers import pipeline

# Multilingual sentiment model
multilingual_sentiment = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

texts = [
    "This is great!",           # English
    "C'est magnifique!",        # French
    "Das ist wunderbar!",       # German
]

for text in texts:
    result = multilingual_sentiment(text)[0]
    print(f"{text} -> {result['label']}: {result['score']:.3f}")
```

## Aspect-Based Sentiment Analysis

### Extraction and Classification

```python
from transformers import pipeline

# ABSA pipeline (aspect extraction + sentiment)
absa_pipeline = pipeline(
    "text-classification",
    model="yangheng/deberta-v3-base-absa-v1.1"
)

text = "The food was excellent but the service was terrible."

# For each aspect, classify sentiment
aspects = ["food", "service", "price", "ambiance"]

for aspect in aspects:
    input_text = f"{text} [SEP] {aspect}"
    result = absa_pipeline(input_text)
    print(f"{aspect}: {result[0]['label']} ({result[0]['score']:.3f})")
```

### Custom ABSA Implementation

```python
import torch
from transformers import AutoModel, AutoTokenizer

class ABSAModel(torch.nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size * 2,
            num_labels
        )

    def forward(self, input_ids, attention_mask, aspect_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Get [CLS] representation
        cls_repr = hidden_states[:, 0, :]

        # Get aspect representation (average of aspect tokens)
        aspect_repr = (hidden_states * aspect_mask.unsqueeze(-1)).sum(dim=1)
        aspect_repr = aspect_repr / aspect_mask.sum(dim=1, keepdim=True)

        # Concatenate for classification
        combined = torch.cat([cls_repr, aspect_repr], dim=-1)

        return self.classifier(combined)
```

## Evaluation

### Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Standard classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

# Detailed report
print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))
```

### Common Benchmarks

| Dataset | Domain | Size | Labels |
|---------|--------|------|--------|
| SST-2 | Movie reviews | 67K sentences | Binary |
| SST-5 | Movie reviews | 11K sentences | 5 classes |
| IMDB | Movie reviews | 50K reviews | Binary |
| Yelp | Business reviews | 700K reviews | 5 stars |
| SemEval | Twitter | Varies | 3 classes |
| Amazon Reviews | Products | 4M reviews | 5 stars |

## Production Considerations

### Preprocessing

```python
import re
import emoji

def preprocess_for_sentiment(text):
    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'ll": " will",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Handle emojis (for social media)
    text = emoji.demojize(text)  # Convert emoji to text

    # Normalize whitespace
    text = ' '.join(text.split())

    return text
```

### Confidence Thresholds

```python
def classify_with_confidence(text, model, tokenizer, threshold=0.7):
    """Return prediction only if confidence exceeds threshold."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_class = probabilities.max(dim=-1)

    if confidence.item() < threshold:
        return {"label": "uncertain", "confidence": confidence.item()}

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return {
        "label": label_map[predicted_class.item()],
        "confidence": confidence.item()
    }
```

### Aggregating Document Sentiment

For long documents, aggregate sentence-level sentiment:

```python
import numpy as np
from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def document_sentiment(text, sentence_tokenizer):
    """Aggregate sentence-level sentiment for documents."""
    sentences = sentence_tokenizer(text)

    if len(sentences) == 0:
        return {"label": "neutral", "confidence": 0.0}

    results = sentiment_model(sentences)

    # Weighted average by confidence
    scores = []
    weights = []

    for result in results:
        if result["label"] == "POSITIVE":
            scores.append(1)
        elif result["label"] == "NEGATIVE":
            scores.append(-1)
        else:
            scores.append(0)
        weights.append(result["score"])

    weighted_score = np.average(scores, weights=weights)

    if weighted_score > 0.2:
        label = "positive"
    elif weighted_score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": weighted_score,
        "sentence_results": results
    }
```

## Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Lexicon (VADER) | Fast, no training, interpretable | Limited vocabulary, no context | Quick prototyping, social media |
| Traditional ML | Interpretable, fast inference | Feature engineering required | Smaller datasets, interpretability needs |
| LSTM/CNN | Learns representations | Needs training data | Medium datasets |
| Transformers | State-of-the-art accuracy | Slow, resource intensive | High accuracy requirements |

## Best Practices

1. Start with pre-trained models before training custom ones
2. Consider domain: general models may underperform on specific domains
3. Handle class imbalance: most datasets have more neutral/positive than negative
4. Validate on held-out data from same distribution as production
5. Monitor model drift: language and sentiment expressions evolve
6. Combine with rule-based filters for edge cases (profanity, specific phrases)
7. Provide confidence scores alongside predictions
8. Consider multi-label for texts expressing mixed sentiments
