# Abstractive Summarization

## Summary

Abstractive summarization generates novel text that captures the essential meaning of the source document. Unlike extractive methods, abstractive summarizers can paraphrase, compress, and synthesize information using words and phrases not present in the original. Modern approaches use sequence-to-sequence models with attention mechanisms, primarily transformer-based architectures like BART, T5, and Pegasus.

Key points to remember:

- Generation-based: Creates new text rather than selecting existing sentences
- Paraphrasing: Can rephrase and compress information
- Encoder-decoder: Transformer architectures with cross-attention
- Pre-training: Models like BART, Pegasus use summarization-specific objectives
- Hallucination: Primary challengemay generate factually incorrect content
- Controllability: Length, style, and focus can be controlled
- Evaluation: ROUGE, BERTScore, factuality metrics
- Fine-tuning: Domain adaptation improves performance significantly

## Architecture

### Encoder-Decoder Framework

```python
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

class AbstractiveSummarizer:
    """BART-based abstractive summarization."""

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text, max_length=150, min_length=50):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### Key Model Architectures

```python
# BART: Denoising autoencoder pre-training
from transformers import BartForConditionalGeneration

# T5: Text-to-text framework
from transformers import T5ForConditionalGeneration

# Pegasus: Gap-sentence generation pre-training
from transformers import PegasusForConditionalGeneration

# LED: Longformer Encoder Decoder for long documents
from transformers import LEDForConditionalGeneration

class MultiModelSummarizer:
    """Compare different summarization models."""

    MODELS = {
        'bart': ('facebook/bart-large-cnn', BartForConditionalGeneration),
        't5': ('t5-large', T5ForConditionalGeneration),
        'pegasus': ('google/pegasus-cnn_dailymail', PegasusForConditionalGeneration),
        'led': ('allenai/led-large-16384-arxiv', LEDForConditionalGeneration)
    }

    def __init__(self, model_name='bart'):
        model_path, model_class = self.MODELS[model_name]
        self.model = model_class.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_name = model_name

    def summarize(self, text, **kwargs):
        if self.model_name == 't5':
            text = "summarize: " + text

        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Pre-training Objectives

### BART Denoising

```python
class BARTPretraining:
    """
    BART pre-training corruptions:
    1. Token masking
    2. Token deletion
    3. Text infilling
    4. Sentence permutation
    5. Document rotation
    """

    def text_infilling(self, text, mask_ratio=0.3):
        """Replace spans with single [MASK] token."""
        tokens = text.split()
        num_masks = int(len(tokens) * mask_ratio)

        # Sample span lengths from Poisson distribution
        span_lengths = np.random.poisson(3, num_masks)

        # Apply masking
        masked_tokens = tokens.copy()
        i = 0
        masks_applied = 0

        while i < len(masked_tokens) and masks_applied < num_masks:
            span_len = min(span_lengths[masks_applied], len(masked_tokens) - i)
            masked_tokens[i:i+span_len] = ['[MASK]']
            i += 1
            masks_applied += 1

        return ' '.join(masked_tokens)
```

### Pegasus Gap-Sentence Generation

```python
class PegasusPretraining:
    """
    Pegasus GSG: Mask entire sentences that are
    most similar to the rest of the document.
    """

    def select_gap_sentences(self, sentences, gap_ratio=0.3):
        """Select sentences to mask based on ROUGE importance."""
        # Compute importance of each sentence
        importance = []
        for i, sent in enumerate(sentences):
            other_sents = sentences[:i] + sentences[i+1:]
            other_text = ' '.join(other_sents)
            # ROUGE-1 F1 between sentence and rest
            score = compute_rouge1(sent, other_text)
            importance.append((i, score))

        # Select top sentences as gaps (pseudo-summary)
        num_gaps = int(len(sentences) * gap_ratio)
        importance.sort(key=lambda x: x[1], reverse=True)
        gap_indices = [idx for idx, _ in importance[:num_gaps]]

        return gap_indices
```

## Decoding Strategies

### Beam Search with Constraints

```python
class ConstrainedDecoder:
    """Decoding with various constraints."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, text, constraints=None):
        inputs = self.tokenizer(text, return_tensors="pt")

        generate_kwargs = {
            "max_length": 150,
            "min_length": 50,
            "num_beams": 4,
            "length_penalty": 2.0,
            "no_repeat_ngram_size": 3,  # Prevent repetition
            "early_stopping": True
        }

        if constraints:
            if 'force_words' in constraints:
                # Force certain words to appear
                force_words_ids = [
                    self.tokenizer(word, add_special_tokens=False).input_ids
                    for word in constraints['force_words']
                ]
                generate_kwargs['force_words_ids'] = force_words_ids

            if 'bad_words' in constraints:
                # Prevent certain words
                bad_words_ids = [
                    self.tokenizer(word, add_special_tokens=False).input_ids
                    for word in constraints['bad_words']
                ]
                generate_kwargs['bad_words_ids'] = bad_words_ids

        outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Diverse Beam Search

```python
def diverse_beam_search(model, tokenizer, text, num_beams=6, num_beam_groups=3):
    """Generate diverse summaries using beam groups."""
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=0.5,
        num_return_sequences=num_beam_groups
    )

    summaries = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    return summaries
```

### Nucleus Sampling

```python
def sample_summary(model, tokenizer, text, temperature=0.7, top_p=0.9):
    """Generate with nucleus sampling for variety."""
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=50
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Controllable Summarization

### Length Control

```python
class LengthControlledSummarizer:
    """Control summary length precisely."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def summarize_to_length(self, text, target_words):
        """Generate summary with approximately target word count."""
        # Estimate tokens from words (rough approximation)
        target_tokens = int(target_words * 1.3)

        inputs = self.tokenizer(text, return_tensors="pt")

        # Use length penalty and constraints
        outputs = self.model.generate(
            **inputs,
            max_length=target_tokens + 20,
            min_length=target_tokens - 20,
            length_penalty=1.0,
            num_beams=4
        )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def summarize_with_prefix(self, text, length_token="<short>"):
        """Use control tokens for length."""
        # Some models support control tokens
        controlled_text = f"{length_token} {text}"
        return self.summarize(controlled_text)
```

### Focus Control

```python
class FocusedSummarizer:
    """Summarize focusing on specific aspects."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def summarize_with_focus(self, text, focus_query):
        """Generate query-focused summary."""
        # Prepend focus query
        prompt = f"Focus on {focus_query}: {text}"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, max_length=150)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize_by_aspect(self, text, aspects):
        """Generate separate summaries for each aspect."""
        summaries = {}
        for aspect in aspects:
            summaries[aspect] = self.summarize_with_focus(text, aspect)
        return summaries

# Usage
summarizer = FocusedSummarizer(model, tokenizer)
article = "Long article about company earnings..."
aspects = ["financial performance", "future outlook", "risks"]
aspect_summaries = summarizer.summarize_by_aspect(article, aspects)
```

## Hallucination Mitigation

### Factuality Checking

```python
from transformers import pipeline

class FactualSummarizer:
    """Generate summaries with factuality verification."""

    def __init__(self, summarizer_model, nli_model="roberta-large-mnli"):
        self.summarizer = summarizer_model
        self.nli = pipeline("text-classification", model=nli_model)

    def summarize_with_verification(self, text, num_candidates=5):
        """Generate multiple candidates and select most factual."""
        candidates = self._generate_candidates(text, num_candidates)

        # Score each candidate for factuality
        scored_candidates = []
        for candidate in candidates:
            score = self._check_factuality(text, candidate)
            scored_candidates.append((candidate, score))

        # Return most factual
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def _check_factuality(self, source, summary):
        """Check if summary claims are entailed by source."""
        # Split summary into sentences
        sentences = summary.split('. ')

        entailment_scores = []
        for sent in sentences:
            result = self.nli(f"{source} [SEP] {sent}")
            if result[0]['label'] == 'ENTAILMENT':
                entailment_scores.append(result[0]['score'])
            elif result[0]['label'] == 'CONTRADICTION':
                entailment_scores.append(-result[0]['score'])
            else:
                entailment_scores.append(0)

        return sum(entailment_scores) / len(entailment_scores)
```

### Copy Mechanism

```python
class CopyAugmentedSummarizer(nn.Module):
    """Pointer-generator network for faithful summarization."""

    def __init__(self, encoder, decoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Pointer mechanism
        self.p_gen_linear = nn.Linear(decoder.hidden_size * 2, 1)

    def forward(self, source, target=None):
        # Encode source
        encoder_outputs, encoder_hidden = self.encoder(source)

        # Decode with copy mechanism
        outputs = []
        hidden = encoder_hidden

        for t in range(max_length):
            # Decoder step
            decoder_output, hidden, attention = self.decoder(
                input_token, hidden, encoder_outputs
            )

            # Compute p_gen (probability of generating vs copying)
            p_gen = torch.sigmoid(self.p_gen_linear(
                torch.cat([decoder_output, hidden], dim=-1)
            ))

            # Vocabulary distribution
            vocab_dist = F.softmax(self.vocab_projection(decoder_output), dim=-1)

            # Copy distribution (attention over source)
            copy_dist = attention

            # Combined distribution
            final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist

            outputs.append(final_dist)

        return outputs
```

## Long Document Summarization

### Longformer Encoder Decoder

```python
from transformers import LEDForConditionalGeneration, LEDTokenizer

class LongDocSummarizer:
    """Summarize documents up to 16K tokens."""

    def __init__(self):
        self.model = LEDForConditionalGeneration.from_pretrained(
            "allenai/led-large-16384-arxiv"
        )
        self.tokenizer = LEDTokenizer.from_pretrained(
            "allenai/led-large-16384-arxiv"
        )

    def summarize(self, long_document):
        inputs = self.tokenizer(
            long_document,
            return_tensors="pt",
            max_length=16384,
            truncation=True
        )

        # Set global attention on first token
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask,
            max_length=512,
            num_beams=4
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Hierarchical Summarization

```python
class HierarchicalSummarizer:
    """Summarize long documents in stages."""

    def __init__(self, model, tokenizer, chunk_size=512):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def summarize(self, document):
        # Stage 1: Split into chunks
        chunks = self._split_into_chunks(document)

        # Stage 2: Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self._summarize_chunk(chunk)
            chunk_summaries.append(summary)

        # Stage 3: Combine and summarize again
        combined = ' '.join(chunk_summaries)

        if len(combined.split()) > self.chunk_size:
            # Recursively summarize
            return self.summarize(combined)
        else:
            return self._summarize_chunk(combined)

    def _split_into_chunks(self, text, overlap=50):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks
```

## Evaluation

### Comprehensive Metrics

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

class SummarizationEvaluator:
    """Comprehensive evaluation suite."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

    def evaluate(self, predictions, references, sources=None):
        results = {}

        # ROUGE scores
        rouge_scores = self._compute_rouge(predictions, references)
        results.update(rouge_scores)

        # BERTScore
        P, R, F1 = bert_score(predictions, references, lang='en')
        results['bertscore_f1'] = F1.mean().item()

        # Factuality (if sources provided)
        if sources:
            results['factuality'] = self._compute_factuality(
                predictions, sources
            )

        # Length statistics
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        results['avg_pred_length'] = np.mean(pred_lengths)
        results['avg_ref_length'] = np.mean(ref_lengths)

        return results

    def _compute_rouge(self, predictions, references):
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for pred, ref in zip(predictions, references):
            result = self.rouge_scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        return {k: np.mean(v) for k, v in scores.items()}
```

### Human Evaluation Criteria

```python
def human_evaluation_protocol():
    """Standard criteria for human evaluation."""
    criteria = {
        'fluency': {
            'description': 'Is the summary grammatically correct and readable?',
            'scale': '1-5 (1=incomprehensible, 5=perfectly fluent)'
        },
        'coherence': {
            'description': 'Does the summary flow logically?',
            'scale': '1-5 (1=incoherent, 5=perfectly organized)'
        },
        'relevance': {
            'description': 'Does the summary cover the main points?',
            'scale': '1-5 (1=irrelevant, 5=captures all key information)'
        },
        'consistency': {
            'description': 'Is the summary factually consistent with source?',
            'scale': '1-5 (1=contradicts source, 5=fully consistent)'
        }
    }
    return criteria
```

## Fine-tuning

### Domain Adaptation

```python
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

def fine_tune_summarizer(model, tokenizer, train_data, val_data):
    """Fine-tune for specific domain."""

    def preprocess(examples):
        inputs = tokenizer(
            examples['document'],
            max_length=1024,
            truncation=True
        )
        targets = tokenizer(
            examples['summary'],
            max_length=128,
            truncation=True
        )
        inputs['labels'] = targets['input_ids']
        return inputs

    train_dataset = train_data.map(preprocess, batched=True)
    val_dataset = val_data.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="./summarizer",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    return model
```

## Benchmark Datasets

| Dataset | Domain | Size | Avg Doc Length | Avg Summary Length |
|---------|--------|------|----------------|-------------------|
| CNN/DailyMail | News | 300K | 781 words | 56 words |
| XSum | News | 227K | 431 words | 23 words |
| arXiv | Scientific | 215K | 6K words | 292 words |
| PubMed | Biomedical | 133K | 3K words | 214 words |
| SAMSum | Dialogue | 16K | 94 words | 20 words |
| BookSum | Books | 12K | 5K+ words | 505 words |

## Production Considerations

### Batch Processing

```python
class BatchSummarizer:
    """Efficient batch summarization."""

    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def summarize_batch(self, documents):
        summaries = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            )

            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4
            )

            batch_summaries = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            summaries.extend(batch_summaries)

        return summaries
```

### Caching and Optimization

```python
import hashlib

class CachedSummarizer:
    """Summarizer with result caching."""

    def __init__(self, model, tokenizer, cache_dir="./cache"):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}

    def summarize(self, text):
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate summary
        summary = self._generate(text)

        # Cache result
        self.cache[cache_key] = summary
        return summary
```

## Best Practices

### Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| News articles | BART-CNN, Pegasus-CNN |
| Scientific papers | LED, BigBird |
| Dialogues | BART-SAMSum |
| General purpose | FLAN-T5 |
| Extreme compression | Pegasus-XSum |

### Quality Assurance

```python
def validate_summary(summary, source):
    """Check summary quality before serving."""
    issues = []

    # Length check
    if len(summary.split()) < 10:
        issues.append("Summary too short")
    if len(summary.split()) > 200:
        issues.append("Summary too long")

    # Repetition check
    words = summary.lower().split()
    if len(words) != len(set(words)) * 0.7:  # Too many repeated words
        issues.append("Excessive repetition detected")

    # Hallucination check (basic)
    source_lower = source.lower()
    for sentence in summary.split('.'):
        # Check if key entities exist in source
        entities = extract_entities(sentence)
        for entity in entities:
            if entity.lower() not in source_lower:
                issues.append(f"Potential hallucination: {entity}")

    return issues
```
