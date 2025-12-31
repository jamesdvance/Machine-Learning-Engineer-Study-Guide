# Event Extraction

## Summary

Event extraction identifies and structures events mentioned in text, including the event type, trigger words, and participating arguments with their roles. Unlike relation extraction which focuses on static relationships between entities, event extraction captures dynamic occurrences with temporal aspects. It is essential for news analysis, intelligence gathering, financial monitoring, and biomedical literature mining.

Key points to remember:

- Event components: Trigger (word indicating event) + Arguments (participants with roles)
- Event types: Domain-specific schemas (ACE, ERE, RAMS) define event ontologies
- Pipeline: Trigger detection followed by argument extraction
- Joint extraction: Modern approaches detect triggers and arguments simultaneously
- Document-level: Many events span multiple sentences requiring coreference
- Nested events: Arguments can themselves be events
- Zero-shot: Generative models enable extraction without predefined schemas
- Challenges: Implicit triggers, scattered arguments, event coreference

## Event Structure

### Components

```
Sentence: "Apple announced that Tim Cook will step down as CEO next year."

Event Type: Personnel.End-Position
Trigger: "step down"
Arguments:
  - Person: Tim Cook
  - Organization: Apple
  - Position: CEO
  - Time: next year
```

### Event Schema Example (ACE)

| Event Type | Subtypes | Typical Arguments |
|------------|----------|-------------------|
| Life | Be-Born, Die, Marry, Divorce | Person, Place, Time |
| Movement | Transport | Agent, Artifact, Origin, Destination |
| Transaction | Transfer-Ownership, Transfer-Money | Buyer, Seller, Artifact, Price |
| Business | Start-Org, End-Org, Merge-Org | Organization, Place, Time |
| Conflict | Attack, Demonstrate | Attacker, Target, Instrument, Place |
| Contact | Meet, Phone-Write | Participants, Place, Time |
| Personnel | Start-Position, End-Position, Elect | Person, Organization, Position |
| Justice | Arrest, Trial, Sentence | Defendant, Agent, Crime |

## Pipeline Approach

### Step 1: Trigger Detection

Identify words that indicate an event:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class TriggerDetector:
    def __init__(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.id2label = self.model.config.id2label

    def detect_triggers(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0]

        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

        triggers = []
        current_trigger = None

        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.id2label[pred.item()]

            if label.startswith("B-"):
                if current_trigger:
                    triggers.append(current_trigger)
                current_trigger = {
                    "type": label[2:],
                    "start": offset[0].item(),
                    "end": offset[1].item(),
                    "text": text[offset[0]:offset[1]]
                }
            elif label.startswith("I-") and current_trigger:
                current_trigger["end"] = offset[1].item()
                current_trigger["text"] = text[current_trigger["start"]:current_trigger["end"]]
            else:
                if current_trigger:
                    triggers.append(current_trigger)
                    current_trigger = None

        if current_trigger:
            triggers.append(current_trigger)

        return triggers
```

### Step 2: Argument Extraction

For each detected trigger, extract arguments:

```python
class ArgumentExtractor:
    def __init__(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def extract_arguments(self, text, trigger):
        # Mark trigger in text
        marked_text = self._mark_trigger(text, trigger)

        inputs = self.tokenizer(marked_text, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Parse argument spans
        arguments = self._parse_arguments(predictions, inputs)

        return {
            "trigger": trigger,
            "arguments": arguments
        }

    def _mark_trigger(self, text, trigger):
        # Insert markers around trigger
        before = text[:trigger["start"]]
        trigger_text = text[trigger["start"]:trigger["end"]]
        after = text[trigger["end"]:]
        return f"{before}<trigger>{trigger_text}</trigger>{after}"
```

## Joint Extraction Models

### BERT-Based Joint Model

```python
import torch
import torch.nn as nn
from transformers import BertModel

class JointEventExtractor(nn.Module):
    def __init__(self, model_name, num_event_types, num_argument_roles):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Trigger detection head
        self.trigger_classifier = nn.Linear(hidden_size, num_event_types)

        # Argument role classification head
        # For each token pair (trigger, argument candidate)
        self.argument_classifier = nn.Linear(hidden_size * 2, num_argument_roles)

    def forward(self, input_ids, attention_mask, trigger_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Trigger predictions
        trigger_logits = self.trigger_classifier(sequence_output)

        # If triggers provided (training) or detected (inference)
        if trigger_mask is not None:
            # Extract trigger representations
            trigger_repr = self._get_trigger_repr(sequence_output, trigger_mask)

            # For each token, classify argument role relative to trigger
            argument_logits = self._classify_arguments(sequence_output, trigger_repr)
        else:
            argument_logits = None

        return trigger_logits, argument_logits

    def _get_trigger_repr(self, sequence_output, trigger_mask):
        # Average pooling over trigger tokens
        trigger_mask = trigger_mask.unsqueeze(-1).float()
        trigger_repr = (sequence_output * trigger_mask).sum(dim=1) / trigger_mask.sum(dim=1)
        return trigger_repr

    def _classify_arguments(self, sequence_output, trigger_repr):
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Expand trigger representation to match sequence
        trigger_repr = trigger_repr.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate for each token
        combined = torch.cat([sequence_output, trigger_repr], dim=-1)

        return self.argument_classifier(combined)
```

### Sequence-to-Sequence Approach

Generate structured event output:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class GenerativeEventExtractor:
    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def extract_events(self, text):
        prompt = f"Extract events from: {text}"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_output(decoded)

    def _parse_output(self, text):
        """
        Parse structured output like:
        Event: Attack | Trigger: bombed | Attacker: rebels | Target: building | Place: Kabul
        """
        events = []
        for line in text.split("\n"):
            if line.startswith("Event:"):
                event = self._parse_event_line(line)
                events.append(event)
        return events
```

## Document-Level Event Extraction

### Handling Cross-Sentence Arguments

```python
class DocumentEventExtractor:
    def __init__(self, model):
        self.model = model
        self.coref_model = load_coreference_model()

    def extract_document_events(self, document):
        sentences = split_sentences(document)

        # Step 1: Extract sentence-level events
        sentence_events = []
        for sent in sentences:
            events = self.model.extract_events(sent)
            sentence_events.append(events)

        # Step 2: Resolve coreference
        coref_clusters = self.coref_model.predict(document)

        # Step 3: Link arguments across sentences
        document_events = self._merge_cross_sentence_events(
            sentence_events, coref_clusters
        )

        # Step 4: Event coreference (same event mentioned multiple times)
        merged_events = self._merge_coreferent_events(document_events)

        return merged_events

    def _merge_cross_sentence_events(self, sentence_events, coref_clusters):
        """Link arguments that refer to same entity across sentences."""
        # Implementation depends on coreference format
        pass
```

### RAMS Dataset Approach

Document-level with scattered arguments:

```python
def process_rams_example(document, trigger_info, argument_annotations):
    """
    RAMS format: trigger in one sentence, arguments may be in other sentences.
    """
    # Find sentence containing trigger
    trigger_sent_idx = find_sentence_with_trigger(document, trigger_info)

    # Extract context window around trigger
    context_start = max(0, trigger_sent_idx - 2)
    context_end = min(len(document), trigger_sent_idx + 3)
    context = document[context_start:context_end]

    # Arguments can be anywhere in context
    arguments = []
    for arg in argument_annotations:
        arg_sent_idx = find_sentence_with_span(document, arg['span'])
        if context_start <= arg_sent_idx < context_end:
            arguments.append({
                'role': arg['role'],
                'text': arg['text'],
                'relative_position': arg_sent_idx - trigger_sent_idx
            })

    return {
        'context': context,
        'trigger': trigger_info,
        'arguments': arguments
    }
```

## Zero-Shot Event Extraction

Extract events without training on specific schema:

```python
from transformers import pipeline

class ZeroShotEventExtractor:
    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-xl"
        )

    def extract_events(self, text, event_schema=None):
        if event_schema:
            schema_str = self._format_schema(event_schema)
            prompt = f"""Extract events from the text according to this schema:
{schema_str}

Text: {text}

Events (format as JSON):"""
        else:
            prompt = f"""Extract all events from the following text.
For each event, identify:
- Event type
- Trigger word
- Participants and their roles
- Time and location if mentioned

Text: {text}

Events:"""

        result = self.generator(prompt, max_length=500, do_sample=False)
        return self._parse_response(result[0]['generated_text'])

    def _format_schema(self, schema):
        lines = []
        for event_type, roles in schema.items():
            lines.append(f"- {event_type}: {', '.join(roles)}")
        return "\n".join(lines)
```

## Evaluation

### Metrics

```python
def evaluate_event_extraction(predictions, ground_truth):
    """
    Evaluate trigger detection and argument extraction.
    """
    # Trigger identification (type-agnostic)
    trigger_id_scores = evaluate_spans(
        [p['trigger']['span'] for p in predictions],
        [g['trigger']['span'] for g in ground_truth]
    )

    # Trigger classification (with type)
    trigger_class_scores = evaluate_spans_with_type(
        [(p['trigger']['span'], p['trigger']['type']) for p in predictions],
        [(g['trigger']['span'], g['trigger']['type']) for g in ground_truth]
    )

    # Argument identification
    arg_id_scores = evaluate_arguments(predictions, ground_truth, match_type=False)

    # Argument classification (with role)
    arg_class_scores = evaluate_arguments(predictions, ground_truth, match_type=True)

    return {
        'trigger_id': trigger_id_scores,
        'trigger_class': trigger_class_scores,
        'argument_id': arg_id_scores,
        'argument_class': arg_class_scores
    }

def evaluate_spans(pred_spans, gold_spans):
    """Evaluate span detection (exact match)."""
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### Benchmarks

| Dataset | Domain | Event Types | Annotation Level |
|---------|--------|-------------|------------------|
| ACE 2005 | News | 33 types | Sentence |
| ERE | News | 38 types | Sentence |
| RAMS | News | 139 types | Document |
| MAVEN | Wikipedia | 168 types | Document |
| M2E2 | Multimedia | 8 types | Multimodal |
| GENIA | Biomedical | 36 types | Sentence |

## Domain-Specific Applications

### Biomedical Event Extraction

```python
# GENIA-style biomedical events
biomedical_schema = {
    "Gene_expression": ["Theme"],
    "Transcription": ["Theme"],
    "Protein_catabolism": ["Theme"],
    "Phosphorylation": ["Theme", "Site"],
    "Localization": ["Theme", "ToLoc", "FromLoc"],
    "Binding": ["Theme", "Theme2", "Site"],
    "Regulation": ["Theme", "Cause"],
    "Positive_regulation": ["Theme", "Cause"],
    "Negative_regulation": ["Theme", "Cause"]
}

# Example
text = "IL-2 gene expression is regulated by NF-kB."
# Event 1: Gene_expression(Theme: IL-2 gene)
# Event 2: Regulation(Theme: Event1, Cause: NF-kB)
```

### Financial Event Extraction

```python
financial_schema = {
    "Acquisition": ["Acquirer", "Acquired", "Price", "Date"],
    "Merger": ["Company1", "Company2", "Date"],
    "IPO": ["Company", "Exchange", "Price", "Date"],
    "Bankruptcy": ["Company", "Date"],
    "Executive_Change": ["Person", "Company", "Position", "Type"],
    "Earnings": ["Company", "Amount", "Period"],
    "Dividend": ["Company", "Amount", "Date"]
}

text = "Apple acquired Beats for $3 billion in May 2014."
# Acquisition(Acquirer: Apple, Acquired: Beats, Price: $3 billion, Date: May 2014)
```

## Challenges and Solutions

### Implicit Triggers

Events without explicit trigger words:

```
"He's no longer with the company."
# Implicit End-Position event
```

Solution: Include implicit trigger detection or use frame semantics

### Scattered Arguments

Arguments spread across sentences:

```
"The explosion occurred in Baghdad. Three soldiers were killed."
# Attack event with arguments in both sentences
```

Solution: Document-level models with cross-sentence attention

### Nested Events

Events as arguments of other events:

```
"The arrest caused protests."
# Cause event with Arrest event as argument
```

Solution: Hierarchical event representation

```python
class HierarchicalEvent:
    def __init__(self, event_type, trigger, arguments):
        self.event_type = event_type
        self.trigger = trigger
        self.arguments = arguments  # Can contain other HierarchicalEvent objects

    def to_dict(self):
        return {
            'type': self.event_type,
            'trigger': self.trigger,
            'arguments': {
                role: arg.to_dict() if isinstance(arg, HierarchicalEvent) else arg
                for role, arg in self.arguments.items()
            }
        }
```

## Best Practices

### Data Annotation

1. Clear guidelines for trigger identification
2. Consistent argument boundary decisions
3. Handle overlapping events explicitly
4. Document edge cases and ambiguities

### Model Selection

| Scenario | Recommended Approach |
|----------|---------------------|
| ACE-style extraction | Joint BERT model |
| Document-level | RAMS-style with long context |
| New domain | Zero-shot with LLM then fine-tune |
| Real-time | Pipeline with cached NER |
| Biomedical | Domain-specific pre-training |

### Error Analysis

```python
def analyze_event_errors(predictions, ground_truth):
    errors = {
        'missed_triggers': [],
        'false_triggers': [],
        'wrong_event_type': [],
        'missed_arguments': [],
        'false_arguments': [],
        'wrong_argument_role': []
    }

    # Analyze trigger errors
    pred_triggers = {(p['trigger']['start'], p['trigger']['end']): p for p in predictions}
    gold_triggers = {(g['trigger']['start'], g['trigger']['end']): g for g in ground_truth}

    for span, gold in gold_triggers.items():
        if span not in pred_triggers:
            errors['missed_triggers'].append(gold)
        elif pred_triggers[span]['trigger']['type'] != gold['trigger']['type']:
            errors['wrong_event_type'].append((pred_triggers[span], gold))

    for span, pred in pred_triggers.items():
        if span not in gold_triggers:
            errors['false_triggers'].append(pred)

    return errors
```
