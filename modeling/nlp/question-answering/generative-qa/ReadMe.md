# Generative Question Answering

## Summary

Generative Question Answering (QA) produces free-form text answers rather than extracting spans from context. Using sequence-to-sequence or decoder-only language models, generative QA can synthesize information from multiple sources, handle questions requiring reasoning, and generate answers not explicitly stated in the source material. This approach powers modern conversational AI systems and retrieval-augmented generation (RAG) pipelines.

Key points to remember:

- Answer generation: Produces natural language answers rather than span extraction
- Architecture: Encoder-decoder (T5, BART) or decoder-only (GPT, LLaMA) models
- RAG: Combines retrieval with generation for knowledge-grounded responses
- Fusion-in-Decoder: Encodes multiple passages independently, fuses in decoder
- Hallucination: Primary challengemodel may generate plausible but incorrect answers
- Attribution: Modern systems link claims to source documents
- Open-domain: No predefined context; retrieves relevant documents first
- Evaluation: ROUGE, BERTScore, human evaluation for fluency and factuality

## Architecture Patterns

### Encoder-Decoder (T5/BART)

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5QA:
    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def answer(self, question, context):
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
qa = T5QA("google/flan-t5-large")
answer = qa.answer(
    "What causes climate change?",
    "Climate change is primarily caused by greenhouse gas emissions..."
)
```

### Decoder-Only (GPT-style)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class DecoderQA:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def answer(self, question, context):
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract only the generated answer
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()
        return answer
```

## Comparison: Extractive vs Generative QA

| Aspect | Extractive QA | Generative QA |
|--------|---------------|---------------|
| Output | Span from context | Free-form text |
| Answer source | Must exist in context | Can synthesize |
| Hallucination risk | None (by design) | Significant |
| Multi-hop reasoning | Limited | Natural |
| Abstraction | None | Can paraphrase |
| Evaluation | Exact/F1 match | ROUGE, human eval |
| Latency | Lower | Higher |
| Use case | Factoid lookup | Conversational AI |

## Retrieval-Augmented Generation (RAG)

### Basic RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, retriever_model, generator_model):
        self.encoder = SentenceTransformer(retriever_model)
        self.generator = generator_model
        self.index = None
        self.documents = []

    def index_documents(self, documents):
        self.documents = documents
        embeddings = self.encoder.encode(documents)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5):
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.documents[i], scores[0][j]) for j, i in enumerate(indices[0])]

    def answer(self, question, top_k=5):
        # Retrieve relevant documents
        retrieved = self.retrieve(question, top_k)

        # Combine into context
        context = "\n\n".join([doc for doc, score in retrieved])

        # Generate answer
        return self.generator.answer(question, context)
```

### Fusion-in-Decoder (FiD)

Process multiple passages independently, fuse in decoder:

```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class FusionInDecoder:
    """
    FiD encodes each passage separately with the question,
    then concatenates encoded representations for the decoder.
    """

    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def answer(self, question, passages, max_passage_length=256):
        # Encode each passage with the question
        encoded_passages = []

        for passage in passages:
            text = f"question: {question} context: {passage}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_passage_length,
                truncation=True,
                padding="max_length"
            )
            encoded_passages.append(inputs)

        # Concatenate all encoded passages
        input_ids = torch.cat([p["input_ids"] for p in encoded_passages], dim=1)
        attention_mask = torch.cat([p["attention_mask"] for p in encoded_passages], dim=1)

        # Run through encoder
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Generate with decoder attending to all passages
        outputs = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=128
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### RAG vs Fine-tuning

| Factor | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge updates | Real-time | Requires retraining |
| Hallucination | Reduced with attribution | More common |
| Domain adaptation | Add documents | Train on domain data |
| Cost | Retrieval overhead | Training cost |
| Transparency | Can cite sources | Black box |
| Maintenance | Update document store | Retrain periodically |

## Hallucination Mitigation

### Detection Approaches

```python
class HallucinationDetector:
    def __init__(self, nli_model):
        from transformers import pipeline
        self.nli = pipeline("text-classification", model=nli_model)

    def check_entailment(self, context, answer):
        """Check if answer is entailed by context."""
        result = self.nli(f"{context} [SEP] {answer}")
        return result[0]['label'] == 'ENTAILMENT'

    def detect_hallucination(self, question, context, answer):
        """
        Returns hallucination score (0-1).
        Higher = more likely hallucinated.
        """
        # Check factual consistency
        is_entailed = self.check_entailment(context, answer)

        # Check for claims not in context
        claims = self._extract_claims(answer)
        unsupported_claims = []

        for claim in claims:
            if not self.check_entailment(context, claim):
                unsupported_claims.append(claim)

        hallucination_ratio = len(unsupported_claims) / max(len(claims), 1)

        return {
            'is_entailed': is_entailed,
            'hallucination_score': hallucination_ratio,
            'unsupported_claims': unsupported_claims
        }
```

### Mitigation Strategies

```python
class GroundedGenerator:
    """Generate answers with explicit grounding."""

    def __init__(self, model):
        self.model = model

    def answer_with_citations(self, question, passages):
        """Generate answer with inline citations."""
        prompt = f"""Answer the question based ONLY on the provided passages.
Include [1], [2], etc. to cite which passage supports each claim.
If the answer cannot be found in the passages, say "I cannot find this information."

Passages:
{self._format_passages(passages)}

Question: {question}

Answer with citations:"""

        return self.model.generate(prompt)

    def _format_passages(self, passages):
        return "\n\n".join([
            f"[{i+1}] {passage}"
            for i, passage in enumerate(passages)
        ])

    def answer_with_confidence(self, question, context):
        """Generate answer with confidence indication."""
        prompt = f"""Answer the question based on the context.
Rate your confidence (HIGH/MEDIUM/LOW) based on how well the context supports the answer.

Context: {context}

Question: {question}

Provide your answer in this format:
Answer: [your answer]
Confidence: [HIGH/MEDIUM/LOW]
Reasoning: [why you have this confidence level]"""

        return self.model.generate(prompt)
```

## Open-Domain QA

### Full Pipeline

```python
class OpenDomainQA:
    """
    Complete open-domain QA: no pre-defined context,
    retrieve from large corpus then generate.
    """

    def __init__(self, retriever, generator, corpus_index):
        self.retriever = retriever
        self.generator = generator
        self.corpus_index = corpus_index

    def answer(self, question):
        # Step 1: Retrieve relevant passages
        passages = self.retriever.retrieve(
            question,
            index=self.corpus_index,
            top_k=10
        )

        # Step 2: Rerank for relevance
        reranked = self.rerank(question, passages)[:5]

        # Step 3: Generate answer with retrieved context
        answer = self.generator.answer(question, reranked)

        # Step 4: Verify and attribute
        verified = self.verify_answer(answer, reranked)

        return {
            'answer': answer,
            'sources': reranked,
            'verification': verified
        }

    def rerank(self, question, passages):
        """Cross-encoder reranking for better relevance."""
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [(question, p['text']) for p in passages]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(passages, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [p for p, s in ranked]
```

### Wikipedia-Scale QA

```python
class WikipediaQA:
    """QA over Wikipedia corpus."""

    def __init__(self):
        self.passage_encoder = SentenceTransformer('facebook/dpr-ctx_encoder-single-nq-base')
        self.question_encoder = SentenceTransformer('facebook/dpr-question_encoder-single-nq-base')

    def build_index(self, wikipedia_passages):
        """Build dense index over Wikipedia passages."""
        # Encode all passages (can be done in batches)
        embeddings = self.passage_encoder.encode(
            wikipedia_passages,
            batch_size=128,
            show_progress_bar=True
        )

        # Build FAISS index with IVF for efficiency
        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        index.train(embeddings)
        index.add(embeddings)

        return index

    def retrieve(self, question, index, passages, top_k=100):
        """Retrieve relevant Wikipedia passages."""
        question_embedding = self.question_encoder.encode([question])
        scores, indices = index.search(question_embedding, top_k)
        return [passages[i] for i in indices[0]]
```

## Evaluation Metrics

### Automatic Metrics

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_generative_qa(predictions, references):
    """Comprehensive evaluation for generative QA."""

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = {
        'rouge1': [], 'rouge2': [], 'rougeL': []
    }

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang='en')

    # Exact match (for factoid questions)
    exact_matches = [
        normalize_answer(p) == normalize_answer(r)
        for p, r in zip(predictions, references)
    ]

    return {
        'rouge1': sum(rouge_scores['rouge1']) / len(predictions),
        'rouge2': sum(rouge_scores['rouge2']) / len(predictions),
        'rougeL': sum(rouge_scores['rougeL']) / len(predictions),
        'bert_score_f1': F1.mean().item(),
        'exact_match': sum(exact_matches) / len(predictions)
    }

def normalize_answer(text):
    """Normalize for comparison."""
    import re
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())
```

### Factuality Evaluation

```python
class FactualityEvaluator:
    """Evaluate factual accuracy of generated answers."""

    def __init__(self):
        from transformers import pipeline
        self.qa_model = pipeline("question-answering")
        self.nli_model = pipeline("text-classification",
                                   model="roberta-large-mnli")

    def evaluate(self, generated_answer, gold_answer, context):
        """
        Multi-faceted factuality evaluation.
        """
        results = {}

        # 1. Answer equivalence
        results['answer_equivalence'] = self._check_equivalence(
            generated_answer, gold_answer
        )

        # 2. Context consistency
        results['context_consistency'] = self._check_consistency(
            generated_answer, context
        )

        # 3. Claim verification
        claims = self._extract_claims(generated_answer)
        verified_claims = [
            self._verify_claim(claim, context) for claim in claims
        ]
        results['claim_accuracy'] = sum(verified_claims) / max(len(claims), 1)

        return results

    def _check_equivalence(self, pred, gold):
        """Check if answers are semantically equivalent."""
        result = self.nli_model(f"{pred} [SEP] {gold}")
        return result[0]['label'] == 'ENTAILMENT'

    def _check_consistency(self, answer, context):
        """Check if answer is consistent with context."""
        result = self.nli_model(f"{context} [SEP] {answer}")
        return result[0]['label'] != 'CONTRADICTION'
```

## Training Generative QA Models

### Fine-tuning T5

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

def prepare_dataset(examples):
    """Prepare training examples."""
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples['question'], examples['context'])
    ]
    targets = examples['answer']

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding='max_length'
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding='max_length'
    )
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

# Training setup
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

training_args = TrainingArguments(
    output_dir="./qa_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

### RLHF for QA

```python
class RLHFQATrainer:
    """
    Reinforcement Learning from Human Feedback for QA.
    Rewards: factuality, helpfulness, safety.
    """

    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model

    def compute_reward(self, question, context, answer):
        """Compute reward for generated answer."""
        rewards = {}

        # Factuality reward
        rewards['factuality'] = self.reward_model.score_factuality(
            answer, context
        )

        # Helpfulness reward
        rewards['helpfulness'] = self.reward_model.score_helpfulness(
            question, answer
        )

        # Fluency reward
        rewards['fluency'] = self.reward_model.score_fluency(answer)

        # Combine with weights
        total = (
            0.5 * rewards['factuality'] +
            0.3 * rewards['helpfulness'] +
            0.2 * rewards['fluency']
        )

        return total, rewards
```

## Production Considerations

### Latency Optimization

```python
class OptimizedRAG:
    """Production-ready RAG with caching and batching."""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.cache = {}  # Query cache
        self.embedding_cache = {}  # Embedding cache

    def answer(self, question):
        # Check cache
        cache_key = self._hash_query(question)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Retrieve with cached embeddings
        passages = self._cached_retrieve(question)

        # Generate answer
        answer = self.generator.answer(question, passages)

        # Cache result
        self.cache[cache_key] = answer
        return answer

    def batch_answer(self, questions):
        """Process multiple questions efficiently."""
        # Batch encode questions
        embeddings = self.retriever.encode_batch(questions)

        # Batch retrieve
        all_passages = self.retriever.batch_retrieve(embeddings)

        # Batch generate (if model supports)
        answers = self.generator.batch_generate(questions, all_passages)

        return answers
```

### Streaming Responses

```python
class StreamingQA:
    """Stream answers token by token."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def stream_answer(self, question, context):
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate with streaming
        for output in self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            streamer=self._create_streamer()
        ):
            yield self.tokenizer.decode(output, skip_special_tokens=True)

    def _create_streamer(self):
        from transformers import TextIteratorStreamer
        return TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
```

## Datasets and Benchmarks

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| Natural Questions | Open-domain | 307K | Google search questions |
| TriviaQA | Open-domain | 650K | Trivia questions |
| MS MARCO | Reading comprehension | 1M | Bing queries |
| ELI5 | Long-form | 270K | Explain Like I'm 5 |
| HotpotQA | Multi-hop | 113K | Reasoning over multiple docs |
| ASQA | Ambiguous | 6.3K | Questions with multiple answers |

## Best Practices

### When to Use Generative QA

| Scenario | Recommendation |
|----------|----------------|
| Factoid lookup | Extractive (more accurate) |
| Explanation needed | Generative |
| Multiple sources | Generative with RAG |
| Conversational | Generative |
| High-stakes facts | Extractive or generative with verification |
| Creative answers | Generative |

### Quality Checklist

```python
def qa_quality_check(question, context, answer):
    """Pre-deployment quality checks."""
    issues = []

    # Length check
    if len(answer) < 10:
        issues.append("Answer too short")
    if len(answer) > 1000:
        issues.append("Answer too long")

    # Relevance check
    if not has_keyword_overlap(question, answer):
        issues.append("Answer may not address question")

    # Hallucination check
    if contains_specific_claims(answer):
        if not claims_supported_by_context(answer, context):
            issues.append("Potential hallucination detected")

    # Confidence check
    uncertain_phrases = ["I think", "maybe", "possibly", "I'm not sure"]
    if any(phrase in answer.lower() for phrase in uncertain_phrases):
        issues.append("Low confidence answer")

    return {
        'passed': len(issues) == 0,
        'issues': issues
    }
```
