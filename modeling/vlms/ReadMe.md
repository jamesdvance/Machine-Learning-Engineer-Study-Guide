# Vision-Language Models (VLMs)

## Summary

Vision-Language Models combine visual perception with language understanding to process and reason about images, documents, and videos. These models bridge two modalities that humans naturally integrate: seeing and communicating. VLMs have evolved from task-specific architectures (image classifiers, caption generators) to general-purpose multimodal systems capable of following arbitrary instructions about visual content.

Key points to remember:

- VLMs process both images and text, generating text outputs
- Core challenge is aligning visual representations with language model understanding
- Architectural approaches range from simple projection (LLaVA) to complex cross-attention (Flamingo)
- Commercial VLMs (GPT-4V, Gemini) offer highest capability; open models enable customization
- Common tasks: captioning, VQA, document understanding, OCR
- Key limitations: spatial reasoning, counting, hallucination
- Model selection depends on task complexity, volume, and deployment constraints
- Rapid progress continues with new models and capabilities appearing regularly

## The Vision-Language Problem

### Why It's Hard

Connecting vision and language requires solving several subproblems:

**Representation Alignment**: Visual features (from CNNs or ViTs) exist in a different space than language embeddings. Models must learn to translate between these representations.

**Grounding**: Connecting words to visual regions. When a model says "the red car," it should refer to an actual red car in the image.

**Compositional Understanding**: Interpreting novel combinations of visual concepts. Understanding "a cat riding a skateboard" even if never trained on that exact combination.

**Reasoning**: Drawing conclusions that go beyond direct observation. Inferring the season from visual cues or predicting what might happen next.

### Evolution of Solutions

**Early Approaches** (2014-2018):
- CNN image encoder + LSTM caption decoder
- Separate encoders fused with attention
- Task-specific architectures

**Vision-Language Pretraining** (2019-2022):
- CLIP: Contrastive alignment of image and text embeddings
- ViLBERT, UNITER: Joint transformers for vision and language
- BLIP: Unified encoder-decoder for multiple tasks

**Large VLMs** (2022-present):
- LLaVA: Simple projection from CLIP to LLM
- Flamingo: Cross-attention for few-shot learning
- GPT-4V, Gemini: Native multimodal training at scale

## Architecture Overview

### The Vision Encoder

Converts images to embeddings:

**CLIP ViT**: Most common choice for open models
- Pretrained on 400M image-text pairs
- Produces semantically meaningful features
- Multiple sizes (B/32, B/16, L/14)

**SigLIP**: Improved CLIP training
- Sigmoid loss instead of softmax
- Better scaling properties
- Increasingly preferred in new models

**Native Vision**: Some models (Gemini) use custom vision encoders
- Trained jointly with language model
- Potentially better integration
- Not available for open use

### The Bridge

Connects vision encoder to language model:

**Projection Layer** (LLaVA):
```
Visual features -> Linear/MLP -> LLM token space
```
Simple but effective. Visual tokens processed like text tokens.

**Perceiver Resampler** (Flamingo):
```
Variable visual features -> Fixed visual tokens
```
Compresses visual information. Enables variable resolution/length inputs.

**Cross-Attention** (Flamingo):
```
LLM layers receive visual information via cross-attention
```
Deeper integration. Better for few-shot learning.

**Q-Former** (BLIP-2):
```
Learned queries extract information from visual features
```
Aggressive compression. Fewer visual tokens.

### The Language Model

Generates text output:

**Frozen LLM**: Keep pretrained weights
- Preserves language capabilities
- Faster training
- Common in research models

**Fine-tuned LLM**: Update during multimodal training
- Better integration
- Risk of degrading text capabilities
- More expensive

**Native Multimodal**: Trained on all modalities from start
- Best theoretical integration
- Requires massive resources
- GPT-4V, Gemini approach

## Model Landscape

### Commercial APIs

| Model | Provider | Strengths | Access |
|-------|----------|-----------|--------|
| GPT-4V/4o | OpenAI | General capability | API |
| Gemini | Google | Long context, video, audio | API |
| Claude | Anthropic | Document understanding | API |

### Open Models

| Model | Size | Architecture | Best For |
|-------|------|--------------|----------|
| LLaVA-1.5 | 7B/13B | Projection | General use, fine-tuning |
| LLaVA-NeXT | 7B-34B | Dynamic resolution | High-res images |
| IDEFICS-2 | 8B/80B | Flamingo-style | Few-shot learning |
| Qwen-VL | 7B | Full fine-tune | Multilingual |
| InternVL | 6B-26B | Strong vision | Visual tasks |

### Specialized Models

| Model | Specialty |
|-------|-----------|
| Donut | Document parsing (no OCR) |
| LayoutLM | Form understanding |
| PaLI | Multilingual vision |
| Pix2Struct | Charts and diagrams |

## Capabilities and Limitations

### What VLMs Do Well

**General Visual Understanding**:
- Scene description
- Object identification
- Activity recognition
- Attribute detection

**Text in Images**:
- OCR (with varying accuracy)
- Document comprehension
- Sign and label reading
- Handwriting (basic)

**Visual Reasoning**:
- Answering questions about content
- Making inferences from visual evidence
- Comparing images
- Following visual instructions

### Known Limitations

**Counting**: VLMs consistently struggle with counting objects, especially in crowded scenes.

**Spatial Reasoning**: Understanding precise spatial relationships ("left of," "behind") is unreliable.

**Fine-Grained Recognition**: Distinguishing visually similar categories (bird species, car models) remains challenging.

**Hallucination**: All VLMs can confidently describe objects or text that aren't present.

**Small Details**: High-resolution images may have details missed due to encoding resolution.

**Multi-Image Consistency**: Answers about the same image may vary between queries.

## Practical Considerations

### Model Selection Framework

```
What's the task complexity?
  Low (OCR, classification) -> Specialized model
  High (reasoning, QA) -> General VLM

What's the volume?
  High -> Self-hosted or efficient API (Flash)
  Low -> Best available (GPT-4V, Gemini Pro)

Need customization?
  Yes -> Open model (LLaVA, InternVL)
  No -> Commercial API

Privacy concerns?
  High -> Self-hosted
  Low -> API acceptable
```

### Cost Considerations

| Approach | Cost Profile |
|----------|--------------|
| GPT-4V API | ~$0.01-0.03 per image |
| Gemini Flash | ~$0.001 per image |
| Self-hosted 7B | GPU cost amortized |
| Self-hosted 70B | Significant GPU investment |

### Deployment Patterns

**API-Based**:
- Simplest integration
- Per-request costs
- Latency depends on provider
- Limited customization

**Self-Hosted**:
- Fixed infrastructure costs
- Full control over model
- Can fine-tune
- Requires ML ops expertise

**Hybrid**:
- Simple tasks: self-hosted small model
- Complex tasks: API to large model
- Balances cost and capability

## Evaluation

### Benchmarks

| Benchmark | What It Tests |
|-----------|---------------|
| VQAv2 | Visual question answering |
| GQA | Compositional reasoning |
| TextVQA | Reading text in images |
| DocVQA | Document understanding |
| MMMU | Multi-discipline multimodal |
| ChartQA | Chart and graph understanding |
| POPE | Object hallucination |

### Evaluation Practices

**Quantitative**: Run benchmarks for comparison, but benchmarks may not reflect your use case.

**Qualitative**: Test on representative examples from your domain.

**Error Analysis**: Categorize failure modes. Are errors random or systematic?

**Consistency**: Same question should give consistent answers.

**Adversarial**: Test with tricky cases, edge cases, potential misuse.

## Future Directions

### Near-Term Trends

- Higher resolution without proportional cost
- Better video understanding
- Improved spatial reasoning
- Reduced hallucination
- Faster, cheaper inference

### Emerging Capabilities

- Real-time video processing
- Multi-image reasoning at scale
- Embodied AI (robots with VLM brains)
- Multimodal agents
- Scientific image understanding

### Open Challenges

- Reliable counting and measurement
- Consistent multi-image reasoning
- Verifiable and grounded outputs
- Efficient long video processing
- Domain-specific adaptation at scale

## Further Reading

### Architectures
Detailed exploration of how VLMs are built:
- [CLIP](architectures/clip/ReadMe.md): Foundation contrastive model
- [LLaVA](architectures/llava/ReadMe.md): Simple but effective projection
- [Flamingo](architectures/flamingo/ReadMe.md): Cross-attention for few-shot
- [GPT-4V](architectures/gpt-4v/ReadMe.md): OpenAI's commercial VLM
- [Gemini](architectures/gemini/ReadMe.md): Google's native multimodal

### Tasks
Common applications of VLMs:
- [Image Captioning](tasks/image-captioning/ReadMe.md): Generating descriptions
- [Visual Question Answering](tasks/visual-question-answering/ReadMe.md): Answering questions
- [Document Understanding](tasks/document-understanding/ReadMe.md): Extracting structured data
- [OCR](tasks/ocr/ReadMe.md): Reading text in images
