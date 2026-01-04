# VLM Architectures

## Summary

Vision-Language Models (VLMs) combine visual perception with language understanding to enable multimodal reasoning. The core architectural challenge is integrating information from two very different domains: high-dimensional visual inputs and sequential text. Different approaches to this integration produce models with distinct strengths, from CLIP's efficient contrastive learning to Gemini's native multimodality.

Key points to remember:

- Three main architectural paradigms: contrastive (CLIP), projection-based (LLaVA), and cross-attention (Flamingo)
- Contrastive models align image and text embeddings in a shared space, enabling zero-shot classification
- Projection-based models map visual features directly into LLM token space with minimal architecture changes
- Cross-attention models insert visual attention layers into LLMs, enabling deeper integration
- Native multimodal models (Gemini) train on interleaved modalities from scratch
- Commercial APIs (GPT-4V, Gemini) offer highest capability but limited customization
- Open models (LLaVA, OpenFlamingo) enable fine-tuning and on-premise deployment
- Architecture choice depends on task requirements, compute budget, and customization needs

## Architectural Paradigms

### Contrastive Dual-Encoder (CLIP)

The contrastive approach trains separate encoders for images and text to produce aligned embeddings:

```
Image -> Image Encoder -> [embedding]
                              |
                         cosine similarity
                              |
Text  -> Text Encoder  -> [embedding]
```

**Characteristics**:
- Both encoders produce fixed-size embeddings
- Trained with contrastive loss to maximize matching pair similarity
- Enables zero-shot classification by comparing to text descriptions
- Cannot generate text, only classify

**Best for**: Zero-shot image classification, image-text retrieval, building blocks for larger VLMs

### Projection-Based (LLaVA)

Projection-based architectures map visual tokens directly into the LLM's input space:

```
Image -> Vision Encoder -> [visual tokens] -> Projection -> [LLM-compatible tokens]
                                                                    |
                                                                   LLM -> Text output
                                                                    |
Text  -> Tokenizer      -> [text tokens]  ---------------------------
```

**Characteristics**:
- Simple architecture: projection layer is often a single linear layer or small MLP
- Visual and text tokens concatenated and processed identically by LLM
- LLM architecture unchanged, only projection is trained (or LLM fine-tuned)
- Easy to implement and adapt to new LLMs

**Best for**: General visual question answering, instruction following, research and fine-tuning

### Cross-Attention (Flamingo)

Cross-attention architectures insert new attention layers into the LLM specifically for visual information:

```
Image -> Vision Encoder -> Perceiver Resampler -> [visual tokens]
                                                        |
                                                 cross-attention
                                                        |
Text  -> LLM layers -------------------------> gated cross-attn -> next LLM layer -> ...
```

**Characteristics**:
- Gated cross-attention allows controlled injection of visual information
- Perceiver compresses visual tokens to fixed count
- LLM weights typically frozen, only cross-attention trained
- More complex but enables sophisticated few-shot learning

**Best for**: Few-shot learning, multi-image reasoning, video understanding

### Native Multimodal (Gemini)

Native multimodal models process all modalities through a unified architecture from the start:

```
Image  --|
Video  --|--> Unified Tokenizer --> Multimodal Transformer --> Text output
Audio  --|
Text   --|
```

**Characteristics**:
- Trained on interleaved multimodal data from scratch
- No separate vision encoder that needs alignment
- Typically uses Mixture-of-Experts for efficiency
- Highest capacity but requires massive training resources

**Best for**: Complex multimodal reasoning, long context, video and audio understanding

## Architecture Comparison

### Integration Depth

How deeply visual and text information interact:

| Architecture | Integration Depth | Description |
|--------------|-------------------|-------------|
| CLIP | Minimal | Only compare embeddings, no generation |
| LLaVA | Token-level | Visual tokens participate in all LLM layers |
| Flamingo | Layer-level | Cross-attention at specific layers |
| Gemini | Full | Native multimodal from ground up |

Deeper integration generally enables more sophisticated reasoning but requires more training.

### Training Requirements

| Architecture | What's Trained | Training Data | Compute |
|--------------|----------------|---------------|---------|
| CLIP | Both encoders | Image-text pairs | Moderate |
| LLaVA | Projection + LLM | Instruction data | Moderate |
| Flamingo | Cross-attention only | Interleaved web data | High |
| Gemini | Everything | All modalities | Massive |

### Multi-Image Handling

| Architecture | Multi-Image Support | Mechanism |
|--------------|---------------------|-----------|
| CLIP | No | Single image encoding |
| LLaVA | Concatenation | Multiple visual token sequences |
| Flamingo | Native | Interleaved attention |
| Gemini | Native | Unified sequence |

Flamingo and Gemini handle multiple images most naturally due to their interleaved training.

### Few-Shot Capability

| Architecture | Few-Shot Learning | Mechanism |
|--------------|-------------------|-----------|
| CLIP | Limited | Fixed embedding comparison |
| LLaVA | Moderate | In-context examples |
| Flamingo | Strong | Trained on interleaved examples |
| Gemini | Strong | Native in-context learning |

Flamingo specifically excels at few-shot learning due to its training on web pages with multiple images.

## Open vs Commercial Models

### Open Models

**LLaVA Family**:
- Fully open weights and training code
- Multiple size variants (7B, 13B, 34B)
- Active community development
- Easy to fine-tune for specific domains

**OpenFlamingo / IDEFICS**:
- Open replication of Flamingo
- Few-shot learning capabilities
- Larger models available

**Qwen-VL, InternVL, Llama 3.2 Vision**:
- Production-quality open models
- Competitive with commercial offerings
- Multiple languages supported

### Commercial Models

**GPT-4V / GPT-4o**:
- Highest general capability
- API-only access
- Best for production applications
- Limited customization

**Gemini**:
- Native multimodal including video and audio
- Extremely long context (2M tokens)
- Google Cloud integration
- Multiple model sizes

**Claude (Anthropic)**:
- Strong document understanding
- 200K context
- Conservative safety approach

## Choosing an Architecture

### Decision Framework

```
Need zero-shot classification?
  Yes -> CLIP or SigLIP
  No  -> Continue

Need to fine-tune for specific domain?
  Yes -> LLaVA or open model
  No  -> Continue

Need few-shot learning with multiple examples?
  Yes -> Flamingo-style (IDEFICS, Otter)
  No  -> Continue

Need video or audio understanding?
  Yes -> Gemini
  No  -> Continue

Need highest capability, cost is acceptable?
  Yes -> GPT-4V or Gemini Pro
  No  -> LLaVA or Gemini Flash
```

### By Use Case

| Use Case | Recommended | Alternative |
|----------|-------------|-------------|
| Image classification | CLIP | SigLIP |
| Visual chat | LLaVA-1.5 | GPT-4V |
| Document extraction | GPT-4V | Gemini |
| Video understanding | Gemini | GPT-4V (frames) |
| Few-shot tasks | Flamingo/IDEFICS | Gemini |
| On-premise deployment | LLaVA | InternVL |
| Mobile/edge | Gemini Nano | MobileVLM |
| Research | LLaVA | Any open model |

### By Resource Constraints

| Constraint | Recommended Approach |
|------------|---------------------|
| Low GPU memory (<16GB) | Quantized LLaVA, small models |
| No GPU | CLIP on CPU, API models |
| No internet/API | Open models only |
| High volume | Gemini Flash, self-hosted |
| Low latency | Gemini Flash, optimized serving |

## Vision Encoder Choices

Most VLMs use a pretrained vision encoder:

| Encoder | Architecture | Training | Used By |
|---------|--------------|----------|---------|
| CLIP ViT | ViT-L/14 | Contrastive | LLaVA, many others |
| SigLIP | ViT | Sigmoid contrastive | PaliGemma, newer models |
| EVA-CLIP | ViT | Enhanced CLIP | InternVL |
| NFNet | ConvNet | Contrastive | Flamingo |
| DINOv2 | ViT | Self-supervised | Some research models |

SigLIP is increasingly preferred over CLIP for new models due to better scaling properties.

## Future Directions

### Emerging Trends

**Higher Resolution**: Models handling 4K+ images for detailed analysis.

**Video-Native**: More models with native video understanding like Gemini.

**Unified Architectures**: Single architectures for all modalities without adapters.

**Efficient Attention**: Linear attention and other methods for longer sequences.

**On-Device**: Smaller models for mobile and edge deployment.

### Open Challenges

**Spatial Reasoning**: Precise counting and positioning remains difficult.

**Hallucination**: All architectures can generate incorrect visual descriptions.

**Temporal Understanding**: Long video comprehension is still limited.

**Fine-Grained Recognition**: Distinguishing similar categories (bird species, car models).

## Further Reading

For detailed information on each architecture:
- [CLIP](clip/ReadMe.md): Contrastive learning for vision-language alignment
- [LLaVA](llava/ReadMe.md): Simple projection-based visual instruction following
- [Flamingo](flamingo/ReadMe.md): Cross-attention for few-shot multimodal learning
- [GPT-4V](gpt-4v/ReadMe.md): OpenAI's commercial multimodal model
- [Gemini](gemini/ReadMe.md): Google's native multimodal architecture
