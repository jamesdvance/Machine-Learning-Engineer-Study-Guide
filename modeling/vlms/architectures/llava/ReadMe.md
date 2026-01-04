# LLaVA (Large Language and Vision Assistant)

## Summary

LLaVA is a multimodal model that connects a pretrained vision encoder (CLIP) to a large language model (Vicuna/LLaMA) through a simple projection layer. This architecture enables visual instruction following, where users can ask questions about images and receive detailed, conversational responses. LLaVA demonstrated that relatively simple vision-language integration combined with instruction tuning could achieve competitive performance with much more complex architectures.

Key points to remember:

- Connects CLIP ViT-L/14 vision encoder to LLaMA-based LLM via a linear or MLP projection
- Uses two-stage training: first alignment pretraining, then visual instruction tuning
- Instruction-tuned on GPT-4 generated visual conversations
- Simple architecture makes it accessible for academic research and fine-tuning
- LLaVA-1.5 added MLP projection and higher resolution for significant improvements
- Open source with weights, data, and training code available
- Foundation for many subsequent open VLM efforts

## Architecture

### Overall Design

LLaVA consists of three components:

1. **Vision Encoder**: CLIP ViT-L/14 that processes images into visual tokens
2. **Projection Layer**: Transforms visual tokens to the LLM embedding space
3. **Language Model**: Vicuna or LLaMA that generates text responses

The vision encoder and LLM are pretrained; only the projection layer is trained from scratch.

### Vision Encoder

LLaVA uses the CLIP ViT-L/14 vision encoder:

- Input: 224x224 image (LLaVA-1.0) or 336x336 (LLaVA-1.5)
- Output: Grid of 16x16 (256) or 24x24 (576) patch embeddings
- Dimension: 1024-dimensional vectors per patch

The vision encoder is frozen during pretraining and optionally fine-tuned during instruction tuning.

```
Image (336x336)
    |
CLIP ViT-L/14
    |
576 patch embeddings (1024-dim each)
```

### Projection Layer

The projection layer maps CLIP embeddings to the LLM's embedding space:

**LLaVA-1.0**: Single linear layer
```python
visual_projection = nn.Linear(clip_dim, llm_dim)
# 1024 -> 4096 for Vicuna-7B
```

**LLaVA-1.5**: Two-layer MLP with GELU activation
```python
visual_projection = nn.Sequential(
    nn.Linear(clip_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim)
)
```

The MLP projection in LLaVA-1.5 provides better alignment and is now standard practice.

### Language Model Integration

Visual tokens are prepended to text tokens in the LLM's input sequence:

```
[BOS] [IMG_1] [IMG_2] ... [IMG_576] User: What is in this image? Assistant:
```

The LLM processes visual and text tokens identically through its transformer layers, with the visual tokens providing context for generation.

### Multi-Modal Attention

LLaVA uses standard causal attention across all tokens:

```
Token positions:
[v1, v2, ..., v576, t1, t2, t3, ...]
  visual tokens      text tokens

Attention:
- Visual tokens attend to preceding visual tokens
- Text tokens attend to all preceding tokens (visual + text)
- Causal masking prevents attending to future tokens
```

This simple approach contrasts with architectures like Flamingo that use specialized cross-attention.

## Training

### Stage 1: Vision-Language Alignment

The first stage aligns the vision encoder's output with the LLM's embedding space:

**Data**: 595K image-text pairs from CC3M (filtered)
**Trainable**: Only the projection layer
**Frozen**: Vision encoder and LLM
**Objective**: Next token prediction on captions

The pretraining task is straightforward captioning:
```
Image + "Describe this image." -> Generate caption
```

This stage teaches the projection layer to produce visual tokens that the LLM can interpret as meaningful language.

### Stage 2: Visual Instruction Tuning

The second stage teaches the model to follow diverse visual instructions:

**Data**: 158K visual instruction-following examples (LLaVA-Instruct)
**Trainable**: Projection layer + LLM (optionally vision encoder)
**Frozen**: Vision encoder (LLaVA-1.0) or unfrozen (LLaVA-1.5)
**Objective**: Next token prediction on instruction-response pairs

### Instruction Data Generation

The instruction-tuning data was generated using GPT-4:

1. Provide GPT-4 with image captions and bounding boxes
2. Ask GPT-4 to generate conversations, detailed descriptions, and reasoning
3. Filter and curate the resulting dialogues

Example instruction types:
- Detailed description: "Describe this image in detail"
- Conversation: Multi-turn Q&A about image content
- Complex reasoning: "What might happen next in this scene?"

This approach creates diverse, high-quality training data without human annotation.

## LLaVA Versions

### LLaVA-1.0

Original version with basic architecture:
- CLIP ViT-L/14 @ 224x224
- Linear projection
- Vicuna-7B/13B
- 595K pretraining + 158K instruction tuning

### LLaVA-1.5

Improved version with better performance:
- CLIP ViT-L/14 @ 336x336 (higher resolution)
- MLP projection (2-layer)
- Vicuna-7B/13B or LLaMA-2
- Additional academic VQA data in training
- Optional vision encoder fine-tuning

Key improvements:
- Higher resolution captures more visual detail
- MLP projection provides better alignment
- Academic data improves benchmark performance

### LLaVA-NeXT (LLaVA-1.6)

Latest version with dynamic resolution:
- Supports images up to 672x672 or variable aspect ratios
- Splits large images into tiles processed separately
- Stronger LLM backends (Vicuna, Mistral, Yi)
- Improved training data and longer training

Dynamic resolution handling:
```
Large image -> Split into 4 tiles + downscaled overview
Each tile -> Processed by CLIP independently
All tokens -> Concatenated as input to LLM
```

## Performance

### Benchmarks

LLaVA-1.5-13B performance on common VLM benchmarks:

| Benchmark | Score | Notes |
|-----------|-------|-------|
| VQAv2 | 80.0 | Visual question answering |
| GQA | 63.3 | Compositional reasoning |
| TextVQA | 61.3 | Reading text in images |
| POPE | 85.9 | Object hallucination |
| MME | 1531 | Comprehensive evaluation |
| MMBench | 67.7 | Multi-modal understanding |

### Comparison with Other Models

| Model | Parameters | VQAv2 | Architecture Complexity |
|-------|------------|-------|------------------------|
| LLaVA-1.5-7B | 7B | 78.5 | Simple (projection only) |
| LLaVA-1.5-13B | 13B | 80.0 | Simple |
| Flamingo-9B | 9B | 56.3 | Complex (cross-attention) |
| BLIP-2 | 12B | 65.0 | Q-Former bridging |
| InstructBLIP | 13B | 82.4 | Q-Former + instruction |

LLaVA achieves competitive results with a much simpler architecture.

## Practical Usage

### Using LLaVA with Transformers

```python
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load model
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf"
)
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Prepare inputs
image = Image.open("photo.jpg")
prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate response
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
```

### Chat Format

LLaVA uses a specific conversation format:

```
USER: <image>
What is the main subject of this image?
ASSISTANT: The main subject is a golden retriever sitting in a park...
USER: What is the dog doing?
ASSISTANT: The dog appears to be...
```

The `<image>` token indicates where visual tokens are inserted.

### Inference Optimization

For efficient inference:

```python
# Load in 4-bit quantization
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# Or use vLLM for faster serving
from vllm import LLM
llm = LLM(model="llava-hf/llava-1.5-7b-hf")
```

## Fine-Tuning

### LoRA Fine-Tuning

LLaVA can be efficiently fine-tuned with LoRA:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Domain-Specific Fine-Tuning

For domain-specific applications:

1. Collect domain-relevant image-text pairs
2. Generate instruction-following data for your domain
3. Fine-tune projection layer + LoRA on LLM
4. Optionally unfreeze vision encoder for specialized visual features

Common domains:
- Medical imaging (X-rays, pathology)
- Document understanding
- Scientific figures
- Satellite imagery

## Comparison with Other Architectures

### LLaVA vs Flamingo

| Aspect | LLaVA | Flamingo |
|--------|-------|----------|
| Vision-LLM connection | Projection layer | Cross-attention |
| Complexity | Simple | Complex |
| Training cost | Lower | Higher |
| Multi-image handling | Concatenated tokens | Native support |
| Performance | Competitive | Similar |

LLaVA demonstrates that sophisticated cross-attention may not be necessary.

### LLaVA vs BLIP-2

| Aspect | LLaVA | BLIP-2 |
|--------|-------|--------|
| Bridge mechanism | Linear/MLP projection | Q-Former (querying transformer) |
| Visual tokens | 576 per image | 32 per image |
| Training efficiency | 2 stages | 3 stages |
| Compute cost | Lower | Higher |

BLIP-2's Q-Former compresses visual information more aggressively, while LLaVA preserves all patch information.

## Limitations

### Visual Understanding

- Resolution limited by CLIP encoder (mitigated in LLaVA-NeXT)
- Struggles with small objects and fine details
- Limited spatial reasoning capabilities
- Object counting is unreliable

### Hallucination

LLaVA can hallucinate visual content:
- Describes objects not present in the image
- Invents text when asked to read
- Confabulates details in complex scenes

Mitigation strategies:
- Fine-tune on curated, accurate data
- Use reinforcement learning from human feedback
- Implement output verification

### Multi-Image Reasoning

- Not designed for comparing multiple images
- Long sequence lengths when using many images
- No explicit support for video (though frames can be provided)

## Best Practices

### Prompt Design

Effective prompting for LLaVA:

```
# Simple question
USER: <image>
What animal is in this image?

# Detailed analysis
USER: <image>
Describe this image in detail, including all visible objects and their relationships.

# Specific focus
USER: <image>
Focus on the text visible in this image. What does it say?
```

### Image Preprocessing

- Resize to model's expected resolution (336x336 for LLaVA-1.5)
- Use the processor's built-in preprocessing
- For multiple images, process and concatenate appropriately

### Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| General chat | LLaVA-1.5-7B |
| Higher quality | LLaVA-1.5-13B |
| Speed-constrained | LLaVA-1.5-7B (quantized) |
| High resolution needs | LLaVA-NeXT |
| Research/fine-tuning | LLaVA-1.5-7B |

### Production Deployment

For production use:
1. Quantize to reduce memory (4-bit or 8-bit)
2. Use vLLM or TGI for efficient batched inference
3. Implement input validation for image formats
4. Set appropriate maximum token limits
5. Monitor for hallucinations in critical applications
