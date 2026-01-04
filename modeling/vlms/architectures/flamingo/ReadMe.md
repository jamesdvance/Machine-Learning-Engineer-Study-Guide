# Flamingo

## Summary

Flamingo is a family of visual language models developed by DeepMind that can handle arbitrarily interleaved sequences of images, video, and text. Unlike simpler approaches that concatenate visual tokens with text, Flamingo uses cross-attention layers that allow the frozen language model to attend to visual features at each layer. This architecture enables strong few-shot learning on vision-language tasks, where providing just a few examples in the prompt dramatically improves performance.

Key points to remember:

- Uses cross-attention (Perceiver Resampler + gated cross-attention) to inject visual information into a frozen LLM
- Supports interleaved image-text inputs with multiple images in a single context
- Strong few-shot learner: providing examples in the prompt significantly improves accuracy
- Trained on interleaved web data (image-text from web pages) plus video-text pairs
- Frozen vision encoder (NFNet) and frozen LLM (Chinchilla), only cross-attention layers trained
- Served as inspiration for many subsequent architectures (IDEFICS, OpenFlamingo, Otter)
- 80B parameter flagship model achieved state-of-the-art few-shot performance in 2022

## Architecture

### High-Level Design

Flamingo consists of four main components:

1. **Vision Encoder**: Frozen NFNet-F6 pretrained with contrastive learning
2. **Perceiver Resampler**: Compresses variable-length visual features to fixed-size tokens
3. **Gated Cross-Attention Layers**: Inserted into frozen LLM to attend to visual features
4. **Language Model**: Frozen Chinchilla LLM that generates text outputs

The key innovation is keeping the vision encoder and LLM frozen while training only the bridging components, preserving the pretrained capabilities of both.

### Vision Encoder

Flamingo uses an NFNet-F6 vision encoder pretrained with contrastive learning (similar to CLIP):

- Processes images at 320x320 resolution
- Produces spatial feature maps, not just global embeddings
- Frozen during Flamingo training to preserve visual understanding
- For video: samples frames and processes each independently

```
Image (320x320)
    |
NFNet-F6 (frozen)
    |
Spatial features (10x10 = 100 tokens, 1536-dim)
```

### Perceiver Resampler

The Perceiver Resampler compresses visual features to a fixed number of tokens regardless of input resolution or number of frames:

**Purpose**: Handle variable-length visual inputs efficiently
**Architecture**: Cross-attention between learned queries and visual features
**Output**: Fixed 64 visual tokens per image/video

```python
# Simplified Perceiver Resampler
class PerceiverResampler(nn.Module):
    def __init__(self, num_queries=64, dim=1536, num_layers=6):
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, visual_features):
        # visual_features: (batch, num_visual_tokens, dim)
        x = self.queries.expand(batch_size, -1, -1)
        for layer in self.layers:
            x = layer(x, visual_features)  # queries attend to visual features
        return x  # (batch, 64, dim)
```

This compression is crucial for efficiency: without it, high-resolution images or long videos would produce thousands of tokens.

### Gated Cross-Attention

Gated cross-attention layers are inserted into the frozen LLM to allow text tokens to attend to visual features:

**Placement**: Inserted between every N transformer blocks (typically N=4)
**Gating**: A learned scalar (initialized to 0) controls the contribution of visual information
**Initialization**: Zero-initialization ensures the model starts by ignoring visual input

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)
        self.gate = nn.Parameter(torch.zeros(1))  # Learned gate, initialized to 0

    def forward(self, text_features, visual_features):
        # text_features: (seq_len, batch, dim)
        # visual_features: (num_visual_tokens, batch, dim)
        attn_out, _ = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features
        )
        return text_features + torch.tanh(self.gate) * attn_out
```

The tanh gating starts near zero, allowing gradual integration of visual information during training without disrupting the pretrained LLM.

### Handling Interleaved Inputs

Flamingo natively handles sequences with multiple images interspersed with text:

```
Input sequence:
[Text: "This is a dog:"] [Image 1] [Text: "This is a cat:"] [Image 2] [Text: "What is this?"] [Image 3]

Visual tokens:
Image 1 -> 64 Perceiver tokens
Image 2 -> 64 Perceiver tokens
Image 3 -> 64 Perceiver tokens

Cross-attention masking:
Text after Image 1 can attend to Image 1's tokens
Text after Image 2 can attend to Image 1 and Image 2's tokens
Text after Image 3 can attend to all image tokens
```

Each text position can only attend to images that appeared before it in the sequence, maintaining causality.

## Training

### Training Data

Flamingo was trained on three data sources:

**M3W (MultiModal MassiveWeb)**: 43 million web pages with interleaved images and text
- Scraped HTML with images and surrounding text
- Teaches the model to understand images in context

**ALIGN**: Image-text pairs (similar to CLIP training data)
- 1.8 billion image-alt-text pairs
- Provides image-text alignment

**Video-Text Pairs**: 27 million video-text pairs from YouTube
- Short video clips with associated text
- Enables video understanding

### Training Strategy

Only the Perceiver Resampler and gated cross-attention layers are trained:

```
Frozen components (pretrained):
- NFNet-F6 vision encoder
- Chinchilla LLM (all layers)

Trained components (from scratch or initialized):
- Perceiver Resampler
- Gated cross-attention layers
- Visual token embeddings
```

This approach:
- Preserves pretrained capabilities
- Dramatically reduces training compute
- Enables scaling to large LLMs without full training cost

### Training Objective

Standard next-token prediction on interleaved image-text sequences:

```
Loss = -log P(next_text_token | previous_tokens, images)
```

The model learns to generate text conditioned on both preceding text and all preceding images.

## Few-Shot Learning

### In-Context Learning with Images

Flamingo's standout capability is few-shot learning. By providing examples in the prompt, performance improves dramatically:

```
Prompt format:
Image1: [image of cat] Label: cat
Image2: [image of dog] Label: dog
Image3: [image of bird] Label: bird
Image4: [test image] Label:

Model generates: cat/dog/bird based on test image
```

### Few-Shot Performance Gains

On VQAv2 benchmark:

| Setting | Accuracy |
|---------|----------|
| 0-shot | 56.3% |
| 4-shot | 63.1% |
| 8-shot | 65.4% |
| 16-shot | 67.6% |
| 32-shot | 69.0% |

Each doubling of examples provides meaningful improvement.

### Why Few-Shot Works

Flamingo's few-shot capability stems from:

1. **Interleaved training**: Trained on web pages with multiple images teaches in-context image understanding
2. **Frozen LLM**: Preserves the LLM's in-context learning abilities
3. **Large scale**: 80B parameters provide capacity for complex pattern matching
4. **Cross-attention**: Allows precise routing of information from examples to the query

## Model Variants

### Flamingo Family

| Model | LLM | Vision | Parameters | Few-shot VQA |
|-------|-----|--------|------------|--------------|
| Flamingo-3B | Chinchilla-3B | NFNet | 3.2B | 49.2% |
| Flamingo-9B | Chinchilla-7B | NFNet | 9.3B | 56.3% |
| Flamingo-80B | Chinchilla-70B | NFNet | 80B | 67.6% |

### Open-Source Alternatives

**OpenFlamingo**: Open replication by LAION
- Uses CLIP vision encoder instead of NFNet
- Uses open LLMs (MPT, LLaMA)
- Trained on open datasets

**IDEFICS**: Hugging Face implementation
- 9B and 80B variants
- Closer to original Flamingo
- Trained on OBELICS dataset

**Otter**: Instruction-tuned OpenFlamingo
- Better instruction-following
- More suitable for chat applications

## Comparison with Other Architectures

### Flamingo vs LLaVA

| Aspect | Flamingo | LLaVA |
|--------|----------|-------|
| Visual injection | Cross-attention | Token concatenation |
| Multi-image | Native support | Concatenated tokens |
| LLM modification | Inserted layers | None (projection only) |
| Training | Vision + LLM frozen | LLM fine-tuned |
| Few-shot | Excellent | Limited |
| Complexity | Higher | Lower |

Flamingo's architecture is more complex but enables stronger few-shot learning and cleaner multi-image handling.

### Flamingo vs BLIP-2

| Aspect | Flamingo | BLIP-2 |
|--------|----------|--------|
| Visual compression | Perceiver (64 tokens) | Q-Former (32 tokens) |
| LLM integration | Cross-attention layers | Prepended tokens |
| Training stages | 1 stage | 3 stages |
| Multi-image | Native | Limited |
| Video support | Native | Limited |

Both use visual compression, but Flamingo's cross-attention provides deeper integration.

## Practical Considerations

### Using OpenFlamingo

```python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="mosaicml/mpt-7b",
    tokenizer_path="mosaicml/mpt-7b",
    cross_attn_every_n_layers=4
)

# Load pretrained weights
model.load_state_dict(torch.load("openflamingo_checkpoint.pt"))

# Prepare inputs
images = [image_processor(img).unsqueeze(0) for img in image_list]
text = tokenizer("What is in this image? <image>", return_tensors="pt")

# Generate
output = model.generate(
    vision_x=torch.cat(images),
    lang_x=text.input_ids,
    max_new_tokens=50
)
```

### Using IDEFICS

```python
from transformers import IdeficsForVisionText2Text, AutoProcessor

model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

# Multi-image input
prompts = [
    "User: What is in these images?",
    image1,
    image2,
    "Assistant:"
]

inputs = processor(prompts, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
```

### Few-Shot Prompting

For best few-shot performance:

```
# Good few-shot format
prompts = [
    image1, "This image shows: a red car on a highway\n",
    image2, "This image shows: a blue bicycle in a park\n",
    image3, "This image shows: a yellow bus at a stop\n",
    test_image, "This image shows:"
]
```

Key principles:
- Use consistent formatting across examples
- Choose examples representative of the test distribution
- More examples generally improve performance (up to context limit)
- Examples should be diverse but relevant

## Limitations

### Computational Cost

- Large models (80B) are expensive to run
- Cross-attention adds overhead compared to simpler architectures
- Video processing multiplies visual tokens

### Context Length

- Each image consumes 64 tokens (after Perceiver)
- Multi-image scenarios quickly fill context
- Trade-off between image count and text length

### Hallucination

Like other VLMs, Flamingo can:
- Describe objects not present
- Miscount or mislocate objects
- Generate plausible but incorrect details

### Availability

- Original Flamingo weights not publicly released
- Open alternatives (OpenFlamingo, IDEFICS) approximate but don't match original
- Training requires significant compute resources

## Best Practices

### Prompt Design

For zero-shot:
```
Image: <image>
Question: What is the primary object in this image?
Answer:
```

For few-shot:
```
Image: <image1>
Question: What color is the car?
Answer: The car is red.

Image: <image2>
Question: What color is the car?
Answer: The car is blue.

Image: <test_image>
Question: What color is the car?
Answer:
```

### Example Selection

For few-shot learning:
- Choose examples similar to test cases
- Include diverse examples covering expected variations
- Ensure examples have clear, correct labels
- Avoid ambiguous or borderline cases

### Model Selection

| Use Case | Recommended |
|----------|-------------|
| Few-shot tasks | Flamingo-80B or IDEFICS-80B |
| Resource-constrained | OpenFlamingo-9B or IDEFICS-9B |
| Instruction-following | Otter |
| Research/fine-tuning | OpenFlamingo |
