# CLIP (Contrastive Language-Image Pre-training)

## Summary

CLIP is a foundational vision-language model developed by OpenAI that learns visual concepts from natural language supervision. Rather than training on fixed image classification categories, CLIP learns to associate images with their text descriptions using contrastive learning on 400 million image-text pairs scraped from the internet. This approach enables zero-shot transfer to virtually any visual classification task by describing the categories in natural language.

Key points to remember:

- Dual-encoder architecture with separate image and text encoders projecting to a shared embedding space
- Trained with contrastive learning to maximize similarity between matching image-text pairs
- Zero-shot classification works by comparing image embeddings to text embeddings of class descriptions
- Strong robustness to distribution shift compared to supervised ImageNet models
- Foundation for many VLMs including LLaVA, which uses CLIP's vision encoder
- Limited to classification-style tasks; cannot generate text or handle complex reasoning
- Available in multiple sizes (ViT-B/32, ViT-B/16, ViT-L/14) trading off speed and accuracy

## Architecture

### Dual Encoder Design

CLIP consists of two separate encoder networks that project inputs to a shared embedding space:

**Image Encoder**: Either a Vision Transformer (ViT) or a modified ResNet. The image is processed independently and projected to an embedding vector. Common variants:
- ViT-B/32: Base ViT with 32x32 patches, fastest
- ViT-B/16: Base ViT with 16x16 patches, better accuracy
- ViT-L/14: Large ViT with 14x14 patches, highest accuracy

**Text Encoder**: A Transformer encoder (similar to GPT-2 architecture) that processes the text description and produces an embedding vector. The model uses byte-pair encoding (BPE) tokenization with a vocabulary of 49,152 tokens.

Both encoders output vectors of the same dimensionality (512 or 768 depending on model size), enabling direct comparison via cosine similarity.

### Projection to Shared Space

The final representations from each encoder are linearly projected to a shared embedding space:

```
image_embedding = image_encoder(image)
text_embedding = text_encoder(text)

image_features = linear_projection_image(image_embedding)
text_features = linear_projection_text(text_embedding)
```

The linear projections ensure that semantically related images and text descriptions cluster together in the shared space.

## Training

### Contrastive Learning Objective

CLIP is trained to correctly match images with their corresponding text descriptions within a batch. Given a batch of N image-text pairs:

1. Compute embeddings for all N images and N texts
2. Calculate N x N similarity matrix of all pairwise cosine similarities
3. The diagonal elements (matching pairs) should have high similarity
4. Off-diagonal elements (non-matching pairs) should have low similarity

The loss function is symmetric cross-entropy:
- For each image, classify which text is correct (N-way classification)
- For each text, classify which image is correct (N-way classification)
- Average both losses

### Training Scale

The original CLIP training used:
- 400 million image-text pairs from internet scraping
- 32,768 batch size (large batches improve contrastive learning)
- 32 epochs, equivalent to seeing 12.8 billion image-text pairs
- Trained on 256 V100 GPUs for several weeks

The massive scale of training data is crucial to CLIP's generalization ability. The diversity of internet-sourced descriptions teaches the model rich visual concepts.

### Data Collection

CLIP's training data was constructed by querying for images with associated alt-text or captions. The collection process:

1. Query image search engines for a variety of concepts
2. Filter for images with natural language descriptions
3. Balance the dataset across visual concepts
4. Clean and deduplicate

This differs from curated datasets like ImageNet, which have fixed category labels and controlled collection.

## Zero-Shot Classification

### How It Works

CLIP performs zero-shot image classification by comparing image embeddings to text embeddings of class descriptions:

```python
# Create text prompts for each class
class_prompts = [f"a photo of a {c}" for c in classes]

# Encode image and all class prompts
image_features = clip.encode_image(image)
text_features = clip.encode_text(class_prompts)

# Compute similarities
similarities = image_features @ text_features.T

# Predict class with highest similarity
predicted_class = similarities.argmax()
```

### Prompt Engineering

The phrasing of class descriptions significantly affects accuracy. Instead of just the class name:

**Basic**: "dog", "cat", "car"

**Contextual**: "a photo of a dog", "a photo of a cat", "a photo of a car"

**Domain-specific**: "a satellite image of a forest", "a photo of food: pizza"

OpenAI released prompt templates that improve accuracy for various domains:
- ImageNet: "a photo of a {class}, a type of pet/vehicle/food"
- CIFAR-10: "a photo of a {class}"
- Satellite imagery: "a centered satellite photo of {class}"

### Prompt Ensembling

Using multiple prompts per class and averaging their embeddings improves robustness:

```python
prompts = [
    f"a photo of a {c}",
    f"a blurry photo of a {c}",
    f"a cropped photo of a {c}",
    f"a good photo of a {c}",
]

# Average embeddings across prompts for each class
class_embedding = mean([encode(p) for p in prompts])
```

This handles variation in how concepts might be described.

## Performance Characteristics

### Zero-Shot vs Supervised

CLIP's zero-shot performance on ImageNet is competitive with supervised models:

| Model | ImageNet Accuracy | Training Data |
|-------|------------------|---------------|
| ResNet-50 (supervised) | 76.1% | 1.2M labeled images |
| CLIP ViT-B/16 (zero-shot) | 68.3% | 400M image-text pairs |
| CLIP ViT-L/14 (zero-shot) | 75.3% | 400M image-text pairs |

While slightly lower than fully supervised models on ImageNet specifically, CLIP's advantage emerges in generalization.

### Distribution Shift Robustness

CLIP significantly outperforms supervised models when test distributions differ from training:

| Dataset | ResNet-50 | CLIP ViT-L/14 |
|---------|-----------|---------------|
| ImageNet | 76.1% | 75.3% |
| ImageNet-V2 | 63.3% | 70.1% |
| ImageNet-R | 17.6% | 77.2% |
| ImageNet-Sketch | 24.1% | 59.6% |

ImageNet-R contains artistic renditions; ImageNet-Sketch contains sketches. CLIP's language supervision teaches it concepts rather than dataset-specific visual patterns.

### Limitations

CLIP struggles with:
- Fine-grained classification (bird species, car models)
- Abstract or systematic tasks (counting objects)
- Novel compositions of familiar concepts
- Tasks requiring spatial reasoning

These limitations reflect the nature of web-scraped training data, which emphasizes common visual concepts over precise attributes.

## Practical Usage

### Loading and Using CLIP

With OpenAI's official implementation:

```python
import clip
import torch
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare image and text
image = preprocess(Image.open("photo.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a dog", "a cat", "a car"]).to(device)

# Get features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Calculate similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)
```

### Using OpenCLIP

OpenCLIP provides additional model variants and training recipes:

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='laion2b_s32b_b82k'
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
text = tokenizer(["a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
```

OpenCLIP includes models trained on LAION datasets, often outperforming the original CLIP.

### Image Retrieval

CLIP enables text-to-image and image-to-image retrieval:

```python
# Build index of image embeddings
image_embeddings = []
for image_path in image_paths:
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    embedding = model.encode_image(image)
    image_embeddings.append(embedding)
image_embeddings = torch.cat(image_embeddings)

# Query with text
query = clip.tokenize(["a sunset over the ocean"])
query_embedding = model.encode_text(query)

# Find most similar images
similarities = query_embedding @ image_embeddings.T
top_k_indices = similarities.topk(k=10).indices
```

## Role in VLM Architectures

### As Vision Encoder

Many VLMs use CLIP's vision encoder as a pretrained component:

**LLaVA**: Uses CLIP ViT-L/14 vision encoder, projects to LLM embedding space via an MLP.

**Flamingo**: Uses frozen CLIP vision encoder with cross-attention layers to inject visual information.

**OpenFlamingo**: Open-source Flamingo using CLIP as the visual backbone.

The intuition is that CLIP's vision encoder already produces semantically meaningful representations aligned with language, making it easier for an LLM to understand.

### Embedding Space Alignment

CLIP's key contribution to VLMs is that its image embeddings are already aligned with text embeddings. This means:

1. Visual features carry semantic meaning interpretable through language
2. Linear projections can map between CLIP space and LLM embedding space
3. The vision encoder requires minimal or no fine-tuning

### Limitations for VLMs

When used in VLMs, CLIP's limitations become apparent:

- Fixed resolution (224x224 or 336x336) limits detail perception
- Patch-based processing loses fine spatial information
- Trained for global image understanding, not localized features
- No native handling of multiple images or video

Many VLMs address these by:
- Using higher resolution variants or multi-scale features
- Adding spatial position encodings
- Training with region-level supervision

## Comparison with Other Vision Encoders

| Model | Architecture | Training Objective | Best Use Case |
|-------|--------------|-------------------|---------------|
| CLIP | ViT/ResNet | Contrastive image-text | Zero-shot classification, VLM backbone |
| SigLIP | ViT | Sigmoid contrastive | Improved accuracy over CLIP |
| DINOv2 | ViT | Self-supervised | Dense prediction tasks |
| MAE | ViT | Masked autoencoding | Representation learning |

SigLIP is a recent alternative that uses sigmoid loss instead of softmax, enabling better scaling and accuracy.

## Model Selection

### Choosing a CLIP Variant

| Variant | Parameters | Speed | Accuracy | Use Case |
|---------|------------|-------|----------|----------|
| ViT-B/32 | 88M | Fastest | Good | Real-time applications |
| ViT-B/16 | 86M | Fast | Better | Balanced performance |
| ViT-L/14 | 304M | Slow | Best | Maximum accuracy |
| ViT-L/14@336px | 304M | Slower | Highest | When resolution matters |

### OpenCLIP Variants

OpenCLIP provides models trained on larger datasets:

| Model | Training Data | Improvement |
|-------|---------------|-------------|
| CLIP (OpenAI) | WIT-400M | Baseline |
| LAION-400M | LAION-400M | Comparable |
| LAION-2B | LAION-2B | Better zero-shot |
| DataComp-1B | DataComp-1B | Strong all-around |

For production use, LAION-2B trained models often provide the best quality.

## Best Practices

### Preprocessing

Always use the model-specific preprocessing:

```python
model, preprocess = clip.load("ViT-B/32")
image = preprocess(raw_image)  # Handles resize, crop, normalize
```

Incorrect preprocessing significantly degrades performance.

### Batch Processing

For efficiency, batch image and text encoding:

```python
# Batch images
images = torch.stack([preprocess(img) for img in image_list])
image_features = model.encode_image(images)

# Batch texts
texts = clip.tokenize(text_list)
text_features = model.encode_text(texts)
```

### Normalization

Normalize embeddings before computing similarity:

```python
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

This ensures cosine similarity is computed correctly.

### Memory Efficiency

For large-scale retrieval, use half precision:

```python
model = model.half()
image_features = model.encode_image(image.half())
```

Or use int8 quantization for embeddings after extraction.
