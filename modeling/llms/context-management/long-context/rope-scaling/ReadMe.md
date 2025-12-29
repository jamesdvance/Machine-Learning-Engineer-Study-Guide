# RoPE Scaling

## Summary

RoPE (Rotary Position Embedding) scaling extends LLMs to longer contexts than their training length without full retraining. Since RoPE encodes positions through rotations in embedding space, scaling methods modify these rotations to handle longer sequences. The main approaches are position interpolation (compress positions to fit), NTK-aware scaling (adjust rotation frequencies), and YaRN (combine interpolation with frequency adjustments). These techniques enable models trained on 4K contexts to work with 32K+ sequences with minimal quality loss.

Key points to remember:

- RoPE uses rotation to encode positions; scaling modifies these rotations
- Position interpolation: Compress positions by a scaling factor
- NTK-aware: Adjust rotation frequencies based on neural tangent kernel theory
- YaRN: Combines interpolation with frequency-based scaling for best results
- Fine-tuning helps: Brief continued training improves scaled model quality
- Trade-offs: Longer context typically reduces per-position quality slightly
- Common in practice: LLaMA extended from 4K to 32K+ using these methods

## RoPE Fundamentals

### How RoPE Works

RoPE encodes position by rotating query and key vectors:

```python
def apply_rope(x, position, dim):
    """Apply rotary position embedding."""
    # Create rotation frequencies
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

    # Position * frequency gives rotation angle
    angles = position * freqs

    # Apply rotation to pairs of dimensions
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated = torch.stack([
        x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
        x_pairs[..., 0] * sin + x_pairs[..., 1] * cos
    ], dim=-1)

    return rotated.flatten(-2)
```

### The Extrapolation Problem

RoPE trained on positions 0-4095 struggles with position 8000:

```
Training: positions 0, 1, 2, ..., 4095
Inference: positions 0, 1, 2, ..., 8000
           ^^^^^^^^^^^^^^^^^^^^^^^^^
           Unseen rotation angles for positions 4096-8000
           -> Unpredictable attention patterns
           -> Quality degrades severely
```

## Position Interpolation

### Concept

Compress positions to fit within training range:

```python
def position_interpolation(position, scale_factor):
    """Linear position interpolation."""
    # If trained on 4096 and want 8192, scale = 2
    # Position 8000 -> 8000 / 2 = 4000 (within training range)
    return position / scale_factor
```

### Implementation

```python
class RoPEWithInterpolation(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000, scale_factor=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_factor = scale_factor

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        # Apply interpolation scaling
        scaled_positions = positions / self.scale_factor

        # Compute rotation angles
        freqs = torch.outer(scaled_positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        return torch.cos(emb), torch.sin(emb)
```

### Limitations

- All frequencies scaled equally
- High-frequency rotations (local positions) become too compressed
- Requires fine-tuning to recover quality

## NTK-Aware Scaling

### Concept

Rather than scaling positions, adjust the base frequency:

```python
def ntk_aware_scaling(base, scale_factor, dim):
    """Adjust RoPE base for extended context."""
    # From Neural Tangent Kernel theory:
    # Scale the base to spread rotations more evenly
    alpha = scale_factor
    new_base = base * (alpha ** (dim / (dim - 2)))
    return new_base
```

### Implementation

```python
class NTKAwareRoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000, scale_factor=1.0):
        super().__init__()
        self.dim = dim

        # Adjust base for longer context
        if scale_factor > 1.0:
            base = base * (scale_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)
```

### Dynamic NTK

Adjust scaling dynamically based on sequence length:

```python
class DynamicNTKRoPE(nn.Module):
    def __init__(self, dim, original_max_len, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max_len = original_max_len

    def forward(self, positions, seq_len):
        if seq_len > self.original_max_len:
            # Compute scale factor based on current length
            scale = seq_len / self.original_max_len

            # Dynamically adjust base
            base = self.base * (scale ** (self.dim / (self.dim - 2)))
        else:
            base = self.base

        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        freqs = torch.outer(positions.float(), inv_freq.to(positions.device))

        return torch.cos(freqs), torch.sin(freqs)
```

## YaRN (Yet another RoPE extensioN)

### Concept

YaRN combines interpolation with frequency-specific scaling:

- Low frequencies (long-range): Apply interpolation
- High frequencies (local): Keep unchanged
- Middle frequencies: Blend between approaches

### Implementation

```python
class YaRNRoPE(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000,
                 scale_factor=1.0, beta_fast=32, beta_slow=1,
                 original_max_len=4096):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.original_max_len = original_max_len

        # Compute frequency-specific scaling
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # YaRN ramp function
        low_freq_wavelen = original_max_len / beta_slow
        high_freq_wavelen = original_max_len / beta_fast

        wavelens = 2 * math.pi / inv_freq
        ratios = wavelens / original_max_len

        # Compute scaling factors per frequency
        yarn_factors = torch.where(
            ratios > low_freq_wavelen,
            torch.ones_like(ratios) * scale_factor,  # Interpolate
            torch.where(
                ratios < high_freq_wavelen,
                torch.ones_like(ratios),  # No scaling
                # Linear ramp between
                (ratios - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
                * (scale_factor - 1) + 1
            )
        )

        self.register_buffer('inv_freq', inv_freq / yarn_factors)

    def forward(self, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)
```

### YaRN Components

| Component | Purpose |
|-----------|---------|
| Position interpolation | Handle unseen positions |
| Frequency-based scaling | Preserve local attention quality |
| Attention temperature | Compensate for distribution changes |
| Fine-tuning | Adapt model to new rotation patterns |

## Scaling Method Comparison

| Method | Quality | Fine-tuning Needed | Complexity |
|--------|---------|-------------------|------------|
| Position Interpolation | Good with FT | Yes (recommended) | Low |
| NTK-Aware | Better than PI | Less critical | Low |
| Dynamic NTK | Good zero-shot | Optional | Low |
| YaRN | Best | Recommended | Medium |

## Practical Implementation

### Using Transformers Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with rope_scaling
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    rope_scaling={
        "type": "dynamic",  # or "linear", "yarn"
        "factor": 2.0
    },
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Custom RoPE Scaling

```python
def apply_rope_scaling(model, scale_factor, method="yarn"):
    """Apply RoPE scaling to existing model."""
    for layer in model.model.layers:
        attn = layer.self_attn

        if method == "linear":
            # Position interpolation
            attn.rotary_emb = RoPEWithInterpolation(
                attn.head_dim,
                model.config.max_position_embeddings * scale_factor,
                scale_factor=scale_factor
            )
        elif method == "ntk":
            attn.rotary_emb = NTKAwareRoPE(
                attn.head_dim,
                model.config.max_position_embeddings * scale_factor,
                scale_factor=scale_factor
            )
        elif method == "yarn":
            attn.rotary_emb = YaRNRoPE(
                attn.head_dim,
                model.config.max_position_embeddings * scale_factor,
                scale_factor=scale_factor,
                original_max_len=model.config.max_position_embeddings
            )

    return model
```

### Fine-tuning After Scaling

```python
from transformers import Trainer, TrainingArguments

# Fine-tune on long-context data
training_args = TrainingArguments(
    output_dir="./llama-extended",
    per_device_train_batch_size=1,  # Small batch for long context
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=1,  # Brief fine-tuning
    max_steps=1000,  # Usually sufficient
    fp16=True,
    gradient_checkpointing=True,  # Essential for long context
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=long_context_dataset,
)
trainer.train()
```

## Quality Considerations

### Needle-in-Haystack Testing

```python
def needle_in_haystack_test(model, tokenizer, context_lengths, depths):
    """Test retrieval at various positions in long context."""
    results = {}

    for length in context_lengths:
        for depth in depths:  # 0.0 = start, 1.0 = end
            # Create test case
            haystack = generate_filler_text(length)
            needle_pos = int(length * depth)
            test_text = insert_needle(haystack, needle_pos, "The secret code is 42.")

            # Test retrieval
            inputs = tokenizer(test_text + "\nWhat is the secret code?",
                             return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.decode(outputs[0])

            results[(length, depth)] = "42" in response

    return results
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor recall at edges | Position distribution shift | More fine-tuning data at edges |
| Degraded quality | Over-compression | Use YaRN instead of linear |
| OOM errors | KV-cache for long context | Use sliding window or paged attention |

## Key Takeaways

1. **RoPE is rotational**: Scaling methods modify rotation patterns for longer sequences.

2. **Interpolation is simple but limited**: Compresses all frequencies equally.

3. **NTK-aware preserves high frequencies**: Better local attention quality.

4. **YaRN is state-of-the-art**: Combines best of interpolation and frequency scaling.

5. **Fine-tuning improves quality**: Even brief training on long data helps significantly.

6. **Test thoroughly**: Use needle-in-haystack and other retrieval tests.

7. **Memory is the real limit**: Context extension requires efficient inference (sliding window, paged attention).
