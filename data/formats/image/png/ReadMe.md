# PNG Image Format

## Summary

PNG (Portable Network Graphics) is a lossless compression format that preserves exact pixel values while supporting transparency through an alpha channel. It excels at graphics, screenshots, and any content with sharp edges, text, or flat colors. Unlike JPEG, PNG introduces no artifacts regardless of how many times an image is edited and saved.

Key points to remember:

- Lossless compression preserves exact pixel data
- Supports full alpha channel transparency (8-bit, 256 levels)
- Better than JPEG for graphics, screenshots, diagrams, and text
- Larger file sizes than JPEG for photographs
- Supports color depths from 1-bit to 48-bit (16-bit per channel)
- No quality degradation from repeated editing and saving
- Uses DEFLATE compression (same as gzip/zlib)

Color modes supported:

- Grayscale (1, 2, 4, 8, 16 bits)
- Grayscale with alpha (8, 16 bits)
- Indexed color/palette (1, 2, 4, 8 bits, up to 256 colors)
- Truecolor RGB (8, 16 bits per channel)
- Truecolor RGBA (8, 16 bits per channel)

When to use PNG:

- Images with transparency requirements
- Screenshots and UI elements
- Graphics with text, logos, or sharp edges
- Diagrams, charts, and technical illustrations
- Source/archival images requiring lossless storage
- Intermediate processing steps before final compression

When to avoid PNG:

- Photographs where file size matters (use JPEG or WebP)
- Animated images (use GIF, APNG, or WebP)
- When legacy system compatibility is critical (though PNG support is nearly universal)

---

## Understanding PNG Compression

PNG uses lossless compression, meaning the decoded image is bit-for-bit identical to the original. This is achieved through a combination of filtering and DEFLATE compression.

### The Compression Pipeline

1. **Filtering**: Before compression, each row of pixels is filtered to improve compressibility. The filter predicts pixel values based on neighboring pixels, then stores the difference. Five filter types are available:
   - None: No filtering
   - Sub: Difference from pixel to the left
   - Up: Difference from pixel above
   - Average: Difference from average of left and above
   - Paeth: Predictor based on left, above, and upper-left

2. **DEFLATE Compression**: The filtered data is compressed using DEFLATE, which combines LZ77 (finding repeated patterns) with Huffman coding. This is the same algorithm used in gzip and zlib.

### Compression Levels

PNG encoders typically offer compression levels from 0-9:
- Level 0: No compression, fastest encoding, largest files
- Level 1-3: Fast compression, slightly larger files
- Level 4-6: Balanced compression speed and ratio
- Level 7-9: Maximum compression, slowest encoding

Higher compression levels do not affect image quality since PNG is lossless. They only trade encoding time for smaller file sizes.

```python
from PIL import Image

# Default compression
image.save("output.png")

# Maximum compression (slower)
image.save("output.png", compress_level=9)

# Fast compression (larger file)
image.save("output.png", compress_level=1)
```

### Optimizing PNG Files

Several tools can reduce PNG file size beyond what standard encoders produce:

**OptiPNG**: Tries multiple filter and compression combinations to find the smallest result.

**pngquant**: Converts 24-bit or 32-bit PNG to 8-bit palette PNG with alpha support. Technically lossy but often visually indistinguishable with significant size reduction.

**zopfli**: Google's DEFLATE-compatible compressor that produces smaller files at the cost of much longer encoding time.

## Transparency and Alpha Channels

PNG's transparency support is one of its key advantages over JPEG. The alpha channel stores opacity information for each pixel, ranging from fully transparent (0) to fully opaque (255 for 8-bit).

### Working with Transparency

```python
from PIL import Image
import numpy as np

# Load PNG with transparency
image = Image.open("logo.png")
print(image.mode)  # "RGBA" if transparent, "RGB" if opaque

# Check for alpha channel
if image.mode == "RGBA":
    r, g, b, a = image.split()
    # a is the alpha channel

# Convert to numpy array (includes alpha)
array = np.array(image)  # Shape: (height, width, 4) for RGBA

# Create image with transparency
rgba_array = np.zeros((100, 100, 4), dtype=np.uint8)
rgba_array[:, :, 0] = 255  # Red channel
rgba_array[:, :, 3] = 128  # 50% transparent
image = Image.fromarray(rgba_array, mode="RGBA")
image.save("transparent.png")
```

### Handling Transparency in ML Pipelines

Most vision models expect RGB input. When loading PNGs with transparency, you must decide how to handle the alpha channel:

```python
from PIL import Image

def load_png_as_rgb(path, background_color=(255, 255, 255)):
    """Load PNG and composite onto background color."""
    image = Image.open(path)

    if image.mode == "RGBA":
        # Create background
        background = Image.new("RGB", image.size, background_color)
        # Composite using alpha
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode == "RGB":
        return image
    else:
        return image.convert("RGB")
```

Alternatively, some applications use the alpha channel as a mask for segmentation or attention:

```python
def load_png_with_mask(path):
    """Load PNG, returning RGB image and alpha mask separately."""
    image = Image.open(path)

    if image.mode == "RGBA":
        rgb = image.convert("RGB")
        alpha = image.split()[3]
        return rgb, alpha
    else:
        rgb = image.convert("RGB")
        # Create fully opaque mask
        alpha = Image.new("L", image.size, 255)
        return rgb, alpha
```

## PNG in Machine Learning Workflows

### Advantages for Training Data

PNG's lossless nature makes it ideal for:

**Ground Truth Labels**: Segmentation masks, bounding box visualizations, and annotation overlays should be stored as PNG to prevent artifacts from corrupting label boundaries.

**Reproducibility**: Since PNG preserves exact pixel values, results are reproducible. Two researchers loading the same PNG will get identical arrays.

**Quality-Sensitive Applications**: Medical imaging, document analysis, and scientific visualization benefit from lossless storage.

### File Size Considerations

PNG files can be significantly larger than JPEG for photographic content. A 4K photograph might be:
- 2-5 MB as JPEG (quality 85)
- 15-25 MB as PNG

For large datasets, this difference accumulates. Consider:

1. **Storage Costs**: A million-image dataset at 20 MB per image requires 20 TB of storage.

2. **I/O Bandwidth**: Training bottlenecks often occur in data loading. Larger files mean slower throughput.

3. **Hybrid Approaches**: Store original images as PNG for archival, but create JPEG versions for training. Or use dataset formats that handle compression internally (TFRecord, WebDataset).

### Memory Considerations

PNG decoding requires more memory than JPEG for the same image dimensions because:
- The full uncompressed image must be buffered during decompression
- RGBA images use 4 bytes per pixel versus 3 for RGB
- Higher bit depths (16-bit channels) increase memory proportionally

```python
# Memory for a 4K RGBA image (3840 x 2160)
pixels = 3840 * 2160
bytes_per_pixel = 4  # RGBA
memory_bytes = pixels * bytes_per_pixel  # ~33 MB per image
```

## Indexed Color (Palette) PNGs

PNG supports indexed color mode, where each pixel references a palette of up to 256 colors. This significantly reduces file size for images with limited colors.

```python
from PIL import Image

# Convert to palette mode
image = Image.open("graphic.png")
palette_image = image.convert("P", palette=Image.ADAPTIVE, colors=256)
palette_image.save("indexed.png")
```

Indexed PNGs are useful for:
- Simple graphics, icons, and logos
- Segmentation masks with discrete class labels
- Reducing file size when color depth is not critical

Note that indexed PNG with transparency uses a single transparent color, not a full alpha channel. For smooth transparency edges, use RGBA mode.

## Comparison with Other Formats

### PNG vs JPEG

| Characteristic | PNG | JPEG |
|----------------|-----|------|
| Compression | Lossless | Lossy |
| Transparency | Full alpha | None |
| Best for | Graphics, text | Photographs |
| File size (photo) | Larger | Smaller |
| File size (graphic) | Smaller | Larger |
| Artifacts | None | Blocking, ringing |
| Repeated edits | Safe | Degrades quality |

### PNG vs WebP

WebP offers both lossy and lossless modes. Lossless WebP typically achieves 20-30% smaller files than PNG. However:
- PNG has near-universal support; WebP support is still growing
- Some imaging libraries have incomplete WebP support
- PNG is the safer choice for maximum compatibility

### PNG vs TIFF

Both support lossless compression, but:
- PNG is better for web and general use (smaller files, universal support)
- TIFF supports more features (CMYK, layers, multiple compression types)
- TIFF is preferred for professional photography and printing workflows

## Metadata in PNG

PNG supports metadata through text chunks. Common metadata types:

- **tEXt**: Uncompressed text for author, description, copyright
- **zTXt**: Compressed text for longer metadata
- **iTXt**: International text supporting UTF-8

Reading and writing metadata:

```python
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Read metadata
image = Image.open("image.png")
print(image.info)  # Dictionary of metadata

# Write metadata
metadata = PngInfo()
metadata.add_text("Author", "Your Name")
metadata.add_text("Description", "Sample image")
image.save("output.png", pnginfo=metadata)
```

## Interlacing

PNG supports Adam7 interlacing, which displays a low-resolution preview before the full image loads. Similar to progressive JPEG but using a different algorithm.

```python
# Save with interlacing
image.save("output.png", interlace=True)
```

Interlacing slightly increases file size and decode time but improves perceived loading speed for web applications. For training data where images are loaded entirely before use, non-interlaced PNGs are more efficient.
