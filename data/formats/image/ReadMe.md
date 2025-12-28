# Image Formats for Machine Learning

## Summary

Choosing the right image format affects storage costs, data loading speed, and potentially model performance. This chapter compares the major image formats used in ML workflows and provides guidance on when to use each.

Quick reference for format selection:

| Format | Compression | Transparency | Best For |
|--------|-------------|--------------|----------|
| JPEG | Lossy | No | Photographs, web images |
| PNG | Lossless | Yes (alpha) | Graphics, screenshots, labels |
| WebP | Both | Yes (alpha) | Modern web, balanced needs |
| TIFF | Both | Yes | Scientific, medical, geospatial |
| Base64 | None (encoding) | N/A | API payloads, inline embedding |

Key decision factors:

- **Lossless required?** Use PNG, lossless WebP, or TIFF
- **Transparency needed?** Use PNG, WebP, or TIFF
- **Photographs?** Use JPEG or lossy WebP for smaller files
- **Scientific data?** Use TIFF for high bit depth and metadata
- **API transmission?** Encode with Base64 regardless of underlying format
- **Web delivery?** Use WebP where supported, JPEG as fallback

Common pitfalls:

- Training on heavily compressed JPEG data can embed artifacts into model behavior
- Mixing formats with different quality levels introduces inconsistency
- Ignoring file size impacts on training throughput
- Not accounting for decode time in data loading pipelines

---

## Format Comparison

### Compression and File Size

The choice between lossy and lossless compression is fundamental. Lossy compression discards information to achieve smaller files, while lossless preserves exact pixel values.

**Lossy formats** (JPEG, lossy WebP):
- Significantly smaller file sizes (5-20x compression ratios)
- Introduce artifacts that accumulate with re-encoding
- Suitable for photographs and natural images
- Quality degradation may affect fine-grained visual features

**Lossless formats** (PNG, lossless WebP, TIFF with LZW/ZIP):
- Larger files but exact pixel preservation
- No quality degradation from repeated editing
- Essential for ground truth labels and masks
- Better for graphics, text, and sharp edges

Typical file sizes for a 1920x1080 image:

| Format | Configuration | Approximate Size |
|--------|---------------|------------------|
| JPEG | Quality 85 | 200-400 KB |
| JPEG | Quality 95 | 400-800 KB |
| PNG | Default compression | 2-5 MB |
| WebP | Lossy, quality 85 | 150-300 KB |
| WebP | Lossless | 1.5-4 MB |
| TIFF | LZW compression | 3-6 MB |
| TIFF | Uncompressed | 6 MB |

### Decode Performance

Image decoding speed matters for training pipelines where data loading can become a bottleneck:

| Format | Relative Decode Speed |
|--------|----------------------|
| JPEG | Fast |
| PNG | Moderate |
| WebP Lossy | Moderate |
| WebP Lossless | Moderate-Slow |
| TIFF (LZW) | Moderate |
| TIFF (Uncompressed) | Very Fast |

For high-throughput training, consider:
1. Using faster-to-decode formats for training data
2. Pre-decoding and caching as tensors (TFRecord, NumPy arrays)
3. Parallel decoding in data loaders
4. Hardware-accelerated decoding where available (NVIDIA DALI, nvJPEG)

### Color Depth and Dynamic Range

Standard formats store 8 bits per channel (256 levels). For applications requiring higher precision:

| Format | Maximum Bit Depth |
|--------|-------------------|
| JPEG | 8-bit |
| PNG | 16-bit integer |
| WebP | 8-bit |
| TIFF | 32-bit float |

High bit depth is important for:
- Medical imaging (subtle density differences)
- Satellite imagery (wide radiometric range)
- HDR photography and rendering
- Depth maps and distance data

## ML Pipeline Considerations

### Training Data Storage

For large training datasets, format choice significantly impacts infrastructure:

**Storage Costs**: A million-image dataset at 300 KB/image (JPEG) requires 300 GB. The same images as PNG at 3 MB each require 3 TB. At cloud storage rates, this difference compounds.

**Transfer Bandwidth**: Moving data between storage and compute becomes a bottleneck with larger formats. Regional data transfer costs also scale with file size.

**Preprocessing Trade-offs**: Investing in preprocessing time to convert to optimal formats (WebP, or serialized tensors) pays dividends during training iterations.

### Quality Consistency

Inconsistent compression across a dataset can introduce problems:

**Training Bias**: Models may learn compression artifacts rather than genuine image features. This is particularly problematic when artifacts correlate with labels.

**Evaluation Fairness**: Testing on images with different compression than training data affects measured performance.

**Best Practice**: Standardize compression settings during data ingestion. Either use lossless formats for source data, or apply consistent lossy compression across the dataset.

### Augmentation and Artifacts

JPEG compression can be used intentionally as data augmentation:

```python
from albumentations import ImageCompression

augment = ImageCompression(quality_lower=50, quality_upper=90, p=0.5)
```

This improves model robustness to compression artifacts encountered in production. However, be cautious about compounding effects when source data is already compressed.

## Format-Specific Guidance

### When to Use JPEG

JPEG excels for photographs and natural images where:
- File size is a priority
- Perfect pixel accuracy is not required
- No transparency is needed
- Maximum compatibility is important

Avoid JPEG for:
- Segmentation masks and labels (artifacts corrupt boundaries)
- Images with text or sharp graphics
- Archival storage of originals
- Intermediate processing steps

### When to Use PNG

PNG is the right choice for:
- Ground truth labels and segmentation masks
- Images requiring transparency
- Screenshots, diagrams, and graphics
- Archival storage where lossless quality matters
- Any case where artifacts are unacceptable

Consider alternatives when:
- File size is critical (use lossy WebP)
- Higher bit depth is needed (use TIFF)
- Format modernization is possible (use lossless WebP)

### When to Use WebP

WebP offers the best of both worlds:
- Lossy mode for photographs with smaller files than JPEG
- Lossless mode for graphics with smaller files than PNG
- Transparency support in both modes
- Animation support

Limitations:
- Not universally supported in all tools
- Maximum 8-bit color depth
- Some older systems and libraries lack support

### When to Use TIFF

TIFF is essential for:
- Scientific and medical imaging
- Geospatial data (GeoTIFF)
- High bit depth requirements (16-bit, 32-bit float)
- Multi-page image sequences (z-stacks, time series)
- Professional photography workflows

Avoid TIFF for:
- Web delivery
- Mobile applications
- Situations where simpler formats suffice

### When to Use Base64

Base64 encoding is used to transmit images through text-only channels:
- API request/response payloads (JSON, XML)
- Embedding images in HTML/CSS
- Database storage in text fields
- Configuration files

Remember that Base64 is an encoding layer, not a format. The underlying image (JPEG, PNG, etc.) is preserved. Base64 adds approximately 33% to the data size.

## Practical Recommendations

### For Training Datasets

1. **Source Quality**: Store original images in lossless format (PNG, TIFF) if possible
2. **Training Copies**: Create WebP or JPEG copies with consistent quality settings
3. **Labels**: Always use lossless (PNG) for segmentation masks
4. **Batch Formats**: Consider TFRecord or WebDataset for training efficiency

### For Inference Pipelines

1. **Accept Multiple Formats**: Design APIs to handle common formats
2. **Validate on Upload**: Check format, dimensions, and basic quality
3. **Standardize Internally**: Convert to consistent format/size early in pipeline
4. **Return Appropriate Formats**: Match output format to use case

### For Model Development

1. **Document Format Requirements**: Specify expected input formats clearly
2. **Test Across Quality Levels**: Verify model performance with varied compression
3. **Consider Robustness**: Include compression augmentation if production data varies
4. **Monitor for Drift**: Track input image quality in production

## Tools and Libraries

Common Python libraries for image handling:

| Library | Strengths |
|---------|-----------|
| Pillow (PIL) | General purpose, wide format support |
| OpenCV | Fast processing, computer vision focus |
| tifffile | Scientific TIFF, high bit depth |
| rasterio | Geospatial, GeoTIFF |
| imageio | Simple API, format abstraction |
| scikit-image | Scientific image processing |

For high-performance pipelines:

| Tool | Use Case |
|------|----------|
| NVIDIA DALI | GPU-accelerated decoding and augmentation |
| TurboJPEG | Optimized JPEG encode/decode |
| libvips | Memory-efficient processing of large images |
