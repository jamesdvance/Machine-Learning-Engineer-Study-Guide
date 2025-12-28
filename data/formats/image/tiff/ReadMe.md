# TIFF Image Format

## Summary

TIFF (Tagged Image File Format) is a flexible, richly-featured image format designed for professional imaging workflows. It supports multiple compression methods (including lossless), high bit depths, multiple images per file, and extensive metadata. TIFF is the standard format for scientific imaging, medical imaging, satellite imagery, and professional photography where maximum quality and flexibility are required.

Key points to remember:

- Supports lossless compression (LZW, ZIP) and uncompressed storage
- High bit depth support (8, 16, 32 bits per channel, including floating-point)
- Multiple images can be stored in a single file (multi-page TIFF)
- Supports various color spaces (RGB, CMYK, LAB, grayscale)
- Extensive metadata and tag system for arbitrary data storage
- Larger files than PNG or JPEG but maximum flexibility
- Two variants: BigTIFF for files over 4GB, standard TIFF otherwise

Common use cases in ML:

- Geospatial and satellite imagery (GeoTIFF)
- Medical imaging (often paired with DICOM metadata)
- Scientific microscopy
- High dynamic range (HDR) image storage
- Multi-spectral and hyperspectral imaging
- Archival storage where quality is paramount

When to use TIFF:

- Scientific or medical imaging applications
- When higher than 8-bit color depth is needed
- Multi-page documents or image stacks (z-stacks, time series)
- Geospatial data with coordinate system metadata
- Professional photography workflows requiring maximum quality

When to avoid TIFF:

- Web delivery (use JPEG, PNG, or WebP)
- Mobile applications with storage constraints
- Datasets where file size significantly impacts training speed
- General-purpose applications where simpler formats suffice

---

## Understanding TIFF Structure

TIFF files are organized around a tag-based structure that allows tremendous flexibility. Each image is described by a set of tags specifying its properties.

### File Organization

A TIFF file contains:

1. **Header**: 8 bytes identifying the file as TIFF and specifying byte order (big-endian or little-endian).

2. **Image File Directories (IFDs)**: Each IFD describes one image within the file. IFDs contain tags specifying image dimensions, color depth, compression, and data location.

3. **Image Data**: The actual pixel data, which can be stored in strips or tiles.

4. **Metadata**: Additional tags for camera settings, geospatial coordinates, custom application data, and more.

### Tags and Flexibility

The tag system makes TIFF extremely flexible but also complex. Standard tags define common properties, while private tags allow applications to store custom data. Some important standard tags:

- ImageWidth, ImageLength: Dimensions
- BitsPerSample: Bit depth per channel
- SamplesPerPixel: Number of channels
- Compression: Compression method used
- PhotometricInterpretation: How to interpret pixel values
- SampleFormat: Integer, unsigned integer, or floating-point

### Compression Options

TIFF supports numerous compression methods:

| Method | Type | Notes |
|--------|------|-------|
| None (1) | Uncompressed | Largest files, fastest read/write |
| LZW (5) | Lossless | Good general-purpose compression |
| ZIP/Deflate (8) | Lossless | Often slightly better than LZW |
| JPEG (7) | Lossy | For photographs, rarely used in TIFF |
| PackBits (32773) | Lossless | Simple RLE, fast but limited compression |

For ML applications, LZW or ZIP compression provides a good balance of file size and decode speed.

## TIFF in Machine Learning Applications

### Scientific and Medical Imaging

TIFF is the standard format for many scientific imaging applications where data integrity is critical:

**Microscopy**: Z-stacks (3D image stacks) and time-lapse series are often stored as multi-page TIFFs. Each page represents one slice or time point.

**Medical Imaging**: While DICOM is the primary medical format, TIFF is commonly used for pathology slides and other high-resolution medical images. Whole slide images (WSIs) often use pyramidal TIFF structures.

**Satellite Imagery**: GeoTIFF extends TIFF with geospatial metadata, enabling coordinate system awareness essential for Earth observation ML.

### Loading TIFF Files

Basic loading with PIL/Pillow:

```python
from PIL import Image
import numpy as np

# Load single-page TIFF
image = Image.open("image.tif")
array = np.array(image)

# Load multi-page TIFF
from PIL import ImageSequence

def load_tiff_stack(path):
    """Load all pages from a multi-page TIFF."""
    image = Image.open(path)
    pages = []
    for page in ImageSequence.Iterator(image):
        pages.append(np.array(page))
    return np.stack(pages)
```

For scientific applications, tifffile provides more complete TIFF support:

```python
import tifffile

# Load any TIFF, including complex scientific formats
image = tifffile.imread("image.tif")

# Load specific pages
page_3 = tifffile.imread("stack.tif", key=3)

# Read metadata
with tifffile.TiffFile("image.tif") as tif:
    for page in tif.pages:
        print(page.tags)
```

### High Bit Depth Images

TIFF commonly stores 16-bit or 32-bit images, which require careful handling:

```python
import tifffile
import numpy as np

# Load 16-bit image
image_16bit = tifffile.imread("microscopy.tif")
print(image_16bit.dtype)  # uint16

# Normalize to 0-1 range for model input
image_float = image_16bit.astype(np.float32) / 65535.0

# For display or 8-bit model input, scale down
# Simple linear scaling (may lose dynamic range)
image_8bit = (image_16bit / 256).astype(np.uint8)

# Better: percentile-based scaling to handle outliers
p_low, p_high = np.percentile(image_16bit, [1, 99])
image_scaled = np.clip((image_16bit - p_low) / (p_high - p_low), 0, 1)
image_8bit = (image_scaled * 255).astype(np.uint8)
```

### Floating-Point TIFF

TIFF supports 32-bit floating-point pixels, useful for:
- Depth maps and distance data
- Probability maps and heatmaps from model outputs
- HDR image data
- Scientific measurements with physical units

```python
import tifffile
import numpy as np

# Save floating-point predictions
predictions = model(input_batch)  # Shape: (H, W), values 0-1
tifffile.imwrite("predictions.tif", predictions.astype(np.float32))

# Load and verify
loaded = tifffile.imread("predictions.tif")
print(loaded.dtype)  # float32
```

## GeoTIFF for Geospatial Data

GeoTIFF embeds coordinate reference system (CRS) information and geotransform data within TIFF tags. This is essential for satellite imagery and Earth observation ML.

### Reading GeoTIFFs

```python
import rasterio
import numpy as np

with rasterio.open("satellite.tif") as src:
    # Read all bands
    image = src.read()  # Shape: (bands, height, width)

    # Get coordinate reference system
    crs = src.crs

    # Get bounds in CRS units
    bounds = src.bounds

    # Convert pixel coordinates to geographic coordinates
    lon, lat = src.xy(row=100, col=200)
```

### Common GeoTIFF Considerations

**Multi-Band Images**: Satellite images often have many bands (multispectral or hyperspectral). The shape is typically (bands, height, width) rather than (height, width, channels).

**Large File Sizes**: A single Sentinel-2 tile can exceed 1 GB. Use windowed reading to process large images in chunks:

```python
with rasterio.open("large_image.tif") as src:
    # Read a window instead of entire image
    from rasterio.windows import Window
    window = Window(col_off=1000, row_off=1000, width=512, height=512)
    chip = src.read(window=window)
```

**Cloud Optimized GeoTIFF (COG)**: A specific TIFF organization optimized for cloud storage and partial reads. COGs allow efficient access to small regions of large images without downloading the entire file.

## Multi-Page and Pyramidal TIFFs

### Multi-Page TIFF

Multiple images in a single file, common for:
- Z-stacks in microscopy
- Time series
- Document scanning (multi-page documents)

```python
import tifffile
import numpy as np

# Write multi-page TIFF
stack = np.random.randint(0, 255, (10, 512, 512), dtype=np.uint8)
tifffile.imwrite("stack.tif", stack)

# Read specific page
page_5 = tifffile.imread("stack.tif", key=5)
```

### Pyramidal TIFF

Multiple resolution levels stored in a single file, enabling efficient viewing at different zoom levels. Common for whole slide imaging and large geospatial datasets.

```python
import tifffile

# Read with pyramid structure
with tifffile.TiffFile("slide.tif") as tif:
    # Access different resolution levels
    full_res = tif.pages[0].asarray()
    half_res = tif.pages[1].asarray()  # If stored as separate pages

    # Or access as pyramid
    for level, page in enumerate(tif.series[0].levels):
        print(f"Level {level}: {page.shape}")
```

## Performance Considerations

### Read Performance

TIFF reading can be slower than JPEG or PNG due to:
- Larger file sizes
- Complex decompression for some compression types
- Additional metadata processing

For training with many TIFF files, consider:

1. **Converting to Training-Optimized Formats**: Convert TIFFs to TFRecord, WebDataset, or other formats optimized for sequential access.

2. **Memory Mapping**: For very large TIFFs, use memory-mapped access:

```python
import tifffile

# Memory-mapped reading (doesn't load entire file)
image = tifffile.memmap("large_image.tif")
# Access regions on-demand
region = image[1000:1512, 2000:2512]
```

3. **Preprocessing Pipeline**: Decode TIFFs during data preprocessing, saving processed tensors for faster training iteration.

### Write Performance

When saving model outputs as TIFF:

```python
import tifffile
import numpy as np

# Fast uncompressed write
tifffile.imwrite("output.tif", data, compress=False)

# Compressed write (slower but smaller)
tifffile.imwrite("output.tif", data, compress="zlib")

# With specific tile size for large images
tifffile.imwrite("output.tif", data, tile=(512, 512), compress="zlib")
```

## Comparison with Other Formats

### TIFF vs PNG

| Characteristic | TIFF | PNG |
|----------------|------|-----|
| Bit depth | Up to 32-bit float | Up to 16-bit integer |
| Multi-page | Yes | No |
| Compression options | Many | DEFLATE only |
| Metadata | Extensive tags | Limited text chunks |
| File size | Larger | Smaller |
| Web support | Poor | Excellent |
| Scientific use | Standard | Limited |

### TIFF vs HDF5

For scientific data, HDF5 is an alternative:

| Characteristic | TIFF | HDF5 |
|----------------|------|------|
| Primary purpose | Images | Any numerical data |
| Hierarchical structure | Limited | Full hierarchy |
| Compression | Per-image | Per-dataset |
| Tool support | Image tools | Scientific tools |
| Geospatial metadata | GeoTIFF | Custom |

## Practical Tips

1. **Library Selection**: Use tifffile for scientific TIFFs, Pillow for simple cases, rasterio for GeoTIFFs.

2. **Memory Management**: Large TIFFs can exceed available RAM. Use windowed reading or memory mapping.

3. **Compression Choice**: LZW for general use, ZIP for slightly better compression, uncompressed when read speed is critical.

4. **Bit Depth Awareness**: Always check dtype after loading. Normalize appropriately for your model's expected input range.

5. **Metadata Preservation**: When processing scientific data, preserve important metadata tags in outputs.
