# H.265 / HEVC Video Codec

## Summary

H.265, also known as HEVC (High Efficiency Video Coding), is the successor to H.264 offering approximately 40-50% better compression at equivalent quality. Standardized in 2013, it was designed primarily for 4K and higher resolutions where H.264's bitrate requirements become impractical. For machine learning practitioners, H.265 is increasingly common in high-resolution video datasets, particularly for 4K content, surveillance systems, and streaming platforms.

Key points to remember:

- 40-50% bitrate reduction compared to H.264 at equivalent quality
- Higher encode and decode complexity than H.264
- Better suited for 4K/8K resolutions where compression gains matter most
- Hardware decode support is widespread but not universal
- Complex licensing situation with multiple patent pools
- Container formats: MP4, MKV, MOV, TS

Profiles:

- Main: 8-bit 4:2:0, standard consumer content
- Main 10: 10-bit 4:2:0, HDR content
- Main 4:2:2 10: Professional video production
- Main 4:4:4: Full chroma, color-critical applications

When to use H.265:

- 4K or higher resolution content
- Storage or bandwidth is constrained
- Hardware decode is available
- HDR content with 10-bit color
- Long-term archival with better compression

When to consider alternatives:

- Maximum compatibility needed (use H.264)
- Royalty-free requirement (use VP9 or AV1)
- Even better compression needed (use AV1)
- Fast encoding is critical (use H.264)

---

## Compression Improvements Over H.264

H.265 achieves better compression through several architectural improvements while following the same general hybrid coding approach.

### Larger and More Flexible Block Structures

**Coding Tree Units (CTUs)**: H.265 uses CTUs up to 64x64 pixels, compared to H.264's 16x16 macroblocks. Larger blocks improve compression for uniform regions.

**Flexible Partitioning**: CTUs can be recursively subdivided into Coding Units (CUs) as small as 8x8. This quad-tree structure adapts to image content, using large blocks for smooth areas and small blocks for detailed regions.

**Prediction Units and Transform Units**: Separate from CUs, allowing independent optimization of prediction and transform block sizes.

### Improved Prediction

**More Intra Prediction Modes**: 35 angular modes (vs 9 in H.264) plus DC and planar modes. Better captures directional patterns in images.

**Advanced Motion Compensation**: Improved motion vector prediction, merge mode for efficient signaling, and advanced motion vector prediction (AMVP).

**Sample Adaptive Offset (SAO)**: Post-deblocking filter that reduces banding and ringing artifacts, improving quality especially at low bitrates.

### Transform Improvements

- Transform sizes from 4x4 to 32x32 (vs 4x4 and 8x8 in H.264)
- Transform skip mode for screen content
- Improved entropy coding with CABAC throughout

## H.265 in Machine Learning Workflows

### Decoding H.265 Video

Hardware-accelerated decode is strongly recommended for H.265 due to higher complexity.

**OpenCV with FFmpeg backend**:
```python
import cv2

# Ensure OpenCV is built with FFmpeg and HEVC support
cap = cv2.VideoCapture("video_hevc.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
```

**PyAV**:
```python
import av

container = av.open("video_hevc.mp4")
for frame in container.decode(video=0):
    array = frame.to_ndarray(format="rgb24")
```

**Decord with GPU acceleration**:
```python
from decord import VideoReader, gpu

# GPU decode is especially beneficial for H.265
vr = VideoReader("video_hevc.mp4", ctx=gpu(0))
frames = vr.get_batch([0, 30, 60, 90]).asnumpy()
```

### Hardware Decode Support

Check for H.265 hardware decode capability:

```python
import subprocess

# Check NVIDIA GPU decode support
result = subprocess.run(['nvidia-smi', '-q', '-d', 'SUPPORTED_CLOCKS'],
                       capture_output=True, text=True)

# FFmpeg hardware decode
# -hwaccel cuda for NVIDIA
# -hwaccel vaapi for Intel/AMD on Linux
# -hwaccel videotoolbox for macOS
```

Most GPUs from 2015 onward support H.265 decode:
- NVIDIA: Maxwell generation (GTX 900 series) and later
- AMD: GCN 3rd generation and later
- Intel: Skylake and later integrated graphics
- Apple: A9 chip and later, M1/M2

### Encoding for ML Outputs

When creating H.265 video from model outputs:

```python
import subprocess

process = subprocess.Popen([
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}',
    '-pix_fmt', 'rgb24',
    '-r', '30',
    '-i', '-',
    '-c:v', 'libx265',  # H.265 encoder
    '-preset', 'medium',
    '-crf', '28',  # H.265 CRF scale differs from H.264
    '-pix_fmt', 'yuv420p',
    'output_hevc.mp4'
], stdin=subprocess.PIPE)

for frame in predictions:
    process.stdin.write(frame.tobytes())

process.stdin.close()
process.wait()
```

Note: H.265 encoding is significantly slower than H.264. For real-time or batch processing of many videos, hardware encoding (NVENC) may be necessary.

**NVENC hardware encoding**:
```bash
ffmpeg -i input.mp4 -c:v hevc_nvenc -preset medium -crf 28 output.mp4
```

## Encoding Parameters

### CRF Scale

H.265 uses a different CRF scale than H.264. Equivalent quality typically requires CRF values about 4-6 points higher:

| H.264 CRF | Approximate H.265 CRF | Quality Level |
|-----------|----------------------|---------------|
| 18 | 22-24 | Visually lossless |
| 23 | 27-28 | Good quality |
| 28 | 32-34 | Lower quality |

### Presets

Like H.264, presets trade speed for compression:

| Preset | Relative Speed | Compression | Notes |
|--------|----------------|-------------|-------|
| ultrafast | 1x | Poor | Testing only |
| superfast | 2x | Poor | Low-latency streaming |
| veryfast | 4x | Moderate | Fast encoding |
| faster | 6x | Moderate-good | General use |
| fast | 8x | Good | General use |
| medium | 10x | Good | Default |
| slow | 20x | Very good | Quality focus |
| slower | 40x | Excellent | Archival |
| veryslow | 80x | Best | Maximum compression |

The speed penalty for H.265 encoding is significant. A "medium" preset H.265 encode might take 10x longer than H.264 at the same preset.

### Tuning Options

```bash
# Animation/cartoon content
ffmpeg -i input.mp4 -c:v libx265 -tune animation output.mp4

# Low-latency streaming
ffmpeg -i input.mp4 -c:v libx265 -tune zerolatency output.mp4

# Grain preservation
ffmpeg -i input.mp4 -c:v libx265 -tune grain output.mp4
```

## 10-Bit and HDR Content

H.265 Main 10 profile is the standard for HDR content. Understanding this is important when working with HDR video datasets.

### HDR Formats in H.265

- **HDR10**: Static metadata, 10-bit, most common
- **HDR10+**: Dynamic metadata per scene
- **Dolby Vision**: Proprietary, dual-layer format

### Working with 10-Bit Video

```python
import av

container = av.open("hdr_video.mp4")

for frame in container.decode(video=0):
    # 10-bit video decoded to 16-bit array
    array = frame.to_ndarray(format="rgb48le")
    # Scale to 0-1 float
    array_float = array.astype(np.float32) / 65535.0
```

For ML training on HDR content:
1. Consider whether HDR information is relevant to your task
2. If not, tone-map to SDR for consistency
3. If yes, ensure your pipeline handles higher bit depths correctly

## Performance Considerations

### Decode Performance

H.265 decode is more computationally intensive than H.264:

| Platform | H.264 (1080p) | H.265 (1080p) | H.265 (4K) |
|----------|---------------|---------------|------------|
| CPU (1 core) | 200+ fps | 50-100 fps | 15-30 fps |
| GPU (NVDEC) | 1000+ fps | 500+ fps | 200+ fps |

For ML training pipelines, hardware decode is strongly recommended for H.265 content.

### Memory Requirements

H.265's larger CTUs and additional reference pictures increase memory requirements:
- More decoded picture buffer memory
- Larger intermediate buffers during decode
- 10-bit content doubles pixel storage

### Encoding Time

H.265 encoding is substantially slower:
- 3-10x slower than H.264 at equivalent quality
- Hardware encoding (NVENC, QSV) reduces this dramatically
- Consider encoding once and reusing for multiple training runs

## Comparison with Other Codecs

### H.265 vs H.264

| Aspect | H.265 | H.264 |
|--------|-------|-------|
| Compression | 40-50% smaller | Baseline |
| Encode time | 3-10x slower | Faster |
| Decode complexity | Higher | Lower |
| Hardware support | Widespread | Universal |
| 4K/8K suitability | Excellent | Marginal |
| HDR support | Native (Main 10) | Limited |

### H.265 vs VP9

| Aspect | H.265 | VP9 |
|--------|-------|-----|
| Compression | Similar | Similar |
| Royalties | Yes (complex) | Royalty-free |
| Hardware decode | Widespread | Good |
| Browser support | Limited (no Firefox) | Excellent |
| HDR support | Excellent | Good |

### H.265 vs AV1

| Aspect | H.265 | AV1 |
|--------|-------|-----|
| Compression | Baseline | 20-30% smaller |
| Encode speed | Slow | Very slow |
| Decode complexity | High | Higher |
| Royalties | Yes | Royalty-free |
| Hardware support | Widespread | Growing |
| Maturity | Established | Newer |

## Practical Recommendations

### For Dataset Preparation

1. **4K and Higher**: H.265 is appropriate when source content is 4K or higher resolution.

2. **Storage Efficiency**: For large video datasets, H.265's improved compression can significantly reduce storage costs.

3. **Decode Pipeline**: Ensure your training infrastructure has hardware H.265 decode support before committing to H.265 datasets.

4. **Re-encoding**: If converting from H.265 to another format, avoid quality loss by using high CRF values or lossless intermediate formats.

### For Model Outputs

1. **Compatibility Check**: Verify target platforms support H.265 playback before using for outputs.

2. **Hardware Encoding**: Use NVENC or similar for reasonable encoding speeds.

3. **Quality Settings**: CRF 24-28 typically provides good quality for visualization purposes.

### For Training Pipelines

1. **GPU Decode**: Always use hardware-accelerated decoding for H.265.

2. **Preprocessing**: Consider extracting frames to images for random access if decode overhead is significant.

3. **Mixed Datasets**: When combining H.264 and H.265 sources, normalize to consistent parameters during preprocessing.

## Licensing Considerations

H.265 has a complex patent licensing situation with multiple pools:
- MPEG LA HEVC Patent Pool
- HEVC Advance
- Velos Media

This complexity has driven adoption of royalty-free alternatives (VP9, AV1) for web delivery. For internal ML workflows, the licensing typically does not apply, but verify your organization's situation for any commercial distribution.
