# Video Formats for Machine Learning

## Summary

Video codecs significantly impact ML workflows through their effects on storage costs, decode performance, and visual quality. This chapter compares the four major video codecs and provides guidance for choosing the right format for your use case.

Quick reference for codec selection:

| Codec | Compression | Encode Speed | Decode HW Support | Royalty-Free |
|-------|-------------|--------------|-------------------|--------------|
| H.264 | Baseline | Fast | Universal | No |
| H.265 | +40-50% | Slow | Widespread | No |
| VP9 | +30-50% | Moderate | Good | Yes |
| AV1 | +50-60% | Very slow | Growing | Yes |

Key decision factors:

- **Maximum compatibility?** Use H.264
- **Best compression?** Use AV1 (if encode time permits)
- **Royalty-free requirement?** Use VP9 or AV1
- **4K/HDR content?** Use H.265 or AV1
- **Real-time encoding?** Use H.264 (or VP9 with hardware)
- **Web delivery?** Use VP9 (YouTube standard) or AV1

Common pitfalls:

- Underestimating AV1/H.265 encode time for large datasets
- Assuming hardware decode support without verification
- Ignoring compression artifacts that may affect model training
- Not accounting for GOP structure when random frame access is needed

---

## Codec Comparison

### Compression Efficiency

Compression efficiency determines storage requirements and bandwidth. Higher efficiency means smaller files at the same quality.

Approximate bitrate savings at equivalent quality (relative to H.264):

| Resolution | H.265 | VP9 | AV1 |
|------------|-------|-----|-----|
| 720p | 30-40% | 25-35% | 40-50% |
| 1080p | 35-45% | 30-40% | 45-55% |
| 4K | 40-50% | 35-45% | 50-60% |

The advantage of newer codecs increases at higher resolutions because their larger block structures (64x64 for H.265/VP9, 128x128 for AV1) provide more benefit with more pixels.

### Encode Performance

Encoding speed varies dramatically between codecs:

| Codec | Relative Encode Time (1080p) |
|-------|------------------------------|
| H.264 (x264, medium) | 1x (baseline) |
| H.265 (x265, medium) | 3-5x |
| VP9 (libvpx, cpu-used 4) | 5-8x |
| AV1 (libaom, cpu-used 4) | 30-50x |
| AV1 (SVT-AV1, preset 6) | 8-15x |

For large-scale dataset preparation, these differences compound. Processing 100,000 videos:
- H.264: Days
- H.265: Weeks
- AV1 (libaom): Months
- AV1 (SVT-AV1): Weeks

Hardware encoders (NVENC, QSV) dramatically reduce encoding time but may not achieve the same compression as software encoders.

### Decode Performance

Decode performance affects training throughput:

**CPU decode (1080p, multi-threaded):**
| Codec | Approximate FPS |
|-------|-----------------|
| H.264 | 400-600 |
| H.265 | 200-400 |
| VP9 | 300-500 |
| AV1 | 200-400 |

**GPU decode (1080p):**
| Codec | Approximate FPS |
|-------|-----------------|
| H.264 | 1000+ |
| H.265 | 800+ |
| VP9 | 700+ |
| AV1 | 500+ |

For training pipelines, hardware decode prevents data loading from becoming a bottleneck.

### Hardware Support

Hardware decode availability affects deployment flexibility:

| Codec | Desktop GPU | Mobile | Browsers | Embedded |
|-------|-------------|--------|----------|----------|
| H.264 | Universal | Universal | Universal | Universal |
| H.265 | Widespread | Widespread | Limited | Good |
| VP9 | Good | Good | Excellent | Limited |
| AV1 | Growing | Growing | Good | Limited |

H.264 remains the safest choice for maximum compatibility.

## Video in ML Workflows

### Loading Video for Training

Common approaches for loading video frames:

**OpenCV (simple, widely compatible)**:
```python
import cv2

def load_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames
```

**Decord (ML-optimized, GPU support)**:
```python
from decord import VideoReader, cpu, gpu

# CPU decode
vr = VideoReader("video.mp4", ctx=cpu(0))

# GPU decode
vr = VideoReader("video.mp4", ctx=gpu(0))

# Random access
frames = vr.get_batch([0, 30, 60, 90]).asnumpy()

# Uniform sampling
indices = np.linspace(0, len(vr)-1, 16, dtype=int)
frames = vr.get_batch(indices).asnumpy()
```

**PyAV (FFmpeg bindings, full control)**:
```python
import av

def decode_video(path):
    container = av.open(path)
    for frame in container.decode(video=0):
        yield frame.to_ndarray(format="rgb24")
```

### Temporal Sampling Strategies

For video understanding tasks, frame sampling matters:

**Uniform sampling**: Equal spacing across video duration
```python
indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
```

**Random sampling**: Random frames for data augmentation
```python
indices = np.random.choice(total_frames, num_samples, replace=False)
indices.sort()  # Sort for efficient sequential access
```

**Dense sampling**: Consecutive frames for motion analysis
```python
start = np.random.randint(0, total_frames - clip_length)
indices = np.arange(start, start + clip_length)
```

**Strided sampling**: Every Nth frame
```python
indices = np.arange(0, total_frames, stride)[:num_samples]
```

### GOP Structure and Random Access

Video codecs use Groups of Pictures (GOPs) with I-frames (complete), P-frames (predicted), and B-frames (bidirectional). Random access requires decoding from the nearest I-frame.

For training requiring random frame access:
- Use shorter GOP lengths (fewer frames between I-frames)
- Accept decode overhead from sequential access
- Extract all frames to images once, then load images during training
- Use formats with all I-frames (higher storage but instant access)

### Encoding Model Outputs

When saving video predictions or visualizations:

```python
import subprocess

def save_video(frames, output_path, fps=30, codec='h264', crf=23):
    height, width = frames[0].shape[:2]

    codec_map = {
        'h264': ['libx264', '-crf', str(crf)],
        'h265': ['libx265', '-crf', str(crf)],
        'vp9': ['libvpx-vp9', '-crf', str(crf), '-b:v', '0'],
        'av1': ['libsvtav1', '-crf', str(crf)],
    }

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', codec_map[codec][0],
        *codec_map[codec][1:],
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
```

## Quality Considerations for ML

### Compression Artifacts

Video compression introduces artifacts that may affect model training:

**Blocking**: Visible block boundaries, especially at low bitrates. May cause models to learn artificial edge patterns.

**Ringing**: Halos around high-contrast edges. Can affect edge detection and segmentation tasks.

**Color bleeding**: Chroma information spreading across sharp boundaries due to chroma subsampling.

**Temporal artifacts**: Flickering, ghosting, or inconsistent quality between frames.

### Quality Settings for Training Data

Recommended quality settings for training data:

| Codec | CRF/Quality | Notes |
|-------|-------------|-------|
| H.264 | CRF 18-23 | Good balance for most tasks |
| H.265 | CRF 22-28 | Equivalent visual quality |
| VP9 | CRF 24-32 | Different scale than H.264 |
| AV1 | CRF 25-35 | Best quality per bit |

Higher quality (lower CRF) is recommended for:
- Fine-grained visual tasks (OCR, small object detection)
- Training source models
- Archival copies

Lower quality is acceptable for:
- Large-scale pretraining
- Coarse visual tasks
- Data augmentation variants

### Augmentation with Compression

Compression can be used as data augmentation to improve robustness:

```python
import cv2
import numpy as np

def jpeg_augment(image, quality_range=(30, 90)):
    """Apply JPEG compression as augmentation."""
    quality = np.random.randint(*quality_range)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

# For video, consider re-encoding at varied quality levels
```

## Dataset Preparation Strategies

### Format Standardization

For consistent training, standardize video properties:

1. **Resolution**: Resize to consistent dimensions
2. **Frame rate**: Resample to common FPS
3. **Codec**: Transcode to single format
4. **Quality**: Apply consistent compression settings

```bash
# Example: Standardize to 720p, 30fps, H.264
ffmpeg -i input.mp4 \
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1" \
    -r 30 \
    -c:v libx264 -crf 23 \
    output.mp4
```

### Storage Optimization

Balance storage costs against processing overhead:

| Strategy | Storage | Processing | Use Case |
|----------|---------|------------|----------|
| Original formats | Varies | Low | Diverse sources |
| Standardized H.264 | Medium | Medium | Balanced approach |
| Standardized AV1 | Low | High encode | Storage-constrained |
| Extracted frames | High | Low | Random access heavy |

### Preprocessing Pipeline

Typical video dataset preparation:

1. **Download/collect** source videos
2. **Validate** format, resolution, duration
3. **Transcode** to consistent format/quality
4. **Extract** metadata (duration, fps, resolution)
5. **Split** into train/val/test
6. **Generate** frame indices or clips for training

## Containers vs Codecs

Codecs (H.264, VP9) define compression; containers (MP4, WebM) define packaging.

| Container | Common Codecs | Notes |
|-----------|---------------|-------|
| MP4 | H.264, H.265, AV1 | Most compatible |
| WebM | VP8, VP9, AV1 | Web-focused |
| MKV | Any | Most flexible |
| MOV | H.264, H.265, ProRes | Apple ecosystem |
| AVI | Legacy codecs | Outdated |

For ML workflows, MP4 or MKV are typically the best choices for flexibility and tool support.

## Practical Recommendations

### For Training Datasets

1. **Verify decode support** before committing to a codec
2. **Use hardware decode** for large-scale processing
3. **Standardize format** during data ingestion
4. **Balance quality vs size** based on task requirements
5. **Consider frame extraction** for random-access-heavy workloads

### For Inference/Production

1. **Use H.264** for maximum compatibility
2. **Use VP9/AV1** for web delivery
3. **Match input format** to expected production data
4. **Validate** model performance across compression levels

### For Research

1. **Document codec and settings** for reproducibility
2. **Use high quality** to minimize compression confounds
3. **Test across quality levels** to understand robustness
4. **Publish preprocessing scripts** with datasets

## Common Issues and Solutions

**Slow data loading**: Enable hardware decode, use Decord instead of OpenCV, prefetch data.

**Random access overhead**: Extract frames to images, use shorter GOPs, or accept sequential decode.

**Inconsistent frame counts**: Some tools count frames differently. Use FFprobe for accurate counts.

**Color space issues**: Videos use YUV internally. Verify RGB conversion is handled correctly.

**Variable frame rate**: Some sources have VFR. Consider resampling to constant FPS during preprocessing.
