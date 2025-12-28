# AV1 Video Codec

## Summary

AV1 (AOMedia Video 1) is the newest major video codec, developed by the Alliance for Open Media and finalized in 2018. It achieves approximately 30-50% better compression than H.265/VP9 while being completely royalty-free. AV1 represents the state of the art in video compression but comes with significantly higher encoding complexity. For ML practitioners, AV1 is increasingly relevant as it becomes the preferred codec for major streaming platforms and next-generation video content.

Key points to remember:

- Best-in-class compression efficiency (30-50% better than H.265/VP9)
- Royalty-free with backing from major tech companies
- Very slow encoding (10-100x slower than H.264)
- Hardware decode support growing rapidly
- Native HDR and wide color gamut support
- Container formats: MP4, WebM, MKV

Profiles:

- Main: 8-bit and 10-bit 4:2:0
- High: Adds 4:4:4 support
- Professional: Adds 12-bit support

Alliance for Open Media members include:
- Amazon, Apple, Arm, Cisco, Facebook, Google, Intel, Microsoft, Mozilla, Netflix, NVIDIA, Samsung

When to use AV1:

- Storage cost is a primary concern (best compression)
- Long-term archival of video content
- Streaming where encoding is done once, played many times
- Next-generation content pipelines
- When hardware decode is available

When to consider alternatives:

- Real-time or fast encoding needed (use H.264 or VP9)
- Hardware encode required (limited support)
- Older hardware decode compatibility (use H.264)
- Time-sensitive preprocessing pipelines

---

## AV1 Technical Innovations

AV1 builds on VP9's foundation while adding numerous improvements that collectively achieve its compression advantage.

### Advanced Block Structure

**Superblocks**: Increased maximum size to 128x128 pixels (vs 64x64 in VP9/H.265). Larger blocks improve efficiency for high-resolution content.

**Flexible Partitioning**: More partition options including 1:4 and 4:1 rectangular blocks, T-shaped and L-shaped partitions. This flexibility better matches content structure.

### Improved Prediction

**Intra Prediction**:
- 56 directional modes (vs 10 in VP9, 35 in H.265)
- Paeth predictor for smooth gradients
- Recursive filtering for sharper edges
- Chroma-from-luma prediction

**Inter Prediction**:
- Up to 7 reference frames
- Compound prediction with masks
- Warped motion compensation
- Overlapped block motion compensation (OBMC)

### Advanced In-Loop Filters

**Constrained Directional Enhancement Filter (CDEF)**: Removes ringing artifacts while preserving edges.

**Loop Restoration**: Three tools working together:
- Wiener filter for noise reduction
- Self-guided filter for detail preservation
- Switchable restoration at block level

**Film Grain Synthesis**: Models and removes grain before encoding, re-synthesizes during decode. Dramatically improves compression for grainy content.

### Transform and Entropy Coding

- Transform sizes from 4x4 to 64x64
- Multiple transform types per block
- Improved symbol coding with multi-symbol arithmetic coding

## AV1 in Machine Learning Workflows

### Decoding AV1 Video

AV1 decode support is still maturing in some libraries:

**PyAV with dav1d decoder (recommended)**:
```python
import av

# dav1d is the fastest software AV1 decoder
container = av.open("video.mp4")
for frame in container.decode(video=0):
    array = frame.to_ndarray(format="rgb24")
```

**FFmpeg command line**:
```bash
# Extract frames with dav1d decoder
ffmpeg -c:v libdav1d -i input.mp4 frames/%04d.png

# Hardware decode on supported systems
ffmpeg -hwaccel cuda -c:v av1_cuvid -i input.mp4 output.mp4
```

**Decord**:
```python
from decord import VideoReader, cpu

# Check if AV1 is supported in your build
vr = VideoReader("video_av1.mp4", ctx=cpu(0))
frames = vr.get_batch([0, 30, 60]).asnumpy()
```

### Hardware Decode Support

AV1 hardware decode is rapidly expanding:

| Platform | Support | Notes |
|----------|---------|-------|
| NVIDIA RTX 30/40 series | Full | NVDEC support |
| AMD RDNA2+ | Full | VCN decode |
| Intel Arc, 12th Gen+ | Full | Integrated decode |
| Apple M3+ | Full | ProRes not AV1 on M1/M2 |
| Apple M1/M2 | CPU only | Fast software decode |
| Older GPUs | CPU only | Use dav1d |

### Encoding AV1 Video

AV1 encoding is computationally intensive. Multiple encoders exist with different trade-offs:

**libaom (reference encoder)**:
```bash
# Very slow but best quality
ffmpeg -i input.mp4 -c:v libaom-av1 -crf 30 -cpu-used 4 output.mp4

# Two-pass for better quality
ffmpeg -i input.mp4 -c:v libaom-av1 -b:v 2M -pass 1 -f null /dev/null
ffmpeg -i input.mp4 -c:v libaom-av1 -b:v 2M -pass 2 output.mp4
```

**SVT-AV1 (fastest production encoder)**:
```bash
# Much faster, good quality
ffmpeg -i input.mp4 -c:v libsvtav1 -crf 30 -preset 6 output.mp4
```

**rav1e (Rust implementation)**:
```bash
# Good balance of speed and quality
ffmpeg -i input.mp4 -c:v librav1e -qp 80 -speed 6 output.mp4
```

**Python encoding with SVT-AV1**:
```python
import subprocess

def encode_av1(input_path, output_path, crf=30, preset=6):
    """Encode video to AV1 using SVT-AV1."""
    subprocess.run([
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libsvtav1',
        '-crf', str(crf),
        '-preset', str(preset),
        '-pix_fmt', 'yuv420p',
        output_path
    ], check=True)
```

### Hardware Encoding

AV1 hardware encoding is limited but growing:

**NVIDIA NVENC (RTX 40 series)**:
```bash
ffmpeg -i input.mp4 -c:v av1_nvenc -cq 30 -preset p4 output.mp4
```

**Intel Arc (QSV)**:
```bash
ffmpeg -i input.mp4 -c:v av1_qsv -global_quality 25 output.mp4
```

Hardware encoding is significantly faster but may not achieve the same compression as software encoders.

## Encoding Parameters

### Encoder Comparison

| Encoder | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| libaom | Very slow | Best | Archival, quality-critical |
| SVT-AV1 | Moderate | Very good | Production, streaming |
| rav1e | Moderate | Good | General purpose |
| NVENC | Fast | Good | Real-time, batch processing |

### SVT-AV1 Presets

SVT-AV1 is recommended for most production use:

| Preset | Relative Speed | Quality | Notes |
|--------|----------------|---------|-------|
| 0-2 | Very slow | Best | Research, archival |
| 3-5 | Slow | Excellent | Quality-focused |
| 6-8 | Moderate | Very good | Balanced |
| 9-11 | Fast | Good | Time-constrained |
| 12-13 | Very fast | Lower | Real-time capable |

### Quality Settings

AV1 uses CRF (Constant Rate Factor) for quality:

| CRF Value | Quality Level | Typical Use |
|-----------|---------------|-------------|
| 18-23 | Very high | Archival, source |
| 24-30 | High | General use |
| 31-40 | Medium | Web delivery |
| 41-50 | Lower | Bandwidth-limited |

AV1's CRF scale is similar to x264/x265, but AV1 achieves the same visual quality at higher CRF values due to better compression.

## Performance Considerations

### Decode Performance

AV1 software decode is viable thanks to highly optimized decoders:

**dav1d performance (1080p)**:
- Single-threaded: 50-100 fps
- Multi-threaded (4 cores): 200+ fps
- Multi-threaded (8 cores): 400+ fps

Hardware decode on supported GPUs achieves 500+ fps at 1080p.

### Encode Performance

AV1 encoding is the primary challenge. Approximate times for 1 minute of 1080p video:

| Encoder | Preset | Encode Time |
|---------|--------|-------------|
| libaom | cpu-used 4 | 30-60 minutes |
| SVT-AV1 | preset 6 | 5-10 minutes |
| SVT-AV1 | preset 10 | 1-2 minutes |
| NVENC | preset p4 | 10-30 seconds |

For large datasets, encoding time is a significant consideration. Pre-encoding all training data to AV1 may not be practical.

### Memory Usage

AV1 decode requires more memory than older codecs:
- 1080p: 200-400 MB
- 4K: 800 MB - 1.5 GB

This is due to larger reference buffers and complex filtering.

## AV1 for ML Applications

### Advantages for Training Data

**Storage Efficiency**: A dataset of 100,000 videos at 30% better compression translates to significant storage savings.

**Quality Preservation**: At a given file size, AV1 preserves more visual detail than alternatives.

**Future-Proofing**: As the newest major codec, AV1 will remain relevant for years.

### Challenges for Training Pipelines

**Encode Time**: Pre-processing source videos to AV1 takes substantially longer than H.264.

**Hardware Requirements**: Fast AV1 decode requires recent hardware.

**Library Support**: Some older ML video libraries may not support AV1.

### Practical Strategies

1. **Decode-Heavy Pipelines**: If videos are decoded many times during training, AV1's smaller size reduces I/O.

2. **Storage-Constrained**: When storage is expensive, AV1's compression advantage matters.

3. **Hybrid Approach**: Store master copies in AV1, extract frames or convert to H.264 for training.

4. **Incremental Adoption**: Use AV1 for new content while maintaining existing H.264/VP9.

## Comparison with Other Codecs

### AV1 vs H.264

| Aspect | AV1 | H.264 |
|--------|-----|-------|
| Compression | 50-60% smaller | Baseline |
| Encode speed | Very slow | Fast |
| Hardware decode | Growing | Universal |
| Royalties | Free | Yes |

### AV1 vs H.265

| Aspect | AV1 | H.265 |
|--------|-----|-------|
| Compression | 20-30% smaller | Baseline |
| Encode speed | Slower | Slow |
| Royalties | Free | Complex |
| Hardware support | Growing | Widespread |

### AV1 vs VP9

| Aspect | AV1 | VP9 |
|--------|-----|-----|
| Compression | 30% smaller | Baseline |
| Encode speed | Much slower | Moderate |
| Hardware decode | Growing | Widespread |
| Maturity | Newer | Established |

## Film Grain Handling

AV1's film grain synthesis is particularly relevant for ML:

### How It Works

1. Encoder analyzes and models grain patterns
2. Grain is removed before compression
3. Parameters stored in bitstream
4. Decoder synthesizes grain during playback

### ML Implications

**Training Data**: If training on content with natural grain, be aware that AV1 may reconstruct grain differently than the original.

**Denoising Models**: AV1's grain removal could affect training for denoising tasks.

**Feature Extraction**: Grain-related features may differ between original and AV1-compressed video.

You can disable grain synthesis:
```bash
ffmpeg -i input.mp4 -c:v libaom-av1 -denoise-noise-level 0 output.mp4
```

## Streaming and Adaptive Bitrate

AV1 is well-suited for adaptive streaming:

- Defined in CMAF (Common Media Application Format)
- Supported by DASH and HLS
- Better quality per bitrate at all quality levels

Major streaming platforms using or transitioning to AV1:
- Netflix (for 4K content)
- YouTube (growing adoption)
- Twitch (for VOD)
- Disney+ (for new content)

## Practical Recommendations

### For Dataset Preparation

1. **Evaluate Trade-offs**: Calculate whether AV1's storage savings justify the encoding time.

2. **Use SVT-AV1**: Best balance of speed and quality for production use.

3. **Choose Appropriate Preset**: Preset 6-8 for quality, 10-12 for speed.

4. **Verify Decode Support**: Ensure training infrastructure can decode AV1.

### For Model Outputs

1. **Consider Audience**: Use AV1 only if target platforms support it.

2. **Hardware Encode**: Use NVENC for acceptable quality at fast speeds.

3. **Fallback Format**: Provide H.264 alternative for compatibility.

### For Training Pipelines

1. **Hardware Decode**: Use GPU AV1 decode where available.

2. **Software Fallback**: dav1d is fast enough for moderate-scale pipelines.

3. **Frame Caching**: Extract frames once for repeated training runs.

4. **Mixed Formats**: Accept multiple formats in data loaders.

## Future Outlook

AV1 adoption is accelerating:
- Hardware support in most new devices
- Growing software encoder efficiency
- Increasing streaming platform adoption
- Foundation for future video AI applications

For ML practitioners, AV1 will become increasingly important as more content is produced and distributed in this format.
