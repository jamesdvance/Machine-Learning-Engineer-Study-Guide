# TensorRT

## Summary

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It analyzes trained models, applies optimizations like layer fusion and precision calibration, and generates optimized inference engines that achieve maximum performance on NVIDIA GPUs. TensorRT is the gold standard for production inference on NVIDIA hardware.

Key points to remember:

- 2-5x faster inference compared to framework-native execution
- Supports FP32, FP16, INT8, and FP8 precision
- Performs layer fusion, kernel auto-tuning, and memory optimization
- Works with ONNX, TensorFlow, and PyTorch models
- Generates hardware-specific optimized engines
- Engines are GPU-architecture specific (not portable)
- Best for static shapes and batch inference
- Dynamic shapes supported with optimization profiles

## How TensorRT Works

### Optimization Pipeline

```
Trained Model (ONNX/TF/PyTorch)
    |
    v
Parser (Import model)
    |
    v
Builder (Optimize network)
    - Layer fusion
    - Kernel selection
    - Precision calibration
    - Memory optimization
    |
    v
Engine (Serialized runtime)
    |
    v
Runtime (Execute inference)
```

### Key Optimizations

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| Layer fusion | Combine Conv+BN+ReLU | Fewer kernel launches |
| Kernel auto-tuning | Select best CUDA kernel | Hardware-optimized |
| Precision reduction | FP16/INT8 execution | 2-4x speedup |
| Memory optimization | Tensor memory reuse | Lower memory |
| Dynamic tensor memory | Minimal allocations | Reduced latency |

## Model Conversion

### From ONNX

```python
import tensorrt as trt

# Create builder and network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# Parse ONNX model
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

engine = builder.build_serialized_network(network, config)

# Save engine
with open("model.engine", "wb") as f:
    f.write(engine)
```

### From PyTorch

```python
import torch
import torch_tensorrt

model = MyModel().eval().cuda()
example_input = torch.randn(1, 3, 224, 224).cuda()

# Method 1: torch_tensorrt.compile
compiled = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16}
)

# Method 2: Export to ONNX first
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}}
)
# Then use ONNX pipeline above
```

### From TensorFlow

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# SavedModel to TensorRT
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="saved_model",
    precision_mode=trt.TrtPrecisionMode.FP16
)

converter.convert()
converter.save("tensorrt_model")
```

## Precision Modes

### FP32 (Default)

```python
# Full precision, baseline accuracy
config = builder.create_builder_config()
# FP32 is default, no special flag needed
```

### FP16 (Half Precision)

```python
# 2x faster, minimal accuracy loss
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
```

### INT8 (Quantized)

```python
# 4x faster, requires calibration
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

# Calibration dataset
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.batch_size = 32

        # Allocate device memory
        self.device_input = cuda.mem_alloc(
            self.batch_size * 3 * 224 * 224 * 4
        )

    def get_batch(self, names):
        try:
            batch = next(self.dataloader)
            cuda.memcpy_htod(self.device_input, batch.numpy())
            return [int(self.device_input)]
        except StopIteration:
            return None

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)

config.int8_calibrator = Calibrator(calibration_loader)
```

### Precision Comparison

| Precision | Speed | Memory | Accuracy |
|-----------|-------|--------|----------|
| FP32 | 1x | 1x | Baseline |
| FP16 | 2x | 0.5x | ~Same |
| INT8 | 4x | 0.25x | Slight drop |
| FP8 | 2-3x | 0.5x | ~Same (H100+) |

## Dynamic Shapes

### Optimization Profiles

```python
# Define min/opt/max shapes for dynamic dimensions
profile = builder.create_optimization_profile()

profile.set_shape(
    "input",
    min=(1, 3, 224, 224),    # Minimum shape
    opt=(32, 3, 224, 224),   # Optimal shape (tuned for)
    max=(64, 3, 224, 224)    # Maximum shape
)

config.add_optimization_profile(profile)
```

### Multiple Profiles

```python
# Different profiles for different use cases
# Profile 1: Small batches (latency-optimized)
profile1 = builder.create_optimization_profile()
profile1.set_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
config.add_optimization_profile(profile1)

# Profile 2: Large batches (throughput-optimized)
profile2 = builder.create_optimization_profile()
profile2.set_shape("input", (16, 3, 224, 224), (32, 3, 224, 224), (64, 3, 224, 224))
config.add_optimization_profile(profile2)
```

## Inference Execution

### Basic Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load engine
with open("model.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Allocate buffers
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

# Create stream
stream = cuda.Stream()

# Run inference
def infer(input_data):
    # Copy input to device
    cuda.memcpy_htod_async(d_input, input_data, stream)

    # Execute
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle
    )

    # Copy output to host
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)

    stream.synchronize()
    return output
```

### Batch Inference

```python
def batch_infer(inputs):
    batch_size = len(inputs)

    # Set dynamic shape
    context.set_binding_shape(0, (batch_size, 3, 224, 224))

    # Allocate for batch
    d_input = cuda.mem_alloc(batch_size * 3 * 224 * 224 * 4)
    d_output = cuda.mem_alloc(batch_size * 1000 * 4)

    # Stack inputs
    batch = np.stack(inputs)

    # Copy and execute
    cuda.memcpy_htod(d_input, batch)
    context.execute_v2(bindings=[int(d_input), int(d_output)])

    output = np.empty((batch_size, 1000), dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

## Layer Fusion

### Common Fusion Patterns

```
Conv + BatchNorm + ReLU -> Single kernel
MatMul + Add + ReLU -> Single kernel
Conv + Add (residual) -> Fused residual
LayerNorm components -> Single kernel
```

### Viewing Fusion

```python
# Enable verbose logging to see fusions
logger = trt.Logger(trt.Logger.VERBOSE)

# Or use trtexec
# trtexec --onnx=model.onnx --verbose
```

### Plugin Layers

```python
# For unsupported operations, use plugins
class MyPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "MyCustomOp"
        self.version = "1"

    def create_plugin(self, name, fc):
        return MyPlugin()

    def deserialize_plugin(self, name, data):
        return MyPlugin()

# Register plugin
trt.init_libnvinfer_plugins(logger, "")
```

## Performance Optimization

### Workspace Size

```python
# More workspace = better kernel selection
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    4 << 30  # 4GB
)
```

### Builder Optimization Level

```python
# Higher level = longer build, better performance
config.builder_optimization_level = 5  # 0-5
```

### Timing Cache

```python
# Cache kernel timing for faster rebuilds
timing_cache = config.create_timing_cache(b"")

# After building, save cache
cache_data = config.get_timing_cache().serialize()
with open("timing.cache", "wb") as f:
    f.write(cache_data)

# Load cache for next build
with open("timing.cache", "rb") as f:
    timing_cache = config.create_timing_cache(f.read())
config.set_timing_cache(timing_cache, ignore_mismatch=False)
```

## Benchmarking

### Using trtexec

```bash
# Basic benchmark
trtexec --onnx=model.onnx

# With FP16
trtexec --onnx=model.onnx --fp16

# With specific batch size
trtexec --onnx=model.onnx --shapes=input:32x3x224x224

# Detailed timing
trtexec --onnx=model.onnx --dumpProfile --separateProfileRun
```

### Python Benchmarking

```python
import time

def benchmark(context, input_data, iterations=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        infer(input_data)

    # Benchmark
    cuda.Context.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        infer(input_data)

    cuda.Context.synchronize()
    elapsed = time.perf_counter() - start

    return {
        'total_time': elapsed,
        'avg_latency': elapsed / iterations * 1000,  # ms
        'throughput': iterations / elapsed  # inferences/sec
    }
```

## Integration Patterns

### With Triton Inference Server

```python
# config.pbtxt
name: "my_model"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

### With PyTorch

```python
import torch
import torch_tensorrt

# Compile with torch_tensorrt
model = torch_tensorrt.compile(
    torch_model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 3, 224, 224],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30
)

# Use like normal PyTorch model
output = model(input_tensor)
```

## Debugging

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Build failure | Unsupported ops | Use plugin or simplify |
| Accuracy drop | INT8 quantization | Better calibration data |
| Slow build | Large model | Reduce workspace, cache |
| OOM at runtime | Large batch | Reduce batch, check shapes |

### Verbose Logging

```python
# Maximum verbosity
logger = trt.Logger(trt.Logger.VERBOSE)

# Or use environment variable
import os
os.environ['TRT_SEVERITY'] = 'VERBOSE'
```

### Validating Accuracy

```python
# Compare TensorRT vs original
original_output = pytorch_model(input_tensor)
trt_output = trt_model(input_tensor)

# Check difference
max_diff = (original_output - trt_output).abs().max()
print(f"Max difference: {max_diff}")

# Should be small (< 1e-3 for FP16, < 1e-1 for INT8)
```

## Best Practices

1. **Export to ONNX first**: Cleaner conversion path
2. **Use FP16 by default**: Best speed/accuracy trade-off
3. **Profile before optimizing**: Use trtexec for baselines
4. **Cache timing data**: Faster subsequent builds
5. **Match precision to needs**: INT8 only when accuracy permits
6. **Use static shapes when possible**: Better optimization
7. **Warm up before benchmarking**: First inference is slow
8. **Version match**: Engine only works on same GPU architecture

## GPU Architecture Compatibility

| GPU Architecture | TensorRT Version | Notes |
|------------------|------------------|-------|
| Ampere (A100, A10) | 8.x+ | FP8 on H100+ |
| Turing (T4, RTX 20xx) | 7.x+ | Good INT8 support |
| Volta (V100) | 6.x+ | Tensor cores |
| Pascal (P100) | 5.x+ | No tensor cores |

Engines are NOT portable across architectures. Rebuild for each target GPU.
