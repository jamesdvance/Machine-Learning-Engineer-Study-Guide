# Megatron-LM

## Summary

Megatron-LM is NVIDIA's framework for training large transformer models, designed for maximum scale and efficiency. It pioneered efficient 3D parallelism combining data, tensor, and pipeline parallelism, enabling training of models with hundreds of billions of parameters. Megatron-LM is optimized for NVIDIA hardware and is the foundation for training many state-of-the-art language models.

Key points to remember:

- Designed for training very large transformer models (100B+ parameters)
- Implements efficient 3D parallelism: data + tensor + pipeline
- Sequence parallelism extends tensor parallel to reduce activation memory
- Heavily optimized for NVIDIA GPUs (NVLink, InfiniBand)
- More complex setup than other frameworks but maximum efficiency
- Often combined with DeepSpeed for ZeRO optimizations
- Foundation for NVIDIA NeMo framework
- Primarily focused on transformer architectures

## Architecture

### 3D Parallelism

Megatron-LM combines three parallelism strategies:

**Tensor Parallelism (TP)**: Within a node using NVLink
- Splits attention heads and MLP across GPUs
- Requires high-bandwidth interconnect
- Typically TP=8 within a DGX node

**Pipeline Parallelism (PP)**: Across node groups
- Splits layers into stages
- Uses interleaved 1F1B schedule
- Balances bubble overhead vs memory

**Data Parallelism (DP)**: Across pipeline replicas
- Each pipeline replica processes different data
- Gradient synchronization between replicas
- Scales to many nodes

### Example Configuration

For 128 GPUs (16 nodes with 8 GPUs each):
```
TP = 8  (within node)
PP = 4  (4 stages across node groups)
DP = 4  (4 parallel pipelines)

128 = 8 x 4 x 4
```

## Installation

```bash
# Clone repository
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install dependencies
pip install -r requirements.txt

# Install Apex (recommended)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Basic Usage

### Training GPT

```bash
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers 48 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-iters 100000 \
    --lr 1e-4 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --fp16 \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints
```

### Multi-Node Launch

```bash
# Using torchrun
torchrun --nnodes=16 --nproc_per_node=8 \
    --rdzv_id=123 --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    ...

# Using Slurm
srun --nodes=16 --ntasks-per-node=8 --gpus-per-node=8 \
    python pretrain_gpt.py ...
```

## Tensor Parallelism Implementation

### Column-Parallel Linear

```python
# Megatron splits attention QKV and first MLP projection column-wise
# Input: [batch, seq, hidden]
# Weight: [hidden, output] split to [hidden, output/TP]
# Output: [batch, seq, output/TP] (different on each rank)
```

### Row-Parallel Linear

```python
# Megatron splits attention output and second MLP projection row-wise
# Input: [batch, seq, hidden/TP] (different on each rank)
# Weight: [hidden/TP, output]
# Output: [batch, seq, output] (all-reduced)
```

### Attention

```python
# Each TP rank handles hidden_size/TP attention heads
# QKV computed locally
# Attention output: row-parallel (all-reduce)
```

### MLP

```python
# First linear: column-parallel (no communication)
# Activation: GeLU (local)
# Second linear: row-parallel (all-reduce)
```

Total: 2 all-reduce operations per transformer layer.

## Pipeline Parallelism

### Interleaved Schedule

Megatron uses interleaved 1F1B for better pipeline efficiency:

```python
# Virtual pipeline stages > physical pipeline stages
# GPU 0 handles stages 0, 4
# GPU 1 handles stages 1, 5
# GPU 2 handles stages 2, 6
# GPU 3 handles stages 3, 7

# Reduces bubble by increasing virtual depth
```

Configuration:
```bash
--pipeline-model-parallel-size 4 \
--num-layers-per-virtual-pipeline-stage 2
```

### Pipeline Communication

```python
# Activations sent between stages
# Gradients sent back during backward
# Uses P2P send/recv operations
```

## Sequence Parallelism

Extends tensor parallelism to sequence dimension:

```python
# Between tensor-parallel regions:
# - LayerNorm and Dropout are sequence-parallel
# - Activations partitioned along sequence

# Benefits:
# - Reduces activation memory per GPU
# - Enables longer sequences
```

Enable:
```bash
--sequence-parallel
```

## Data Loading

### Indexed Dataset

Megatron uses memory-mapped indexed datasets:

```bash
# Preprocess data
python tools/preprocess_data.py \
    --input /path/to/data.json \
    --output-prefix /path/to/output \
    --vocab /path/to/vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /path/to/merges.txt \
    --workers 32
```

### Blended Dataset

```bash
# Mix multiple datasets with weights
--data-path "0.7 /path/to/dataset1 0.3 /path/to/dataset2"
```

## Checkpointing

### Save Configuration

```bash
--save /path/to/checkpoints \
--save-interval 1000
```

### Load and Resume

```bash
--load /path/to/checkpoints \
--finetune  # Start fresh optimizer state
# Or without --finetune to resume training
```

### Distributed Checkpointing

Megatron saves sharded checkpoints:
```
checkpoints/
  iter_001000/
    mp_rank_00/
    mp_rank_01/
    ...
```

## Integration with DeepSpeed

### Megatron-DeepSpeed

Combined for ZeRO optimizations:

```bash
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --deepspeed \
    --deepspeed_config ds_config.json \
    ...
```

DeepSpeed config:
```json
{
    "zero_optimization": {
        "stage": 1
    },
    "bf16": {
        "enabled": true
    }
}
```

### Benefits of Combination

- Megatron: Efficient 3D parallelism
- DeepSpeed: ZeRO memory optimization, additional features
- Together: Maximum scale and efficiency

## Model Architectures

### GPT

```bash
python pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    ...
```

### BERT

```bash
python pretrain_bert.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    ...
```

### T5

```bash
python pretrain_t5.py \
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 768 \
    ...
```

## Performance Optimization

### Flash Attention

```bash
--use-flash-attn
```

### Fused Kernels

```bash
--fused-bias-fc
--fused-bias-fc-loss-head
```

### Selective Activation Recomputation

```bash
--recompute-activations
--recompute-granularity selective
```

### Distributed Optimizer

```bash
--use-distributed-optimizer
```

Shards optimizer state across data-parallel ranks.

## Typical Configurations

### 7B Model (8 GPUs)

```bash
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 1 \
--num-layers 32 \
--hidden-size 4096 \
--num-attention-heads 32 \
--micro-batch-size 4 \
--global-batch-size 256
```

### 70B Model (64 GPUs)

```bash
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 4 \
--num-layers 80 \
--hidden-size 8192 \
--num-attention-heads 64 \
--micro-batch-size 1 \
--global-batch-size 1024 \
--sequence-parallel
```

### 175B Model (512 GPUs)

```bash
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 16 \
--num-layers 96 \
--hidden-size 12288 \
--num-attention-heads 96 \
--micro-batch-size 1 \
--global-batch-size 2048 \
--sequence-parallel \
--use-flash-attn
```

## Debugging

### Common Issues

**Hang during initialization**:
- Check NCCL environment variables
- Verify network connectivity
- Ensure consistent parallelism config

**Out of memory**:
- Reduce micro-batch-size
- Enable activation checkpointing
- Increase pipeline parallel size

**Poor scaling**:
- Check tensor parallel within NVLink domain
- Verify network bandwidth
- Profile communication vs compute

### Logging

```bash
--log-interval 1 \
--tensorboard-dir /path/to/tb_logs \
--tensorboard-log-interval 10
```

### Profiling

```bash
--profile
--profile-step-start 10
--profile-step-end 20
```

## Best Practices

1. **Tensor parallel = NVLink GPUs**: Keep TP within single node
2. **Pipeline stages = layers / 4-8**: Balance bubble vs memory
3. **Use sequence parallelism**: For long sequences and memory efficiency
4. **Enable Flash Attention**: For speed and memory
5. **Profile before scaling**: Understand bottlenecks
6. **Start from known configs**: Use NVIDIA's published configurations
7. **Combine with DeepSpeed**: For additional optimizations
