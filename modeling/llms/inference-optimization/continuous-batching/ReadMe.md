# Continuous Batching

## Summary

Continuous batching (also called iteration-level batching or in-flight batching) dramatically improves LLM serving throughput by dynamically adding and removing requests from batches at each generation step. Unlike static batching where all requests must complete before new ones begin, continuous batching allows finished requests to leave and new requests to join mid-generation. This eliminates the "convoy effect" where short requests wait for long ones, achieving 10-20x higher throughput in production systems like vLLM, TensorRT-LLM, and Triton.

Key points to remember:

- Dynamic batching: Requests join/leave at each generation step
- Eliminates padding waste: No waiting for longest sequence
- Higher throughput: 10-20x improvement over static batching
- Memory management: Requires efficient KV-cache handling
- Request scheduling: Preemption and priority queues
- Production systems: vLLM, TensorRT-LLM, text-generation-inference
- Prefill vs decode: Different compute characteristics to balance

## The Static Batching Problem

### Why Static Batching Wastes Resources

```
Static Batching (Traditional):
Request A: [prompt====][gen][gen][gen][gen][gen]
Request B: [prompt][gen][gen][DONE][pad][pad][pad]
Request C: [prompt===][gen][DONE][pad][pad][pad][pad]
                                   ^^^^^^^^^^^^
                                   Wasted compute
All requests start together, short ones wait for longest

Timeline:
t=0: All 3 requests start
t=3: B finishes, but waits for A
t=4: C finishes, but waits for A
t=6: A finishes, batch completes
t=7: New batch can finally start

Throughput: 3 requests / 7 time units = 0.43 req/unit
```

### Continuous Batching Solution

```
Continuous Batching:
Request A: [prompt====][gen][gen][gen][gen][gen]
Request B: [prompt][gen][gen][DONE]
Request D:              [prompt==][gen][gen][gen]
Request C: [prompt===][gen][DONE]
Request E:                  [prompt][gen][gen][gen]

Timeline:
t=0: A, B, C start
t=3: B finishes ’ D joins immediately
t=4: C finishes ’ E joins immediately
t=6: A, D finish
t=7: E finishes, F can join...

Throughput: 5 requests / 7 time units = 0.71 req/unit (1.6x)
With more requests: 10-20x improvement
```

## How It Works

### Iteration-Level Scheduling

```python
class ContinuousBatcher:
    def __init__(self, model, max_batch_size=64, max_seq_len=4096):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.running_requests = []  # Currently generating
        self.waiting_queue = []     # Pending requests
        self.kv_cache_manager = KVCacheManager()

    def step(self):
        """One iteration of continuous batching."""
        # 1. Remove completed requests
        self.running_requests = [
            req for req in self.running_requests
            if not req.is_finished()
        ]

        # 2. Free KV cache for completed requests
        for req in self.get_finished():
            self.kv_cache_manager.free(req.request_id)

        # 3. Add new requests if space available
        while (len(self.running_requests) < self.max_batch_size
               and self.waiting_queue):

            new_req = self.waiting_queue.pop(0)

            # Check if we can allocate KV cache
            if self.kv_cache_manager.can_allocate(new_req.prompt_len):
                self.kv_cache_manager.allocate(new_req)
                self.running_requests.append(new_req)
            else:
                # No space, put back and wait
                self.waiting_queue.insert(0, new_req)
                break

        # 4. Run one generation step for all running requests
        if self.running_requests:
            self.generate_step()

    def generate_step(self):
        """Generate one token for each running request."""
        # Gather inputs (different positions for each request)
        input_tokens = []
        positions = []
        kv_cache_refs = []

        for req in self.running_requests:
            input_tokens.append(req.get_next_input())
            positions.append(req.current_position)
            kv_cache_refs.append(
                self.kv_cache_manager.get_cache(req.request_id)
            )

        # Batched forward pass
        logits = self.model.forward(
            input_tokens,
            positions,
            kv_cache_refs
        )

        # Sample and update each request
        for i, req in enumerate(self.running_requests):
            next_token = sample(logits[i], req.sampling_params)
            req.append_token(next_token)
```

### Request Lifecycle

```python
class Request:
    def __init__(self, prompt_tokens, sampling_params, max_tokens):
        self.request_id = generate_id()
        self.prompt_tokens = prompt_tokens
        self.prompt_len = len(prompt_tokens)
        self.generated_tokens = []
        self.sampling_params = sampling_params
        self.max_tokens = max_tokens
        self.state = "waiting"  # waiting -> prefill -> decode -> finished

    @property
    def current_position(self):
        return self.prompt_len + len(self.generated_tokens)

    def get_next_input(self):
        if self.state == "prefill":
            return self.prompt_tokens  # Full prompt for prefill
        else:
            return [self.generated_tokens[-1]]  # Last token for decode

    def append_token(self, token):
        self.generated_tokens.append(token)
        if token == EOS_TOKEN or len(self.generated_tokens) >= self.max_tokens:
            self.state = "finished"

    def is_finished(self):
        return self.state == "finished"
```

## Prefill vs Decode Phases

### Different Compute Characteristics

```
Prefill Phase (processing prompt):
- Compute-bound: Processing many tokens at once
- High parallelism: All prompt tokens processed together
- Memory write: Fills KV cache with prompt representations
- Longer latency: Proportional to prompt length

Decode Phase (generating tokens):
- Memory-bound: Processing one token at a time
- Sequential: Token-by-token generation
- Memory read: Reads from KV cache
- Lower latency per token

Challenge: Mixing prefill and decode in same batch
- Prefill: Large matrix multiplies
- Decode: Small operations, dominated by memory access
```

### Chunked Prefill

```python
class ChunkedPrefillBatcher:
    """Split long prefills to avoid blocking decodes."""

    def __init__(self, model, chunk_size=512):
        self.model = model
        self.chunk_size = chunk_size

    def step(self):
        # Separate prefill and decode requests
        prefill_reqs = [r for r in self.running if r.state == "prefill"]
        decode_reqs = [r for r in self.running if r.state == "decode"]

        # Process prefills in chunks
        for req in prefill_reqs:
            remaining = req.prompt_len - req.prefill_progress
            chunk = min(remaining, self.chunk_size)

            self.process_prefill_chunk(req, chunk)

            if req.prefill_progress >= req.prompt_len:
                req.state = "decode"

        # Process all decode requests
        if decode_reqs:
            self.process_decode_batch(decode_reqs)
```

### Prefill Scheduling Strategies

```
Strategy 1: Prefill-First
- Complete all prefills before any decodes
- Simple but adds latency to decode requests
- Good for throughput, bad for latency

Strategy 2: Interleaved
- Mix prefill and decode in same batch
- Complex scheduling, hardware utilization challenges
- Balance throughput and latency

Strategy 3: Separate Queues
- Different GPU resources for prefill vs decode
- Disaggregated serving (Splitwise, DistServe)
- Best of both worlds, more infrastructure
```

## Memory Management

### Paged Attention (vLLM)

```python
class PagedKVCache:
    """
    Manage KV cache like OS virtual memory.
    Allocate fixed-size blocks instead of contiguous space.
    """

    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        self.block_size = block_size  # Tokens per block (e.g., 16)
        self.num_blocks = num_blocks

        # Physical blocks: (num_blocks, 2, num_heads, block_size, head_dim)
        self.blocks = torch.zeros(
            num_blocks, 2, num_heads, block_size, head_dim,
            dtype=torch.float16, device="cuda"
        )

        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # request_id -> list of block indices

    def allocate(self, request_id, num_tokens):
        """Allocate blocks for a request."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return False  # Not enough blocks

        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            allocated.append(block_idx)

        self.block_tables[request_id] = allocated
        return True

    def free(self, request_id):
        """Return blocks to free list."""
        blocks = self.block_tables.pop(request_id, [])
        self.free_blocks.extend(blocks)

    def append_token(self, request_id, position, k, v):
        """Write KV for new token, allocating block if needed."""
        block_idx = position // self.block_size
        block_offset = position % self.block_size

        # Allocate new block if needed
        if block_idx >= len(self.block_tables[request_id]):
            if not self.free_blocks:
                return False  # Trigger preemption
            new_block = self.free_blocks.pop()
            self.block_tables[request_id].append(new_block)

        # Write to block
        physical_block = self.block_tables[request_id][block_idx]
        self.blocks[physical_block, 0, :, block_offset, :] = k  # Keys
        self.blocks[physical_block, 1, :, block_offset, :] = v  # Values
        return True
```

### Why Paging Matters

```
Without Paging (Contiguous Allocation):
Request A: [===============================]  Max length reserved
Request B: [=============]..................  Wasted reserved space
Request C: [==]..............................  Most space wasted
                          ^^^^^^^^^^^^^^^^^
                          Reserved but unused

Memory fragmentation: Can't fit new requests even with "free" space

With Paging:
Block 0: [A][A][A][A]  Used by A
Block 1: [B][B][B][_]  Partially used by B
Block 2: [A][A][C][C]  Shared by A and C
Block 3: [_][_][_][_]  Free

Benefits:
- No fragmentation: Blocks allocated on demand
- Copy-on-write: Share blocks between requests (beam search)
- Preemption: Swap out blocks instead of whole sequences
- Near-zero waste: Only allocate what's needed
```

## Request Scheduling

### Priority Queues

```python
class PriorityScheduler:
    """Schedule requests by priority and waiting time."""

    def __init__(self):
        self.queues = {
            "high": [],
            "normal": [],
            "low": []
        }

    def add_request(self, request):
        priority = request.priority or "normal"
        heapq.heappush(
            self.queues[priority],
            (request.arrival_time, request)
        )

    def get_next_batch(self, max_size):
        """Get next batch respecting priorities."""
        batch = []

        # High priority first
        for priority in ["high", "normal", "low"]:
            while self.queues[priority] and len(batch) < max_size:
                _, request = heapq.heappop(self.queues[priority])
                batch.append(request)

        return batch
```

### Preemption Strategies

```python
class PreemptiveScheduler:
    """Handle memory pressure by preempting requests."""

    def handle_memory_pressure(self):
        if not self.can_allocate_new_block():
            # Strategy 1: Swap to CPU
            victim = self.select_victim()
            self.swap_out(victim)

            # Strategy 2: Recompute (discard KV cache)
            # victim = self.select_victim()
            # self.recompute_later(victim)

    def select_victim(self):
        """Select request to preempt."""
        # Options:
        # - Longest running (FCFS preemption)
        # - Shortest remaining (minimize wasted work)
        # - Lowest priority
        # - Random

        # Common: Preempt last arrived (LIFO)
        return self.running_requests[-1]

    def swap_out(self, request):
        """Move KV cache to CPU memory."""
        blocks = self.kv_cache.block_tables[request.request_id]
        cpu_copy = self.copy_to_cpu(blocks)
        self.swapped_requests[request.request_id] = cpu_copy
        self.kv_cache.free(request.request_id)
        request.state = "swapped"

    def swap_in(self, request):
        """Restore KV cache from CPU."""
        cpu_copy = self.swapped_requests.pop(request.request_id)
        blocks = self.kv_cache.allocate(request.request_id, len(cpu_copy))
        self.copy_to_gpu(cpu_copy, blocks)
        request.state = "decode"
```

## Production Systems

### vLLM Architecture

```python
# vLLM: PagedAttention + Continuous Batching
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_num_batched_tokens=4096,
    max_num_seqs=256,  # Max concurrent requests
)

# Process many requests efficiently
prompts = ["Hello" for _ in range(1000)]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# Continuous batching happens automatically
outputs = llm.generate(prompts, sampling_params)
```

### Text-Generation-Inference (TGI)

```python
# TGI: Production-ready serving
from text_generation import Client

client = Client("http://localhost:8080")

# Async generation with continuous batching
response = client.generate(
    "Hello, how are you?",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)
```

### Throughput Comparison

| System | Method | Throughput (tok/s) | Notes |
|--------|--------|-------------------|-------|
| HuggingFace | Static batch | 100-200 | Baseline |
| vLLM | Continuous + PagedAttn | 2000-3000 | 10-15x |
| TGI | Continuous batch | 1500-2500 | Production-ready |
| TensorRT-LLM | Continuous + optimized | 2500-4000 | NVIDIA optimized |

## Advanced Techniques

### Disaggregated Serving

```
Traditional: Same GPU does prefill + decode
            [GPU: prefill====][decode][decode][decode]

Disaggregated: Separate prefill and decode clusters
Prefill GPU:  [prefill====][prefill====][prefill====]
              Optimized for compute-bound work

Decode GPUs:  [decode][decode][decode][decode][decode]
              Optimized for memory-bound work

KV Cache Transfer: Prefill GPU -> Decode GPU
Benefits:
- Better hardware utilization
- Independent scaling
- Reduced interference
```

### Speculative Decoding Integration

```python
class SpeculativeContinuousBatcher:
    """Combine speculative decoding with continuous batching."""

    def step(self):
        # Draft model generates multiple candidates
        draft_tokens = self.draft_model.generate(
            self.running_requests,
            num_tokens=4  # Generate 4 candidates
        )

        # Target model verifies in single forward pass
        verified = self.target_model.verify(
            self.running_requests,
            draft_tokens
        )

        # Update requests with verified tokens
        for req, tokens in zip(self.running_requests, verified):
            req.extend_tokens(tokens)  # May add 1-4 tokens
```

## Key Takeaways

1. **Dynamic vs static**: Continuous batching allows requests to join/leave at each step.

2. **Eliminates convoy effect**: Short requests don't wait for long ones.

3. **10-20x throughput**: Real-world improvements from proper implementation.

4. **Memory management critical**: Paged attention enables efficient KV cache.

5. **Prefill vs decode trade-off**: Different compute characteristics need balancing.

6. **Preemption for fairness**: Handle memory pressure gracefully.

7. **Production systems exist**: vLLM, TGI, TensorRT-LLM handle complexity.
