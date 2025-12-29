# Speculative Decoding

## Summary

Speculative decoding accelerates LLM inference by using a small, fast "draft" model to generate multiple candidate tokens, then verifying them in parallel with the large "target" model. Since the target model can verify K tokens in roughly the same time as generating one (due to parallel processing), accepted speculation yields up to Kx speedup. The key insight is that verification is cheaper than generation because attention over candidates is parallelizable. This technique is lossless - it produces the exact same output distribution as standard decoding.

Key points to remember:

- Draft and verify: Small model proposes, large model checks
- Lossless: Produces identical output distribution to standard decoding
- 2-3x speedup: Typical improvement for well-matched draft models
- Acceptance rate matters: Higher draft quality = more speedup
- Parallel verification: Check K tokens in ~1 forward pass
- Memory overhead: Must hold both draft and target models
- Works with any decoding: Greedy, sampling, beam search

## The Core Insight

### Why Speculative Decoding Works

```
Standard Autoregressive Generation:
Token 1: [Forward Pass] ’ 100ms
Token 2: [Forward Pass] ’ 100ms
Token 3: [Forward Pass] ’ 100ms
Token 4: [Forward Pass] ’ 100ms
Total: 400ms for 4 tokens

Speculative Decoding:
Draft Model (fast):  [4 tokens] ’ 20ms
Target Model (verify): [check 4] ’ 120ms (parallel attention)
If all accepted: 140ms for 4 tokens (2.8x speedup)

Key Insight:
- Generating 1 token: Expensive (sequential attention building)
- Verifying K tokens: ~Same cost (parallel attention over K)
- Draft model: 10-50x faster than target
```

### Acceptance and Rejection

```python
def speculative_decode(target_model, draft_model, prompt, K=4):
    """
    Generate tokens using speculative decoding.
    K: Number of draft tokens per iteration
    """
    tokens = prompt.copy()

    while not done:
        # Step 1: Draft K candidate tokens (fast)
        draft_tokens = []
        draft_probs = []
        draft_state = tokens.copy()

        for _ in range(K):
            logits = draft_model(draft_state)
            prob = softmax(logits[-1])
            token = sample(prob)
            draft_tokens.append(token)
            draft_probs.append(prob)
            draft_state.append(token)

        # Step 2: Verify with target model (parallel)
        # Single forward pass checks all K candidates
        target_logits = target_model(tokens + draft_tokens)
        target_probs = [softmax(l) for l in target_logits[-K-1:]]

        # Step 3: Accept/reject each token
        accepted = []
        for i, (draft_tok, p_draft, p_target) in enumerate(
            zip(draft_tokens, draft_probs, target_probs[:-1])
        ):
            # Acceptance criterion (see below)
            if accept(draft_tok, p_draft, p_target):
                accepted.append(draft_tok)
            else:
                # Reject this and all following tokens
                # Sample correction token from adjusted distribution
                correction = sample_correction(p_draft, p_target)
                accepted.append(correction)
                break

        # If all K accepted, sample one more from final target probs
        if len(accepted) == K:
            bonus_token = sample(target_probs[-1])
            accepted.append(bonus_token)

        tokens.extend(accepted)

    return tokens
```

## The Acceptance Criterion

### Maintaining Output Distribution

```python
def accept_token(draft_token, p_draft, p_target):
    """
    Acceptance criterion that preserves target distribution.
    Based on rejection sampling.
    """
    # Get probabilities for the drafted token
    q = p_draft[draft_token]  # Draft model's probability
    p = p_target[draft_token]  # Target model's probability

    # Accept with probability min(1, p/q)
    acceptance_prob = min(1.0, p / q)
    return random.random() < acceptance_prob

def sample_correction(p_draft, p_target):
    """
    When rejecting, sample from adjusted distribution.
    This maintains exactness of target distribution.
    """
    # Compute residual distribution
    # norm(max(0, p_target - p_draft))
    residual = torch.clamp(p_target - p_draft, min=0)
    residual = residual / residual.sum()

    return sample(residual)
```

### Why This Is Lossless

```
Claim: Speculative decoding produces identical output to standard decoding

Proof intuition:
1. If p(token) >= q(token): Accept with prob q/q = 1
   Token appears with probability p (from target)

2. If p(token) < q(token): Accept with prob p/q
   - Accepted: Contributes p to final probability
   - Rejected: Falls through to correction sampling

3. Correction sampling from max(0, p-q):
   - Covers tokens under-represented by draft
   - Combined with acceptance, recovers exact p

Result: Each token drawn from exact target distribution
```

## Draft Model Selection

### Options for Draft Models

```
Option 1: Smaller Model from Same Family
- LLaMA-7B as draft for LLaMA-70B
- Good vocabulary alignment
- 2-3x speedup typical

Option 2: Distilled Model
- Train small model to mimic target
- Better acceptance rates
- Requires training compute

Option 3: N-gram or Retrieval
- Very fast lookup
- Works for repetitive/predictable text
- Lower acceptance on novel content

Option 4: Early Exit
- Use first few layers of target as draft
- No additional model memory
- Requires architecture modification

Option 5: Self-Speculative
- Target model generates drafts at lower quality
- No separate model needed
- Medusa heads approach
```

### Acceptance Rate Analysis

```python
def estimate_acceptance_rate(draft_model, target_model, test_data):
    """Measure draft model quality."""
    total_proposed = 0
    total_accepted = 0

    for prompt in test_data:
        tokens = prompt.copy()

        for _ in range(100):  # Generate 100 tokens
            # Draft
            draft_logits = draft_model(tokens)
            draft_probs = softmax(draft_logits[-1])
            draft_token = sample(draft_probs)

            # Target
            target_logits = target_model(tokens)
            target_probs = softmax(target_logits[-1])

            # Check acceptance
            total_proposed += 1
            if accept_token(draft_token, draft_probs, target_probs):
                total_accepted += 1
                tokens.append(draft_token)
            else:
                tokens.append(sample(target_probs))

    return total_accepted / total_proposed

# Typical acceptance rates:
# LLaMA-7B ’ LLaMA-70B: 60-80%
# Distilled draft: 70-90%
# N-gram draft: 30-50%
```

## Speedup Analysis

### Theoretical Speedup

```python
def calculate_speedup(K, acceptance_rate, draft_cost, target_cost):
    """
    Calculate expected speedup from speculative decoding.

    K: Draft length
    acceptance_rate: Probability draft token accepted
    draft_cost: Time for K draft tokens
    target_cost: Time for verification forward pass
    """
    # Expected accepted tokens per iteration
    # If acceptance_rate = alpha, expected = sum(alpha^i for i in 1..K)
    # Plus 1 for the bonus/correction token
    expected_tokens = sum(acceptance_rate ** i for i in range(1, K + 1)) + 1

    # Time per iteration
    iteration_time = draft_cost + target_cost

    # Standard decoding: 1 token per target forward pass
    standard_time = target_cost

    # Speedup
    tokens_per_time = expected_tokens / iteration_time
    standard_tokens_per_time = 1 / standard_time

    return tokens_per_time / standard_tokens_per_time

# Example:
# K=4, acceptance=0.7, draft_cost=20ms, target_cost=100ms
# Expected tokens: 0.7 + 0.49 + 0.34 + 0.24 + 1 = 2.77
# Time: 120ms
# Standard: 100ms/token
# Speedup: 2.77 / 1.2 = 2.3x
```

### Optimal Draft Length

```
Finding optimal K:
- Larger K: More potential tokens, higher draft cost
- Smaller K: Less overhead, fewer tokens per iteration

Trade-off:
- K too small: Verification overhead dominates
- K too large: Many rejected tokens waste draft compute

Optimal K depends on:
1. Acceptance rate (higher ’ larger K)
2. Draft model speed (faster ’ larger K)
3. Target model speed (slower ’ larger K)

Typical optimal: K = 3-8 tokens
```

## Advanced Techniques

### Medusa: Self-Speculative Heads

```python
class MedusaModel(nn.Module):
    """
    Add multiple prediction heads to generate draft tokens.
    No separate draft model needed.
    """

    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size

        # Additional heads predict future tokens
        # Head 0: Next token (standard)
        # Head 1: Token after next
        # Head 2: Two tokens ahead, etc.
        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size)
            for _ in range(num_heads)
        ])

    def forward(self, input_ids):
        # Get base model hidden states
        outputs = self.base_model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        # Standard next-token prediction
        base_logits = outputs.logits

        # Medusa head predictions (speculative)
        medusa_logits = [head(hidden) for head in self.medusa_heads]

        return base_logits, medusa_logits

    def speculative_generate(self, input_ids, K=4):
        """Generate using Medusa heads for speculation."""
        base_logits, medusa_logits = self.forward(input_ids)

        # Construct tree of candidate sequences
        candidates = self.build_candidate_tree(
            base_logits, medusa_logits, K
        )

        # Verify candidates in single forward pass
        verified = self.verify_candidates(candidates)

        return verified
```

### Tree-Based Speculation

```python
class TreeSpeculation:
    """
    Generate tree of candidates instead of single sequence.
    Increases acceptance probability.
    """

    def generate_tree(self, draft_model, tokens, depth=4, width=3):
        """Generate tree of candidate continuations."""
        root = TreeNode(tokens)

        def expand(node, remaining_depth):
            if remaining_depth == 0:
                return

            logits = draft_model(node.tokens)
            probs = softmax(logits[-1])

            # Take top-k candidates
            top_tokens = torch.topk(probs, width).indices

            for token in top_tokens:
                child = TreeNode(node.tokens + [token])
                node.children.append(child)
                expand(child, remaining_depth - 1)

        expand(root, depth)
        return root

    def verify_tree(self, target_model, tree):
        """Verify all tree paths efficiently."""
        # Flatten tree to batch of sequences
        all_paths = self.get_all_paths(tree)

        # Single batched forward pass
        target_logits = target_model.forward_batch(all_paths)

        # Find best verified path
        best_path = self.find_best_path(all_paths, target_logits)

        return best_path

# Tree speculation can achieve higher acceptance
# But requires more memory and complex implementation
```

### Lookahead Decoding

```python
class LookaheadDecoding:
    """
    Generate and verify in sliding window fashion.
    Jacobi iteration approach.
    """

    def generate(self, model, prompt, window_size=5):
        """Lookahead decoding with parallel verification."""
        tokens = prompt.copy()
        # Initialize window with guesses
        window = [self.initial_guess() for _ in range(window_size)]

        while not done:
            # Parallel forward pass for all window positions
            all_tokens = tokens + window
            logits = model(all_tokens)

            # Update window: each position uses previous position's prediction
            new_window = []
            for i in range(window_size):
                pos = len(tokens) + i
                # New guess based on model output
                new_window.append(sample(softmax(logits[pos])))

            # Check for convergence (fixed point)
            if new_window[0] == window[0]:
                # First position converged, accept it
                tokens.append(new_window[0])
                window = new_window[1:] + [self.initial_guess()]
            else:
                window = new_window

        return tokens
```

## Practical Implementation

### With HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load target and draft models
target = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

draft = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# Generate with assisted decoding (HuggingFace's speculative decoding)
inputs = tokenizer("The meaning of life is", return_tensors="pt")

outputs = target.generate(
    **inputs,
    assistant_model=draft,  # Speculative decoding
    max_new_tokens=100,
    do_sample=False  # Works with greedy
)

print(tokenizer.decode(outputs[0]))
```

### With vLLM

```python
from vllm import LLM, SamplingParams

# vLLM supports speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5,
    tensor_parallel_size=4
)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=100
)

outputs = llm.generate(prompts, sampling_params)
```

## Speedup Benchmarks

### Typical Results

| Target Model | Draft Model | Acceptance | Speedup |
|--------------|-------------|------------|---------|
| LLaMA-70B | LLaMA-7B | 65% | 2.0x |
| LLaMA-70B | LLaMA-13B | 75% | 2.3x |
| GPT-4 | GPT-3.5 | ~70% | 2.0-2.5x |
| Codex | Small Codex | 80% | 2.5x |

### When Speculative Decoding Helps Most

```
Best scenarios:
- Large target model (high per-token cost)
- Fast draft model (low speculation cost)
- Predictable text (high acceptance)
- Greedy or low-temperature sampling

Less effective:
- Small target model (already fast)
- High-temperature sampling (lower acceptance)
- Very creative/unpredictable text
- Memory-constrained (can't fit both models)
```

## Key Takeaways

1. **Draft and verify**: Small model proposes, large model checks in parallel.

2. **Lossless**: Produces exact same distribution as standard decoding.

3. **2-3x typical speedup**: Depends on draft quality and model sizes.

4. **Acceptance rate is key**: Better draft = more speedup.

5. **Memory trade-off**: Need to hold both models in memory.

6. **Works with any sampling**: Greedy, temperature, top-p all supported.

7. **Production ready**: Supported in HuggingFace, vLLM, TensorRT-LLM.
