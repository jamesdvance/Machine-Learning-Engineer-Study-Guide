# Direct Preference Optimization (DPO)

## Summary

Direct Preference Optimization is a simpler alternative to RLHF that eliminates the need for a separate reward model and reinforcement learning. DPO directly optimizes the language model on human preference data by reformulating the RL objective as a supervised learning problem. Instead of training a reward model and then using PPO to optimize against it, DPO uses a clever mathematical trick: the optimal policy can be derived directly from the preference data through a classification-like loss. This makes training more stable, faster, and easier to implement while achieving comparable or better results than RLHF.

Key points to remember:

- No reward model: Skips reward modeling step entirely
- No RL: Uses supervised-style loss instead of PPO
- Simpler training: One-stage process, more stable
- Same objective: Mathematically equivalent to RLHF with optimal reward
- Preference pairs: Needs (prompt, chosen, rejected) data
- Reference model: Uses frozen copy to prevent distribution collapse
- Production ready: Widely adopted in open-source model training

## The Key Insight

### RLHF vs DPO

```
RLHF Pipeline:
1. Collect preference data (chosen, rejected pairs)
2. Train reward model on preferences
3. Use PPO to optimize policy against reward model
4. Manage KL penalty to stay near reference

DPO Pipeline:
1. Collect preference data (chosen, rejected pairs)
2. Train policy directly with DPO loss
   (implicitly defines reward = log policy ratio)

Mathematical insight:
- The optimal policy under RLHF has a closed form
- DPO loss directly optimizes for this closed form
- No need to explicitly model reward or run RL
```

### Why This Works

```
RLHF objective:
max E[r(x,y)] - ² * KL(À || À_ref)

Optimal policy (from RL theory):
À*(y|x)  À_ref(y|x) * exp(r(x,y)/²)

Rearranging for reward:
r(x,y) = ² * log(À*(y|x) / À_ref(y|x)) + const

Key observation:
- Reward can be expressed in terms of policy
- We don't need to learn r explicitly
- Just optimize policy to match preferences directly
```

## The DPO Loss

### Mathematical Formulation

```python
def dpo_loss(policy, reference, chosen, rejected, beta=0.1):
    """
    DPO loss function.

    L_DPO = -E[log Ã(² * (log À(y_w|x)/À_ref(y_w|x) - log À(y_l|x)/À_ref(y_l|x)))]

    where:
    - y_w = chosen (winner) response
    - y_l = rejected (loser) response
    - ² = temperature parameter
    - Ã = sigmoid function
    """
    # Compute log probabilities
    chosen_logps = get_log_probs(policy, chosen)
    rejected_logps = get_log_probs(policy, rejected)

    ref_chosen_logps = get_log_probs(reference, chosen)
    ref_rejected_logps = get_log_probs(reference, rejected)

    # Compute log ratios
    chosen_ratio = chosen_logps - ref_chosen_logps
    rejected_ratio = rejected_logps - ref_rejected_logps

    # DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss


def get_log_probs(model, sequences):
    """Compute log probabilities of sequences."""
    outputs = model(sequences.input_ids, attention_mask=sequences.attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = sequences.input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # Sum over tokens (considering attention mask)
    mask = sequences.attention_mask[:, 1:]
    return (token_log_probs * mask).sum(-1) / mask.sum(-1)
```

### Intuition Behind the Loss

```
What DPO optimizes:

Increase probability of chosen response relative to reference
Decrease probability of rejected response relative to reference

The ratio (chosen_ratio - rejected_ratio) measures:
- How much more the policy prefers chosen over rejected
- Compared to the reference model

Sigmoid converts to a probability:
- High ratio ’ high probability ’ low loss (good)
- Low ratio ’ low probability ’ high loss (bad)

² (beta) controls:
- Lower ²: Stronger preference enforcement
- Higher ²: More conservative, stay closer to reference
```

## Implementation

### Basic DPO Training

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

class DPOTrainer:
    def __init__(self, model_name, beta=0.1, lr=1e-6):
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False

        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train_step(self, batch):
        """Single training step."""
        self.policy.train()
        self.reference.eval()

        # Tokenize
        chosen = self.tokenizer(batch['chosen'], return_tensors='pt', padding=True)
        rejected = self.tokenizer(batch['rejected'], return_tensors='pt', padding=True)

        # Compute loss
        loss = dpo_loss(
            self.policy, self.reference,
            chosen, rejected,
            self.beta
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, epochs=1, batch_size=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss

            print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
```

### With Hugging Face TRL

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Reference model (frozen copy)
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load preference dataset
# Format: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("Anthropic/hh-rlhf")

# DPO config
config = DPOConfig(
    beta=0.1,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=512,
    max_prompt_length=256,
)

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

## Data Format

### Preference Pair Structure

```python
# Standard DPO data format
preference_example = {
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "France's capital is Lyon."
}

# With conversation context
conversation_example = {
    "prompt": "Human: What is the capital of France?\nAssistant:",
    "chosen": " The capital of France is Paris, known for the Eiffel Tower.",
    "rejected": " I think it's probably a city with nice weather."
}
```

### Data Collection Strategies

```python
def generate_preference_pairs(model, prompts, annotator):
    """Generate preference data from model outputs."""
    pairs = []

    for prompt in prompts:
        # Generate multiple responses
        responses = [
            model.generate(prompt, temperature=0.8)
            for _ in range(2)
        ]

        # Get human preference
        preference = annotator.compare(prompt, responses[0], responses[1])

        pairs.append({
            "prompt": prompt,
            "chosen": responses[preference["winner"]],
            "rejected": responses[preference["loser"]]
        })

    return pairs

# Or use existing datasets
from datasets import load_dataset

# Anthropic HH-RLHF
hh_rlhf = load_dataset("Anthropic/hh-rlhf")

# Stanford SHP (human preferences)
shp = load_dataset("stanfordnlp/SHP")

# UltraFeedback
ultrafeedback = load_dataset("openbmb/UltraFeedback")
```

## Hyperparameters

### Key Parameters

```python
dpo_config = {
    # Core DPO parameters
    "beta": 0.1,  # KL penalty strength (0.05-0.5 typical)

    # Learning rate
    "learning_rate": 1e-6,  # Lower than SFT (1e-6 to 5e-6)

    # Batch size
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch 16

    # Length limits
    "max_length": 512,  # Max total sequence length
    "max_prompt_length": 256,  # Max prompt length

    # Training
    "num_train_epochs": 1,  # Often 1-3 epochs sufficient
    "warmup_ratio": 0.1,

    # Regularization
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
```

### Beta Selection

```
Beta (²) controls preference strength:

² = 0.05: Strong preference enforcement
- Large policy updates
- Risk of distribution collapse
- Use with high-quality data

² = 0.1: Balanced (common default)
- Good trade-off
- Stable training

² = 0.5: Conservative
- Stay close to reference
- Slower improvement
- More stable with noisy data

Choosing ²:
- Start with 0.1
- If training unstable, increase ²
- If insufficient alignment, decrease ²
```

## Variants

### IPO (Identity Preference Optimization)

```python
def ipo_loss(policy, reference, chosen, rejected, tau=0.1):
    """
    IPO: Alternative loss that's more robust to label noise.
    L_IPO = (log_ratio - 1/(2*tau))^2
    """
    chosen_ratio = log_prob_ratio(policy, reference, chosen)
    rejected_ratio = log_prob_ratio(policy, reference, rejected)

    log_ratio = chosen_ratio - rejected_ratio
    loss = (log_ratio - 1/(2*tau)) ** 2

    return loss.mean()
```

### cDPO (Conservative DPO)

```python
def cdpo_loss(policy, reference, chosen, rejected, beta=0.1, label_smoothing=0.1):
    """
    Conservative DPO: Add label smoothing for robustness.
    """
    chosen_ratio = log_prob_ratio(policy, reference, chosen)
    rejected_ratio = log_prob_ratio(policy, reference, rejected)

    logits = beta * (chosen_ratio - rejected_ratio)

    # Label smoothing
    loss = -((1 - label_smoothing) * F.logsigmoid(logits) +
             label_smoothing * F.logsigmoid(-logits))

    return loss.mean()
```

### KTO (Kahneman-Tversky Optimization)

```python
def kto_loss(policy, reference, sequences, is_desirable, beta=0.1):
    """
    KTO: Works with single responses (not pairs).
    Uses Kahneman-Tversky prospect theory.
    """
    log_ratio = log_prob_ratio(policy, reference, sequences)

    # Different treatment for desirable vs undesirable
    desirable_loss = 1 - F.sigmoid(beta * log_ratio)
    undesirable_loss = 1 - F.sigmoid(-beta * log_ratio)

    loss = torch.where(is_desirable, desirable_loss, undesirable_loss)
    return loss.mean()
```

## Comparison with RLHF

### Trade-offs

| Aspect | RLHF | DPO |
|--------|------|-----|
| Components | RM + PPO | Single loss |
| Stability | Can be unstable | Generally stable |
| Memory | Need RM + policy | Just policy + reference |
| Compute | Multiple forward passes | Two forward passes |
| Hyperparameters | Many (PPO) | Few (mainly ²) |
| Implementation | Complex | Simple |
| Performance | Strong | Comparable |

### When to Use Each

```
Use DPO when:
- Want simpler training pipeline
- Limited compute/memory
- New to preference optimization
- Preference data is clean

Use RLHF when:
- Need fine-grained reward control
- Have noisy preference data
- Want to use reward model for evaluation
- Need online data collection
```

## Practical Tips

### Common Issues

```
Problem: Training loss doesn't decrease
Solutions:
- Increase learning rate slightly
- Check data format (prompt/chosen/rejected)
- Verify reference model is frozen
- Check tokenization includes response

Problem: Model becomes too different from reference
Solutions:
- Increase ²
- Reduce learning rate
- Train for fewer steps

Problem: Chosen/rejected become indistinguishable
Solutions:
- Use higher quality preference data
- Ensure clear preference signal in data
- Check for data contamination
```

### Evaluation

```python
def evaluate_dpo_model(model, reference, eval_data):
    """Evaluate DPO training quality."""
    metrics = {
        "chosen_prob": [],
        "rejected_prob": [],
        "accuracy": []
    }

    for example in eval_data:
        # Get log probs
        chosen_lp = get_log_probs(model, example["chosen"])
        rejected_lp = get_log_probs(model, example["rejected"])

        metrics["chosen_prob"].append(chosen_lp.exp().item())
        metrics["rejected_prob"].append(rejected_lp.exp().item())
        metrics["accuracy"].append(chosen_lp > rejected_lp)

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

## Key Takeaways

1. **Simpler than RLHF**: No reward model, no PPO, just classification-style loss.

2. **Same objective**: Mathematically equivalent to RLHF at optimum.

3. **Reference model**: Frozen copy prevents distribution collapse.

4. **Beta controls trade-off**: Between preference strength and staying near reference.

5. **Data quality matters**: Clean preference pairs are essential.

6. **Widely adopted**: Standard in open-source model alignment.

7. **Many variants**: IPO, cDPO, KTO extend the basic approach.
