# LLM Alignment

## Summary

Alignment techniques make language models helpful, harmless, and honest by training them to follow human preferences and values. Raw pre-trained models are next-token predictors with no inherent goal of being useful; alignment bridges the gap between predicting text and behaving as intended. The main approaches are RLHF (training a reward model on human preferences and optimizing with reinforcement learning), DPO (directly optimizing on preference pairs without RL), and Constitutional AI (self-critique using explicit principles). Modern chatbots like ChatGPT and Claude are aligned models, and alignment is what transforms a base LLM into a useful assistant.

Key points to remember:

- Bridge training and deployment: Align model behavior with user expectations
- Human preferences: Learn from comparison data (which response is better)
- RLHF: Reward model + PPO optimization, the original approach
- DPO: Simpler alternative, no reward model or RL needed
- Constitutional AI: Self-critique using written principles
- Helpful, harmless, honest: The three H's of alignment
- Post-SFT step: Alignment follows supervised fine-tuning

## The Alignment Pipeline

```
Pre-training (CLM)
     “
Supervised Fine-Tuning (SFT)
     “
Alignment (RLHF / DPO / CAI)
     “
Deployed Model

Each step:
1. Pre-training: Learn language from massive text
2. SFT: Learn to follow instructions from demonstrations
3. Alignment: Learn preferences from comparisons
```

## Method Comparison

| Method | Components | Complexity | Stability | Performance |
|--------|------------|------------|-----------|-------------|
| RLHF | RM + PPO | High | Moderate | Excellent |
| DPO | Single loss | Low | High | Excellent |
| CAI | Self-critique | Moderate | Good | Good |
| ORPO | Combined loss | Low | High | Good |

### When to Use Each

```
RLHF:
- Gold standard performance
- Have resources for complex training
- Need fine-grained reward control
- Can handle training instability

DPO:
- Want simpler training
- Limited compute
- New to preference optimization
- Have clean preference data

CAI:
- Need to scale without proportional labeling
- Want explicit, auditable principles
- Consistency is important
- Can articulate desired behavior clearly

ORPO:
- Want to skip RM/PPO entirely
- Simple end-to-end training
- Combine SFT and alignment
```

## Data Requirements

### Preference Data Format

```python
# Standard format for alignment
preference_example = {
    "prompt": "How do I make pasta?",
    "chosen": "Boil water, add pasta, cook for 8-10 minutes...",
    "rejected": "Just put it in the microwave for 5 minutes."
}

# Collection strategies:
# 1. Human annotation (expensive, high quality)
# 2. AI feedback (Constitutional AI)
# 3. Synthetic (model-generated comparisons)
```

### Scale Guidelines

```
Preference data requirements:
- Minimum viable: 1K-5K pairs
- Good quality: 10K-50K pairs
- Production: 50K-500K pairs

Quality matters more than quantity:
- Clear preference signal
- Diverse prompts
- Consistent annotators
```

## Practical Alignment

### Quick Start with DPO

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM

# Load SFT model (already instruction-tuned)
model = AutoModelForCausalLM.from_pretrained("sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("sft_model")

# DPO training
config = DPOConfig(beta=0.1, learning_rate=1e-6)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=preference_data
)
trainer.train()
```

### RLHF Pipeline

```python
# Stage 1: Train reward model
reward_model = train_reward_model(preference_data)

# Stage 2: PPO optimization
from trl import PPOTrainer

ppo_trainer = PPOTrainer(
    model=sft_model,
    ref_model=sft_model.clone(),
    reward_model=reward_model
)

for batch in prompts:
    responses = ppo_trainer.generate(batch)
    rewards = reward_model(responses)
    ppo_trainer.step(batch, responses, rewards)
```

## Common Challenges

### Reward Hacking

```
Problem: Model finds shortcuts to maximize reward

Examples:
- Excessively verbose responses
- Repetitive confident-sounding phrases
- Gaming length or format

Solutions:
- KL penalty (stay near reference)
- Diverse training data
- Reward model ensembles
- Human spot-checks
```

### Training Stability

```
Problem: RLHF training can be unstable

Symptoms:
- Loss spikes
- Reward collapse
- KL explosion

Solutions:
- Lower learning rate
- Increase KL penalty
- Gradient clipping
- Use DPO instead
```

## Key Takeaways

1. **Alignment is essential**: Transforms base models into assistants.

2. **Preferences over demonstrations**: Learn from comparisons, not just examples.

3. **DPO simplifies**: No reward model or RL, just classification-like loss.

4. **RLHF is powerful**: But complex and potentially unstable.

5. **CAI scales**: Self-critique reduces human labeling needs.

6. **Data quality matters**: Clean preference pairs are essential.

7. **Part of the pipeline**: Comes after pre-training and SFT.

## Further Reading

For detailed coverage of alignment techniques, see:

- [RLHF](rlhf/ReadMe.md) - Reinforcement Learning from Human Feedback
- [DPO](dpo/ReadMe.md) - Direct Preference Optimization
- [Constitutional AI](constitutional-ai/ReadMe.md) - Self-critique with principles
