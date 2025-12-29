# Constitutional AI (CAI)

## Summary

Constitutional AI is an alignment approach that reduces the need for human feedback by having the AI critique and revise its own outputs according to a set of principles (a "constitution"). Instead of training a reward model on human preferences, CAI uses the model itself to evaluate responses against explicit rules like "be helpful, harmless, and honest." The model generates an initial response, critiques it based on constitutional principles, and revises it. This self-improvement process generates preference pairs for training without requiring extensive human labeling. CAI scales alignment more efficiently while making safety principles explicit and auditable.

Key points to remember:

- Self-critique: Model evaluates its own outputs against principles
- Explicit constitution: Written rules define desired behavior
- Reduced human labeling: Generate training data through self-revision
- Two phases: Supervised Learning from AI Feedback (SL-CAI) + Reinforcement Learning from AI Feedback (RL-CAI)
- Transparency: Principles are documented and auditable
- Scalable: Less human feedback needed per training iteration
- Complementary to RLHF: Can be combined with human preferences

## The Constitution

### Defining Principles

```python
# Example constitutional principles
CONSTITUTION = [
    {
        "name": "harmlessness",
        "principle": "The response should not encourage harmful, illegal, or unethical activities.",
        "critique_prompt": "Does this response encourage harmful behavior?",
        "revision_prompt": "Revise to avoid harmful content while remaining helpful."
    },
    {
        "name": "helpfulness",
        "principle": "The response should directly address the user's question.",
        "critique_prompt": "Does this response actually help the user?",
        "revision_prompt": "Revise to be more directly helpful while maintaining safety."
    },
    {
        "name": "honesty",
        "principle": "The response should be truthful and acknowledge uncertainty.",
        "critique_prompt": "Is this response truthful? Does it admit limitations?",
        "revision_prompt": "Revise to be more accurate and honest about limitations."
    },
]
```

### Principle Categories

```
Anthropic's Constitution Categories:

1. Helpfulness
   - Be maximally helpful to the user
   - Provide accurate information
   - Complete requested tasks

2. Harmlessness
   - Avoid generating harmful content
   - Don't assist with illegal activities
   - Refuse dangerous requests appropriately

3. Honesty
   - Don't deceive or mislead
   - Acknowledge uncertainty
   - Distinguish fact from opinion

4. Ethics
   - Promote well-being
   - Respect autonomy
   - Be fair and unbiased
```

## The CAI Process

### Phase 1: Supervised Learning from AI Feedback (SL-CAI)

```python
def generate_sl_cai_data(model, prompts, constitution):
    """Generate supervised training data through self-revision."""
    training_data = []

    for prompt in prompts:
        # Step 1: Generate initial response
        initial_response = model.generate(prompt, temperature=1.0)

        # Step 2: Critique using constitutional principles
        critique = generate_critique(model, prompt, initial_response, constitution)

        # Step 3: Revise based on critique
        revised_response = generate_revision(
            model, prompt, initial_response, critique
        )

        training_data.append({
            "prompt": prompt,
            "response": revised_response
        })

    return training_data


def generate_critique(model, prompt, response, constitution):
    """Generate critique based on constitutional principles."""
    critiques = []

    for principle in constitution:
        critique_prompt = f"""Human: Consider the following response:

Question: {prompt}
Response: {response}

{principle['critique_prompt']}
Identify any issues based on: {principle['principle']}

Critique:"""

        critique = model.generate(critique_prompt)
        critiques.append(critique)

    return "\n".join(critiques)


def generate_revision(model, prompt, response, critique):
    """Revise response based on critique."""
    revision_prompt = f"""Human: Here is a response that needs improvement:

Question: {prompt}
Original response: {response}

Critique: {critique}

Please provide an improved response that addresses the critique while remaining helpful.

Improved response:"""

    return model.generate(revision_prompt)
```

### Phase 2: Reinforcement Learning from AI Feedback (RL-CAI)

```python
def generate_rl_cai_data(model, prompts, constitution):
    """Generate preference pairs for RL training."""
    preference_data = []

    for prompt in prompts:
        # Generate multiple candidate responses
        responses = [
            model.generate(prompt, temperature=1.0)
            for _ in range(2)
        ]

        # Use model to rank responses based on constitution
        ranking = rank_by_constitution(model, prompt, responses, constitution)

        preference_data.append({
            "prompt": prompt,
            "chosen": ranking[0],
            "rejected": ranking[-1]
        })

    return preference_data


def rank_by_constitution(model, prompt, responses, constitution):
    """Rank responses according to constitutional principles."""
    ranking_prompt = f"""Human: Consider these responses to: {prompt}

Response A: {responses[0]}

Response B: {responses[1]}

Based on these principles:
{format_principles(constitution)}

Which response better follows the principles? Explain why and state your choice (A or B).

Analysis:"""

    analysis = model.generate(ranking_prompt)
    choice = parse_choice(analysis)

    if choice == "A":
        return [responses[0], responses[1]]
    else:
        return [responses[1], responses[0]]
```

## Multi-Turn Critique

### Iterative Revision

```python
def iterative_cai_revision(model, prompt, initial_response, constitution, max_rounds=3):
    """
    Multiple rounds of critique and revision for better quality.
    """
    current_response = initial_response

    for round in range(max_rounds):
        # Critique current response
        issues = []

        for principle in constitution:
            issue = check_principle(model, prompt, current_response, principle)
            if issue:
                issues.append(issue)

        # If no issues, done
        if not issues:
            break

        # Revise addressing the issues
        revision_prompt = f"""The response to "{prompt}" was:

{current_response}

Issues found:
{format_issues(issues)}

Please provide an improved response:"""

        current_response = model.generate(revision_prompt)

    return current_response
```

### Chain-of-Thought Critique

```python
def cot_critique(model, prompt, response, principle):
    """Detailed chain-of-thought critique."""
    critique_prompt = f"""Let's carefully analyze this response.

Question: {prompt}
Response: {response}

Principle to check: {principle['principle']}

Let's think step by step:
1. What is the response claiming or doing?
2. Does this violate the principle in any way?
3. If so, what specifically is problematic?
4. How could it be improved?

Analysis:"""

    return model.generate(critique_prompt)
```

## Training Pipeline

### Full CAI Pipeline

```python
class ConstitutionalAIPipeline:
    def __init__(self, base_model, constitution):
        self.base_model = base_model
        self.constitution = constitution
        self.sl_model = None
        self.rl_model = None

    def train_sl_phase(self, prompts, epochs=3):
        """Phase 1: Supervised learning from self-critique."""
        # Generate revised training data
        training_data = []

        for prompt in prompts:
            initial = self.base_model.generate(prompt)
            revised = self.revise_response(prompt, initial)
            training_data.append({"prompt": prompt, "response": revised})

        # Fine-tune on revised responses
        self.sl_model = self.supervised_train(
            self.base_model,
            training_data,
            epochs=epochs
        )

    def train_rl_phase(self, prompts):
        """Phase 2: RL from AI-generated preferences."""
        # Generate preference pairs
        preferences = []

        for prompt in prompts:
            responses = [
                self.sl_model.generate(prompt, temperature=0.8)
                for _ in range(2)
            ]
            chosen, rejected = self.rank_responses(prompt, responses)
            preferences.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

        # Train reward model or use DPO
        self.rl_model = self.preference_train(self.sl_model, preferences)

    def revise_response(self, prompt, response):
        """Apply constitutional critique and revision."""
        for principle in self.constitution:
            critique = self.critique(prompt, response, principle)
            if critique.indicates_issue():
                response = self.revise(prompt, response, critique)
        return response
```

## Comparison with RLHF

### CAI vs RLHF

| Aspect | RLHF | Constitutional AI |
|--------|------|------------------|
| Feedback source | Human labelers | AI self-critique |
| Cost | Expensive | Lower |
| Scale | Limited by labelers | Scales with compute |
| Principles | Implicit in data | Explicit in constitution |
| Transparency | Opaque preferences | Documented principles |
| Consistency | Labeler variance | Consistent application |

### When to Use Each

```
Use CAI when:
- Need to scale alignment without proportional labeling costs
- Want explicit, auditable safety principles
- Principles can be articulated clearly
- Consistency is important

Use RLHF when:
- Capturing nuanced human preferences
- Tasks where principles are hard to articulate
- Human judgment is irreplaceable
- Ground truth requires human evaluation

Combine both when:
- Use CAI for initial alignment at scale
- Add RLHF for nuanced human preferences
- Layer CAI principles on top of RLHF training
```

## Practical Implementation

### With Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

def cai_training_pipeline(model_name, constitution_principles):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Phase 1: Generate SL data
    sl_data = []
    for prompt in training_prompts:
        revised = constitutional_revision(model, tokenizer, prompt, constitution_principles)
        sl_data.append({"prompt": prompt, "completion": revised})

    # SFT training
    model = sft_train(model, sl_data)

    # Phase 2: Generate preference data
    pref_data = []
    for prompt in training_prompts:
        responses = generate_candidates(model, tokenizer, prompt)
        chosen, rejected = constitutional_ranking(model, tokenizer, prompt, responses)
        pref_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # DPO training (RLHF alternative)
    trainer = DPOTrainer(model=model, train_dataset=pref_data)
    trainer.train()

    return model
```

## Key Takeaways

1. **Self-improvement**: Model critiques and revises its own outputs.

2. **Explicit principles**: Constitution documents desired behavior.

3. **Scalable**: Less human labeling required than pure RLHF.

4. **Two phases**: Supervised learning + RL from AI feedback.

5. **Transparent**: Principles are auditable and modifiable.

6. **Complementary**: Works alongside human preference data.

7. **Iterative**: Multiple revision rounds improve quality.
