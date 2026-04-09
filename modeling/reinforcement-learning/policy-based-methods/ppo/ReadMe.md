# PPO (Proximal Policy Optimization)

## Summary

Proximal Policy Optimization (PPO) is the most widely used policy gradient algorithm in modern reinforcement learning. Introduced by Schulman et al. at OpenAI in 2017, PPO strikes an exceptional balance between implementation simplicity, sample efficiency, and training stability. It has become the default algorithm for continuous control tasks, game-playing agents, and most prominently, the reinforcement learning from human feedback (RLHF) stage of large language model alignment. When practitioners reach for a general-purpose RL algorithm today, PPO is almost always the first choice.

PPO belongs to the family of policy gradient methods, meaning it directly optimizes a parameterized policy rather than learning a value function and deriving a policy from it. Its key innovation is the clipped surrogate objective, which constrains how far the new policy can deviate from the old policy at each update step. This prevents the catastrophically large policy updates that plague vanilla policy gradients while avoiding the computational overhead of second-order methods like TRPO. The result is an algorithm that is nearly as simple to implement as basic policy gradient methods but nearly as stable as trust region methods.

Key points to remember:

- PPO is an on-policy, actor-critic algorithm that collects a batch of experience, then performs multiple epochs of minibatch SGD on that batch before discarding it
- The clipped surrogate objective limits the policy ratio r(theta) = pi_new(a|s) / pi_old(a|s) to the range [1 - epsilon, 1 + epsilon], typically with epsilon = 0.2
- PPO uses Generalized Advantage Estimation (GAE) to compute advantage estimates that balance bias and variance through a parameter lambda
- A shared or separate value network (critic) is trained alongside the policy network (actor) to provide baseline estimates
- PPO handles both discrete and continuous action spaces without modification to its core algorithm
- The algorithm is highly parallelizable through vectorized environments, collecting experience from many environments simultaneously
- PPO is the standard algorithm for the RLHF step in training large language models (InstructGPT, ChatGPT, Claude, Llama)

When to use PPO:

- Continuous or discrete action spaces with moderate dimensionality
- Environments where you can run many parallel instances (games, simulations, robotics simulators)
- RLHF for language model alignment
- When you want a reliable baseline that works well across diverse tasks without heavy hyperparameter tuning
- When implementation simplicity matters (research prototyping, production systems)

When not to use PPO:

- When sample efficiency is critical and you cannot afford to discard data (use off-policy methods like SAC or TD3)
- Very high-dimensional continuous action spaces where SAC may be more effective
- Purely offline RL settings where you only have logged data (use CQL, IQL, or Decision Transformer)
- Simple discrete-action problems where DQN variants may suffice with less engineering overhead

---

## From Policy Gradients to PPO

### The Policy Gradient Foundation

Policy gradient methods optimize a policy pi_theta(a|s) directly by computing the gradient of the expected return with respect to the policy parameters theta. The foundational result is the policy gradient theorem:

```
grad_theta J(theta) = E [ sum_t grad_theta log pi_theta(a_t | s_t) * A_t ]
```

where A_t is an advantage estimate measuring how much better action a_t is compared to the average action in state s_t. The vanilla REINFORCE algorithm uses this directly, but it suffers from high variance and takes very small steps to remain stable.

### The Problem of Step Size

The fundamental challenge in policy optimization is choosing the right step size. Take too small a step and learning is painfully slow. Take too large a step and the policy can change so dramatically that performance collapses, sometimes irrecoverably. This is especially problematic because small changes in parameter space can cause large changes in the probability distribution over actions, and a single bad update can send the policy into a region of state space from which it cannot recover.

### TRPO: The Trust Region Approach

Trust Region Policy Optimization (TRPO), also by Schulman et al. (2015), addressed the step size problem by formulating each policy update as a constrained optimization problem: maximize the expected improvement subject to a constraint on the KL divergence between old and new policies:

```
maximize  E [ (pi_theta(a|s) / pi_theta_old(a|s)) * A_t ]
subject to  E [ KL(pi_theta_old(. | s) || pi_theta(. | s)) ] <= delta
```

TRPO solves this using conjugate gradient descent and a line search with the Fisher information matrix. While theoretically elegant and empirically effective, TRPO is complex to implement, difficult to extend to architectures with shared parameters between policy and value function, and computationally expensive per update due to the second-order optimization.

### PPO: Achieving Trust Region Behavior Simply

PPO achieves similar monotonic improvement guarantees as TRPO but through a first-order method that requires only standard SGD. Instead of imposing a hard KL constraint, PPO modifies the objective function itself to penalize large deviations. This makes PPO compatible with standard deep learning tooling, automatic differentiation, and arbitrary network architectures.

---

## The Clipped Surrogate Objective

### Probability Ratio

The core quantity in PPO is the probability ratio between the new and old policies:

```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

When r_t = 1, the new policy assigns the same probability to the action as the old policy. When r_t > 1, the new policy is more likely to take this action. When r_t < 1, it is less likely.

The standard policy gradient objective (in its importance-sampled form) is:

```
L^CPI(theta) = E_t [ r_t(theta) * A_t ]
```

This is called the conservative policy iteration (CPI) objective. Without any constraint, maximizing this objective can lead to excessively large policy updates because r_t can become very large.

### The Clipping Mechanism

PPO-Clip modifies this objective by clipping the ratio r_t:

```
L^CLIP(theta) = E_t [ min( r_t(theta) * A_t, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_t ) ]
```

where epsilon is a hyperparameter, typically 0.2. The clip function restricts r_t to the interval [1 - epsilon, 1 + epsilon], which is [0.8, 1.2] with the default epsilon.

The min operation is the crucial part. It takes the lower (more pessimistic) value between the unclipped and clipped objectives. This creates two cases:

**When the advantage is positive (A_t > 0)**: The action was better than average. The objective wants to increase r_t (make this action more likely). But the clipped term caps the benefit at (1 + epsilon) * A_t. So the gradient vanishes once r_t exceeds 1 + epsilon, preventing the policy from moving too far in the direction of this good action.

**When the advantage is negative (A_t < 0)**: The action was worse than average. The objective wants to decrease r_t (make this action less likely). But the clipped term caps the benefit at (1 - epsilon) * A_t. The gradient vanishes once r_t falls below 1 - epsilon, preventing the policy from moving too far away from this bad action.

In both cases, the clipping creates a "trust region" around the old policy. Inside this region, the gradient flows normally. Outside it, the gradient is zero (from the clipped branch) and the unclipped branch provides a lower objective value, so the min selects the clipped (flat) branch instead. The net effect is that policy updates are naturally bounded.

### Why Clipping Works Better Than a Penalty

The clipping mechanism is more robust than a fixed KL penalty because it is adaptive at the sample level. Each (state, action) pair has its own ratio r_t, and clipping acts independently on each. Actions where the policy has already changed enough get zero gradient, while actions where the policy has not yet changed enough continue to receive gradient signal. This is fundamentally different from a global KL penalty, which applies a uniform pressure across all state-action pairs.

---

## PPO-Clip vs PPO-Penalty

Schulman et al. proposed two variants of PPO. The clip variant (PPO-Clip) is described above and is overwhelmingly the default in practice. The penalty variant (PPO-Penalty) uses an adaptive KL penalty instead of clipping:

```
L^KLPEN(theta) = E_t [ r_t(theta) * A_t - beta * KL(pi_theta_old(. | s_t) || pi_theta(. | s_t)) ]
```

where beta is a penalty coefficient that is automatically adjusted based on the observed KL divergence:

```
If KL_observed < KL_target / 1.5:  beta <- beta / 2
If KL_observed > KL_target * 1.5:  beta <- beta * 2
```

The adaptive scheme increases beta when the policy is changing too fast and decreases it when the policy is changing too slowly. While this is theoretically sound, it introduces additional hyperparameters (KL_target, the adaptation thresholds) and is more sensitive to tuning.

In the original paper, PPO-Clip matched or outperformed PPO-Penalty across tested benchmarks while being simpler to implement. Virtually all modern PPO implementations use the clip variant. The penalty variant occasionally appears in RLHF pipelines where explicit KL control relative to a reference policy is desired (this is a different KL penalty from the PPO-Penalty variant, applied between the learned policy and a frozen supervised fine-tuned model).

---

## Generalized Advantage Estimation (GAE)

### The Bias-Variance Tradeoff in Advantage Estimation

Computing good advantage estimates is critical for PPO performance. The advantage A_t = Q(s_t, a_t) - V(s_t) measures how much better an action is than the expected value of the state. Different estimators trade off bias and variance:

**Monte Carlo return (high variance, zero bias)**: Use the full discounted return from the trajectory as the Q estimate.

```
A_t^MC = (r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...) - V(s_t)
```

**One-step TD (low variance, high bias)**: Use a single-step bootstrapped estimate.

```
A_t^TD = r_t + gamma * V(s_{t+1}) - V(s_t) = delta_t
```

where delta_t is the TD error. This has low variance because it only depends on a single reward, but it is biased because V is only an approximation.

### GAE: Exponentially-Weighted TD Errors

GAE (Schulman et al., 2016) provides a smooth interpolation between these extremes. It computes the advantage as an exponentially-weighted sum of multi-step TD errors:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

A_t^GAE = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
        = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
```

The parameter lambda controls the tradeoff:

- lambda = 0: Reduces to one-step TD advantage (low variance, high bias)
- lambda = 1: Reduces to Monte Carlo advantage minus the baseline (high variance, low bias)
- lambda = 0.95 (typical): A good balance that works well across most environments

In practice, GAE is computed recursively from the end of the trajectory backward:

```
A_{T} = 0
A_{t} = delta_t + gamma * lambda * A_{t+1}
```

This backward pass is simple and efficient. The GAE parameter lambda is one of the most important hyperparameters in PPO, and 0.95 is a strong default.

### Return Targets for the Value Function

The value function is typically trained to predict the GAE-augmented return:

```
V_target(s_t) = A_t^GAE + V_old(s_t)
```

or equivalently, the discounted return computed from the GAE advantages. The value function loss is simply:

```
L^VF = (V_theta(s_t) - V_target(s_t))^2
```

---

## Value Function Clipping

Some PPO implementations apply clipping to the value function loss, analogous to the policy clipping. The clipped value loss is:

```
V_clipped = V_old(s_t) + clip(V_theta(s_t) - V_old(s_t), -epsilon, +epsilon)

L^VF_clipped = max( (V_theta(s_t) - V_target)^2, (V_clipped - V_target)^2 )
```

The intention is to prevent the value function from changing too drastically in a single update, analogous to the policy clipping. However, empirical evidence on value function clipping is mixed. The paper "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" (Engstrom et al., 2020) found that value clipping sometimes helps and sometimes hurts, depending on the environment. Many modern implementations include it as an option but not a default. The original PPO paper did not include value function clipping.

If you do use value clipping, use the same epsilon as the policy clipping (0.2) as a starting point.

---

## Multiple Epochs of Minibatch Updates

### The Core Training Loop

A defining feature of PPO is that it reuses collected experience for multiple gradient updates before discarding it. The training loop proceeds as follows:

1. Collect T timesteps of experience across N parallel environments using the current policy (total batch size = N * T)
2. Compute advantages and returns for the entire batch using GAE
3. For K epochs (typically K = 3 to 10):
   a. Shuffle the batch and split it into M minibatches
   b. For each minibatch, compute the clipped surrogate loss and take a gradient step
4. Discard the batch and return to step 1

This is possible because the probability ratio r_t(theta) naturally accounts for the difference between the data-collection policy (pi_old) and the current policy (pi_theta). The clipping mechanism ensures that even after several epochs, the policy does not drift too far from the data-collection policy, keeping the importance-sampled estimates valid.

### Why Multiple Epochs Help

In vanilla policy gradient methods, each batch of experience is used for exactly one gradient update and then discarded. This is wasteful because collecting experience is often the bottleneck (expensive environment interactions). Multiple epochs improve sample efficiency by extracting more learning signal from each batch.

The clipping mechanism is what makes this safe. Without it, multiple gradient steps on the same batch would quickly push the policy far from the data-collection policy, making the importance sampling ratios unreliable and causing training instability. With clipping, the ratios are bounded, and once a sample has been "used up" (its ratio is clipped), it contributes zero gradient, effectively removing it from further updates.

### Typical Values

- Number of epochs (K): 3 to 10. More epochs extract more value from each batch but increase the risk of overfitting to the batch. For continuous control, 10 epochs is common. For LLM RLHF, 1 to 4 epochs is typical because the data distribution is more complex.
- Number of minibatches (M): Chosen so that each minibatch has 32 to 512 samples. Larger minibatches give more stable gradients but fewer updates per epoch.

---

## Implementation Details That Matter

The paper "Implementation Matters in Deep Policy Gradients" and the blog post "The 37 Implementation Details of Proximal Policy Optimization" (Huang et al., 2022) demonstrated that PPO's performance depends heavily on implementation details beyond the core algorithm. Many of these details are not mentioned in the original paper but are present in high-performing codebases.

### Vectorized Environments

Running many environment instances in parallel is essential for PPO performance. Rather than collecting a long trajectory from a single environment, PPO typically runs N = 8 to 128 environments simultaneously and collects T = 128 to 2048 steps from each, producing a batch of N * T timesteps.

Vectorized environments improve both wall-clock training speed (parallel data collection) and learning quality (the batch contains diverse experiences from different environment states). Libraries like Gymnasium provide `SyncVectorEnv` and `AsyncVectorEnv` wrappers for this purpose.

### Advantage Normalization

Before using the computed advantages in the loss function, it is standard practice to normalize them to zero mean and unit variance within each minibatch:

```
A_normalized = (A - mean(A)) / (std(A) + 1e-8)
```

This is not mentioned in the original PPO paper but has a significant impact on training stability. Without normalization, the magnitude of the advantage estimates can vary widely across batches, leading to inconsistent gradient scales. Normalization ensures the gradients are consistently scaled regardless of the reward structure.

Important: normalize per minibatch, not across the entire batch. This provides fresh normalization statistics for each gradient step and is the convention in major implementations.

### Observation Normalization and Reward Scaling

For continuous control tasks, it is common to maintain running statistics of observations and normalize them to approximately zero mean and unit variance:

```
s_normalized = (s - running_mean) / (running_std + 1e-8)
```

Similarly, rewards can be normalized using a running estimate of the return variance. These normalizations make the learning problem easier for the neural networks and reduce sensitivity to the specific scales of a given environment.

### Orthogonal Initialization

Initializing the policy and value network weights using orthogonal initialization (with a scale of sqrt(2) for hidden layers and 0.01 for the policy output layer) is a common practice in PPO implementations. The small scale on the policy output layer ensures the initial policy is close to uniform (for discrete actions) or low-variance (for continuous actions), preventing large initial policy updates.

### Entropy Bonus

An entropy bonus is added to the loss to encourage exploration:

```
L = L^CLIP - c_1 * L^VF + c_2 * H(pi_theta)
```

where H(pi_theta) is the entropy of the policy, c_1 is the value loss coefficient (typically 0.5), and c_2 is the entropy coefficient (typically 0.01). The entropy bonus discourages the policy from becoming too deterministic too quickly. For continuous action spaces, this is the differential entropy of the Gaussian policy.

### Learning Rate Annealing

Linearly annealing the learning rate from its initial value to zero over the course of training is a common practice that improves final performance. This allows large updates early when the policy is far from optimal and small, fine-grained updates later.

### Global Gradient Clipping

Clipping the global gradient norm (typically to 0.5) prevents destructive updates from occasional large gradients. This is applied to the combined loss (policy + value + entropy).

---

## PPO for RLHF

### The Connection to LLM Alignment

PPO's most prominent application today is in reinforcement learning from human feedback (RLHF), the process used to align large language models with human preferences. The pipeline, introduced in the InstructGPT paper (Ouyang et al., 2022), has three stages:

1. **Supervised fine-tuning (SFT)**: Fine-tune a pretrained language model on high-quality demonstration data.
2. **Reward model training**: Train a reward model on human preference comparisons (which response is better).
3. **PPO fine-tuning**: Use PPO to optimize the SFT model against the reward model, with a KL penalty to prevent divergence from the SFT model.

In this setting, the RL formulation is:

- **State**: The prompt (input text)
- **Action**: The entire generated response (or token-by-token generation)
- **Reward**: The reward model's score for the (prompt, response) pair, minus a KL penalty term: R = R_model(prompt, response) - beta * KL(pi_theta || pi_ref)
- **Policy**: The language model itself, pi_theta(response | prompt)

### Why PPO for RLHF

PPO is used for RLHF because of several properties:

- **Stability**: The clipped objective prevents the language model from changing too drastically in a single update, which could cause mode collapse or degenerate outputs.
- **Simplicity**: PPO requires only first-order gradients and standard training infrastructure. Given the already enormous complexity of LLM training pipelines, algorithmic simplicity is highly valued.
- **Compatibility with KL constraints**: The KL penalty between the policy and the reference (SFT) model is naturally incorporated as part of the reward. This keeps the model close to the SFT model, preventing reward hacking.
- **Proven track record**: PPO was used in InstructGPT, ChatGPT, and many subsequent systems, establishing a strong empirical precedent.

### RLHF-Specific Implementation Details

RLHF PPO differs from standard RL PPO in several ways:

- **Large model sizes**: The policy, value function, reward model, and reference model may each be billions of parameters. Memory management and distributed training are critical.
- **Token-level vs sequence-level**: Some implementations treat each token as a separate action (token-level MDP), while others treat the entire response as a single action (bandit formulation). Token-level is more common in practice.
- **KL penalty as reward shaping**: The KL divergence between the current policy and the reference model is typically added per-token as a reward signal, not as a separate loss term.
- **Fewer epochs**: RLHF typically uses 1 to 4 PPO epochs per batch, fewer than the 3 to 10 common in standard RL, because the data distribution is more complex and overfitting is a greater risk.
- **Generation cost**: Generating responses during data collection is expensive (autoregressive inference). This makes the ratio of compute for generation vs. training heavily skewed toward generation.

### Alternatives to PPO in RLHF

Several alternatives to PPO for RLHF have been proposed:

- **DPO (Direct Preference Optimization)**: Eliminates the need for a separate reward model and RL training by directly optimizing on preference data. Simpler but may underperform PPO on complex tasks.
- **REINFORCE variants**: Simpler policy gradient methods with various variance reduction techniques. Used in some production systems.
- **GRPO (Group Relative Policy Optimization)**: Samples multiple responses per prompt and uses relative rankings to estimate advantages, eliminating the need for a separate value model. Used in DeepSeek-R1.

PPO remains the most common choice for RLHF as of early 2026, though DPO and its variants are increasingly popular for their simplicity.

---

## Hyperparameters and Practical Tuning

### Key Hyperparameters

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Clip epsilon | 0.2 | The most critical hyperparameter. Rarely needs tuning. |
| GAE lambda | 0.95 | Controls bias-variance tradeoff. 0.95 is a strong default. |
| Discount factor (gamma) | 0.99 | Lower (0.95-0.98) for shorter-horizon tasks. |
| Number of epochs (K) | 3-10 | More epochs = more sample efficiency but risk of overfitting. |
| Number of environments (N) | 8-128 | More environments = more diverse data per batch. |
| Steps per environment (T) | 128-2048 | Longer rollouts capture more temporal structure. |
| Learning rate | 3e-4 | Adam optimizer. Often linearly annealed to 0. |
| Entropy coefficient | 0.01 | Increase if exploration is insufficient. |
| Value loss coefficient | 0.5 | Standard value. |
| Max gradient norm | 0.5 | Global gradient clipping. |
| Minibatch size | 32-512 | Batch size / number of minibatches. |

### Tuning Strategy

1. **Start with defaults**: The default hyperparameters from CleanRL or Stable Baselines3 are well-tuned for standard benchmarks. Begin here and adjust only if needed.

2. **Tune the batch size first**: The total batch size (N * T) has the largest impact on sample efficiency. Larger batches are almost always better if compute allows. For continuous control, N=1, T=2048 or N=8, T=256 are common.

3. **Adjust epochs and minibatch size**: If the policy changes too much per iteration (the approximate KL divergence between old and new policies exceeds 0.01-0.02), reduce the number of epochs or increase the minibatch size. If it changes too little, increase epochs.

4. **Monitor the clip fraction**: The fraction of samples where the ratio is clipped should be between 0.1 and 0.3 during healthy training. If it is near zero, the policy is barely changing (increase learning rate or epochs). If it is above 0.5, updates are too aggressive (decrease learning rate or epochs).

5. **Monitor the approximate KL divergence**: Log the mean KL divergence between old and new policies after each update iteration. Healthy values are typically 0.005 to 0.02. Large spikes indicate training instability.

6. **Monitor the explained variance**: The explained variance of the value function (1 - Var(returns - V_pred) / Var(returns)) should approach 1.0 over training. Low explained variance means the value function is a poor predictor of returns, which degrades advantage estimation.

7. **Entropy**: If the policy entropy collapses early, increase the entropy coefficient. If it remains too high, the policy may not be learning; check the reward signal and advantage computation.

---

## PyTorch Implementation

The following is a complete, working PPO implementation for environments with vector-based state spaces. It follows the CleanRL style: single-file, well-commented, and faithful to the implementation details described above.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for policy/value network layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Combined actor-critic network with shared or separate backbones."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # Critic (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        # Actor (policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value


class PPOAgent:
    """PPO-Clip agent with GAE and minibatch updates."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=4,
        num_minibatches=4,
        device="cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.device = torch.device(device)

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: (T, N) tensor of rewards
            values: (T, N) tensor of value predictions
            dones: (T, N) tensor of done flags
            next_value: (N,) tensor of value prediction for the state after the last step

        Returns:
            advantages: (T, N) tensor
            returns: (T, N) tensor
        """
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns, values_old):
        """Perform PPO update with multiple epochs of minibatch SGD.

        All inputs are flattened tensors of shape (batch_size, ...).
        """
        batch_size = states.shape[0]
        minibatch_size = batch_size // self.num_minibatches

        # Normalize advantages across the full batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.num_epochs):
            # Random permutation for minibatch creation
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Get current policy and value predictions
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(mb_states, mb_actions)
                )
                new_values = new_values.squeeze(-1)

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (unclipped MSE)
                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        # Compute diagnostics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = (
                ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }


# -------------------------------------------------------
# Example training loop (requires gymnasium)
# -------------------------------------------------------
# import gymnasium as gym
#
# # Hyperparameters
# num_envs = 4
# num_steps = 128         # Steps per environment per rollout
# total_timesteps = 500_000
#
# # Create vectorized environments
# envs = gym.vector.SyncVectorEnv(
#     [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
# )
# state_dim = envs.single_observation_space.shape[0]
# action_dim = envs.single_action_space.n
#
# agent = PPOAgent(
#     state_dim=state_dim,
#     action_dim=action_dim,
#     num_epochs=4,
#     num_minibatches=4,
# )
#
# num_updates = total_timesteps // (num_envs * num_steps)
#
# for update in range(num_updates):
#     # Storage for rollout
#     all_states = torch.zeros((num_steps, num_envs, state_dim))
#     all_actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
#     all_log_probs = torch.zeros((num_steps, num_envs))
#     all_rewards = torch.zeros((num_steps, num_envs))
#     all_dones = torch.zeros((num_steps, num_envs))
#     all_values = torch.zeros((num_steps, num_envs))
#
#     obs, _ = envs.reset() if update == 0 else (obs, info)
#     obs = torch.FloatTensor(obs)
#
#     # Collect rollout
#     for step in range(num_steps):
#         with torch.no_grad():
#             action, log_prob, _, value = agent.network.get_action_and_value(obs)
#
#         all_states[step] = obs
#         all_actions[step] = action
#         all_log_probs[step] = log_prob
#         all_values[step] = value.squeeze(-1)
#
#         next_obs, reward, terminated, truncated, info = envs.step(action.numpy())
#         done = np.logical_or(terminated, truncated)
#
#         all_rewards[step] = torch.FloatTensor(reward)
#         all_dones[step] = torch.FloatTensor(done.astype(float))
#
#         obs = torch.FloatTensor(next_obs)
#
#     # Compute GAE
#     with torch.no_grad():
#         next_value = agent.network.get_value(obs).squeeze(-1)
#     advantages, returns = agent.compute_gae(
#         all_rewards, all_values, all_dones, next_value
#     )
#
#     # Flatten (T, N, ...) -> (T*N, ...)
#     batch_states = all_states.reshape(-1, state_dim)
#     batch_actions = all_actions.reshape(-1)
#     batch_log_probs = all_log_probs.reshape(-1)
#     batch_advantages = advantages.reshape(-1)
#     batch_returns = returns.reshape(-1)
#     batch_values = all_values.reshape(-1)
#
#     # PPO update
#     metrics = agent.update(
#         batch_states, batch_actions, batch_log_probs,
#         batch_advantages, batch_returns, batch_values,
#     )
#
#     if update % 10 == 0:
#         print(f"Update {update}, KL: {metrics['approx_kl']:.4f}, "
#               f"Clip: {metrics['clip_fraction']:.3f}, "
#               f"Entropy: {metrics['entropy']:.3f}")
```

### Implementation Notes

- **Separate actor and critic networks**: The actor and critic have separate network bodies (no shared layers). Shared architectures can work but are harder to tune because the policy and value losses can interfere. Separate networks are the safer default.

- **Orthogonal initialization with small policy output scale**: The policy output layer is initialized with std=0.01, making the initial action probabilities nearly uniform. This prevents the policy from confidently taking wrong actions at the start of training, which would create large initial advantages and unstable updates.

- **Tanh activations**: Tanh is used instead of ReLU in many PPO implementations. This is a convention from continuous control (MuJoCo) that works well in practice, though ReLU also works for many tasks.

- **Adam epsilon = 1e-5**: A smaller Adam epsilon than the default (1e-8) prevents numerical issues when gradients are very small, which can happen with the clipped objective.

- **Approximate KL for diagnostics**: The KL divergence approximation (ratio - 1 - log_ratio) is a second-order approximation that is cheaper to compute than the exact KL and sufficient for monitoring purposes.

---

## Comparison with Related Algorithms

### PPO vs TRPO

PPO and TRPO are siblings that share the same theoretical motivation (trust region optimization) but differ in implementation:

| Aspect | TRPO | PPO |
|---|---|---|
| Constraint mechanism | Hard KL constraint solved with conjugate gradients | Clipped surrogate objective with first-order SGD |
| Computational cost per update | High (second-order optimization, line search) | Low (standard gradient descent) |
| Implementation complexity | High (Fisher-vector products, conjugate gradient solver) | Low (fits in ~100 lines of core logic) |
| Shared policy-value network | Difficult (constraint applies to full parameter space) | Easy (just add losses together) |
| Minibatch updates | Typically single full-batch update | Multiple epochs of minibatch SGD |
| Performance | Strong, especially in continuous control | Matches or exceeds TRPO in most settings |
| Adoption | Limited (mostly research) | Universal (research and production) |

TRPO guarantees monotonic improvement under certain assumptions, while PPO provides this only approximately. In practice, PPO's simplicity and scalability have made it the clear winner. TRPO is primarily of historical and theoretical interest.

### PPO vs A2C

A2C (Advantage Actor-Critic) is the synchronous version of A3C and can be viewed as a predecessor to PPO in spirit. Both are on-policy actor-critic methods that use advantage estimation, but they differ in key ways:

| Aspect | A2C | PPO |
|---|---|---|
| Policy update | Single gradient step per batch | Multiple epochs of minibatch SGD per batch |
| Update constraint | None (relies on small learning rate) | Clipped surrogate objective |
| Sample efficiency | Lower (one gradient step per batch) | Higher (multiple reuses of each batch) |
| Stability | Can be unstable with large learning rates | Robust due to clipping |
| Implementation | Simpler | Slightly more complex (clipping, ratio computation) |

PPO strictly dominates A2C in practice: it uses the same data collection procedure but extracts more learning from each batch through multiple epochs, and the clipping mechanism provides a safety net against destructive updates. There is essentially no reason to use A2C when PPO is available.

### PPO vs SAC

SAC (Soft Actor-Critic) is the primary off-policy alternative to PPO for continuous control:

| Aspect | PPO | SAC |
|---|---|---|
| Data usage | On-policy (discard after use) | Off-policy (replay buffer) |
| Sample efficiency | Moderate | High |
| Wall-clock efficiency | High (parallelizable) | Moderate (sequential updates) |
| Action spaces | Discrete and continuous | Primarily continuous |
| Exploration | Entropy bonus, stochastic policy | Maximum entropy framework (built-in) |
| Hyperparameter sensitivity | Low | Moderate |
| Use in RLHF | Standard | Rare |

Choose PPO when you can run many parallel environments and wall-clock time matters more than sample efficiency. Choose SAC when environment interactions are expensive and you need maximum sample efficiency.

---

## Why PPO Dominates

PPO's dominance in the RL landscape is not due to any single factor but the combination of several:

**Simplicity**: PPO can be implemented in under 200 lines of core logic. It uses standard first-order optimization, works with any network architecture, and integrates cleanly with existing deep learning frameworks. This means faster iteration, easier debugging, and lower barrier to entry.

**Stability**: The clipped objective provides robust training dynamics across a wide range of hyperparameters. Unlike vanilla policy gradients, PPO rarely diverges catastrophically. Unlike TRPO, it does not require careful second-order optimization.

**Generality**: PPO works with discrete actions, continuous actions, multi-dimensional action spaces, pixel observations, vector observations, and everything in between. The same algorithm and hyperparameters often transfer across very different domains.

**Scalability**: PPO is embarrassingly parallel during data collection. It scales naturally from a single CPU to thousands of distributed workers, making it suitable for both research experiments and large-scale production systems like RLHF.

**Ecosystem**: PPO has robust implementations in every major RL library (Stable Baselines3, CleanRL, RLlib, TRL). The extensive tooling, documentation, and community knowledge make it the path of least resistance for most RL applications.

**Proven track record**: From Dota 2 (OpenAI Five) to robotic manipulation (OpenAI's robot hand) to LLM alignment (InstructGPT, ChatGPT), PPO has demonstrated strong performance on some of the most challenging RL problems ever attempted. This track record gives practitioners confidence in choosing PPO as their starting point.

PPO is not always the best algorithm for every specific problem. Off-policy methods like SAC can be more sample-efficient. Model-based methods can be faster in low-data regimes. But PPO's combination of simplicity, reliability, and broad applicability makes it the default first choice, and it is frequently good enough to be the last choice as well.
