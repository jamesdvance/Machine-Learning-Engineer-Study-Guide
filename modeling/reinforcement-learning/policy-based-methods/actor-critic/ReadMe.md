# Actor-Critic Methods

## Summary

Actor-critic methods are a family of reinforcement learning algorithms that combine policy gradient methods (the actor) with learned value functions (the critic). The actor maintains a parameterized policy that maps states to action probabilities, while the critic learns a value function that evaluates how good the actor's decisions are. By using the critic's value estimates to reduce the variance of policy gradient updates, actor-critic methods address the central weakness of pure policy gradient algorithms like REINFORCE -- high variance in gradient estimates -- while preserving the low bias that comes from directly optimizing the policy.

The core idea is straightforward. REINFORCE uses the full Monte Carlo return to weight policy gradient updates, which is unbiased but noisy because it depends on the entire trajectory of rewards from a given time step onward. Actor-critic methods replace this Monte Carlo return with a bootstrapped estimate from the critic, typically a one-step temporal-difference (TD) target. This introduces a small amount of bias (because the critic's estimate is imperfect) but dramatically reduces variance, allowing the agent to learn from individual transitions rather than waiting for complete episodes. The result is an algorithm that can learn online, step by step, and converges more reliably in practice.

Key points to remember:

- Actor-critic = policy gradient (actor) + value function approximation (critic)
- The actor outputs a probability distribution over actions; the critic outputs a scalar value estimate
- The critic's role is to provide a lower-variance baseline or advantage signal for the actor's gradient updates
- The advantage function A(s, a) = Q(s, a) - V(s) measures how much better an action is compared to the average; TD error serves as a one-sample estimate of the advantage
- Actor and critic can share a neural network backbone (parameter sharing) or use entirely separate networks
- Generalized Advantage Estimation (GAE) provides a tunable tradeoff between bias and variance in the advantage estimate
- Actor-critic methods can learn online (one step at a time) unlike REINFORCE which requires complete episodes
- Nearly all modern deep RL algorithms (A2C, A3C, PPO, TRPO, SAC, DDPG, TD3) are actor-critic methods

When to use actor-critic methods:

- Continuous or large discrete action spaces where value-based methods struggle
- Online learning settings where you cannot wait for episode completion
- Problems that benefit from stochastic policies (exploration, multimodal action distributions)
- When REINFORCE is too unstable due to high-variance gradient estimates
- As the foundation for more advanced algorithms like PPO or SAC

When not to use actor-critic methods:

- Small discrete action spaces where DQN variants are simpler and well-understood
- When you need off-policy learning with maximum sample efficiency (consider DQN with prioritized replay)
- Extremely simple environments where tabular methods or basic Q-learning suffice

---

## From REINFORCE to Actor-Critic

### REINFORCE Recap

REINFORCE is the simplest policy gradient algorithm. It parameterizes a policy pi_theta(a|s) with a neural network and updates the parameters theta to maximize expected return. The gradient of the expected return with respect to theta is:

```
grad_theta J(theta) = E_pi [ sum_t grad_theta log pi_theta(a_t | s_t) * G_t ]
```

where G_t is the return (cumulative discounted reward) from time step t onward. REINFORCE collects a full episode, computes G_t for each time step, and uses these as weights for the policy gradient.

The problem is variance. G_t is a sum of many random variables (all future rewards), so it fluctuates enormously across episodes. Two episodes starting from the same state can produce wildly different returns due to stochastic transitions and actions. This makes the gradient estimate very noisy, requiring many episodes to average out and converge.

### The Baseline Trick

A standard technique to reduce variance in REINFORCE is to subtract a baseline b(s_t) from the return:

```
grad_theta J(theta) = E_pi [ sum_t grad_theta log pi_theta(a_t | s_t) * (G_t - b(s_t)) ]
```

Any baseline that does not depend on the action preserves the unbiasedness of the gradient estimator (since E_a[grad log pi * b(s)] = 0 for any function of state alone), but a well-chosen baseline can dramatically reduce variance. The optimal baseline in the mean-squared-error sense is approximately the state-value function V(s_t), because it centers the weighting around zero: actions that are better than average get positive weight, worse-than-average actions get negative weight.

This motivates learning V(s) as a baseline. But once you are learning a value function anyway, you can go further.

### The Actor-Critic Idea

Instead of using the full Monte Carlo return G_t and subtracting a learned baseline, actor-critic methods replace G_t entirely with a bootstrapped estimate from the critic. In the simplest one-step form:

```
G_t is replaced by: r_t + gamma * V(s_{t+1})
```

The advantage estimate becomes the TD error:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

and the policy gradient update becomes:

```
grad_theta J(theta) approx grad_theta log pi_theta(a_t | s_t) * delta_t
```

This is the fundamental actor-critic update. The TD error delta_t is a one-sample, one-step estimate of the advantage A(s_t, a_t) = Q(s_t, a_t) - V(s_t). It tells the actor: "this action was delta_t better (or worse) than what the critic expected." If delta_t is positive, the action led to a better-than-expected outcome, so the actor increases its probability. If delta_t is negative, the actor decreases it.

The tradeoff is clear. REINFORCE uses the full return G_t, which is unbiased but high variance. The one-step TD error uses only one reward and bootstraps from V(s_{t+1}), which has much lower variance but introduces bias because V is an approximation. This bias-variance tradeoff is the core design axis of actor-critic methods.

---

## Architecture

### Separate Networks

The simplest architecture uses two entirely independent neural networks: one for the actor (policy) and one for the critic (value function). Each has its own parameters and optimizer.

```
Actor network:  state -> hidden layers -> action probabilities (softmax for discrete, mean/std for continuous)
Critic network: state -> hidden layers -> single scalar V(s)
```

Advantages of separate networks:

- No interference between actor and critic gradients
- Each network can have its own architecture, learning rate, and capacity
- Easier to debug since the two components are decoupled
- The critic can be trained more aggressively without destabilizing the policy

Disadvantages:

- More total parameters and memory usage
- No shared representation learning; both networks must independently learn useful state features

### Shared Network with Separate Heads

A more common architecture in deep RL uses a shared backbone (feature extractor) with two output heads: one for the policy and one for the value function.

```
state -> shared hidden layers -> policy head -> action probabilities
                              -> value head  -> single scalar V(s)
```

Advantages of shared networks:

- Shared feature extraction reduces total parameter count
- The critic's value learning can help the actor learn useful state representations (and vice versa)
- More parameter-efficient, especially for high-dimensional inputs like images

Disadvantages:

- Gradient interference: actor and critic losses flow through the same backbone, and their gradients may conflict
- Requires careful loss weighting (the total loss is typically L_actor + c * L_critic + entropy_bonus)
- A poorly tuned value loss coefficient can degrade policy performance

In practice, the shared-backbone architecture is dominant in algorithms like A2C and PPO when applied to environments with visual observations, because learning a good visual representation is expensive and sharing it is efficient.

### Network Architecture for Continuous Actions

For continuous action spaces, the actor typically outputs the parameters of a probability distribution rather than a softmax over discrete choices. The most common choice is a Gaussian policy:

```
Actor output: mean mu(s) and log standard deviation log_sigma(s)
Action sampling: a ~ N(mu(s), sigma(s)^2)
Log probability: log pi(a|s) = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
```

The standard deviation can be a separate learned parameter (state-independent) or output by the network (state-dependent). State-dependent variance allows the agent to be more uncertain in some states than others, which can improve exploration.

---

## Advantage Function Estimation

The advantage function is central to actor-critic methods. It measures how much better a particular action is compared to the average action in a given state:

```
A(s, a) = Q(s, a) - V(s)
```

The actor's gradient update is weighted by the advantage. Actions with positive advantage are reinforced; actions with negative advantage are suppressed. The question is how to estimate A(s, a) accurately.

### TD Error as Advantage Estimate

The simplest advantage estimate is the one-step TD error:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

This is an unbiased estimate of the advantage if V is the true value function. In practice V is learned and imperfect, so the estimate is biased. However, the variance is low because it depends on only one reward.

### N-Step Returns

We can interpolate between the one-step TD estimate and the full Monte Carlo return using n-step returns:

```
A_t^(n) = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n}) - V(s_t)
```

- n = 1: one-step TD (low variance, higher bias)
- n = infinity: Monte Carlo return (zero bias, high variance)
- Intermediate n: a tradeoff

Choosing n is a hyperparameter. Larger n reduces bias (less reliance on V's accuracy) but increases variance (more stochastic rewards in the sum).

### Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation (Schulman et al., 2016) provides a principled, continuous tradeoff between bias and variance using an exponentially-weighted average of all n-step advantage estimates. GAE is parameterized by lambda in [0, 1]:

```
A_t^GAE(gamma, lambda) = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
```

where delta_{t+l} = r_{t+l} + gamma * V(s_{t+l+1}) - V(s_{t+l}) is the TD error at time step t+l.

The lambda parameter controls the tradeoff:

- lambda = 0: A_t = delta_t (one-step TD error, low variance, higher bias)
- lambda = 1: A_t = G_t - V(s_t) (Monte Carlo advantage, zero bias if V is correct, high variance)
- lambda in (0, 1): exponentially-weighted blend of n-step advantages

In practice, lambda = 0.95 is a common default. GAE can be computed efficiently in a single backward pass through the trajectory:

```
A_{T-1} = delta_{T-1}
A_t = delta_t + gamma * lambda * A_{t+1}    (for t = T-2, T-3, ..., 0)
```

GAE is the standard advantage estimator in PPO and most modern actor-critic implementations. It is one of those ideas that appears in nearly every competitive deep RL system.

---

## Online (One-Step) vs. Batch Actor-Critic

### Online Actor-Critic

The original actor-critic formulation updates both the actor and the critic after every single environment step. The agent observes a transition (s, a, r, s'), computes the TD error, and immediately updates both networks:

```
1. Observe state s, sample action a ~ pi_theta(.|s)
2. Take action, observe reward r and next state s'
3. Compute TD error: delta = r + gamma * V_w(s') - V_w(s)
4. Update critic: w <- w + alpha_w * delta * grad_w V_w(s)
5. Update actor: theta <- theta + alpha_theta * delta * grad_theta log pi_theta(a|s)
6. s <- s'
```

Online actor-critic is fully incremental and can be applied to continuing (non-episodic) tasks. However, it has limitations:

- Each update uses only one sample, so gradient estimates are very noisy
- Requires careful tuning of two learning rates
- Sensitive to the correlation between consecutive samples (similar to Q-learning without replay)

### Batch Actor-Critic (A2C Style)

Modern actor-critic methods typically collect a batch of experience (a rollout of T steps, or multiple parallel rollouts) before performing updates. This is the approach used by A2C and PPO:

```
1. Collect T steps of experience using current policy: {(s_t, a_t, r_t, s_{t+1})}
2. Compute advantage estimates for all steps (using GAE or n-step returns)
3. Compute actor loss: L_actor = -mean_t [ log pi_theta(a_t | s_t) * A_t ]
4. Compute critic loss: L_critic = mean_t [ (V_w(s_t) - target_t)^2 ]
5. Update networks using combined loss (possibly with entropy bonus)
```

Batch updates reduce noise by averaging over multiple transitions. Running multiple environments in parallel (vectorized environments) further increases the batch diversity and throughput. This is the standard approach in practice and is what frameworks like Stable Baselines3 implement.

The batch approach also enables techniques that are impossible in the purely online setting, such as multiple optimization epochs over the same batch (as in PPO) or trust region constraints (as in TRPO).

---

## The Critic's Training Objective

The critic learns V(s) by minimizing the squared error between its predictions and the bootstrapped target:

```
L_critic = (1/T) * sum_t (V_w(s_t) - y_t)^2
```

where the target y_t can be:

- One-step TD target: y_t = r_t + gamma * V_w(s_{t+1})  (with V_w(s_{t+1}) detached from the computation graph)
- N-step return: y_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V_w(s_{t+n})
- GAE-based target: y_t = A_t^GAE + V_w(s_t)  (which simplifies to the lambda-return)

Some implementations clip the value function loss (PPO-style) to prevent large updates that destabilize the critic. Others use the Huber loss instead of MSE for robustness to outlier returns.

An important implementation detail: when computing the advantage for the actor update, the critic's value predictions should be treated as constants (detached from the computation graph). The actor's gradient should not flow through the critic's outputs.

---

## Entropy Regularization

A common addition to actor-critic methods is an entropy bonus in the actor's loss:

```
L_total = L_actor + c_v * L_critic - c_e * H(pi_theta)
```

where H(pi_theta) = -sum_a pi_theta(a|s) * log pi_theta(a|s) is the entropy of the policy, c_v is the value loss coefficient (typically 0.5), and c_e is the entropy coefficient (typically 0.01).

The entropy bonus encourages the policy to maintain some randomness, preventing premature convergence to a deterministic policy. Without it, the actor can collapse to always choosing one action early in training, even if that action is only slightly better according to the (still inaccurate) critic. The entropy term keeps the policy exploring.

---

## PyTorch Implementation

The following is a complete, working actor-critic implementation for environments with discrete action spaces. It uses the batch (A2C-style) approach with GAE.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Value head (critic)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

    def get_action_and_value(self, state):
        """Sample an action and return its log probability and the state value."""
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_action(self, state, action):
        """Compute log probability and value for a given state-action pair."""
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: list of rewards for each time step
        values: list of V(s_t) predictions (detached)
        dones: list of done flags
        next_value: V(s_T) for the state after the last step
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: tensor of GAE advantage estimates
        returns: tensor of target returns for the critic (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # If the episode ended at step t, there is no future value
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = delta + gamma * lam * next_non_terminal * last_gae
        last_gae = advantages[t]

    returns = advantages + torch.tensor(values)
    return advantages, returns


class ActorCriticAgent:
    """A2C-style actor-critic agent."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        rollout_length=128,
        device="cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.device = torch.device(device)

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def collect_rollout(self, env, state):
        """
        Collect a rollout of experience from the environment.

        Args:
            env: Gymnasium environment
            state: current environment state

        Returns:
            Rollout data and the final state
        """
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(self.rollout_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(
                    state_tensor
                )

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            if done:
                state, _ = env.reset()
            else:
                state = next_state

        return states, actions, rewards, dones, log_probs, values, state

    def update(self, states, actions, rewards, dones, log_probs_old, values_old, last_state):
        """
        Compute losses and update the actor-critic network.
        """
        # Compute the value of the state after the last step in the rollout
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state_tensor)
            last_value = last_value.item()

        # Compute GAE advantages and returns
        advantages, returns = compute_gae(
            rewards, values_old, dones, last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages (reduces variance, common practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages_tensor = advantages.to(self.device)
        returns_tensor = returns.to(self.device)

        # Forward pass through current network
        log_probs_new, entropy, values_new = self.network.evaluate_action(
            states_tensor, actions_tensor
        )

        # Actor loss: negative of advantage-weighted log probabilities
        actor_loss = -(log_probs_new * advantages_tensor.detach()).mean()

        # Critic loss: squared error between predicted values and target returns
        critic_loss = (values_new.squeeze() - returns_tensor.detach()).pow(2).mean()

        # Entropy bonus: encourages exploration
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            actor_loss
            + self.value_loss_coef * critic_loss
            + self.entropy_coef * entropy_loss
        )

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss.item(),
        }


# -------------------------
# Example training loop
# -------------------------
# import gymnasium as gym
#
# env = gym.make("CartPole-v1")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
#
# agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim)
#
# state, _ = env.reset()
# total_steps = 0
# episode_reward = 0
# episode_count = 0
#
# for iteration in range(1000):
#     # Collect a rollout
#     states, actions, rewards, dones, log_probs, values, state = (
#         agent.collect_rollout(env, state)
#     )
#     total_steps += agent.rollout_length
#
#     # Update the network
#     losses = agent.update(states, actions, rewards, dones, log_probs, values, state)
#
#     # Track episode rewards (simplified; in practice use a wrapper)
#     episode_reward += sum(rewards)
#     n_dones = sum(dones)
#     if n_dones > 0:
#         episode_count += int(n_dones)
#         if iteration % 10 == 0:
#             print(
#                 f"Step {total_steps}, Episodes: {episode_count}, "
#                 f"Actor Loss: {losses['actor_loss']:.4f}, "
#                 f"Critic Loss: {losses['critic_loss']:.4f}, "
#                 f"Entropy: {losses['entropy']:.4f}"
#             )
#         episode_reward = 0
```

### Implementation Notes

- **Shared backbone**: The `ActorCritic` module uses a shared two-layer MLP with separate linear heads for the policy and value. For image-based environments, replace the MLP backbone with a CNN.

- **GAE computation**: The `compute_gae` function iterates backward through the rollout, accumulating the exponentially-weighted sum of TD errors. The `next_non_terminal` mask ensures that bootstrapping does not cross episode boundaries.

- **Advantage normalization**: Subtracting the mean and dividing by the standard deviation of the advantages within each batch is a standard practice that stabilizes training. Without this, the scale of the advantage can vary widely between batches.

- **`detach()` and gradient flow**: The advantages are detached when computing the actor loss so that gradients do not flow from the actor's loss through the critic's value predictions. Similarly, the returns are detached for the critic loss. Each head's loss should only update the relevant parameters (though in a shared network, both gradients flow through the backbone).

- **Gradient clipping**: `clip_grad_norm_` with a small max norm (0.5) prevents large updates that can destabilize the shared network. This is especially important with shared architectures where actor and critic gradients combine.

---

## Connection to Related Methods

### Predecessor: REINFORCE

REINFORCE is the simplest policy gradient method and the direct predecessor of actor-critic. It uses the full Monte Carlo return G_t to weight the policy gradient, with no learned value function (or optionally, a learned baseline that does not participate in bootstrapping). Actor-critic methods can be seen as REINFORCE where the return is replaced by a bootstrapped, lower-variance advantage estimate from a critic.

The progression is:

1. REINFORCE: grad = log pi * G_t (high variance, no bias)
2. REINFORCE with baseline: grad = log pi * (G_t - V(s)) (reduced variance, no bias)
3. Actor-critic: grad = log pi * delta_t (low variance, some bias from bootstrapping)

### Siblings: A2C and A3C

**A2C (Advantage Actor-Critic)** is the synchronous, batch version of actor-critic. It runs multiple environment instances in parallel, collects rollouts from all of them, and performs a single synchronized gradient update. A2C is essentially the algorithm described in this chapter with parallel environments for efficiency.

**A3C (Asynchronous Advantage Actor-Critic)** (Mnih et al., 2016) was the original deep actor-critic breakthrough. It runs multiple actor-learner threads asynchronously, each interacting with its own environment copy and computing gradients independently. These gradients are applied to a shared parameter server without synchronization. The asynchrony provides natural exploration diversity (different threads explore different parts of the state space) and was originally motivated by the need to decorrelate training data without a replay buffer.

In practice, A2C has largely replaced A3C because synchronous updates are simpler to implement, easier to reproduce, and perform equally well or better when using vectorized environments on a GPU. A3C's asynchronous updates can cause stale gradients and are harder to scale to GPU-based training.

### Siblings: PPO and TRPO

**TRPO (Trust Region Policy Optimization)** (Schulman et al., 2015) adds a constraint to the actor-critic update: the KL divergence between the old and new policy must stay within a trust region. This prevents destructively large policy updates. TRPO solves a constrained optimization problem at each step using conjugate gradient and line search, which makes it more complex to implement.

**PPO (Proximal Policy Optimization)** (Schulman et al., 2017) achieves a similar effect to TRPO but with a much simpler implementation. Instead of an explicit KL constraint, PPO clips the probability ratio between new and old policies:

```
L_clip = min(ratio * A, clip(ratio, 1-epsilon, 1+epsilon) * A)
```

where ratio = pi_new(a|s) / pi_old(a|s) and epsilon is typically 0.2. This clipping prevents the policy from changing too much in a single update. PPO is the most widely used deep RL algorithm in practice due to its simplicity, stability, and strong performance.

Both TRPO and PPO are actor-critic methods at their core. They use the same advantage estimation (typically GAE) and critic training as described in this chapter, with the only difference being how the actor's loss constrains the policy update.

### Continuous Action Extensions: DDPG, TD3, SAC

For continuous action spaces, several actor-critic variants use a deterministic policy (DDPG, TD3) or a stochastic policy with entropy regularization (SAC). These methods learn Q(s, a) as the critic instead of V(s), which enables off-policy learning with a replay buffer. They are covered in detail in their respective chapters but share the same fundamental actor-critic structure: the actor proposes actions, the critic evaluates them, and the critic's feedback guides the actor's improvement.

---

## When to Use Actor-Critic vs. Alternatives

### Actor-Critic vs. Pure Policy Gradient (REINFORCE)

Use actor-critic over REINFORCE in almost all practical settings. REINFORCE is useful for understanding the theory and for very simple problems, but its high variance makes it impractical for anything beyond toy environments. The only situation where REINFORCE might be preferred is when you need a completely unbiased gradient estimate and cannot tolerate any bootstrapping bias -- but this is rare.

### Actor-Critic vs. Pure Value-Based (DQN)

The choice depends primarily on the action space:

- **Discrete actions**: DQN and its variants (Double DQN, Dueling DQN, Rainbow) are strong baselines. Actor-critic methods work too but may be unnecessary complexity for small action spaces. However, actor-critic methods handle large discrete action spaces better because they do not need to enumerate all actions.

- **Continuous actions**: Actor-critic methods are the standard. DQN cannot be directly applied to continuous actions. Methods like SAC, TD3, and PPO are the go-to choices.

- **Stochastic policies**: If the optimal policy is genuinely stochastic (e.g., in partially observable environments or games requiring mixed strategies), actor-critic methods naturally represent stochastic policies. Value-based methods derive deterministic policies by taking the argmax of Q-values.

### Actor-Critic vs. Model-Based RL

Model-based methods learn a model of the environment's dynamics and use it for planning. They are typically more sample-efficient than actor-critic methods but require the model to be accurate enough to plan with. If you have a limited budget of environment interactions and the dynamics are learnable, consider model-based approaches. If you have plentiful environment interaction (simulators, games) or the dynamics are too complex to model, actor-critic is the standard choice.

### Practical Decision Guide

For most problems in 2024-2025, the default recommendation is:

- **Discrete actions, single environment**: Start with DQN or Rainbow.
- **Discrete actions, parallel environments**: Start with PPO.
- **Continuous actions, on-policy**: Start with PPO.
- **Continuous actions, off-policy with replay**: Start with SAC or TD3.
- **Any of the above, maximum simplicity**: Start with PPO. It is the most general and forgiving algorithm.

All of the above except DQN/Rainbow are actor-critic methods. The actor-critic framework is the workhorse of modern deep RL.

---

## Hyperparameters and Practical Tips

### Key Hyperparameters

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Learning rate | 3e-4 | Adam optimizer; lower for complex environments |
| Discount factor (gamma) | 0.99 | Standard for most tasks |
| GAE lambda | 0.95 | Controls bias-variance tradeoff in advantage estimation |
| Rollout length | 128 - 2048 | Longer rollouts reduce bias from bootstrapping |
| Value loss coefficient | 0.5 | Balances critic loss relative to actor loss |
| Entropy coefficient | 0.01 | Too high prevents convergence; too low allows premature collapse |
| Max gradient norm | 0.5 | Prevents destructive updates in shared networks |
| Number of parallel envs | 4 - 64 | More envs increase throughput and batch diversity |

### Common Pitfalls

1. **Not detaching the advantage**: If the advantage (which depends on V(s)) is not detached from the computation graph, the actor's loss will backpropagate through the critic, creating circular gradient flows. Always detach advantages and returns when computing losses.

2. **Forgetting to handle episode boundaries in GAE**: If bootstrapping crosses episode boundaries (using V(s') from a new episode after a terminal state), the advantage estimates will be wrong. The done mask must zero out the bootstrap term at episode terminations.

3. **Learning rate too high for shared networks**: Shared architectures are more sensitive to learning rate because two loss terms compete for the same parameters. If training is unstable, try reducing the learning rate or switching to separate networks.

4. **Skipping advantage normalization**: Normalizing advantages to zero mean and unit variance within each batch is simple but surprisingly important. Without it, the actor loss scale can vary wildly across batches.

5. **Entropy coefficient decay**: For complex environments, starting with a higher entropy coefficient (0.05) and annealing it toward 0.01 over training can help. Early exploration is important, but too much entropy late in training prevents the policy from becoming precise.

---

## Theoretical Foundation

### The Policy Gradient Theorem

The actor-critic framework rests on the policy gradient theorem (Sutton et al., 2000), which states:

```
grad_theta J(theta) = E_{s ~ d_pi, a ~ pi_theta} [ Q_pi(s, a) * grad_theta log pi_theta(a | s) ]
```

where d_pi is the stationary state distribution under policy pi. This theorem tells us that we can estimate the gradient of the expected return by sampling states and actions from the policy and weighting the log-probability gradient by the action-value Q(s, a). Subtracting a baseline V(s) gives us the advantage, which does not change the expected gradient but reduces variance.

### Bias-Variance Tradeoff Formalized

The choice of advantage estimator determines the bias-variance tradeoff:

- **Monte Carlo (lambda = 1)**: Zero bias (assuming V is exact), variance proportional to the trajectory length
- **One-step TD (lambda = 0)**: Bias proportional to the critic's approximation error, variance proportional to one step of reward variance
- **GAE (0 < lambda < 1)**: Bias decays exponentially with lambda, variance grows with lambda

This tradeoff is why GAE with lambda around 0.95 works well in practice: it gets most of the variance reduction of TD while keeping bias small.

### Convergence Properties

The convergence of actor-critic methods is more nuanced than pure value-based methods. Two-timescale stochastic approximation theory (Borkar, 2008) provides the theoretical foundation: if the critic's learning rate is faster than the actor's, the critic converges to an approximately correct value function for the current policy, and the actor converges to a local optimum of the expected return. In practice, using the same learning rate for both works fine because the critic's loss landscape is typically easier to optimize than the actor's.

Actor-critic methods converge to local optima of the expected return, not global optima. This is a fundamental limitation of policy gradient methods in general. However, in deep RL, local optima are often good enough, and the exploration provided by stochastic policies and entropy regularization helps avoid the worst local optima.

---

## Key Takeaways

1. **Actor-critic combines the best of both worlds**: the actor provides a direct, differentiable policy while the critic provides low-variance feedback for training it.

2. **The TD error is a one-step advantage estimate**: delta = r + gamma * V(s') - V(s). Positive delta reinforces the action; negative delta suppresses it.

3. **GAE is the standard advantage estimator**: It smoothly interpolates between one-step TD and Monte Carlo, controlled by lambda. Use lambda = 0.95 as a default.

4. **Shared network architectures are efficient but require care**: Balance the actor and critic losses, clip gradients, and detach advantage estimates properly.

5. **Actor-critic is the foundation of modern deep RL**: A2C, A3C, PPO, TRPO, SAC, DDPG, and TD3 are all actor-critic methods with different update rules and constraints.

6. **Start with PPO**: For most practical problems, PPO (which is actor-critic + clipped policy updates + GAE) is the best starting point. Understand vanilla actor-critic to understand PPO.
