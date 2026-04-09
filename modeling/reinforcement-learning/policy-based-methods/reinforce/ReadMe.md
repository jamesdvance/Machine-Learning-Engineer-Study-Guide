# REINFORCE (Monte Carlo Policy Gradient)

## Summary

REINFORCE, introduced by Ronald Williams in 1992, is the simplest and most foundational policy gradient algorithm in reinforcement learning. Unlike value-based methods such as Q-Learning and DQN that learn an action-value function and derive a policy indirectly, REINFORCE directly parameterizes and optimizes the policy itself. The agent maintains a probability distribution over actions given a state, and updates the parameters by following the gradient of expected cumulative reward. Because it uses complete episode returns to estimate this gradient (rather than bootstrapping from value estimates), REINFORCE is a Monte Carlo method -- it must wait until an episode ends before any learning can occur.

The algorithm is conceptually elegant: collect a full trajectory, compute the return at each time step, and adjust the policy parameters so that actions leading to higher returns become more probable and actions leading to lower returns become less probable. This update is made precise by the policy gradient theorem and the log-probability trick, which together provide an unbiased gradient estimator that can be computed through standard automatic differentiation. However, this simplicity comes at a cost. Because REINFORCE relies on full Monte Carlo returns, its gradient estimates suffer from high variance, which translates into slow and unstable learning. This high variance problem is the central limitation of REINFORCE and the primary motivation for more advanced methods like Actor-Critic and PPO.

Key points to remember:

- REINFORCE is a Monte Carlo policy gradient method: it uses complete episode returns, no bootstrapping
- The policy is parameterized directly (e.g., by a neural network) and optimized via gradient ascent on expected return
- The policy gradient theorem provides the theoretical foundation; the log-probability trick makes it computationally tractable
- Gradient estimates are unbiased but have high variance, leading to slow convergence
- Subtracting a baseline (typically a learned state-value function) from the return reduces variance without introducing bias
- REINFORCE naturally handles both discrete and continuous action spaces
- It is on-policy: trajectories used for updates must come from the current policy, making it sample-inefficient
- REINFORCE is the starting point from which Actor-Critic, A2C/A3C, TRPO, and PPO all evolved

When to use REINFORCE:

- Simple environments with short episodes where high variance is manageable
- Educational and prototyping settings to understand policy gradient fundamentals
- Problems with continuous action spaces where value-based methods like DQN do not apply directly
- Situations where you need a stochastic policy (e.g., for exploration or in adversarial settings)

When not to use REINFORCE:

- Long-horizon tasks where Monte Carlo returns will have extreme variance
- Sample-constrained settings where you cannot afford to discard trajectories after one update
- Production systems requiring stable, reliable training (use PPO or SAC instead)
- Environments where you need off-policy learning or replay buffers

---

## Why Policy Gradients? Motivation from Value-Based Limitations

### The Limits of Value-Based Methods

Value-based methods like Q-Learning and DQN learn an action-value function Q(s, a) and derive a policy by selecting the action with the highest Q-value (argmax). This approach has three fundamental limitations:

1. **Discrete actions only**: The argmax operation requires enumerating all possible actions. In continuous action spaces (e.g., controlling a robot arm's joint torques), you cannot compute argmax over an infinite set. While methods like DDPG work around this, they require separate architectural innovations.

2. **Deterministic policies**: The argmax over Q-values always yields the same action in a given state. In some settings -- multi-agent games, partially observable environments, or problems where stochasticity is inherently optimal -- you need a policy that can randomize over actions.

3. **Sensitivity to value errors**: Small errors in Q-value estimates can cause the argmax to select the wrong action entirely. A slight overestimation of one action's value flips the policy, even if the true values are very close.

### The Policy Gradient Alternative

Policy gradient methods address all three limitations by directly parameterizing the policy as a probability distribution. Instead of learning Q(s, a) and deriving a policy, we maintain a policy network pi_theta(a | s) that outputs the probability of taking action a in state s. The parameters theta are updated to maximize expected cumulative reward. This approach:

- Works natively with continuous action spaces (output a Gaussian distribution over actions)
- Produces stochastic policies that can represent mixed strategies
- Changes policies smoothly, since small parameter updates cause small changes in action probabilities rather than sudden switches

---

## The Policy Gradient Theorem

### Setup and Objective

Consider an agent interacting with an environment over episodes. The agent follows a policy pi_theta(a | s), parameterized by theta. A trajectory tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T) is a sequence of states, actions, and rewards over one episode. The return of a trajectory is the total discounted reward:

```
R(tau) = sum_{t=0}^{T} gamma^t * r_t
```

The objective is to maximize the expected return under the policy:

```
J(theta) = E_{tau ~ pi_theta} [ R(tau) ]
```

We want to compute the gradient of J with respect to theta so we can perform gradient ascent.

### The Core Theorem

The policy gradient theorem (Sutton et al., 1999) states:

```
grad_theta J(theta) = E_{tau ~ pi_theta} [ sum_{t=0}^{T} grad_theta log pi_theta(a_t | s_t) * G_t ]
```

where G_t is the return from time step t onward:

```
G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
```

This result is remarkable because it expresses the gradient of the expected return -- a quantity that depends on the environment dynamics -- purely in terms of the policy and the observed returns. The environment dynamics do not appear in the gradient expression. This is what makes policy gradient methods model-free.

### Derivation Intuition

The derivation proceeds in three steps. First, express the expected return as an integral over trajectories:

```
J(theta) = integral P(tau; theta) R(tau) d_tau
```

where P(tau; theta) is the probability of trajectory tau under policy pi_theta. Second, take the gradient and apply the log-derivative trick (described in the next section):

```
grad_theta J(theta) = integral P(tau; theta) * grad_theta log P(tau; theta) * R(tau) d_tau
                    = E_{tau ~ pi_theta} [ grad_theta log P(tau; theta) * R(tau) ]
```

Third, expand the log-probability of a trajectory:

```
log P(tau; theta) = log p(s_0) + sum_{t=0}^{T} [ log pi_theta(a_t | s_t) + log p(s_{t+1} | s_t, a_t) ]
```

The initial state distribution p(s_0) and the transition dynamics p(s_{t+1} | s_t, a_t) do not depend on theta, so their gradients are zero. Only the log-policy terms survive:

```
grad_theta log P(tau; theta) = sum_{t=0}^{T} grad_theta log pi_theta(a_t | s_t)
```

Substituting back and using the fact that future rewards do not depend on past actions (causality), we arrive at the policy gradient theorem.

---

## The Log-Probability Trick

The log-probability trick (also called the REINFORCE trick or score function estimator) is the key mathematical identity that makes policy gradient computation possible. It converts a gradient of an expectation into an expectation of a gradient, which can then be estimated from samples.

### The Identity

For any probability distribution p_theta(x) parameterized by theta:

```
grad_theta p_theta(x) = p_theta(x) * grad_theta log p_theta(x)
```

This follows directly from the chain rule:

```
grad_theta log p_theta(x) = grad_theta p_theta(x) / p_theta(x)
```

Rearranging gives the identity. The practical consequence is:

```
grad_theta E_{x ~ p_theta} [ f(x) ] = grad_theta integral p_theta(x) f(x) dx
                                      = integral grad_theta p_theta(x) f(x) dx
                                      = integral p_theta(x) grad_theta log p_theta(x) f(x) dx
                                      = E_{x ~ p_theta} [ grad_theta log p_theta(x) * f(x) ]
```

### Why This Matters

Without the log-probability trick, computing the gradient of J(theta) would require differentiating through the environment dynamics (the transition function), which are typically unknown and non-differentiable. The log-probability trick sidesteps this entirely. We only need:

1. The ability to sample trajectories from the current policy (run the policy in the environment)
2. The ability to compute grad_theta log pi_theta(a | s) (automatic differentiation through the policy network)
3. The observed returns (sum up the rewards)

The gradient estimate is formed by multiplying the log-probability gradient by the return and averaging over sampled trajectories. No model of the environment is needed.

---

## Monte Carlo Return Estimation

REINFORCE uses Monte Carlo estimation for the return G_t, meaning it computes the actual sum of discounted rewards observed from time step t to the end of the episode. This is in contrast to temporal-difference (TD) methods that bootstrap, estimating future returns using a value function.

### How It Works

After collecting a complete trajectory (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, r_T), compute G_t for each time step by working backward:

```
G_T = r_T
G_{T-1} = r_{T-1} + gamma * G_T
G_{T-2} = r_{T-2} + gamma * G_{T-1}
...
G_t = r_t + gamma * G_{t+1}
```

This backward computation is efficient (linear in episode length) and produces the exact discounted return for each time step.

### Properties

**Unbiased**: Monte Carlo returns are unbiased estimates of the true expected return from each state. There is no approximation error from a value function. This means the gradient estimate, while noisy, points in the correct direction on average.

**High variance**: The return G_t is a sum of many random variables (future rewards), each depending on stochastic policy choices and stochastic environment transitions. The variance of this sum grows with the episode length. Two trajectories starting from the same state can yield wildly different returns due to the randomness in all subsequent decisions and transitions.

**Requires complete episodes**: You must wait until the episode terminates to compute any G_t value. This makes REINFORCE inapplicable to continuing (non-episodic) tasks and adds latency in episodic tasks with long episodes.

### Contrast with TD Estimation

TD methods estimate returns using a value function:

```
G_t (TD) approx r_t + gamma * V(s_{t+1})
```

This introduces bias (V is an approximation) but dramatically reduces variance because the estimate depends on only one step of randomness rather than all future steps. Actor-Critic methods use TD estimation in place of Monte Carlo returns, directly addressing REINFORCE's variance problem.

---

## The REINFORCE Algorithm

### Pseudocode

```
Algorithm: REINFORCE

Input: Differentiable policy network pi_theta(a | s), learning rate alpha, discount factor gamma
Initialize policy network parameters theta randomly

Repeat for each episode:
    # 1. Collect a trajectory
    Generate episode (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, r_T) by following pi_theta

    # 2. Compute returns
    For t = T down to 0:
        G_t = r_t + gamma * G_{t+1}    (with G_{T+1} = 0)

    # 3. Update policy parameters
    For t = 0 to T:
        theta <- theta + alpha * gamma^t * G_t * grad_theta log pi_theta(a_t | s_t)

Until convergence
```

### Step-by-Step Explanation

1. **Generate episode**: Run the current policy in the environment from start to terminal state, recording all states, actions, and rewards. The policy must be stochastic (outputs a distribution over actions and samples from it) to ensure exploration.

2. **Compute returns**: After the episode ends, compute the discounted return G_t for each time step using the backward recursion. This is the Monte Carlo estimate.

3. **Update parameters**: For each time step in the episode, compute the gradient of the log-probability of the action taken, multiply it by the return G_t (optionally with a discount factor gamma^t), and add this to the parameter update. The update increases the probability of actions that led to high returns and decreases the probability of actions that led to low returns.

In practice, the update is done as a single batch operation: accumulate the loss over all time steps, then perform one backward pass and optimizer step.

---

## Variance Reduction with Baselines

### The High Variance Problem

The raw REINFORCE gradient estimator has high variance because G_t can vary enormously across trajectories. Consider a simple example: an agent in a game where all trajectories yield positive reward (some large, some small). Without a baseline, the gradient update increases the probability of all actions taken, just by different amounts. The relative signal (which actions are better than average) is buried under a large common-mode signal (all returns are positive). It takes many samples for the relative differences to emerge.

Concretely, if G_t ranges from 50 to 150 across trajectories, the gradient is always in the "increase probability" direction for all actions. The useful information -- that some actions led to returns of 150 versus 50 -- is a small fraction of the total gradient magnitude. This wastes gradient updates on uninformative common-mode signal.

### Baselines

We can subtract any function b(s_t) that does not depend on the action from the return without introducing bias:

```
grad_theta J(theta) = E [ sum_t grad_theta log pi_theta(a_t | s_t) * (G_t - b(s_t)) ]
```

This is unbiased for any baseline b because:

```
E_a [ grad_theta log pi_theta(a | s) * b(s) ] = b(s) * E_a [ grad_theta log pi_theta(a | s) ]
                                                = b(s) * grad_theta sum_a pi_theta(a | s)
                                                = b(s) * grad_theta 1
                                                = 0
```

The optimal baseline (in terms of minimizing variance) is approximately the expected return from each state, which is exactly the state-value function V(s). In practice, the most common baseline is a learned value function V_phi(s) trained alongside the policy.

### REINFORCE with Baseline: Pseudocode

```
Algorithm: REINFORCE with Baseline

Input: Policy network pi_theta(a | s), value network V_phi(s), 
       learning rates alpha_theta, alpha_phi, discount factor gamma
Initialize parameters theta, phi randomly

Repeat for each episode:
    # 1. Collect a trajectory
    Generate episode (s_0, a_0, r_0, ..., s_T, r_T) following pi_theta

    # 2. Compute returns
    For t = T down to 0:
        G_t = r_t + gamma * G_{t+1}    (with G_{T+1} = 0)

    # 3. Update value function (baseline)
    Minimize sum_t (G_t - V_phi(s_t))^2 with respect to phi

    # 4. Update policy
    For t = 0 to T:
        Advantage_t = G_t - V_phi(s_t)
        theta <- theta + alpha_theta * gamma^t * Advantage_t * grad_theta log pi_theta(a_t | s_t)

Until convergence
```

The quantity (G_t - V_phi(s_t)) is called the advantage: it measures how much better the actual return was compared to what we expected from that state. Positive advantage means the action was better than average; negative advantage means it was worse. This centered signal has much lower variance than the raw return.

---

## PyTorch Implementation

The following is a complete implementation of REINFORCE with a baseline for environments with discrete action spaces.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Neural network that parameterizes the policy pi_theta(a | s)."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Neural network that approximates the state-value function V_phi(s)."""

    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class REINFORCEAgent:
    """REINFORCE agent with learned baseline."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        device="cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []
        self.states = []

    def select_action(self, state):
        """Sample action from the policy distribution."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = self.policy_net(state_tensor)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.states.append(state_tensor)
        return action.item()

    def store_reward(self, reward):
        """Store reward for the current time step."""
        self.rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns G_t for each time step (backward pass)."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        return returns

    def update(self):
        """Perform REINFORCE update at the end of an episode."""
        returns = self.compute_returns()

        # Stack stored tensors
        log_probs = torch.stack(self.log_probs).to(self.device)
        states = torch.cat(self.states, dim=0).to(self.device)

        # Compute baseline values
        values = self.value_net(states).detach()

        # Compute advantages (return - baseline)
        advantages = returns - values

        # Normalize advantages for stability (optional but recommended)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss: -log_prob * advantage (negative because we do gradient ascent)
        policy_loss = -(log_probs * advantages).sum()

        # Value loss: MSE between predicted values and actual returns
        value_predictions = self.value_net(states)
        value_loss = nn.functional.mse_loss(value_predictions, returns)

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        # Clear episode storage
        self.log_probs = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()


# --------------------------------
# Example training loop
# --------------------------------
# import gymnasium as gym
#
# env = gym.make("CartPole-v1")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
#
# agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim)
#
# for episode in range(2000):
#     state, _ = env.reset()
#     episode_reward = 0
#
#     while True:
#         action = agent.select_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#
#         agent.store_reward(reward)
#         state = next_state
#         episode_reward += reward
#
#         if done:
#             break
#
#     policy_loss, value_loss = agent.update()
#
#     if episode % 100 == 0:
#         print(f"Episode {episode}, Reward: {episode_reward:.1f}, "
#               f"Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}")
```

### Implementation Notes

- **`Categorical(logits=...)`**: PyTorch's Categorical distribution handles the softmax internally when you pass logits. This is numerically more stable than manually computing softmax and then creating a distribution from probabilities.

- **`dist.log_prob(action)`**: This computes log pi_theta(a_t | s_t) for the sampled action. During the backward pass, PyTorch automatically computes grad_theta log pi_theta(a_t | s_t) with respect to the network parameters.

- **`values.detach()`**: When computing advantages, the value network output must be detached from the computation graph. The baseline should not receive gradients through the policy loss -- it has its own loss function (MSE against returns).

- **Advantage normalization**: Normalizing advantages to have zero mean and unit variance is a practical trick that stabilizes training. It ensures that roughly half the actions are reinforced (positive advantage) and half are discouraged (negative advantage), preventing the gradient from being dominated by common-mode signal.

- **Gradient clipping**: Even with a baseline, individual gradient contributions can be large. Clipping prevents occasional large updates from destabilizing the network.

- **On-policy requirement**: The trajectory data is discarded after each update. Unlike DQN, there is no replay buffer. Each trajectory can only be used once because the gradient estimator assumes the data was generated by the current policy pi_theta.

### Continuous Action Spaces

For continuous actions, replace the Categorical distribution with a Gaussian. The policy network outputs a mean and log-standard-deviation for each action dimension:

```python
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)
```

The rest of the algorithm is identical. The `log_prob` and `sample` methods work the same way for Normal distributions as for Categorical. This seamless handling of continuous actions is a major advantage of policy gradient methods over value-based approaches.

---

## The High Variance Problem

### Why Variance Matters

High variance in gradient estimates is the central practical challenge of REINFORCE. Even with a baseline, the gradient estimates can require thousands of episodes to converge on simple problems that DQN solves in hundreds.

The root cause is that Monte Carlo returns accumulate variance from every random event in the trajectory: every stochastic action selection and every stochastic state transition from the current time step to the end of the episode. For an episode of length T, the return at time step 0 depends on approximately T random variables. By the law of total variance, the variance of G_0 grows roughly linearly with T.

### Consequences

1. **Slow convergence**: The policy improves slowly because the noisy gradient estimates cause the optimization to take many small, jittery steps rather than direct progress toward the optimum.

2. **High sample complexity**: You need many trajectories per update to get a reasonable gradient estimate. Collecting many trajectories is expensive, especially in environments with costly interactions (robotics, simulations).

3. **Sensitivity to hyperparameters**: The learning rate must be set conservatively to prevent large, noisy gradient steps from destabilizing the policy. Too high a learning rate causes catastrophic policy collapse; too low causes impractically slow learning.

4. **Reward scale sensitivity**: If rewards are large in magnitude, the gradient estimates have proportionally large variance. This is why advantage normalization and careful reward design are important in practice.

### Mitigation Strategies

Beyond baselines, several techniques can reduce variance:

- **Reward normalization**: Scale returns to have zero mean and unit variance across the batch.

- **Batch updates**: Collect multiple trajectories per update and average the gradient estimates. This reduces variance by a factor of 1/N (where N is the batch size) at the cost of N times more environment interactions per update.

- **Entropy regularization**: Adding an entropy bonus to the objective encourages the policy to remain stochastic, which indirectly helps with exploration and can smooth the optimization landscape. The modified objective becomes J(theta) + beta * H(pi_theta), where H is the entropy of the policy and beta is a small coefficient.

- **Reward shaping**: Designing denser reward signals that provide more frequent feedback reduces the variance of returns by making individual rewards more informative and reducing reliance on long-horizon credit assignment.

---

## Comparison with Value-Based Methods

### REINFORCE vs. Q-Learning / DQN

| Aspect | REINFORCE | Q-Learning / DQN |
|---|---|---|
| What is learned | Policy pi_theta(a given s) directly | Action-value function Q(s, a) |
| Policy type | Stochastic (samples from distribution) | Deterministic (argmax of Q-values) |
| Action spaces | Discrete and continuous | Discrete only (DQN) |
| Sample usage | On-policy (use each trajectory once) | Off-policy (replay buffer, reuse data) |
| Exploration | Built-in via stochastic policy | Requires explicit strategy (epsilon-greedy) |
| Bootstrapping | No (Monte Carlo returns) | Yes (TD targets from value network) |
| Bias | Unbiased gradient estimates | Biased by function approximation errors |
| Variance | High (full episode returns) | Lower (single-step TD targets) |
| Sample efficiency | Low | Higher (replay buffer enables reuse) |
| Convergence | Slow but theoretically sound | Faster in practice but less stable guarantees |

### When Each Approach Wins

**Value-based methods** are better when:
- The action space is discrete and not too large
- Sample efficiency matters (you cannot afford to discard data)
- The environment allows off-policy learning
- You want the simplicity of epsilon-greedy exploration

**REINFORCE / policy gradients** are better when:
- The action space is continuous
- You need a stochastic policy
- The problem has strong theoretical requirements for convergence guarantees
- You are building toward more advanced policy gradient methods (PPO, Actor-Critic)

In practice, most production RL systems use neither pure REINFORCE nor pure DQN. They use methods that combine ideas from both: Actor-Critic methods use a policy (like REINFORCE) but estimate advantages with TD learning (like DQN), getting the best of both worlds.

---

## Connection to Actor-Critic Methods

### From REINFORCE to Actor-Critic

REINFORCE with a baseline is one small step away from Actor-Critic. The key difference is how the advantage is estimated:

- **REINFORCE with baseline**: Advantage_t = G_t - V_phi(s_t), where G_t is the full Monte Carlo return. This is unbiased but high variance.

- **Actor-Critic (one-step)**: Advantage_t = r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t), which is the TD error delta_t. This is biased (because V_phi is an approximation) but much lower variance because it depends on only one step of randomness.

- **Generalized Advantage Estimation (GAE)**: Advantage_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}. This interpolates between Monte Carlo (lambda=1, high variance, no bias) and one-step TD (lambda=0, low variance, more bias) using the parameter lambda.

The "actor" is the policy network pi_theta and the "critic" is the value network V_phi. In REINFORCE with a baseline, the critic only serves as a baseline -- the return estimate is still Monte Carlo. In true Actor-Critic methods, the critic actively participates in return estimation through bootstrapping, which is what reduces variance.

### The Evolution

The lineage from REINFORCE to modern methods is:

1. **REINFORCE**: Monte Carlo returns, no baseline. High variance.
2. **REINFORCE with baseline**: Monte Carlo returns, value baseline. Reduced variance, still high.
3. **Actor-Critic (A2C)**: TD or GAE returns, value baseline. Substantially lower variance, some bias.
4. **A3C**: Actor-Critic with asynchronous parallel workers for faster data collection.
5. **TRPO**: Actor-Critic with a trust region constraint on policy updates for stability.
6. **PPO**: Actor-Critic with a clipped surrogate objective that approximates TRPO's constraint cheaply.

Each step in this lineage addresses a specific limitation of the previous method. Understanding REINFORCE is essential because it provides the conceptual foundation -- the policy gradient theorem, the log-probability trick, the role of baselines -- that all subsequent methods build on.

---

## Hyperparameters and Practical Tips

### Key Hyperparameters

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Policy learning rate | 1e-4 to 3e-4 | Must be small; too large causes policy collapse |
| Value learning rate | 1e-3 to 3e-3 | Can be larger than policy LR since value loss is more stable |
| Discount factor (gamma) | 0.99 | Lower for shorter-horizon tasks |
| Hidden layer size | 64 - 256 | Two layers is usually sufficient |
| Batch size (episodes) | 1 - 10 | More episodes per update reduces variance but costs more samples |
| Entropy coefficient | 0.01 | Encourages exploration; set to 0 to disable |
| Gradient clip norm | 0.5 - 1.0 | Prevents destabilizing large updates |

### Practical Tips

1. **Always use a baseline**: Raw REINFORCE without a baseline is practically unusable on anything but the simplest environments. The variance reduction from a learned baseline is too significant to skip.

2. **Normalize advantages**: After computing advantages, subtract the mean and divide by the standard deviation. This is a simple but highly effective variance reduction technique.

3. **Monitor entropy**: If the policy's entropy drops quickly, the policy is collapsing to a near-deterministic one before adequately exploring. Add an entropy bonus or reduce the learning rate.

4. **Use multiple episodes per update**: Averaging gradients over several episodes (batch REINFORCE) reduces variance at the cost of more interactions. This is a simple and effective technique.

5. **Start simple**: Verify your REINFORCE implementation on CartPole before moving to harder environments. CartPole should converge within 500-1000 episodes with a baseline.

6. **Consider upgrading to PPO**: If REINFORCE is too slow or unstable for your problem, PPO is the most common next step. It uses the same policy gradient foundation but adds clipped surrogate objectives and minibatch updates that dramatically improve sample efficiency and stability.
