# DQN (Deep Q-Network)

## Summary

DQN, or Deep Q-Network, is the algorithm that brought deep learning into reinforcement learning at scale. Published by DeepMind in 2013 and refined in their landmark 2015 Nature paper, DQN demonstrated that a single neural network could learn to play Atari 2600 games directly from raw pixel input, achieving human-level performance on many titles. It extends classical Q-Learning by replacing the tabular Q-table with a deep neural network that approximates the Q-function, enabling the agent to generalize across large or continuous state spaces that would be intractable for table-based methods.

The key insight of DQN is not simply using a neural network as a function approximator -- that idea had been explored before and was known to be unstable. DQN's contribution was solving that instability through two techniques: experience replay and a target network. Experience replay stores past transitions in a buffer and samples mini-batches uniformly at random, breaking the temporal correlations that destabilize training. The target network is a periodically-updated copy of the Q-network used to compute the target values in the loss function, which prevents the moving-target problem where the network chases its own changing predictions.

Key points to remember:

- DQN = Q-Learning + deep neural network function approximator + experience replay + target network
- The Q-network takes a state as input and outputs Q-values for all actions simultaneously
- Experience replay breaks correlation between consecutive samples and improves data efficiency
- The target network is a frozen copy of the Q-network, updated every C steps, that stabilizes training
- The loss function is the mean squared error between predicted Q-values and target Q-values computed using the Bellman equation
- Epsilon-greedy exploration balances exploitation of learned values with random exploration
- DQN only works with discrete action spaces; it cannot directly handle continuous actions
- DQN tends to overestimate Q-values, a problem addressed by Double DQN

When to use DQN:

- Discrete action spaces (e.g., game controls, routing decisions, inventory management)
- High-dimensional state spaces where tabular Q-Learning is infeasible
- Problems where you can collect many interactions (DQN is sample-hungry)
- Offline or batch RL settings where you have logged experience data

When not to use DQN:

- Continuous action spaces (use DDPG, TD3, or SAC instead)
- Extremely large discrete action spaces (combinatorial explosion)
- When sample efficiency is critical (consider model-based methods)
- When you need stochastic policies (consider policy gradient methods)

---

## From Q-Learning to DQN

### Q-Learning Recap

Q-Learning is a model-free, off-policy reinforcement learning algorithm that learns the optimal action-value function Q*(s, a), which represents the expected cumulative discounted reward of taking action a in state s and then following the optimal policy thereafter. The classic update rule is:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

where alpha is the learning rate, gamma is the discount factor, r is the immediate reward, s' is the next state, and the max is taken over all possible next actions a'. In tabular Q-Learning, Q-values are stored in a lookup table indexed by (state, action) pairs.

The limitation is obvious: when the state space is large (e.g., images with millions of pixel values) or continuous, a table cannot represent every possible state. You need function approximation.

### Why Naive Function Approximation Failed

Researchers had tried using neural networks as Q-function approximators before DQN, but training was notoriously unstable. There were two main reasons:

1. **Correlated samples**: In RL, consecutive transitions (s, a, r, s') are highly correlated because they come from a single trajectory. Training a neural network on correlated sequential data violates the i.i.d. assumption that underpins stochastic gradient descent, leading to poor convergence or divergence.

2. **Non-stationary targets**: In Q-Learning, the target value r + gamma * max_a' Q(s', a') depends on the same network being updated. As the network's weights change, the targets shift, creating a moving-target problem. The network is essentially chasing its own tail.

DQN solved both problems with two architectural innovations.

---

## Key Innovations

### Experience Replay

Experience replay maintains a fixed-size buffer (replay memory) D of past transitions (s, a, r, s', done). During training, instead of learning from the most recent transition, the agent samples a random mini-batch of transitions from D. This provides three benefits:

- **Breaks temporal correlation**: Random sampling ensures the mini-batch contains transitions from different episodes and time steps, making the training data approximately i.i.d.
- **Improves data efficiency**: Each transition can be sampled and learned from multiple times, rather than being used once and discarded.
- **Smooths the data distribution**: The replay buffer aggregates transitions from many past policies, averaging over the distribution and preventing oscillations.

The replay buffer is typically implemented as a circular buffer with a fixed maximum size (e.g., 1 million transitions). When the buffer is full, the oldest transitions are overwritten. The standard DQN uses uniform random sampling, though later work (Prioritized Experience Replay) showed that sampling proportional to TD error magnitude can improve learning speed.

### Target Network

The target network is a separate copy of the Q-network with parameters theta_target (often written as theta^-). While the main Q-network (with parameters theta) is updated at every training step, the target network's parameters are held fixed and only synchronized with the main network every C steps (e.g., every 10,000 steps). The target value is computed as:

```
y = r + gamma * max_a' Q(s', a'; theta_target)
```

Because theta_target is fixed between updates, the target y does not change with every gradient step, giving the main network a stable objective to optimize against. This dramatically reduces oscillations and divergence during training.

An alternative to hard updates every C steps is soft (Polyak) updates, where the target network parameters are blended toward the main network at each step:

```
theta_target <- tau * theta + (1 - tau) * theta_target
```

with tau being a small value like 0.005. This approach, used in DDPG and other algorithms, provides smoother target updates.

---

## Architecture

### The Original Atari Architecture

In the original DQN paper, the input is a stack of 4 consecutive 84x84 grayscale frames from the Atari emulator. Stacking frames gives the network temporal information (e.g., the direction and speed of a moving ball). The architecture is:

1. **Input**: 4 x 84 x 84 grayscale image stack
2. **Conv layer 1**: 32 filters, 8x8 kernel, stride 4, ReLU activation
3. **Conv layer 2**: 64 filters, 4x4 kernel, stride 2, ReLU activation
4. **Conv layer 3**: 64 filters, 3x3 kernel, stride 1, ReLU activation
5. **Fully connected layer**: 512 units, ReLU activation
6. **Output layer**: One unit per action (e.g., 18 for Atari), no activation (linear output representing Q-values)

The network takes a state (frame stack) as input and outputs Q-values for all possible actions simultaneously. To select an action, you simply take the argmax of the output vector. This is much more efficient than having a separate forward pass for each action.

### General Architecture Considerations

For non-image state spaces, the CNN layers are replaced with fully connected layers. The key design principle remains: the network maps states to a vector of Q-values, one per discrete action.

- **State input**: Can be any fixed-size vector (e.g., sensor readings, feature vectors) or image-like input
- **Hidden layers**: Typically 2-3 hidden layers with 64-512 units each for vector inputs; CNNs for image inputs
- **Output**: One linear output per discrete action
- **Activation**: ReLU is standard for hidden layers; no activation on the output layer since Q-values are unbounded real numbers

---

## Training Loop

The DQN training loop interleaves environment interaction with network updates. Here is the complete procedure:

### Algorithm: DQN with Experience Replay

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights theta
Initialize target network with weights theta_target = theta
For episode = 1 to M:
    Initialize state s_1 (preprocess if needed)
    For t = 1 to T:
        # Action selection (epsilon-greedy)
        With probability epsilon:
            Select random action a_t
        Otherwise:
            Select a_t = argmax_a Q(s_t, a; theta)

        # Environment step
        Execute action a_t, observe reward r_t and next state s_{t+1}
        Store transition (s_t, a_t, r_t, s_{t+1}, done) in D

        # Learning step (if enough samples in D)
        Sample random mini-batch of transitions (s_j, a_j, r_j, s_{j+1}, done_j) from D
        Compute target:
            If done_j:
                y_j = r_j
            Else:
                y_j = r_j + gamma * max_a' Q(s_{j+1}, a'; theta_target)
        Compute loss: L = (1/batch_size) * sum( (y_j - Q(s_j, a_j; theta))^2 )
        Perform gradient descent step on L with respect to theta

        # Target network update
        Every C steps: theta_target <- theta
```

### Epsilon-Greedy Exploration

The epsilon-greedy strategy is crucial for balancing exploration and exploitation. A common schedule:

- Start with epsilon = 1.0 (fully random)
- Linearly anneal epsilon to 0.1 (or 0.01) over the first 1 million frames
- Keep epsilon at 0.1 for the remainder of training

The initial random exploration fills the replay buffer with diverse experiences. As training progresses and the Q-network improves, the agent increasingly exploits its learned policy while maintaining a small probability of random actions to continue discovering new states.

---

## Loss Function and Gradient Computation

### The DQN Loss

The loss function is the mean squared error (MSE) between the predicted Q-value and the target Q-value:

```
L(theta) = E_{(s,a,r,s') ~ D} [ (y - Q(s, a; theta))^2 ]
```

where the target is:

```
y = r + gamma * max_a' Q(s', a'; theta_target)
```

Note that the target y is treated as a constant during backpropagation. Gradients flow only through Q(s, a; theta), not through the target network. This is sometimes called a semi-gradient method.

### Huber Loss Alternative

In practice, many implementations use the Huber loss (smooth L1 loss) instead of MSE to reduce the impact of large TD errors, which can cause unstable gradient updates:

```
Huber(delta) = 0.5 * delta^2           if |delta| <= 1
             = |delta| - 0.5           otherwise
```

where delta = y - Q(s, a; theta). The Huber loss behaves like MSE for small errors and like MAE for large errors, clipping the gradient magnitude and improving stability.

### Gradient Computation

The gradient of the loss with respect to the network parameters theta is:

```
grad_theta L = E [ (Q(s, a; theta) - y) * grad_theta Q(s, a; theta) ]
```

In practice this is computed via automatic differentiation on the mini-batch. The gradient is only computed with respect to the Q-values corresponding to the actions actually taken. For the output vector Q(s, .; theta), only the component at index a is involved in the loss; the other components have zero gradient contribution for that sample.

### Gradient Clipping

It is common to clip gradients (e.g., clip the gradient norm to 10 or clip individual gradients to [-1, 1]) to prevent exploding gradients, especially early in training when TD errors can be large.

---

## Hyperparameters and Practical Training Tips

### Key Hyperparameters

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Replay buffer size | 100,000 - 1,000,000 | Larger is generally better but uses more RAM |
| Mini-batch size | 32 | Standard across most DQN implementations |
| Discount factor (gamma) | 0.99 | Lower values (0.95) for shorter-horizon tasks |
| Learning rate | 1e-4 to 2.5e-4 | Adam optimizer is standard; RMSProp was used in the original paper |
| Target network update frequency (C) | 1,000 - 10,000 steps | Too frequent causes instability; too infrequent slows learning |
| Epsilon start | 1.0 | Fully random at the beginning |
| Epsilon end | 0.01 - 0.1 | Maintain some exploration |
| Epsilon decay frames | 1,000,000 | Linear annealing over this many frames |
| Learning starts | 50,000 steps | Fill replay buffer before training begins |
| Frame skip | 4 | Agent sees every 4th frame; action is repeated for skipped frames |

### Practical Tips

1. **Reward clipping**: The original DQN clipped all rewards to [-1, +1]. This makes the algorithm robust across games with different reward scales, but loses magnitude information. For custom environments, consider reward normalization instead.

2. **Frame stacking**: Stacking 4 consecutive frames as input channels gives the network velocity information without needing recurrence. This is essential for partially observable environments like Atari.

3. **Frame preprocessing**: Convert to grayscale, resize to 84x84, and skip frames to reduce computation. These steps are standard for Atari but may not apply to other domains.

4. **Warm-up period**: Do not start training until the replay buffer has a minimum number of transitions (e.g., 50,000). This ensures the initial mini-batches are reasonably diverse.

5. **Optimizer choice**: Adam with a learning rate of 1e-4 is a safe default. The original paper used RMSProp with specific settings, but Adam is more commonly used in modern implementations.

6. **Network size**: For simple environments (CartPole, LunarLander), two hidden layers with 64-128 units each are sufficient. Overparameterizing a simple problem can slow convergence.

7. **Monitor Q-value estimates**: If predicted Q-values grow unboundedly, something is wrong (likely the target network is not being used, or there is a bug in the target computation). Logging average Q-values is a useful diagnostic.

8. **Evaluation vs. training policy**: During evaluation, use a greedy policy (epsilon = 0 or epsilon = 0.05) rather than the training epsilon. Report evaluation performance separately from training performance.

---

## PyTorch Implementation

The following is a complete, working DQN implementation for environments with vector-based state spaces (e.g., CartPole, LunarLander). For image-based environments, replace the MLP with the CNN architecture described above.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    """Neural network that approximates the Q-function."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer for storing transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100_000,
        buffer_capacity=100_000,
        batch_size=32,
        target_update_freq=1000,
        learning_starts=1000,
        device="cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.device = torch.device(device)

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is never trained directly

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Step counter
        self.steps = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        """Linearly decay epsilon."""
        fraction = min(1.0, self.steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def train_step(self):
        """Sample a mini-batch and perform one gradient update."""
        if len(self.replay_buffer) < self.learning_starts:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values: Q(s, a; theta)
        # Gather selects the Q-value for the action that was actually taken
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values: r + gamma * max_a' Q(s', a'; theta_target)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def step(self, state, action, reward, next_state, done):
        """Store transition and perform training."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1
        self.update_epsilon()

        loss = self.train_step()

        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss


# ---------------------
# Example training loop
# ---------------------
# import gymnasium as gym
#
# env = gym.make("CartPole-v1")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
#
# agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
#
# for episode in range(1000):
#     state, _ = env.reset()
#     episode_reward = 0
#
#     while True:
#         action = agent.select_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#
#         agent.step(state, action, reward, next_state, done)
#
#         state = next_state
#         episode_reward += reward
#
#         if done:
#             break
#
#     if episode % 50 == 0:
#         print(f"Episode {episode}, Reward: {episode_reward:.1f}, "
#               f"Epsilon: {agent.epsilon:.3f}, Buffer: {len(agent.replay_buffer)}")
```

### Implementation Notes

- **`gather` for action indexing**: The line `q_network(states).gather(1, actions.unsqueeze(1))` selects the Q-value corresponding to the action that was actually taken in each transition. This is essential because we only have a target for the action that was executed.

- **`(1 - dones)` masking**: When a transition leads to a terminal state, there is no future reward, so the target is simply r. Multiplying by (1 - done) zeros out the bootstrapped term for terminal transitions.

- **`torch.no_grad()` for targets**: The target computation must not contribute gradients. Using `torch.no_grad()` prevents PyTorch from building a computation graph for the target, which is both correct and memory-efficient.

- **Gradient clipping**: `clip_grad_norm_` with max_norm=10.0 prevents occasional large TD errors from causing destructive weight updates.

---

## Limitations

### Overestimation Bias

DQN systematically overestimates Q-values. The max operator in the target computation:

```
y = r + gamma * max_a' Q(s', a'; theta_target)
```

uses the same network both to select the best action and to evaluate its value. If the Q-network has any estimation error (which it always does, especially early in training), the max operator will preferentially select actions whose Q-values are overestimated. Over many updates, this positive bias compounds.

This was addressed by **Double DQN** (van Hasselt et al., 2016), which decouples action selection from action evaluation:

```
a* = argmax_a' Q(s', a'; theta)          # Select action with the online network
y  = r + gamma * Q(s', a*; theta_target)  # Evaluate with the target network
```

This simple change significantly reduces overestimation and often improves performance.

### Discrete Actions Only

DQN outputs a Q-value for each discrete action. If the action space is continuous (e.g., controlling torque on a robot joint), DQN cannot be directly applied because you cannot enumerate all possible actions. Extensions like DDPG (Deep Deterministic Policy Gradient), TD3, and SAC were developed to handle continuous action spaces by learning a policy network that outputs continuous actions.

### Sample Inefficiency

DQN requires millions of environment interactions to learn, even for relatively simple tasks. The replay buffer helps by reusing data, but DQN is still far less sample-efficient than model-based methods or methods with better exploration strategies.

### Limited Representational Capacity

A single Q-value conflates the value of being in a state with the advantage of taking a specific action in that state. **Dueling DQN** (Wang et al., 2016) addresses this by decomposing the Q-network output into two streams:

```
Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))
```

where V(s) is the state-value function and A(s, a) is the advantage function. This decomposition allows the network to learn which states are valuable regardless of the action, improving learning efficiency especially in states where the choice of action does not matter much.

---

## Connection to Related Methods

### Predecessor: Q-Learning

DQN is the direct deep learning extension of tabular Q-Learning. The conceptual update rule is identical; the only difference is that the Q-table is replaced by a neural network and the training process is stabilized with experience replay and a target network. If your state space is small enough for a table, use Q-Learning. When it is not, use DQN.

### Siblings: Double DQN and Dueling DQN

- **Double DQN**: Fixes overestimation bias by using the online network to select actions and the target network to evaluate them. The only code change is in the target computation (two lines). This should be considered a default improvement over vanilla DQN.

- **Dueling DQN**: Changes the network architecture to separately estimate state-value and advantage, then combines them. This is an architectural modification that is orthogonal to Double DQN. The two can (and should) be combined.

- **Prioritized Experience Replay**: Replaces uniform sampling from the replay buffer with prioritized sampling based on TD error magnitude. Transitions with larger errors are sampled more often, focusing learning on the most informative experiences.

- **Rainbow DQN**: Combines six extensions of DQN (Double, Dueling, Prioritized Replay, Multi-step returns, Distributional RL, Noisy Nets) into a single agent. Rainbow demonstrated that these improvements are largely complementary.

### Evolution to Policy-Based and Actor-Critic Methods

DQN is a pure value-based method: it learns Q-values and derives a policy by taking the argmax. Policy-based methods (REINFORCE, PPO) directly optimize the policy. Actor-critic methods (A2C, A3C, SAC) combine both approaches, using a value function to reduce variance in policy gradient estimates. Understanding DQN provides the foundation for understanding these more advanced methods.
