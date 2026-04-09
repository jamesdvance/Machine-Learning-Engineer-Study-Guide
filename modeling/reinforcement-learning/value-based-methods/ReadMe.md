# Value-Based Methods

## Summary

Value-based methods are a family of reinforcement learning algorithms that learn a value function -- either a state-value function V(s) or an action-value function Q(s, a) -- and derive a policy indirectly by selecting the action with the highest estimated value. Rather than directly parameterizing and optimizing a policy (as policy-based methods do), value-based methods answer the question "how good is it to take action a in state s?" and then act greedily with respect to those estimates. This approach is grounded in the Bellman equations, which express the value of a state or state-action pair recursively in terms of immediate rewards and the values of successor states.

The unifying computational idea behind all value-based methods is temporal difference (TD) learning. TD methods update value estimates based on other value estimates, bootstrapping from incomplete experience rather than waiting for a full episode to conclude. This makes them applicable to continuing (non-episodic) tasks and generally more sample-efficient than Monte Carlo methods. The simplest TD control algorithm, Q-Learning, maintains a table of Q-values and iteratively refines them using the Bellman optimality equation. When state spaces become too large for a table, deep neural networks serve as function approximators, giving rise to Deep Q-Networks (DQN) and its successors.

The evolution of value-based methods follows a clear trajectory. Tabular Q-Learning (Watkins, 1989) established the foundational update rule. DQN (Mnih et al., 2015) scaled this to high-dimensional state spaces by introducing experience replay and target networks. Double DQN (van Hasselt et al., 2016) corrected the systematic overestimation bias caused by the max operator in the TD target. Dueling DQN (Wang et al., 2016) improved the network architecture by decomposing Q-values into state-value and advantage components. Rainbow (Hessel et al., 2018) combined these and other improvements into a single agent that achieved state-of-the-art Atari performance. Each step in this progression addresses a specific limitation while preserving the core idea of learning action-values through temporal difference updates.

Key points to remember:

- Value-based methods learn Q(s, a) and derive a policy as pi(s) = argmax_a Q(s, a); they do not directly parameterize or optimize a policy
- Temporal difference learning is the shared foundation: all methods in this family update value estimates by bootstrapping from successor-state estimates
- Tabular Q-Learning works for small discrete state-action spaces but cannot scale due to the curse of dimensionality
- DQN replaces the Q-table with a neural network and stabilizes training with experience replay and a target network
- Double DQN fixes overestimation bias with a one-line change to the target computation, decoupling action selection from evaluation
- Dueling DQN restructures the network to separate state value V(s) from action advantage A(s, a), improving learning efficiency in states where action choice matters little
- Rainbow DQN combines six improvements (Double, Dueling, Prioritized Replay, Multi-step, Distributional, Noisy Nets) for the strongest baseline in discrete-action problems
- Value-based methods are best suited to discrete action spaces; continuous action spaces require policy-based or actor-critic methods

## Temporal Difference Learning: The Unifying Idea

Every algorithm in this chapter is built on temporal difference learning. Understanding TD learning is essential to understanding why these methods work and where they differ.

### The Core Mechanism

In reinforcement learning, the agent seeks to maximize cumulative discounted reward. To do so, it needs to estimate how valuable states or actions are. There are three broad approaches to forming these estimates:

1. **Dynamic programming** requires a complete model of the environment (transition probabilities and rewards). It is exact but impractical when the model is unknown.
2. **Monte Carlo methods** estimate values by averaging complete episode returns. They are model-free but must wait until an episode terminates.
3. **Temporal difference methods** estimate values by combining observed rewards with existing value estimates of successor states. They are model-free and can learn from incomplete episodes.

TD methods achieve this by bootstrapping: they update an estimate using, in part, another estimate. The simplest TD prediction rule for a state-value function is:

```
V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]
```

The term `r + gamma * V(s')` is the TD target -- the one-step lookahead estimate of the return. The difference `r + gamma * V(s') - V(s)` is the TD error, which measures how far the current estimate is from the bootstrapped target. The learning rate alpha controls how aggressively the estimate is updated.

For control (learning an optimal policy rather than evaluating a fixed one), the same idea applies to action-value functions. Q-Learning uses the TD target `r + gamma * max_a' Q(s', a')`, while SARSA uses `r + gamma * Q(s', a')` where a' is the action actually taken. The max in Q-Learning makes it off-policy: it learns about the optimal policy regardless of the exploration strategy used to collect data.

### Why Bootstrapping Matters

Bootstrapping gives TD methods two important properties. First, they can learn online, updating after every transition rather than waiting for an episode to end. This is critical for continuing tasks that have no natural episode boundary. Second, bootstrapping reduces variance compared to Monte Carlo estimates, because the TD target uses only one step of randomness (one reward and one transition) rather than the accumulated randomness of an entire trajectory. The tradeoff is bias: the TD target depends on the current (possibly inaccurate) value estimate, introducing bias that only disappears as the estimates converge.

All four algorithms covered in the child chapters -- Q-Learning, DQN, Double DQN, and Dueling DQN -- use one-step TD updates with the Bellman optimality equation as the target. They differ in how the Q-function is represented (table vs. neural network), how the target is computed (single network vs. decoupled selection and evaluation), and how the network is structured (monolithic vs. value-advantage decomposition).

## The Value-Based Approach to Reinforcement Learning

### Learning Q-Values

The action-value function Q(s, a) represents the expected cumulative discounted reward from taking action a in state s and then following the optimal policy:

```
Q*(s, a) = E[ r + gamma * r' + gamma^2 * r'' + ... | s, a, pi* ]
```

The Bellman optimality equation gives a recursive characterization:

```
Q*(s, a) = E[ r + gamma * max_a' Q*(s', a') | s, a ]
```

Value-based methods turn this equation into an iterative learning rule. At each step, the agent observes a transition (s, a, r, s') and moves its estimate of Q(s, a) toward the sample Bellman target `r + gamma * max_a' Q(s', a')`. Over many transitions, under appropriate conditions, the estimates converge to the true optimal Q-values.

### Deriving Policy from Values

Once Q* is known (or well-approximated), the optimal policy is deterministic and trivial to extract:

```
pi*(s) = argmax_a Q*(s, a)
```

The agent simply picks the action with the highest Q-value in every state. This is the defining characteristic of value-based methods: the policy is implicit in the value function. There is no separate policy network to train, no policy gradient to estimate, and no entropy bonus to tune. The simplicity of this policy derivation is both a strength (fewer components to debug) and a limitation (the resulting policy is always deterministic, and the argmax only works over a finite, enumerable set of actions).

### The Exploration Problem

Because the derived policy is greedy with respect to current Q-estimates, a purely greedy agent would never discover better actions. Epsilon-greedy exploration is the standard solution: with probability epsilon the agent takes a random action, and with probability 1 - epsilon it takes the greedy action. Epsilon is typically annealed from 1.0 (fully random) to a small value like 0.01 over the course of training. Other exploration strategies (Boltzmann/softmax, upper confidence bounds, noisy networks) exist but epsilon-greedy remains the default for its simplicity.

## The Evolution: From Q-Learning to Rainbow

The value-based methods in this guide form a natural progression, where each algorithm addresses a specific limitation of its predecessor.

### Tabular Q-Learning (Watkins, 1989)

Q-Learning is the foundational off-policy TD control algorithm. It maintains a table of Q-values indexed by (state, action) pairs and updates them with:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

Q-Learning has a convergence guarantee: given sufficient exploration and an appropriately decaying learning rate, it converges to Q* with probability 1. Its limitation is the curse of dimensionality -- the table has |S| x |A| entries, which is intractable for large or continuous state spaces. A 84x84 grayscale Atari frame has 256^(7056) possible states; no table can represent this.

See the [Q-Learning chapter](q-learning/ReadMe.md) for the full treatment, including the derivation from Bellman optimality, comparison with SARSA, hyperparameter guidance, and a complete Python implementation.

### Deep Q-Network / DQN (Mnih et al., 2015)

DQN replaces the Q-table with a deep neural network Q(s, a; theta) that takes a state as input and outputs Q-values for all actions. The key contribution was not the idea of using a neural network (which had been tried before and was known to be unstable) but two techniques that made it work:

- **Experience replay**: Past transitions are stored in a circular buffer and sampled in random mini-batches for training. This breaks the temporal correlation between consecutive samples that destabilizes SGD.
- **Target network**: A periodically-updated copy of the Q-network provides stable targets for the loss function, preventing the moving-target problem where the network chases its own shifting predictions.

The loss function is the squared TD error:

```
L(theta) = E[ (r + gamma * max_a' Q(s', a'; theta_target) - Q(s, a; theta))^2 ]
```

DQN demonstrated human-level performance on many Atari 2600 games from raw pixel input, establishing deep RL as a viable field. Its main limitations are overestimation bias (from the max operator), restriction to discrete action spaces, and high sample requirements.

See the [DQN chapter](dqn/ReadMe.md) for the complete architecture, training loop, hyperparameter table, PyTorch implementation, and discussion of limitations.

### Double DQN (van Hasselt et al., 2016)

The max operator in the DQN target simultaneously selects and evaluates the best action. When Q-estimates contain noise (as they always do during training), this systematically inflates Q-values because the max preferentially picks actions whose values are overestimated. Double DQN decouples selection from evaluation:

```
Standard DQN:   y = r + gamma * Q_target(s', argmax_a Q_target(s', a))
Double DQN:     y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

The online network selects the best action; the target network evaluates it. Because these two networks have partially decorrelated errors (the target network lags behind), the overestimation bias is substantially reduced. This is a one-line code change that introduces no additional parameters or computational overhead, yet it improved the median human-normalized Atari score from roughly 68% (DQN) to 118%.

See the [Double DQN chapter](double-dqn/ReadMe.md) for the mathematical analysis of overestimation, side-by-side code comparison, benchmark results, and guidance on when the improvement matters most.

### Dueling DQN (Wang et al., 2016)

Dueling DQN modifies the network architecture rather than the target computation. Instead of mapping states directly to Q-values, the network splits into two streams after a shared encoder:

- A value stream producing a scalar V(s): how good is this state regardless of action
- An advantage stream producing A(s, a) for each action: how much better is this action than average

The streams are recombined as:

```
Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]
```

The mean subtraction resolves an identifiability problem (without it, the decomposition is not unique). This architecture is especially beneficial in states where action choice does not matter much -- the value stream captures the shared value, while the advantage stream focuses its capacity on states where careful action selection is critical. Since this is purely an architecture change, it combines naturally with Double DQN (the "D3QN" combination) and other improvements.

See the [Dueling DQN chapter](dueling-dqn/ReadMe.md) for the architecture details, the identifiability problem and its solution, PyTorch implementations for both vector and image inputs, and benchmark comparisons.

### Rainbow DQN (Hessel et al., 2018)

Rainbow combines six independent improvements to DQN into a single agent:

| Component | What It Addresses |
|---|---|
| Double DQN | Overestimation bias in the target |
| Dueling architecture | Inefficient representation of state value vs. action advantage |
| Prioritized Experience Replay | Uniform sampling wastes time on uninformative transitions |
| Multi-step returns | Single-step bootstrapping can be slow to propagate rewards |
| Distributional RL (C51) | Learning only the mean return discards distributional information |
| Noisy Networks | Epsilon-greedy exploration is undirected and state-independent |

The Rainbow paper demonstrated that these improvements are largely complementary -- combining all six yields substantially better performance than any subset. In ablation studies, prioritized replay and multi-step returns contributed the most individually, while Double and Dueling provided consistent moderate gains. Rainbow remains a strong baseline for discrete-action problems, though it is complex to implement from scratch. Libraries like Stable-Baselines3 and CleanRL provide ready-made implementations.

### The Full Progression

```
Q-Learning (Watkins, 1989)
  Tabular, off-policy TD control with convergence guarantees
    |
    |  Replace Q-table with neural network; add replay buffer and target network
    v
DQN (Mnih et al., 2015)
  First deep RL agent to achieve human-level Atari performance
    |
    +--- Double DQN (van Hasselt et al., 2016)
    |      Fix overestimation: decouple action selection from evaluation
    |
    +--- Dueling DQN (Wang et al., 2016)
    |      Better architecture: separate V(s) and A(s, a) streams
    |
    +--- Prioritized Experience Replay (Schaul et al., 2016)
    |      Sample important transitions more often
    |
    +--- Multi-step returns, Distributional RL, Noisy Nets
    |
    v
Rainbow DQN (Hessel et al., 2018)
  Combine all six improvements into a single agent
```

## Value-Based vs. Policy-Based Methods

Understanding when to use value-based methods requires comparing them with the alternative: policy-based methods, which directly parameterize and optimize a policy pi(a|s; theta) without learning a value function.

### Fundamental Differences

| Dimension | Value-Based | Policy-Based |
|---|---|---|
| What is learned | Q(s, a) or V(s) | pi(a\|s; theta) directly |
| Policy derivation | Implicit: argmax over Q-values | Explicit: sample from pi |
| Action spaces | Discrete only (argmax requires enumeration) | Discrete or continuous |
| Policy type | Deterministic (greedy) | Can be stochastic |
| Optimization | Minimize TD error (regression) | Maximize expected return (policy gradient) |
| Variance | Lower (bootstrapping reduces variance) | Higher (Monte Carlo returns in gradient estimates) |
| Bias | Higher (bootstrapping introduces bias) | Lower (unbiased gradient estimates, in theory) |
| Stability | Can diverge with function approximation + off-policy (deadly triad) | Generally more stable optimization landscape |

### When to Use Value-Based Methods

Value-based methods are the right choice when the following conditions hold:

- **Discrete action space**: The argmax operation that extracts a policy from Q-values requires enumerating all actions. If the action space is discrete and not too large (up to a few hundred actions), value-based methods work well.
- **Off-policy learning is valuable**: Value-based methods like DQN are off-policy, meaning they can learn from data collected by any policy. This makes them well-suited for batch/offline RL, learning from demonstrations, and settings where interaction with the environment is expensive.
- **Sample efficiency matters and the action space is small**: By reusing data through experience replay, DQN and its variants are more sample-efficient than on-policy methods like REINFORCE or PPO in many discrete-action settings.
- **The problem has a clear optimal deterministic policy**: If the best action in each state is deterministic (as in most single-player games, routing problems, or inventory management), value-based methods are a natural fit.

### When to Use Policy-Based or Actor-Critic Methods

- **Continuous action spaces**: Policy gradient methods (DDPG, TD3, SAC, PPO) can output continuous-valued actions directly. Value-based methods cannot, because the argmax over a continuous space is intractable.
- **Stochastic policies are needed**: In partially observable environments or multi-agent settings, the optimal policy may be stochastic. Policy-based methods naturally represent stochastic policies.
- **Very large discrete action spaces**: When the number of actions is in the thousands or millions (e.g., recommendation from a large catalog), the argmax over Q-values becomes expensive. Policy-based methods can handle this more naturally.
- **Smooth policy updates are important**: Policy gradient methods update the policy smoothly, which can be beneficial in environments sensitive to sudden policy changes.

### Actor-Critic: The Middle Ground

Actor-critic methods (A2C, A3C, PPO, SAC) combine both approaches: an actor (policy network) selects actions, and a critic (value network) evaluates them. The critic reduces the variance of policy gradient estimates, while the actor handles continuous or stochastic action selection. In practice, actor-critic methods are the most widely used family for complex continuous-control tasks, while value-based methods remain dominant for discrete-action problems.

## When Value-Based Methods Are Appropriate

### Strong Use Cases

- **Game playing with discrete controls**: Atari, board games, card games, and other environments with a small finite set of actions. This is the canonical setting for DQN and its variants.
- **Routing and scheduling**: Discrete decisions like "which server to send this request to" or "which item to process next" are naturally modeled as discrete action spaces.
- **Inventory management and operations research**: Stock/reorder decisions with discrete levels map well to Q-learning frameworks.
- **Offline/batch RL**: When you have a fixed dataset of logged interactions (e.g., from a production system) and cannot run new experiments, off-policy value-based methods can learn from the logged data.
- **Environments with high-dimensional observations but few actions**: Image-based tasks with a handful of discrete actions (e.g., navigation with four directional commands) are well-suited to DQN-style methods.

### Weak Use Cases

- **Continuous control**: Robotics, autonomous driving, and other domains with continuous action spaces require methods like SAC, TD3, or PPO.
- **Very large combinatorial action spaces**: If the number of valid actions per state is in the millions, value-based methods struggle because they must compute Q-values for all actions.
- **Multi-agent competitive settings**: Stochastic policies are often necessary for Nash equilibria, favoring policy-based methods.
- **Real-time systems with extremely tight latency budgets**: The forward pass through a DQN is typically fast, but if the action space is enormous, computing the argmax adds overhead.

## Practical Decision Framework

```
Start
  |
  v
Is the action space discrete and reasonably small?
  |
  +-- No --> Use policy-based or actor-critic methods (PPO, SAC, TD3)
  |
  +-- Yes --> Continue
  |
  v
Is the state space small enough for a table (< ~10,000 states)?
  |
  +-- Yes --> Use tabular Q-Learning (simplest, convergence guaranteed)
  |
  +-- No --> Continue
  |
  v
Do you need a strong baseline with minimal tuning?
  |
  +-- Yes --> Use Double Dueling DQN (D3QN) with Prioritized Replay
  |
  +-- No --> Continue
  |
  v
Do you need state-of-the-art discrete-action performance?
  |
  +-- Yes --> Use Rainbow DQN or a modern variant
  |
  +-- No --> Use standard DQN as a starting point, add improvements as needed
```

## Common Pitfalls and Debugging Tips

1. **Q-values growing without bound**: This almost always indicates a bug in the target computation. Verify that the target network is being used and updated correctly, and that terminal states are handled (the bootstrap term should be zero for terminal transitions).

2. **Performance plateaus early**: The agent may not be exploring enough. Check that epsilon is decaying slowly enough and that the replay buffer is large enough to hold diverse experiences.

3. **Training instability (oscillating returns)**: Try reducing the learning rate, increasing the target network update interval, or switching from hard target updates to soft (Polyak) updates.

4. **Overestimation**: If predicted Q-values are much higher than the actual returns achieved by the greedy policy, switch from standard DQN to Double DQN.

5. **Slow reward propagation**: In environments where the reward is sparse and distant, single-step TD updates propagate the signal slowly. Consider multi-step returns or prioritized experience replay.

6. **Forgetting earlier lessons**: If performance degrades after initial improvement, the replay buffer may be too small (old useful experiences are overwritten) or the learning rate may be too high.

## Key References

- Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. PhD Thesis, Cambridge University.
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-Learning. Machine Learning, 8(3-4), 279-292.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level Control through Deep Reinforcement Learning. Nature, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. Proceedings of AAAI.
- Wang, Z., Schaul, T., Hessel, M., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. Proceedings of ICML.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. Proceedings of ICLR.
- Hessel, M., Modayil, J., van Hasselt, H., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. Proceedings of AAAI.
- Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

## Child Chapters

- [Q-Learning](q-learning/ReadMe.md) - Tabular off-policy TD control: the foundational algorithm
- [DQN](dqn/ReadMe.md) - Deep Q-Network: scaling Q-Learning with neural networks
- [Double DQN](double-dqn/ReadMe.md) - Fixing overestimation bias with decoupled selection and evaluation
- [Dueling DQN](dueling-dqn/ReadMe.md) - Separating state value from action advantage for better representation
