# Q-Learning

## Summary

Q-Learning is the foundational off-policy temporal difference (TD) control algorithm in reinforcement learning. Introduced by Chris Watkins in 1989, it learns an action-value function Q(s, a) that estimates the expected cumulative reward of taking action a in state s and then following the optimal policy thereafter. The algorithm is called "off-policy" because it learns about the optimal policy regardless of the exploration strategy used to collect experience. This separation of the behavior policy (how the agent explores) from the target policy (what the agent is trying to learn) is the defining characteristic that distinguishes Q-Learning from on-policy alternatives like SARSA.

Q-Learning works by maintaining a table of Q-values for every state-action pair and iteratively updating them using observed rewards and the maximum Q-value of the next state. Under standard conditions (all state-action pairs are visited infinitely often, the learning rate decays appropriately), Q-Learning is guaranteed to converge to the optimal action-value function. This theoretical guarantee, combined with its simplicity, makes it the standard starting point for understanding reinforcement learning algorithms.

Key points to remember:

- Off-policy TD control: Q-Learning updates toward the maximum Q-value of the next state, not the Q-value of the action actually taken next. This makes it off-policy.
- Bellman optimality equation: The update rule is derived directly from the Bellman optimality equation for Q-values. Each update moves the current estimate closer to the one-step Bellman target.
- Q-table representation: In the tabular setting, Q-values are stored in a lookup table indexed by (state, action) pairs. This limits applicability to problems with small, discrete state and action spaces.
- Exploration vs exploitation: The agent must balance exploring new state-action pairs (to improve estimates) with exploiting current knowledge (to maximize reward). Epsilon-greedy is the most common strategy.
- Three critical hyperparameters: Learning rate (alpha) controls step size, discount factor (gamma) controls how much future rewards matter, and exploration rate (epsilon) controls the explore-exploit tradeoff.
- Convergence guarantee: Q-Learning converges to the optimal Q-function given sufficient exploration and a decaying learning rate, under the Robbins-Monro conditions.
- Curse of dimensionality: The Q-table grows as |S| x |A|, making tabular Q-Learning impractical for large or continuous state spaces. This limitation directly motivated Deep Q-Networks (DQN).
- Foundation for deep RL: DQN, Double DQN, and Dueling DQN all extend Q-Learning by replacing the Q-table with a neural network function approximator.

## Background: Temporal Difference Learning

To understand Q-Learning, it helps to first understand where it sits in the reinforcement learning landscape. In RL, an agent interacts with an environment modeled as a Markov Decision Process (MDP). At each time step, the agent observes a state s, takes an action a, receives a reward r, and transitions to a new state s'. The goal is to learn a policy that maximizes the expected cumulative discounted reward.

There are three main families of methods for solving this problem:

1. **Dynamic programming**: Requires a complete model of the environment (transition probabilities and reward function). Computes exact solutions but is impractical when the model is unknown.
2. **Monte Carlo methods**: Learn from complete episodes of experience. Do not require a model but must wait until an episode terminates to update value estimates.
3. **Temporal difference (TD) methods**: Learn from incomplete episodes by bootstrapping. They update value estimates based on other value estimates, combining the model-free advantage of Monte Carlo methods with the bootstrapping efficiency of dynamic programming.

Q-Learning is a TD method. Specifically, it is a TD control method, meaning it learns an optimal policy (not just evaluates a given policy). The "control" distinction matters because policy evaluation (prediction) and policy improvement (control) are different problems. TD prediction methods like TD(0) estimate the value of following a fixed policy. TD control methods like Q-Learning and SARSA actively improve the policy.

## The Q-Value and Bellman Optimality

The action-value function Q(s, a) represents the expected return (cumulative discounted reward) starting from state s, taking action a, and then following policy pi:

```
Q_pi(s, a) = E_pi[ R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ... | S_t = s, A_t = a ]
```

The optimal action-value function Q*(s, a) gives the expected return when taking action a in state s and then following the optimal policy:

```
Q*(s, a) = max_pi Q_pi(s, a)
```

The Bellman optimality equation for Q* states that the value of taking action a in state s under the optimal policy equals the immediate reward plus the discounted value of the best action in the next state:

```
Q*(s, a) = E[ R_{t+1} + gamma * max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a ]
```

This recursive relationship is the theoretical foundation of Q-Learning. If we had the true Q* values, we could extract the optimal policy by simply choosing the action with the highest Q-value in each state: pi*(s) = argmax_a Q*(s, a).

## The Q-Learning Update Rule

Q-Learning turns the Bellman optimality equation into an iterative update rule. After observing a transition (s, a, r, s'), the Q-value is updated as:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_{a'} Q(s', a') - Q(s, a) ]
```

Breaking this down:

- **Q(s, a)**: The current estimate of the value of taking action a in state s
- **alpha**: The learning rate (0 < alpha <= 1). Controls how much the new information overrides the old estimate.
- **r + gamma * max_{a'} Q(s', a')**: The TD target. This is the one-step Bellman optimality backup. It uses the actual received reward r plus the discounted value of the best possible action in the next state s'.
- **r + gamma * max_{a'} Q(s', a') - Q(s, a)**: The TD error (delta). This is the difference between the target and the current estimate. When the TD error is zero, the Q-value is self-consistent with the Bellman equation.

The update can be understood as gradient descent on the squared TD error. At each step, the agent observes a sample of the true Bellman backup and adjusts Q(s, a) a fraction alpha of the way toward that sample. Over many updates, the Q-values converge to the true optimal values.

### Derivation from Bellman Optimality

The update rule follows directly from the idea of making Q-values satisfy the Bellman optimality equation. If Q* is the fixed point, then:

```
Q*(s, a) = E[ r + gamma * max_{a'} Q*(s', a') ]
```

We do not know Q*, so we use our current estimate Q in place of Q* on the right-hand side, and we use a single sample (s, a, r, s') instead of the full expectation. This gives us a stochastic approximation:

```
target = r + gamma * max_{a'} Q(s', a')
```

We then move Q(s, a) toward this target by a step of size alpha:

```
Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
```

This is a standard stochastic approximation update. Under the Robbins-Monro conditions (alpha decays such that the sum of alpha diverges but the sum of alpha squared converges), this process converges to the fixed point Q*.

## Off-Policy vs On-Policy: Q-Learning vs SARSA

The critical distinction between Q-Learning and SARSA lies in what value is used for the next state in the update:

**Q-Learning (off-policy)**:
```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_{a'} Q(s', a') - Q(s, a) ]
```

**SARSA (on-policy)**:
```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
```

where a' is the action actually taken in state s' (sampled from the current policy, including exploration).

The difference is a single term: Q-Learning uses `max_{a'} Q(s', a')`, while SARSA uses `Q(s', a')` where a' is the action the agent actually takes next.

This has significant consequences:

- **Q-Learning** learns the value of the optimal policy regardless of what the agent actually does during exploration. Even if the agent takes random actions, Q-Learning will converge to Q*. This is what "off-policy" means: the policy being learned (greedy) differs from the policy being followed (epsilon-greedy or any other exploration strategy).

- **SARSA** learns the value of the policy the agent is actually following, including its exploration behavior. If the agent uses epsilon-greedy exploration, SARSA learns the value of the epsilon-greedy policy, not the fully greedy policy. This is "on-policy" because the target policy and behavior policy are the same.

### Practical Implications

The off-policy nature of Q-Learning has both advantages and disadvantages:

Advantages:
- Can learn from data generated by any policy (including human demonstrations, random exploration, or a different agent)
- Converges to the optimal policy regardless of the exploration strategy
- Can use more aggressive exploration without biasing the learned values

Disadvantages:
- Can overestimate Q-values due to the max operator (this is the maximization bias problem, addressed by Double Q-Learning and Double DQN)
- In the function approximation setting (not tabular), off-policy learning can diverge (the deadly triad of function approximation, bootstrapping, and off-policy learning)
- SARSA can be safer in practice for stochastic environments because it accounts for exploration in its value estimates (the classic "cliff walking" example demonstrates this)

## Algorithm: Tabular Q-Learning

### Pseudocode

```
Algorithm: Q-Learning (Off-Policy TD Control)

Initialize Q(s, a) arbitrarily for all s in S, a in A(s)
Initialize Q(terminal, .) = 0

For each episode:
    Initialize state s
    
    For each step of the episode:
        Choose action a from s using policy derived from Q
            (e.g., epsilon-greedy: with probability epsilon select random action,
             otherwise select argmax_a Q(s, a))
        
        Take action a, observe reward r and next state s'
        
        Update Q-value:
            Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        
        s <- s'
    
    Until s is terminal
```

### Python Implementation

The following implementation demonstrates Q-Learning on a grid world environment. The grid world is a common benchmark: the agent starts in one cell and must navigate to a goal cell while avoiding obstacles, receiving a reward of -1 per step (to encourage finding the shortest path) and a reward of +10 for reaching the goal.

```python
import numpy as np
import random
from collections import defaultdict


class GridWorld:
    """
    Simple grid world environment.
    
    The agent starts at (0, 0) and must reach the goal at (rows-1, cols-1).
    Walls block movement. Each step costs -1, reaching the goal gives +10.
    """
    
    def __init__(self, rows=5, cols=5, walls=None):
        self.rows = rows
        self.cols = cols
        self.walls = walls or set()
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.action_names = ['up', 'down', 'left', 'right']
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.state = self.start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        dr, dc = self.action_deltas[action]
        new_r = self.state[0] + dr
        new_c = self.state[1] + dc
        
        # Check boundaries and walls
        if (0 <= new_r < self.rows and 0 <= new_c < self.cols
                and (new_r, new_c) not in self.walls):
            self.state = (new_r, new_c)
        
        # Determine reward and done flag
        if self.state == self.goal:
            return self.state, 10.0, True
        else:
            return self.state, -1.0, False


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.
    """
    
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Initialize Q-table as a dictionary of dictionaries
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = [a for a in self.actions if q_values[a] == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update rule."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, num_episodes=1000, max_steps=100):
    """Train the Q-Learning agent."""
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return rewards_per_episode


def extract_policy(agent, env):
    """Extract the greedy policy from the learned Q-table."""
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal or state in env.walls:
                continue
            q_values = agent.q_table[state]
            best_action = np.argmax(q_values)
            policy[state] = env.action_names[best_action]
    return policy


# Run the example
if __name__ == "__main__":
    # Create a 5x5 grid world with some walls
    walls = {(1, 1), (1, 3), (2, 3), (3, 1)}
    env = GridWorld(rows=5, cols=5, walls=walls)
    
    agent = QLearningAgent(
        actions=env.actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    rewards = train(env, agent, num_episodes=1000, max_steps=100)
    
    # Display the learned policy
    policy = extract_policy(agent, env)
    print("\nLearned Policy:")
    for r in range(env.rows):
        row_str = []
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                row_str.append("GOAL ")
            elif state in walls:
                row_str.append("WALL ")
            elif state in policy:
                row_str.append(f"{policy[state]:5s}")
            else:
                row_str.append("  ?  ")
        print(" | ".join(row_str))
```

## Q-Table Representation and Convergence

### Q-Table Structure

In tabular Q-Learning, Q-values are stored in a two-dimensional table (or equivalent data structure like a dictionary or NumPy array) indexed by (state, action):

```
              Action 0    Action 1    Action 2    Action 3
State 0       Q(0, 0)     Q(0, 1)     Q(0, 2)     Q(0, 3)
State 1       Q(1, 0)     Q(1, 1)     Q(1, 2)     Q(1, 3)
State 2       Q(2, 0)     Q(2, 1)     Q(2, 2)     Q(2, 3)
...
State N       Q(N, 0)     Q(N, 1)     Q(N, 2)     Q(N, 3)
```

The table has |S| rows and |A| columns, where |S| is the number of states and |A| is the number of actions. Each cell stores a single floating-point number representing the estimated value of that state-action pair.

### Initialization

Q-values are typically initialized to zero, though optimistic initialization (setting Q-values to a high value) is sometimes used to encourage early exploration. With optimistic initialization, the agent is initially "disappointed" by every action it tries, which drives it to explore actions it has not yet attempted.

### Convergence Guarantees

Watkins and Dayan (1992) proved that tabular Q-Learning converges to Q* with probability 1, given the following conditions:

1. **All state-action pairs continue to be visited**: Every (s, a) pair must be updated infinitely often. This is typically ensured by using an exploration strategy like epsilon-greedy with epsilon > 0.

2. **Learning rate conditions (Robbins-Monro)**: The sequence of learning rates alpha_t must satisfy:
   - Sum of alpha_t = infinity (the steps are large enough to overcome initial conditions)
   - Sum of alpha_t^2 < infinity (the steps become small enough to converge)

   In practice, a fixed small learning rate like alpha = 0.1 often works well despite violating the second condition, because perfect convergence to Q* is not required for good policy performance.

3. **The environment is a finite MDP**: The state and action spaces must be finite.

In practice, convergence depends on the problem size, the exploration strategy, and the learning rate schedule. Small problems (hundreds of states) converge in thousands of episodes. Larger problems may require millions of episodes or may not converge at all within a practical time budget, motivating the use of function approximation.

## Hyperparameters

### Learning Rate (alpha)

The learning rate controls how much each new experience changes the Q-value estimate.

- **alpha = 0**: No learning occurs. Q-values never change.
- **alpha = 1**: The Q-value is completely replaced by the new TD target. Previous experience is forgotten. This is appropriate for deterministic environments.
- **alpha between 0 and 1**: A weighted average of old and new information.

Typical values range from 0.01 to 0.5. In deterministic environments, higher values (0.1 to 0.5) work well because there is no noise to average out. In stochastic environments, lower values (0.01 to 0.1) are better because the agent needs to average over many samples of the same transition to get an accurate estimate.

A common schedule is to start with a moderate learning rate (0.1) and optionally decay it over time, though many practitioners use a fixed rate for simplicity.

### Discount Factor (gamma)

The discount factor determines how much the agent values future rewards relative to immediate rewards.

- **gamma = 0**: The agent is completely myopic, only caring about the immediate reward. It will always take the action with the highest immediate reward.
- **gamma = 1**: The agent values all future rewards equally. This can cause issues in continuing (non-episodic) tasks because the sum of rewards may diverge.
- **gamma close to 1 (e.g., 0.99)**: The agent considers long-term consequences. This is the most common setting for episodic tasks.

The choice of gamma affects the effective planning horizon. With gamma = 0.99, rewards 100 steps in the future are discounted by 0.99^100 = 0.366, so they still matter significantly. With gamma = 0.9, rewards 100 steps ahead are discounted by 0.9^100 = 0.0000266, making the agent effectively short-sighted.

### Exploration Rate (epsilon)

In epsilon-greedy exploration, the agent takes a random action with probability epsilon and the greedy action (argmax of Q-values) with probability 1 - epsilon.

- **epsilon = 1**: Fully random exploration. Useful at the start of training when Q-values are meaningless.
- **epsilon = 0**: Fully greedy. No exploration. The agent exploits its current knowledge but may get stuck in a suboptimal policy.
- **Decaying epsilon**: Start with high exploration (epsilon = 1.0) and gradually decrease to a small value (epsilon = 0.01). This allows thorough exploration early on and exploitation of learned knowledge later.

A standard decay schedule multiplies epsilon by a decay factor (e.g., 0.995 or 0.999) after each episode. The minimum epsilon should be slightly above zero to ensure continued exploration.

Other exploration strategies exist (Boltzmann/softmax exploration, UCB, optimistic initialization) but epsilon-greedy is the default for its simplicity and effectiveness.

## Practical Example: Grid World Walkthrough

Consider a 3x3 grid world:

```
[S] [ ] [ ]
[ ] [W] [ ]
[ ] [ ] [G]
```

S is the start (0,0), G is the goal (2,2), W is a wall (1,1). The agent can move up, down, left, or right. Moving into a wall or boundary keeps the agent in place. Each step gives -1 reward, and reaching the goal gives +10.

### Early Training (Episode 1)

The Q-table starts at all zeros. With epsilon = 1.0, the agent moves randomly. Suppose the trajectory is:

```
(0,0) -> right -> (0,1), r=-1
(0,1) -> down  -> (1,1) [wall, stays at (0,1)], r=-1
(0,1) -> right -> (0,2), r=-1
(0,2) -> down  -> (1,2), r=-1
(1,2) -> down  -> (2,2), r=+10
```

The last update is: Q((1,2), down) <- 0 + 0.1 * [10 + 0.99 * 0 - 0] = 1.0

Now Q((1,2), down) = 1.0 while all other Q-values are still 0. The second-to-last update is: Q((0,2), down) <- 0 + 0.1 * [-1 + 0.99 * max Q((1,2), .) - 0] = 0.1 * [-1 + 0.99 * 1.0] = -0.001

The reward signal propagates backward from the goal. After many episodes, Q-values throughout the grid will reflect the expected return of each state-action pair under the optimal policy.

### After Convergence

After sufficient training, the Q-table encodes the optimal path. The greedy policy (argmax of Q-values at each state) yields the shortest path from start to goal, navigating around the wall.

## Limitations of Tabular Q-Learning

### Curse of Dimensionality

The Q-table has |S| x |A| entries. For problems with large state spaces, this becomes intractable:

- **Board games**: Chess has approximately 10^47 possible states. A Q-table cannot fit in memory.
- **Continuous state spaces**: A robot with joint angles, velocities, and positions has a continuous and potentially high-dimensional state space. Discretization is possible but results in exponential growth: discretizing each of D dimensions into B bins yields B^D states.
- **Image observations**: An Atari game frame at 84x84 pixels with 256 grayscale values has 256^(84*84) possible states, far beyond any tabular representation.

### No Generalization

Tabular Q-Learning treats each state independently. Learning that a particular action is good in one state provides no information about nearby states. A function approximator, by contrast, can generalize: learning from one state can update Q-values for similar states.

### Maximization Bias

The max operator in the Q-Learning update can cause systematic overestimation of Q-values. When Q-values are noisy (as they always are during learning), taking the maximum over noisy estimates is biased upward. This is because the maximum of a set of noisy estimates is more likely to include positive noise than negative noise.

Double Q-Learning (Hasselt, 2010) addresses this by maintaining two Q-tables and using one to select the action and the other to evaluate it. This idea was later incorporated into Deep Q-Networks as Double DQN.

### Slow Propagation of Rewards

In tabular Q-Learning, reward information propagates one step per update. If the goal is 100 steps away, it takes at least 100 episodes for the reward signal to reach the start state (and many more for the values to stabilize). Eligibility traces (Q(lambda)) and prioritized experience replay (in DQN) help address this.

## Connection to Deep Q-Networks and Extensions

Tabular Q-Learning is the direct ancestor of several important deep reinforcement learning algorithms. Understanding Q-Learning thoroughly is essential for understanding these extensions.

### Deep Q-Network (DQN)

DQN (Mnih et al., 2015) replaces the Q-table with a deep neural network that takes the state as input and outputs Q-values for all actions. This allows Q-Learning to scale to high-dimensional state spaces like raw pixel observations. DQN introduced two key innovations to stabilize training:

- **Experience replay**: Store transitions in a replay buffer and sample random mini-batches for training. This breaks the correlation between consecutive samples and improves data efficiency.
- **Target network**: Use a separate, periodically updated copy of the Q-network to compute TD targets. This prevents the instability caused by the Q-values and the target changing simultaneously.

The DQN update is the same Q-Learning update, but with a neural network Q(s, a; theta) instead of a table:

```
Loss = (r + gamma * max_{a'} Q(s', a'; theta_target) - Q(s, a; theta))^2
```

See the [DQN chapter](../dqn/ReadMe.md) for a detailed treatment.

### Double DQN

Double DQN (van Hasselt et al., 2016) addresses the maximization bias of DQN by decoupling action selection from action evaluation:

```
target = r + gamma * Q(s', argmax_{a'} Q(s', a'; theta); theta_target)
```

The online network selects the best action, but the target network evaluates it. This reduces overestimation and generally improves performance.

See the [Double DQN chapter](../double-dqn/ReadMe.md) for details.

### Dueling DQN

Dueling DQN (Wang et al., 2016) modifies the network architecture to separately estimate the state value V(s) and the advantage A(s, a) of each action:

```
Q(s, a) = V(s) + A(s, a) - mean_{a'} A(s, a')
```

This decomposition helps the network learn which states are valuable regardless of action, improving learning efficiency especially in states where the choice of action does not matter much.

See the [Dueling DQN chapter](../dueling-dqn/ReadMe.md) for details.

### Summary of the Evolution

```
Tabular Q-Learning (1989)
    |
    v
Deep Q-Network / DQN (2015) -- adds neural network function approximation,
    |                           experience replay, target networks
    v
Double DQN (2016) ------------ fixes maximization bias
    |
    v
Dueling DQN (2016) ----------- architectural improvement separating
    |                           value and advantage streams
    v
Rainbow (2018) --------------- combines DQN, Double DQN, Dueling DQN,
                                prioritized replay, multi-step returns,
                                distributional RL, and noisy nets
```

## Practical Tips

1. **Start with a simple environment** to verify your implementation is correct. FrozenLake and CliffWalking from Gymnasium (formerly OpenAI Gym) are standard test environments for tabular Q-Learning. If Q-Learning does not converge on these, there is a bug.

2. **Monitor the TD error** during training. If the average TD error is not decreasing, the learning rate may be too high or the exploration rate may not be decaying.

3. **Decay epsilon gradually**. A common mistake is decaying epsilon too fast, which causes the agent to exploit a poor policy before it has explored sufficiently. Err on the side of exploring too much.

4. **Use a discount factor close to 1** (0.95 to 0.99) for most problems. A lower gamma makes the agent myopic and can prevent it from finding long-horizon solutions.

5. **Visualize the Q-table** during development. For grid worlds, plotting the learned policy (arrows showing the greedy action in each cell) is the fastest way to verify the agent is learning correctly.

6. **If your state space is too large for a table**, do not try to discretize it into millions of bins. Switch to DQN or another function approximation method.

7. **Compare with SARSA** to understand the off-policy vs on-policy distinction experimentally. The Cliff Walking environment in Gymnasium is specifically designed to illustrate this difference: Q-Learning finds the optimal path along the cliff edge, while SARSA learns a safer path farther from the edge.

8. **For reproducibility**, seed both the environment and the agent's random number generators. Q-Learning with epsilon-greedy exploration is inherently stochastic, and results can vary significantly across runs.

## See Also

- [DQN](../dqn/ReadMe.md) - Deep Q-Network: scaling Q-Learning with neural networks
- [Double DQN](../double-dqn/ReadMe.md) - Fixing maximization bias in DQN
- [Dueling DQN](../dueling-dqn/ReadMe.md) - Architectural improvement separating value and advantage
- [Bellman Equations](../../concepts/bellman-equations/ReadMe.md) - The mathematical foundation
- [Markov Decision Processes](../../concepts/markov-decision-processes/ReadMe.md) - The formal problem setting
- [Exploration vs Exploitation](../../concepts/exploration-vs-exploitation/ReadMe.md) - The exploration challenge
