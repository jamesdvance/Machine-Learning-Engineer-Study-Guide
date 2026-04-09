# Markov Decision Processes (MDPs)

## Summary

A Markov Decision Process is the mathematical framework that underpins virtually all of reinforcement learning. It formalizes sequential decision-making problems where an agent interacts with an environment by observing states, taking actions, receiving rewards, and transitioning to new states. The MDP framework gives us the precise language and structure needed to define what it means to act optimally over time, and every major RL algorithm -- from Q-learning to policy gradient methods -- is either solving an MDP directly or approximating a solution to one.

An MDP is defined by a five-tuple (S, A, P, R, gamma) where S is the set of states, A is the set of actions, P is the state transition probability function, R is the reward function, and gamma is the discount factor. The critical assumption is the Markov property: the future depends only on the current state and action, not on the history of how the agent arrived there. This memoryless property is what makes the problem tractable and allows us to define recursive relationships like the Bellman equations.

Key points to remember:

- An MDP is defined by (S, A, P, R, gamma) -- states, actions, transition probabilities, rewards, and discount factor
- The Markov property means the next state depends only on the current state and action, not on the full history
- A policy maps states to actions; the goal is to find the optimal policy that maximizes cumulative discounted reward
- The state-value function V(s) gives expected return from a state; the action-value function Q(s, a) gives expected return from taking an action in a state
- The discount factor gamma controls the tradeoff between immediate and future rewards
- Episodic tasks have a terminal state; continuing tasks run indefinitely and require discounting
- Bellman equations express value functions recursively and are the foundation for solving MDPs
- POMDPs extend MDPs to cases where the agent cannot fully observe the state
- MDPs are not just theoretical -- they model real problems like inventory management, robotics control, and game playing

## The MDP Framework

### Why MDPs Matter for RL

Reinforcement learning is concerned with how an agent should take actions in an environment to maximize some notion of cumulative reward. Before we can design algorithms to solve this problem, we need a mathematical framework that captures all the essential elements: what the agent observes, what it can do, how the world changes, and what constitutes good behavior. The MDP provides exactly this.

Every RL algorithm you encounter -- whether it is tabular Q-learning, deep Q-networks (DQN), REINFORCE, PPO, or actor-critic methods -- is solving some form of MDP. When you see a paper describe "the environment," it is describing an MDP (or a variant of one). Understanding MDPs is therefore not optional background; it is the shared vocabulary of the entire field.

### Agent-Environment Interaction

The basic loop in an MDP is:

1. At time step t, the agent observes the current state s_t
2. The agent selects an action a_t according to its policy
3. The environment transitions to a new state s_{t+1} according to the transition probability P(s_{t+1} | s_t, a_t)
4. The environment emits a reward r_{t+1} = R(s_t, a_t, s_{t+1})
5. The process repeats from step 1

```
Time step t:    Agent observes s_t
                Agent chooses a_t
                Environment transitions to s_{t+1} ~ P(. | s_t, a_t)
                Agent receives r_{t+1}
Time step t+1:  Agent observes s_{t+1}
                ...
```

This loop continues either until a terminal state is reached (episodic tasks) or indefinitely (continuing tasks).

## Formal Definition

An MDP is a five-tuple (S, A, P, R, gamma):

### States (S)

The state space S is the set of all possible situations the agent can be in. States should capture all information necessary for decision-making.

- In a grid world, the state might be the (row, column) position of the agent
- In a game of chess, the state is the full board configuration plus whose turn it is
- In inventory management, the state could be (current stock level, day of week, pending orders)
- In robotics, the state might include joint angles, velocities, and positions of objects

States can be discrete (a finite or countable set) or continuous (a subset of real-valued vectors). The choice of state representation is one of the most important design decisions in applied RL.

### Actions (A)

The action space A defines what the agent can do. Like states, actions can be discrete or continuous:

- Discrete: move up/down/left/right in a grid, buy/sell/hold a stock
- Continuous: apply a torque of x Newton-meters, set a throttle to a value between 0 and 1

In some formulations, the available actions depend on the current state, written A(s). For example, in a board game, legal moves depend on the board position.

### Transition Function (P)

The transition function (also called the dynamics or model) defines how the environment changes in response to the agent's actions:

```
P(s' | s, a) = Probability of transitioning to state s' given current state s and action a
```

This function satisfies:

```
For all s in S, a in A:  sum over all s' of P(s' | s, a) = 1
```

The transition function encodes the "rules" of the environment. It may be:

- **Deterministic**: Each (s, a) pair leads to exactly one next state. P(s' | s, a) is 1 for one state and 0 for all others.
- **Stochastic**: The same action in the same state can lead to different outcomes. For example, a robot's movement may have noise, or a card game involves random draws.

In model-based RL, the agent tries to learn or is given P. In model-free RL, the agent learns to act well without explicitly knowing P.

### Reward Function (R)

The reward function provides the signal that defines what the agent should optimize. Several equivalent formulations exist:

- R(s, a, s'): reward depends on state, action, and next state
- R(s, a): reward depends on state and action
- R(s): reward depends only on the state

The reward is a scalar value. This is a key design constraint: no matter how complex the objective, it must be expressed as a single number at each time step. In practice, designing a good reward function (reward shaping) is one of the hardest parts of applied RL.

Examples:

| Problem | State | Reward |
|---------|-------|--------|
| Grid world | Agent position | -1 per step, +10 at goal |
| Chess | Board configuration | +1 for win, -1 for loss, 0 otherwise |
| Inventory | Stock level | -holding_cost - stockout_penalty + profit |
| Robot locomotion | Joint angles, velocities | Forward velocity - energy cost |

### Discount Factor (gamma)

The discount factor gamma is a value in [0, 1] that determines how much the agent values future rewards relative to immediate ones. The agent's objective is to maximize the expected discounted return:

```
G_t = r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ...
    = sum from k=0 to infinity of gamma^k * r_{t+k+1}
```

The discount factor serves several purposes:

- **Mathematical**: When gamma < 1, the infinite sum converges as long as rewards are bounded. This is necessary for continuing (non-episodic) tasks.
- **Economic**: A dollar today is worth more than a dollar tomorrow. Discounting captures the time value of rewards.
- **Practical**: gamma close to 1 (e.g., 0.99) makes the agent far-sighted, considering long-term consequences. gamma close to 0 (e.g., 0.1) makes the agent myopic, focused on immediate rewards.

| gamma Value | Behavior | Use Case |
|-------------|----------|----------|
| 0 | Purely greedy, only cares about immediate reward | Bandit problems |
| 0.9 | Moderate foresight | Many practical tasks |
| 0.99 | Strong long-term planning | Games, complex planning |
| 1.0 | No discounting (valid only for episodic tasks) | Finite-horizon problems |

## The Markov Property

The defining assumption of an MDP is the Markov property:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)
```

In words: the probability of the next state depends only on the current state and action, not on any prior history. The current state contains all the information needed to predict the future.

### Why the Markov Property Matters

1. **Tractability**: Without this assumption, the agent would need to remember and reason about its entire history, which grows unboundedly. The Markov property means we only need to consider the current state.

2. **Recursive structure**: The Markov property is what allows us to write Bellman equations, which express the value of a state in terms of the values of its successor states. This recursive structure is the foundation of dynamic programming and, by extension, nearly all RL algorithms.

3. **Sufficient statistic**: The state acts as a sufficient statistic for the future. If you know the current state, knowing the history does not help you make better predictions or decisions.

### When the Markov Property Holds (and When It Does Not)

In many engineered environments (board games, simulated physics with full observation), the Markov property holds exactly. In real-world problems, it often does not hold perfectly because the agent's observation does not capture all relevant information. For example:

- A stock trading agent that only sees the current price does not have a Markov state -- recent price trends and volume matter
- A robot with a camera observing a scene may not know velocities from a single frame
- A medical treatment agent may need the patient's full history, not just current vitals

When the Markov property does not hold, the problem becomes a Partially Observable MDP (POMDP), discussed later in this chapter. In practice, there are several approaches to handle non-Markov observations:

- Include history in the state (e.g., stack the last N frames, as in Atari DQN)
- Use recurrent neural networks (LSTMs, GRUs) to maintain hidden state
- Engineer features that summarize relevant history (e.g., moving averages of stock prices)

## Policies

A policy defines the agent's behavior -- how it selects actions given states.

### Deterministic Policies

A deterministic policy is a function that maps each state to a single action:

```
pi(s) = a
```

For example, in a grid world, a deterministic policy might say: "In cell (2,3), always move right."

### Stochastic Policies

A stochastic policy gives a probability distribution over actions for each state:

```
pi(a | s) = P(A_t = a | S_t = s)
```

For example: "In cell (2,3), move right with probability 0.7 and move down with probability 0.3."

Stochastic policies are important for several reasons:

- They allow exploration during learning
- In some environments, the optimal policy is inherently stochastic (e.g., rock-paper-scissors)
- Policy gradient methods naturally produce stochastic policies
- They are necessary when the agent has partial observability

### Optimal Policy

The optimal policy pi* is the policy that maximizes expected cumulative discounted reward from every state. A fundamental result in MDP theory is that for any finite MDP, there always exists at least one deterministic optimal policy. This means that even though we may use stochastic policies during learning, the best possible behavior can always be expressed deterministically (assuming full observability).

## Value Functions

Value functions estimate how good it is for the agent to be in a given state (or to take a given action in a given state) under a particular policy.

### State-Value Function V(s)

The state-value function for policy pi gives the expected return starting from state s and following policy pi thereafter:

```
V_pi(s) = E_pi[G_t | S_t = s]
        = E_pi[r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ... | S_t = s]
```

V_pi(s) answers the question: "How good is it to be in state s if I follow policy pi?"

### Action-Value Function Q(s, a)

The action-value function for policy pi gives the expected return starting from state s, taking action a, and then following policy pi:

```
Q_pi(s, a) = E_pi[G_t | S_t = s, A_t = a]
           = E_pi[r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ... | S_t = s, A_t = a]
```

Q_pi(s, a) answers the question: "How good is it to take action a in state s, and then follow policy pi?"

### Relationship Between V and Q

The two value functions are closely related:

```
V_pi(s) = sum over a of pi(a | s) * Q_pi(s, a)
```

The state-value is the expected action-value under the policy. If the policy is deterministic with pi(s) = a*, then V_pi(s) = Q_pi(s, a*).

### Why Q Functions Are Especially Useful

In practice, Q functions are often more directly useful than V functions because they allow the agent to select actions without knowing the transition model. To act greedily with respect to V(s), the agent needs to know which action leads to which next state (i.e., it needs the transition function P). With Q(s, a), the agent simply picks the action with the highest Q-value:

```
a* = argmax over a of Q(s, a)
```

This is why algorithms like Q-learning and DQN learn Q functions rather than V functions.

### Optimal Value Functions

The optimal state-value function and optimal action-value function are:

```
V*(s) = max over pi of V_pi(s)     for all s in S
Q*(s, a) = max over pi of Q_pi(s, a)   for all s in S, a in A
```

Once Q* is known, the optimal policy is simply:

```
pi*(s) = argmax over a of Q*(s, a)
```

## Connection to Bellman Equations

The Markov property gives rise to a recursive relationship between the value of a state and the values of its successor states. These are the Bellman equations, covered in detail in the sibling chapter on Bellman Equations.

### Bellman Expectation Equation

For a given policy pi:

```
V_pi(s) = sum over a of pi(a | s) * sum over s' of P(s' | s, a) * [R(s, a, s') + gamma * V_pi(s')]
```

This says: the value of a state is the expected immediate reward plus the discounted value of the next state, averaged over the policy's action choices and the environment's transitions.

### Bellman Optimality Equation

For the optimal value function:

```
V*(s) = max over a of sum over s' of P(s' | s, a) * [R(s, a, s') + gamma * V*(s')]
```

And for the optimal action-value function:

```
Q*(s, a) = sum over s' of P(s' | s, a) * [R(s, a, s') + gamma * max over a' of Q*(s', a')]
```

The Bellman optimality equation states that the value of a state under the optimal policy equals the maximum expected return achievable by any single action, followed by optimal behavior thereafter.

### Why This Matters

The Bellman equations transform the problem of finding an optimal policy (which seems to require searching over all possible sequences of actions) into a system of equations. If the transition function P is known, these equations can be solved directly using dynamic programming methods like value iteration and policy iteration. If P is unknown, they motivate sample-based algorithms like Q-learning, SARSA, and temporal difference learning that estimate value functions from experience.

## Finite vs Infinite Horizon

### Finite Horizon

In a finite-horizon MDP, the agent interacts with the environment for a fixed number of steps T. The objective is:

```
G_t = r_{t+1} + r_{t+2} + ... + r_T
```

Note that discounting is optional in finite-horizon problems since the sum is naturally bounded. The optimal policy in a finite-horizon MDP can be non-stationary, meaning the best action in a given state may depend on how many steps remain. For example, in a game with a time limit, the agent should play more aggressively as the deadline approaches.

### Infinite Horizon

In an infinite-horizon MDP, there is no fixed endpoint. The agent interacts with the environment indefinitely. Discounting (gamma < 1) is essential here to ensure the cumulative reward is finite:

```
G_t = sum from k=0 to infinity of gamma^k * r_{t+k+1}
```

The optimal policy in an infinite-horizon discounted MDP is always stationary -- the best action in a state does not depend on the time step. This is a significant simplification.

### Episodic vs Continuing Tasks

| Property | Episodic | Continuing |
|----------|----------|------------|
| Termination | Has a terminal state; resets | Runs forever |
| Horizon | Finite (variable length) | Infinite |
| Discounting | Optional (gamma can be 1) | Required (gamma < 1) |
| Examples | Games, mazes, episodes of robot trials | Process control, ongoing trading |
| Return | Finite sum of rewards | Infinite discounted sum |

Episodic tasks are a special case. Each episode has a natural ending (the agent reaches a goal, the game ends, etc.), and then the environment resets. Many practical RL problems are episodic. The agent's objective is typically to maximize the expected return per episode.

Continuing tasks have no terminal state. The agent must balance immediate and future rewards indefinitely. The discount factor is critical here -- without it, the total reward would be infinite and comparisons between policies would be meaningless.

## Practical Examples

### Example 1: Grid World

Grid world is the canonical MDP example used throughout RL textbooks.

```
+---+---+---+---+
|   |   |   | G |
+---+---+---+---+
|   | X |   |   |
+---+---+---+---+
| S |   |   |   |
+---+---+---+---+

S = Start, G = Goal (+10 reward), X = Wall (impassable)
```

MDP components:

- **States**: Each cell the agent can occupy, represented as (row, col). There are 11 non-wall cells.
- **Actions**: {up, down, left, right}. Moving into a wall or boundary leaves the agent in the same cell.
- **Transitions**: Deterministic (action always succeeds) or stochastic (e.g., 80% intended direction, 10% each perpendicular direction).
- **Rewards**: -1 per step (encourages shortest path), +10 for reaching G, -5 for falling into a trap if one exists.
- **Discount factor**: gamma = 0.9

With deterministic transitions, the optimal policy is the shortest path from S to G. With stochastic transitions, the optimal policy may avoid paths near dangerous cells even if they are shorter, because the noise could push the agent into a bad state.

### Example 2: Inventory Management

A retailer must decide how much product to order each day.

MDP components:

- **States**: Current inventory level (0, 1, 2, ..., max_capacity). Could also include day of week, season, or pending deliveries.
- **Actions**: Quantity to order (0, 1, 2, ..., max_order).
- **Transitions**: Next inventory = current inventory + order - demand. Demand is stochastic and drawn from some distribution (e.g., Poisson). Inventory cannot go below 0 (excess demand is lost sales) or above max_capacity (excess orders are wasted).
- **Rewards**: Revenue from sales minus holding costs minus ordering costs minus stockout penalty.

```
reward = (units_sold * price) - (inventory_held * holding_cost)
         - (units_ordered * order_cost) - (unmet_demand * stockout_penalty)
```

- **Discount factor**: gamma = 0.95 (a dollar of profit tomorrow is worth slightly less than today).

This is a real-world MDP where the transition function is not known exactly (demand is uncertain), making it a natural fit for model-free RL or model-based RL with a learned demand model.

### Example 3: Robot Navigation

A mobile robot must navigate from its current position to a target location in a warehouse.

MDP components:

- **States**: (x, y, theta) -- position and orientation, possibly discretized or continuous.
- **Actions**: (linear_velocity, angular_velocity) -- continuous action space.
- **Transitions**: Governed by physics (kinematics), but with noise in actuation and possible collisions.
- **Rewards**: +100 for reaching the target, -1 per time step, -50 for collisions.
- **Discount factor**: gamma = 0.99 (the robot should plan ahead significantly).

This example illustrates continuous state and action spaces, which require function approximation (e.g., neural networks) rather than tabular methods to represent value functions and policies.

## Partially Observable MDPs (POMDPs)

In many real-world problems, the agent does not have access to the full state of the environment. Instead, it receives observations that provide partial or noisy information about the state. This is formalized as a Partially Observable MDP.

### POMDP Definition

A POMDP extends the MDP tuple to (S, A, P, R, gamma, O, Z):

- S, A, P, R, gamma are the same as in a standard MDP
- **O**: Set of observations
- **Z**: Observation function, Z(o | s', a) = probability of observing o after taking action a and transitioning to state s'

The agent does not see the state s directly. Instead, at each step it receives an observation o drawn from Z. Because the agent cannot distinguish between states that produce the same observation, it must maintain a belief state -- a probability distribution over possible states.

### Examples of Partial Observability

- A poker player cannot see opponents' cards. The true state includes all cards, but the observation only includes the player's own hand and community cards.
- A self-driving car's cameras provide a partial view of the environment. Objects may be occluded, and the car cannot directly observe the intentions of other drivers.
- A dialogue system does not know the user's true goal or knowledge state; it only observes their utterances.

### Practical Approaches

Solving POMDPs exactly is computationally intractable for all but the smallest problems (PSPACE-complete). In practice, practitioners use several strategies:

1. **State augmentation**: Include enough history in the observation to approximate the Markov property. For example, DQN for Atari stacks the last 4 frames to capture velocity information that a single frame does not provide.

2. **Recurrent neural networks**: Use an LSTM or GRU as part of the policy or value network. The hidden state of the RNN serves as an approximate belief state, summarizing relevant history.

3. **Belief-state planning**: Explicitly maintain a probability distribution over possible states and plan in belief space. This is computationally expensive but principled.

4. **Memory-augmented architectures**: Transformer-based RL agents that attend over a window of past observations.

## How MDPs Underpin All RL Methods

Understanding where different RL algorithms sit relative to the MDP framework helps organize the field:

### Model-Based vs Model-Free

- **Model-based RL**: The agent learns or is given the transition function P and reward function R. It can then plan using dynamic programming, Monte Carlo tree search, or learned world models (e.g., MuZero, Dreamer). These methods explicitly use the MDP structure.
- **Model-free RL**: The agent learns a policy or value function directly from experience without learning P. Methods like Q-learning, SARSA, and policy gradient algorithms sample transitions from the environment and update estimates accordingly. They implicitly solve the MDP.

### Value-Based vs Policy-Based

- **Value-based methods** (Q-learning, DQN): Learn the optimal value function Q*(s, a), then derive the policy as the argmax of Q.
- **Policy-based methods** (REINFORCE, PPO): Directly parameterize and optimize the policy pi(a | s).
- **Actor-critic methods** (A3C, SAC): Combine both -- learn a policy (actor) and a value function (critic).

All of these are solving the same underlying MDP. They differ in what they represent explicitly and how they use experience to improve.

### The MDP Assumptions in Practice

When applying RL to real problems, it is worth asking:

- **Is the state representation sufficient?** If the Markov property does not hold for the chosen state, consider enriching the state with history or using recurrent architectures.
- **Is the reward well-defined?** A poorly designed reward function can lead to unexpected behavior (reward hacking). The MDP framework assumes the reward function perfectly captures the objective.
- **Is the environment stationary?** MDPs assume the transition and reward functions do not change over time. Non-stationary environments require adaptation strategies.
- **Is the action space manageable?** Large or continuous action spaces require specialized algorithms (e.g., DDPG, SAC for continuous actions).

## Key Takeaways

1. **MDPs are the foundation of RL.** Every RL problem is formalized as an MDP or a variant (POMDP, multi-agent MDP, etc.). Understanding the (S, A, P, R, gamma) tuple is essential.

2. **The Markov property is the key assumption.** It enables recursive value functions, Bellman equations, and tractable algorithms. When it does not hold, you need POMDP techniques or state augmentation.

3. **Policies and value functions are the core objects.** A policy tells the agent what to do; a value function tells the agent how good a state or action is. The optimal policy maximizes the value function.

4. **The discount factor is not just a technicality.** It shapes the agent's planning horizon and is necessary for well-defined objectives in continuing tasks. Choosing gamma is a modeling decision.

5. **Bellman equations connect everything.** The recursive structure of value functions under the Markov property is what makes MDPs solvable. See the Bellman Equations chapter for detailed coverage.

6. **Real-world problems rarely have perfect Markov states.** Practical RL engineering often involves designing state representations that are "Markov enough" for the algorithm to work well.

7. **The reward function defines the problem.** The MDP framework is only as good as the reward signal. Getting the reward right is often the hardest part of applied RL.
