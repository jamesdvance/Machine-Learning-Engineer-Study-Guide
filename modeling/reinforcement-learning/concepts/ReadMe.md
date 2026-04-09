# Reinforcement Learning Concepts

## Summary

Reinforcement learning (RL) is a paradigm of machine learning in which an agent learns to make sequential decisions by interacting with an environment and receiving reward signals. Unlike supervised learning, where the correct output is provided for each input, an RL agent must discover which actions yield the most reward through trial and error. The mathematical foundations of RL rest on three interconnected pillars: Markov Decision Processes define the problem structure, Bellman equations provide the recursive machinery for computing solutions, and the exploration-exploitation trade-off captures the fundamental challenge that every learning agent must navigate.

Understanding these foundations is not optional for an ML engineer working with RL. Every algorithm in the field, from tabular Q-learning to proximal policy optimization to model-based planning with learned world models, is grounded in the same conceptual framework. The agent observes a state, takes an action, receives a reward, transitions to a new state, and repeats. The goal is to find a policy (a mapping from states to actions) that maximizes the cumulative reward over time. How we formalize "cumulative reward over time," how we decompose the problem into tractable subproblems, and how we balance learning with performing are the subjects of this chapter and its children.

Key points to remember:

- Reinforcement learning is a framework for sequential decision-making under uncertainty, where an agent learns from interaction rather than labeled examples
- Markov Decision Processes (MDPs) provide the formal mathematical structure: states, actions, transitions, rewards, and a discount factor
- Bellman equations express value functions recursively, decomposing the value of a state into immediate reward plus discounted future value
- The exploration-exploitation dilemma is the central practical challenge: the agent must balance gathering information with using what it already knows
- Returns, discount factors, episodes, value functions, and policies are the building blocks that connect the three pillars
- Value functions come in two forms: state-value V(s) and action-value Q(s,a), each answering a different question about expected future reward
- Policies can be deterministic or stochastic, and the optimal policy maximizes expected return from every state
- The discount factor gamma shapes the agent's planning horizon and is necessary for well-defined objectives in non-terminating tasks
- Model-free methods learn without knowing the environment dynamics; model-based methods learn or leverage a model of the transitions

## The RL Problem: An Integrated View

The central question of reinforcement learning is deceptively simple: given an environment with unknown dynamics and a reward signal, how should an agent act to accumulate the most reward over time? Answering this question requires three things. First, a precise mathematical formulation of the problem. Second, a way to compute or approximate the solution. Third, a strategy for learning when the answer is not known in advance. These correspond directly to the three child chapters of this section.

### MDPs Define the Problem

The Markov Decision Process is the standard formulation for RL problems. An MDP is specified by a five-tuple (S, A, P, R, gamma): a set of states S that describe the environment, a set of actions A available to the agent, a transition function P that governs how the environment evolves, a reward function R that provides the learning signal, and a discount factor gamma that controls how much the agent values future versus immediate rewards.

The MDP framework gives us a common language. When a researcher describes "the environment," they are describing an MDP. When an engineer designs a reward function, they are specifying the R component. When a paper discusses "the optimal policy," it means the policy that maximizes expected cumulative discounted reward within the MDP. The framework is general enough to capture problems ranging from board games to robotic manipulation to recommendation systems.

The critical assumption underlying MDPs is the Markov property: the next state depends only on the current state and action, not on the history of prior states and actions. This memoryless property is what makes the problem tractable. It allows us to reason about the future based solely on where we are now, rather than how we got here.

For full details on MDPs, including the formal definition, the Markov property, policies, value functions, POMDPs, and worked examples, see [Markov Decision Processes](markov-decision-processes/).

### Bellman Equations Provide the Solution Structure

Once the problem is defined as an MDP, the Bellman equations tell us how to solve it. Named after Richard Bellman, these equations express a recursive relationship: the value of being in a state equals the immediate reward the agent expects to receive plus the discounted value of whatever state comes next. This decomposition transforms an apparently intractable problem (reasoning about all possible future trajectories) into a system of equations that can be solved iteratively or approximated with samples.

There are two families of Bellman equations. The Bellman Expectation Equations evaluate a given policy, answering the question: "If I follow this specific policy, how much reward will I accumulate from each state?" The Bellman Optimality Equations characterize the best possible policy, answering: "What is the maximum reward achievable from each state, and which actions achieve it?" The expectation equations are linear and can be solved exactly for small problems. The optimality equations are nonlinear due to the max operator and require iterative methods.

The Bellman equations are not just theoretical elegance. They are the operational core of RL algorithms. Value iteration applies the Bellman Optimality Equation as an update rule. Policy iteration alternates between solving the Bellman Expectation Equation and improving the policy greedily. Q-learning approximates the Bellman Optimality Equation using sampled experience. Deep Q-Networks train a neural network to minimize the squared Bellman error. Understanding these equations means understanding why these algorithms work.

For detailed derivations, code examples, the contraction mapping property, and connections to dynamic programming and Q-learning, see [Bellman Equations](bellman-equations/).

### Exploration vs Exploitation Is the Fundamental Challenge

Even with the MDP formulation and the Bellman equations in hand, a critical practical problem remains: the agent does not know the environment's dynamics or reward structure in advance. It must learn them through interaction. This creates a tension. The agent can exploit its current best estimate of which actions are good, or it can explore actions it has tried less frequently in hopes of finding something better. Too much exploitation locks the agent into suboptimal behavior. Too much exploration wastes time on actions the agent already knows are poor.

This dilemma is present at every level of RL, from the simplest multi-armed bandit to the most complex deep RL system. The strategies developed to address it, including epsilon-greedy, Upper Confidence Bound, Thompson Sampling, intrinsic motivation, and entropy regularization, represent some of the most practically important ideas in the field. Choosing an appropriate exploration strategy and tuning its parameters is often the difference between an RL system that works and one that does not.

For a comprehensive treatment of exploration methods, from bandit algorithms to curiosity-driven exploration in deep RL, see [Exploration vs Exploitation](exploration-vs-exploitation/).

## Returns: Defining the Objective

The agent's goal in an RL problem is to maximize the expected return. The return G_t is the cumulative reward from time step t onward:

```
G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
    = sum_{k=0}^{infinity} gamma^k * R_{t+k+1}
```

The return is not simply the sum of all future rewards. Each future reward is multiplied by gamma raised to a power that increases with the time delay. This discounting is what makes the infinite sum finite (when gamma < 1 and rewards are bounded) and encodes the idea that rewards received sooner are more valuable than rewards received later.

The recursive structure of the return is the key insight that enables the entire Bellman equation framework:

```
G_t = R_{t+1} + gamma * G_{t+1}
```

This says the return from time t equals the immediate reward plus gamma times the return from time t+1. This one-step decomposition is what allows us to express value functions recursively rather than as intractable infinite sums.

### Undiscounted Returns

In episodic tasks that are guaranteed to terminate, it is sometimes valid to use gamma = 1, making the return simply the sum of all rewards until termination:

```
G_t = R_{t+1} + R_{t+2} + ... + R_T
```

where T is the terminal time step. This is the undiscounted return. It is appropriate when the episode length is bounded and there is no reason to prefer earlier rewards over later ones. Many game-playing tasks use undiscounted returns because each game has a definite end.

## The Discount Factor: More Than a Technicality

The discount factor gamma, a scalar in [0, 1], appears in the definition of the return and propagates into every Bellman equation and every RL algorithm. It plays multiple roles simultaneously.

**Mathematical role.** When gamma < 1, the geometric series sum_{k=0}^{infinity} gamma^k converges to 1/(1-gamma). This guarantees that the return is finite as long as rewards are bounded, which is necessary for continuing (non-episodic) tasks where the agent runs indefinitely.

**Economic role.** Discounting captures the intuition that a reward received now is worth more than the same reward received in the future. This parallels the concept of present value in economics and finance. An agent with gamma = 0.99 values a reward received 100 steps from now at roughly 37% of its face value (0.99^100 is approximately 0.366).

**Planning horizon role.** The effective planning horizon of the agent is approximately 1/(1-gamma). With gamma = 0.9, the agent effectively looks about 10 steps ahead. With gamma = 0.99, it looks about 100 steps ahead. With gamma = 0.999, about 1000 steps. This means gamma directly controls how far-sighted the agent is.

**Convergence role.** In iterative algorithms like value iteration, gamma determines the convergence rate. The Bellman operator is a contraction mapping with factor gamma, meaning errors shrink by a factor of gamma at each iteration. Smaller gamma values produce faster convergence but more myopic agents.

| gamma | Effective Horizon | Behavior | Typical Use |
|-------|-------------------|----------|-------------|
| 0     | 1 step            | Purely greedy, immediate reward only | Bandit problems |
| 0.9   | ~10 steps         | Short-term planning | Simple tasks |
| 0.95  | ~20 steps         | Moderate planning | Many practical tasks |
| 0.99  | ~100 steps        | Long-term planning | Games, robotics |
| 0.999 | ~1000 steps       | Very long-term planning | Complex planning |
| 1.0   | Infinite          | No discounting | Episodic tasks only |

In practice, gamma is a hyperparameter that must be chosen based on the problem. Setting it too low makes the agent ignore consequences that are important but delayed. Setting it too high increases variance in value estimates and slows convergence.

## Episodes, Tasks, and Time

RL problems fall into two broad categories based on their temporal structure, and this distinction affects how returns are defined, how algorithms are structured, and which discount factors are valid.

### Episodic Tasks

An episodic task has a natural endpoint. The agent interacts with the environment until a terminal state is reached, at which point the environment resets and a new episode begins. Examples include games (the game ends in a win, loss, or draw), navigation tasks (the agent reaches a goal or runs out of time), and dialog systems (the conversation ends).

Each episode is an independent trial. The agent's objective is to maximize the expected return per episode. Because episodes have finite length, the return is a finite sum even with gamma = 1. This makes undiscounted returns valid for episodic tasks.

Training in episodic tasks proceeds by running many episodes, updating the policy or value function after each episode (or each step within an episode), and measuring performance as the average return over recent episodes.

### Continuing Tasks

A continuing task has no terminal state. The agent runs indefinitely, and there is no natural point at which to reset. Examples include process control in manufacturing, ongoing portfolio management, and autonomous vehicle operation (the car does not stop learning after one trip).

In continuing tasks, discounting with gamma < 1 is essential. Without it, the total reward over infinite time is infinite for any non-trivial policy, making it impossible to compare policies. Discounting ensures the return is a well-defined finite quantity.

Some formulations of continuing tasks use the average reward criterion instead of discounted return. Under this criterion, the agent maximizes the long-run average reward per step rather than a discounted sum. This avoids the need for a discount factor but introduces different algorithmic challenges.

### The Episode Boundary in Practice

Even problems that are naturally continuing are often converted to episodic form for practical training purposes. A trading agent might train on fixed-length trading windows. A robot might train on fixed-duration trials. This conversion simplifies training and allows the use of standard episodic algorithms, though it introduces an artificial time horizon that may affect the learned policy.

## Value Functions: Measuring Expected Performance

Value functions are the central objects of RL. They estimate how good it is to be in a particular state or to take a particular action, where "how good" is measured by the expected return.

### State-Value Function V(s)

The state-value function under policy pi gives the expected return starting from state s and following policy pi:

```
V_pi(s) = E_pi[G_t | S_t = s]
```

V_pi(s) answers the question: "If I am in state s and follow policy pi from here on, how much total discounted reward can I expect?" A high V(s) means the state is desirable under the current policy. A low V(s) means the agent expects poor outcomes from that state.

### Action-Value Function Q(s, a)

The action-value function under policy pi gives the expected return from taking action a in state s and then following policy pi:

```
Q_pi(s, a) = E_pi[G_t | S_t = s, A_t = a]
```

Q_pi(s, a) answers the question: "If I take action a in state s and then follow policy pi, how much total discounted reward can I expect?" The action-value function is more directly useful for decision-making because it allows the agent to compare actions without knowing the environment's transition dynamics. To act optimally, the agent simply picks argmax_a Q(s, a).

### The Relationship Between V and Q

The two value functions are connected through the policy:

```
V_pi(s) = sum_a pi(a|s) * Q_pi(s, a)
```

The state-value is the expected action-value under the policy's action distribution. If the policy is deterministic with pi(s) = a*, then V_pi(s) = Q_pi(s, a*). Going the other direction, Q values provide strictly more information than V values: knowing Q for all actions lets you recover V, but knowing V does not let you recover Q without also knowing the transition model.

### Optimal Value Functions

The optimal value functions represent the best possible performance:

```
V*(s) = max_pi V_pi(s)     for all s
Q*(s, a) = max_pi Q_pi(s, a)   for all s, a
```

A fundamental result in MDP theory guarantees that for any finite MDP, there exists at least one deterministic policy that achieves V* and Q* simultaneously. Once Q* is known, the optimal policy is simply pi*(s) = argmax_a Q*(s, a). This is why many RL algorithms focus on learning Q* directly.

### Why Value Functions Matter for Algorithms

Value functions are not just theoretical constructs. They are the primary objects that RL algorithms learn and manipulate:

- **Value iteration** iteratively applies the Bellman Optimality Equation to converge on V* or Q*.
- **Q-learning** and **DQN** learn Q* from sampled experience, using the Bellman equation as the update target.
- **Policy evaluation** computes V_pi for a given policy, which is then used for policy improvement.
- **Actor-critic methods** learn both a policy (actor) and a value function (critic), using the critic to reduce variance in policy gradient estimates.
- **Advantage functions** A(s, a) = Q(s, a) - V(s) measure how much better an action is compared to the average, and are central to algorithms like A2C, A3C, and PPO.

## Policies: Defining Agent Behavior

A policy is the agent's strategy. It specifies what action the agent takes in each state. Policies are the object that RL algorithms ultimately optimize.

### Deterministic Policies

A deterministic policy maps each state to a single action:

```
pi(s) = a
```

For a given state, the agent always takes the same action. Deterministic policies are sufficient for optimal behavior in fully observable MDPs. The fundamental theorem guarantees that for any finite MDP, there exists a deterministic optimal policy.

### Stochastic Policies

A stochastic policy specifies a probability distribution over actions for each state:

```
pi(a | s) = P(A_t = a | S_t = s)
```

Stochastic policies serve several purposes in RL:

- They provide a natural mechanism for exploration during learning. A policy that assigns nonzero probability to all actions will eventually try every action in every state.
- Policy gradient methods naturally parameterize stochastic policies. The output of a neural network policy is typically a probability distribution (categorical for discrete actions, Gaussian for continuous actions).
- In partially observable settings, the optimal policy can be inherently stochastic because different underlying states may map to the same observation.
- Entropy regularization, used in algorithms like SAC, explicitly encourages stochastic policies to maintain exploration capacity.

### On-Policy vs Off-Policy Learning

A distinction closely tied to policies is whether the agent learns about the same policy it uses to collect data:

- **On-policy** methods (SARSA, REINFORCE, PPO) learn about the policy currently being used to make decisions. The data-collecting policy and the policy being improved are the same. This means old experience becomes stale whenever the policy changes.
- **Off-policy** methods (Q-learning, DQN, SAC) learn about one policy (typically the optimal policy) while following a different policy (typically one that explores more). This allows reuse of old experience through replay buffers, improving sample efficiency.

The on-policy/off-policy distinction affects sample efficiency, stability, and the choice of exploration strategy. Off-policy methods can use more aggressive exploration without degrading the policy being learned, because the exploration and target policies are separate.

## How the Three Pillars Connect

The three child chapters of this section are not independent topics. They form an integrated framework for understanding reinforcement learning.

**MDPs set the stage.** The five-tuple (S, A, P, R, gamma) defines what the agent observes, what it can do, how the world responds, and what constitutes success. Without this formalization, there is no precise problem to solve.

**Bellman equations provide the computational machinery.** Given an MDP, the Bellman equations express the value of every state as a function of the values of neighboring states. This recursive structure is what makes it possible to compute optimal policies without exhaustively searching over all possible action sequences. The Bellman equations are the bridge between "here is the problem" and "here is how to solve it."

**Exploration vs exploitation is the learning challenge.** The Bellman equations assume knowledge of the transition function P and reward function R. In practice, the agent does not have this knowledge and must learn through interaction. Exploration strategies determine how the agent gathers the information needed to approximate Bellman-equation-based solutions. Poor exploration means the agent never collects the data it needs to find good policies, regardless of how sophisticated its planning algorithm is.

The interplay between these three components is visible in every RL algorithm:

| Algorithm | MDP Component Used | Bellman Equation | Exploration Strategy |
|-----------|-------------------|------------------|---------------------|
| Value Iteration | Full model (P, R) | Optimality equation directly | Not needed (full model known) |
| Policy Iteration | Full model (P, R) | Expectation equation + greedy improvement | Not needed (full model known) |
| Q-learning | Samples from environment | Approximates optimality equation | Epsilon-greedy |
| DQN | Samples from environment | Minimizes Bellman error with neural net | Epsilon-greedy with decay |
| SARSA | Samples from environment | Approximates expectation equation | Epsilon-greedy (on-policy) |
| PPO | Samples from environment | Advantage estimation via critic | Stochastic policy + entropy bonus |
| SAC | Samples from environment | Soft Bellman equation | Maximum entropy objective |
| AlphaGo/AlphaZero | Learned model + search | MCTS backed by value network | UCB in tree search |

## Model-Free vs Model-Based RL

A high-level distinction that cuts across all three pillars is whether the agent has or learns a model of the environment.

### Model-Free Methods

Model-free methods learn a policy or value function directly from experience without explicitly learning the transition dynamics P or reward function R. The agent interacts with the environment, collects (state, action, reward, next state) tuples, and uses them to update its estimates.

- **Value-based model-free**: Q-learning, DQN, and their variants learn Q* directly. They use the Bellman Optimality Equation as an update target, replacing the expectation over P with single sampled transitions.
- **Policy-based model-free**: REINFORCE, PPO, and their variants directly optimize the policy by estimating the gradient of expected return with respect to policy parameters.
- **Actor-critic model-free**: A2C, A3C, SAC combine both. The critic estimates value functions, the actor updates the policy.

Model-free methods are simpler to implement and do not require a differentiable model of the environment. Their primary disadvantage is sample inefficiency: the agent must interact extensively with the environment because it cannot "imagine" outcomes.

### Model-Based Methods

Model-based methods learn or are given a model of the environment's dynamics (the transition function P and reward function R). The agent can then use this model to plan, simulating trajectories without interacting with the real environment.

- **Planning with a known model**: Dynamic programming (value iteration, policy iteration) directly solves the Bellman equations using the known model. Monte Carlo Tree Search uses the model to simulate rollouts from promising states.
- **Learning a model**: The agent learns an approximate model from experience, then plans using that learned model. Approaches like Dyna, MuZero, and Dreamer fall into this category.
- **Hybrid approaches**: Many modern systems combine model learning with model-free updates, using the model to generate synthetic experience that supplements real experience.

Model-based methods are generally more sample-efficient because the agent can extract multiple updates from each real interaction by replaying it through the model. The tradeoff is that model errors can compound, especially over long planning horizons, leading to policies that are optimal for the learned model but suboptimal in the real environment.

## A Conceptual Roadmap

For an ML engineer building understanding of reinforcement learning, the following sequence provides a logical path through the material.

**Start with MDPs.** Understand the formal structure: states, actions, transitions, rewards, and the discount factor. Internalize the Markov property and why it matters. Work through simple examples like gridworlds to build intuition for how policies, value functions, and rewards interact. See [Markov Decision Processes](markov-decision-processes/).

**Move to Bellman equations.** Understand how the recursive structure of returns leads to the Bellman Expectation and Optimality Equations. See how dynamic programming methods (value iteration, policy iteration) directly apply these equations when the model is known. This is the bridge from problem formulation to computation. See [Bellman Equations](bellman-equations/).

**Study exploration vs exploitation.** Understand the multi-armed bandit as the simplest case, then see how exploration challenges escalate in full RL problems with large state spaces and sparse rewards. Learn the major strategies (epsilon-greedy, UCB, Thompson Sampling, intrinsic motivation) and when each is appropriate. See [Exploration vs Exploitation](exploration-vs-exploitation/).

**From there, branch into algorithms.** With the conceptual foundations in place, the major algorithm families (value-based methods, policy-based methods, actor-critic, model-based RL) become variations on the same theme: different ways to approximate the Bellman equations using different representations, different update rules, and different exploration strategies. Each sibling chapter in this study guide builds on the foundation laid here.

## Key Takeaways

1. **RL is sequential decision-making under uncertainty.** The agent must learn from interaction, balancing the use of current knowledge with the acquisition of new information.

2. **MDPs provide the mathematical formulation.** The five-tuple (S, A, P, R, gamma) defines the problem. The Markov property makes it tractable.

3. **Bellman equations provide the solution strategy.** The recursive decomposition of value into immediate reward plus discounted future value is the foundation of virtually every RL algorithm.

4. **Exploration vs exploitation is the core practical challenge.** Without adequate exploration, the agent cannot gather the data needed to find good policies. Without adequate exploitation, the agent wastes resources on known-bad actions.

5. **Returns and discount factors define "good."** The return is the discounted cumulative reward. The discount factor controls the planning horizon and is necessary for well-defined objectives in continuing tasks.

6. **Value functions measure expected performance.** V(s) and Q(s, a) answer different but related questions about expected future reward. Q functions are more directly useful for action selection because they do not require knowledge of the transition model.

7. **Policies are what the agent ultimately learns.** Whether deterministic or stochastic, on-policy or off-policy, the policy is the agent's complete strategy for interacting with the environment.

8. **These concepts are not isolated.** MDPs define the problem, Bellman equations provide the solution structure, and exploration strategies provide the learning mechanism. Every RL algorithm combines all three.
