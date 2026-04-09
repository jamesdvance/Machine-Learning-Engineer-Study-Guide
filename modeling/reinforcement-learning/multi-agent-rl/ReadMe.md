# Multi-Agent Reinforcement Learning

## Summary

Multi-Agent Reinforcement Learning (MARL) extends reinforcement learning to settings where multiple agents interact within a shared environment, each learning a policy to maximize its own (or a team's) cumulative reward. Unlike single-agent RL, where the environment is stationary from the learner's perspective, MARL introduces fundamental complications: other agents are simultaneously learning and changing their behavior, making the environment non-stationary for every participant. This transforms the problem from a Markov Decision Process into a stochastic game (also called a Markov game), where the transition dynamics and reward signals depend on the joint actions of all agents. MARL has driven some of the most visible achievements in AI, from OpenAI Five defeating professional Dota 2 players to DeepMind's AlphaStar reaching Grandmaster level in StarCraft II, and it is increasingly relevant in real-world domains like autonomous driving, robotics coordination, and market design.

The field spans a broad spectrum of interaction types. In cooperative settings, agents share a common reward and must learn to coordinate. In competitive (zero-sum) settings, one agent's gain is another's loss, and the goal is to learn strategies that are robust against adversaries. Mixed cooperative-competitive settings combine both, as in team-based games where agents cooperate with teammates while competing against opponents. Each setting introduces distinct algorithmic challenges and solution concepts.

Key points to remember:

- MARL operates on stochastic games (Markov games) rather than MDPs, where the state transition and reward functions depend on the joint action of all agents
- The centralized training with decentralized execution (CTDE) paradigm is the dominant architectural pattern, allowing agents to share information during training while acting independently at deployment
- Non-stationarity is the core challenge: from any single agent's perspective, the environment appears to change as other agents update their policies
- Credit assignment in cooperative settings is difficult because the team reward does not reveal which agent's action contributed to success or failure
- QMIX, MAPPO, and MADDPG are the most widely used algorithms, each addressing different aspects of the multi-agent problem
- Self-play is a powerful training paradigm for competitive settings, where agents improve by playing against copies of themselves
- Communication between agents can be learned end-to-end, allowing agents to develop emergent protocols for coordination
- Scalability is a persistent challenge: the joint action space grows exponentially with the number of agents

When to use MARL:

- Multiple autonomous entities must coordinate or compete in a shared environment
- The optimal policy for one agent depends on the behavior of other agents
- You need decentralized execution but can afford centralized training (robotics swarms, multi-vehicle coordination)
- Game AI where opponents or teammates must exhibit diverse, adaptive behavior
- Market design, auction mechanisms, or economic simulations where strategic interaction is the core problem

When not to use MARL:

- A single-agent RL formulation adequately captures the problem (even if there are multiple entities, if they can be controlled by one policy)
- The agents do not meaningfully interact or their interactions are negligible
- The number of agents is extremely large and a mean-field approximation or population-level model is more tractable
- You need strong theoretical guarantees on convergence, which are harder to obtain in multi-agent settings

---

## From Single-Agent to Multi-Agent RL

### The Markov Game Framework

Single-agent RL is formulated as a Markov Decision Process (MDP) defined by (S, A, P, R, gamma), where S is the state space, A is the action space, P is the transition function, R is the reward function, and gamma is the discount factor. MARL generalizes this to a stochastic game (or Markov game) for N agents:

```
Stochastic Game = (S, A_1, ..., A_N, P, R_1, ..., R_N, gamma)
```

- S is the global state space
- A_i is the action space of agent i
- P(s' | s, a_1, ..., a_N) is the transition function depending on the joint action
- R_i(s, a_1, ..., a_N) is the reward function for agent i
- gamma is the discount factor

Each agent i observes some observation o_i (which may be a partial view of the full state s) and selects an action a_i according to its policy pi_i(a_i | o_i). The joint action (a_1, ..., a_N) determines the next state and each agent's reward. This joint dependence is what makes multi-agent problems fundamentally harder than single-agent ones.

### Why Single-Agent Methods Fail

A natural first attempt is to apply standard single-agent RL to each agent independently, treating all other agents as part of the environment. This approach, called independent learners (IL), has a critical flaw: because other agents are simultaneously updating their policies, the environment dynamics appear non-stationary to each agent. The convergence guarantees of standard RL algorithms rely on a stationary MDP, and this assumption is violated. In practice, independent learners can oscillate, diverge, or converge to poor joint policies.

Despite its theoretical shortcomings, independent learning sometimes works surprisingly well in practice, especially for cooperative tasks with few agents or when combined with parameter sharing. Independent PPO (IPPO), where each agent runs its own PPO instance, is a surprisingly strong baseline that outperforms more sophisticated methods on several benchmarks.

### Observation and Information Structure

In many real-world MARL problems, agents do not observe the full global state. Each agent receives a local observation o_i that depends on its position, sensor range, or communication capabilities. This partial observability transforms the problem into a Decentralized Partially Observable MDP (Dec-POMDP) in the cooperative case, or a Partially Observable Stochastic Game (POSG) in the general case. Dec-POMDPs are NEXP-complete, meaning they are fundamentally harder than single-agent POMDPs (which are themselves PSPACE-complete).

---

## Cooperative, Competitive, and Mixed Settings

### Cooperative MARL

In fully cooperative settings, all agents share a single team reward R(s, a_1, ..., a_N). The goal is to find a joint policy that maximizes the expected cumulative team reward. This is the Dec-POMDP setting when agents have partial observability.

Cooperative MARL is central to applications like:

- Multi-robot coordination (warehouse logistics, search and rescue)
- Network packet routing where routers collectively optimize throughput
- Traffic signal control where intersections cooperate to minimize congestion
- Cooperative game AI where teammates coordinate strategies

The main challenge in cooperative MARL is credit assignment: when the team receives a reward, it is unclear which agent's actions were responsible. If a team of five robots successfully completes a task, did all five contribute equally, or did one robot's action matter far more than the others? Without solving this, agents cannot learn effective individual policies from team-level feedback alone.

### Competitive MARL

In fully competitive (zero-sum) settings, agents have directly opposing objectives. For two players, R_1 = -R_2. The solution concept shifts from optimal joint policies to Nash equilibria, where no agent can improve its expected return by unilaterally changing its policy. Classic examples include board games (Go, chess), fighting games, and adversarial security scenarios.

Competitive MARL has a rich connection to game theory. The minimax theorem guarantees the existence of Nash equilibria in two-player zero-sum games, and value-based methods can converge to these equilibria under certain conditions. Self-play, where an agent trains against copies of itself, is the dominant paradigm for competitive settings and has produced superhuman performance in numerous games.

### Mixed Cooperative-Competitive Settings

Most real-world multi-agent problems involve a mix of cooperation and competition. Agents may belong to teams that cooperate internally while competing against other teams. Examples include:

- Team-based video games (Dota 2, StarCraft II) where players coordinate with teammates and compete against opponents
- Economic markets where firms cooperate within supply chains but compete for market share
- Multi-party negotiation where partial alignment of interests creates complex strategic dynamics

These mixed settings are the most challenging because agents must simultaneously learn to cooperate with allies and compete against adversaries, and the boundaries between cooperation and competition may be fluid.

---

## Centralized Training with Decentralized Execution (CTDE)

### The Core Idea

CTDE is the dominant paradigm in modern MARL. The key insight is that during training, we often have access to information that will not be available at deployment time. During training, a simulator provides the global state, all agents' observations, and all agents' actions. At execution time, each agent must act based only on its local observation.

CTDE exploits this asymmetry by allowing the learning algorithm to use global information during training to learn better decentralized policies. This takes two main forms:

1. Centralized critic: A shared value function or critic that conditions on the global state or joint observations, used during training to provide better gradient estimates. Each agent's policy (actor) still conditions only on its local observation.

2. Value decomposition: A centralized value function is decomposed into per-agent utility functions that can be evaluated locally. The decomposition is learned during training so that decentralized greedy action selection recovers the optimal joint action.

### Why CTDE Works

CTDE addresses non-stationarity by incorporating other agents' information into the training signal. When agent i's critic can observe agent j's actions, the environment appears stationary to agent i's critic even as agent j's policy changes. This stabilizes training while preserving the ability to deploy fully decentralized policies.

The CTDE framework also enables efficient use of shared experience. Since all agents' data is collected in the same episodes, a centralized training process can batch and reuse this data more effectively than fully independent training.

---

## Key Algorithms

### QMIX: Monotonic Value Decomposition

QMIX (Rashid et al., 2018) is a value-based cooperative MARL algorithm built on the idea of value decomposition. It addresses the credit assignment problem by factoring the joint action-value function Q_tot into individual per-agent utility functions Q_i.

The key constraint in QMIX is monotonicity: Q_tot must be a monotonically increasing function of each Q_i. Formally:

```
dQ_tot / dQ_i >= 0  for all i
```

This ensures that if each agent greedily maximizes its own Q_i, the joint action also maximizes Q_tot. Decentralized greedy action selection is therefore consistent with centralized optimization.

Architecture:

```
Agent i observation o_i --> Agent network --> Q_i(o_i, a_i)
All Q_i values --> Mixing network (with hypernetworks conditioned on global state s) --> Q_tot
```

The mixing network is a feed-forward network whose weights are generated by hypernetworks that take the global state as input. The hypernetwork outputs are passed through an absolute value function to ensure non-negative weights, which enforces the monotonicity constraint.

Training uses standard DQN-style updates on Q_tot, but execution is fully decentralized: each agent selects arg max_a_i Q_i(o_i, a_i) using only its local observation.

Strengths and limitations:

- QMIX handles credit assignment elegantly through the factored value function
- The monotonicity constraint is both a strength (enables decentralized execution) and a limitation (it cannot represent all possible joint value functions)
- QMIX struggles with tasks requiring complex coordination where the optimal joint action does not correspond to individual greedy actions under any monotonic decomposition
- Extensions like QPLEX and Weighted QMIX relax the monotonicity constraint to handle a broader class of problems

### MAPPO: Multi-Agent Proximal Policy Optimization

MAPPO (Yu et al., 2022) applies PPO to multi-agent settings using the CTDE paradigm. Despite its simplicity, MAPPO has proven to be a remarkably strong baseline that often matches or exceeds more complex methods on standard benchmarks including StarCraft Multi-Agent Challenge (SMAC), Hanabi, and Multi-Agent Particle Environments.

MAPPO uses:

- A shared policy network (with parameter sharing across agents) that maps each agent's local observation to an action distribution
- A centralized value function V(s) that conditions on the global state, used as the critic during training
- Standard PPO clipped surrogate objective and GAE for advantage estimation

```
Each agent i:
  Actor:  pi_theta(a_i | o_i)      -- local observation only
  Critic: V_phi(s)                  -- global state (training only)
```

The parameter sharing variant trains a single policy network shared by all agents. Agents are distinguished by their observations (which include agent-specific information like ID or position). This dramatically reduces the number of parameters and speeds up training, but assumes a degree of homogeneity among agents.

Key implementation details:

- Input normalization and value function clipping are critical for stability
- Agent-specific features (agent ID, position) concatenated to observations enable a shared policy to produce diverse behaviors
- Large batch sizes and many PPO epochs per batch improve performance significantly
- MAPPO benefits from the same implementation details that matter for single-agent PPO (see the "37 implementation details" literature)

MAPPO's success challenges the assumption that multi-agent problems require fundamentally different algorithms. A well-tuned application of a strong single-agent algorithm with CTDE can go a long way.

### MADDPG: Multi-Agent Deep Deterministic Policy Gradient

MADDPG (Lowe et al., 2017) extends DDPG to multi-agent settings and was one of the first successful CTDE algorithms. It is designed for continuous action spaces and mixed cooperative-competitive environments.

Each agent i has:

- An actor pi_i(a_i | o_i) that maps local observations to deterministic actions
- A centralized critic Q_i(s, a_1, ..., a_N) that conditions on the global state and all agents' actions

```
Agent i:
  Actor:  mu_i(o_i)                          -- decentralized
  Critic: Q_i(o_i, o_-i, a_i, a_-i)         -- centralized, sees all observations and actions
```

During training, each agent's critic observes the actions of all other agents, which makes the environment appear stationary. During execution, only the actor is needed, which uses only local observations. The critic is discarded at deployment.

MADDPG trains using off-policy data stored in a shared replay buffer. The actor is updated using the deterministic policy gradient through the critic, and the critic is updated using standard Bellman backup with target networks.

Key features:

- Handles competitive, cooperative, and mixed settings naturally since each agent has its own reward function and critic
- Policy ensemble: during training, each agent maintains an ensemble of policies for other agents to improve robustness
- Supports continuous action spaces natively through the deterministic policy gradient
- Off-policy learning enables better sample efficiency than on-policy methods like MAPPO

Limitations:

- Scales poorly to many agents because each critic must process all agents' observations and actions
- The centralized critic can be difficult to train when the number of agents is large
- Requires careful tuning of replay buffer size, learning rates, and exploration noise

### Other Notable Algorithms

**VDN (Value Decomposition Networks):** The simplest value decomposition method. Q_tot is the sum of individual Q_i values. This is a special case of QMIX with the mixing network fixed to simple addition. Surprisingly effective for many cooperative tasks.

**COMA (Counterfactual Multi-Agent Policy Gradients):** Uses a centralized critic to compute a counterfactual baseline for each agent: what would the expected return have been if agent i had taken a different action while all other agents acted as they did? This directly addresses credit assignment in cooperative settings.

**QTRAN:** Attempts to learn the full joint value function without the monotonicity restriction of QMIX. It uses additional constraints and auxiliary losses but can be unstable in practice.

**MAT (Multi-Agent Transformer):** Applies the transformer architecture to multi-agent coordination, treating the sequential decision-making of multiple agents as a sequence modeling problem. Agents' actions are generated autoregressively, capturing dependencies between agents.

---

## Communication Between Agents

### Learned Communication

In many multi-agent tasks, agents benefit from exchanging information beyond what they can observe locally. Rather than hand-designing communication protocols, MARL research has explored learning communication end-to-end through backpropagation.

**CommNet (Sukhbaatar et al., 2016):** Agents produce continuous message vectors that are averaged and broadcast to all agents. The communication channel is differentiable, so the entire system (policies and messages) is trained jointly. CommNet demonstrated that agents can learn meaningful communication protocols without any supervision on the message content.

**DIAL (Differentiable Inter-Agent Learning):** Similar to CommNet but uses discrete messages during execution (continuous during training with a straight-through estimator). This is more realistic for settings where communication bandwidth is limited.

**TarMAC (Targeted Multi-Agent Communication):** Introduces an attention mechanism for communication, allowing agents to send targeted messages to specific other agents rather than broadcasting to everyone. Agents learn both the content of messages and who to communicate with.

**IC3Net:** Adds a gating mechanism that allows agents to decide when to communicate. This is important because not every situation requires communication, and unnecessary messages add noise and computational overhead.

### Communication Challenges

- Bandwidth constraints: real-world communication channels have limited capacity, so messages must be compact
- Latency: in real-time systems, communication delays can make shared information stale
- Scalability: broadcasting messages to all agents creates O(N^2) communication overhead; attention-based or sparse communication is needed for large agent populations
- Emergent vs. interpretable protocols: learned communication protocols are typically not human-readable, making debugging difficult

### When Communication Helps

Communication is most valuable when:

- Agents have complementary partial observations that, when shared, provide a more complete picture
- Coordination failures are costly (collision avoidance, joint manipulation)
- The task has a sequential structure where early information from one agent can guide later decisions by others
- The observation function is highly local relative to the scale of the environment

---

## Self-Play as a Training Paradigm

### Concept

Self-play trains an agent by having it play against copies of itself (or past versions of itself). The agent acts as both learner and opponent, creating an ever-improving curriculum. As the agent gets stronger, so does its opponent, driving continuous improvement.

Self-play has a long history in game AI, from Arthur Samuel's checkers program in the 1950s to TD-Gammon in the 1990s, and has powered the most dramatic recent successes in competitive AI.

### Mechanisms

**Naive self-play:** The agent always plays against the current version of itself. This is simple but can lead to forgetting: the agent specializes against its current strategy and loses the ability to counter earlier strategies. Policies can cycle rather than converge.

**Fictitious self-play (FSP):** The agent plays against the average of all past policies, approximating the fictitious play algorithm from game theory. This converges to Nash equilibria in certain game classes but is expensive to implement exactly.

**Population-based training (PBT) with self-play:** Maintains a population of agents that train against each other. This provides diverse opponents and reduces overfitting to any single strategy. Used in AlphaStar to produce a league of agents with diverse playstyles.

**Prioritized fictitious self-play (PFSP):** Used by AlphaStar, this selects opponents from a league based on which past agents the current agent struggles most against. This focuses training on weaknesses rather than wasting time on opponents that are already easy to beat.

### Self-Play in Practice

AlphaGo and AlphaZero used self-play combined with Monte Carlo Tree Search (MCTS) to achieve superhuman performance in Go, chess, and shogi. The training loop is:

```
1. Play games using MCTS guided by current policy and value networks
2. Store game outcomes as training data
3. Train policy and value networks on the collected data
4. Repeat, with the improved networks guiding better MCTS
```

OpenAI Five used self-play with PPO to train a team of five agents to play Dota 2. The training consumed enormous computational resources (equivalent to 180 years of gameplay per day across thousands of GPUs) but produced agents that defeated professional human teams.

Key lessons from self-play:

- Self-play provides an automatic curriculum that adapts to the agent's skill level
- Diversity of opponents is critical to avoid strategy cycling and forgetting
- Self-play requires symmetric (or nearly symmetric) games to be directly applicable
- The computational cost is high because both the agent and opponent must be evaluated at each step

---

## Challenges in Multi-Agent RL

### Non-Stationarity

From any single agent's perspective, the environment dynamics change as other agents update their policies. Consider agent 1 learning in a two-agent game: the transition function effectively becomes P'(s' | s, a_1) = sum_a_2 P(s' | s, a_1, a_2) * pi_2(a_2 | s), which changes whenever agent 2 updates pi_2. This violates the stationary MDP assumption underlying standard RL convergence proofs.

Mitigation strategies:

- CTDE with centralized critics that observe other agents' actions
- Experience replay with importance correction for non-stationarity
- Opponent modeling: explicitly modeling and predicting other agents' policies
- Slower, coordinated policy updates to reduce the rate of environmental change

### Credit Assignment

In cooperative MARL with shared team rewards, determining each agent's contribution is fundamentally difficult. The credit assignment problem has two aspects:

- Temporal credit assignment (as in single-agent RL): which past actions led to the current reward?
- Multi-agent credit assignment: which agent's actions contributed to the team reward?

Solutions include:

- Value decomposition (QMIX, VDN, QTRAN) that learns per-agent value functions
- Counterfactual baselines (COMA) that estimate what would have happened with different individual actions
- Reward shaping with domain-specific individual rewards (but this requires manual engineering and can introduce unintended incentives)
- Shapley value-based methods that draw on cooperative game theory to attribute rewards fairly

### Scalability

The joint action space grows exponentially with the number of agents: |A_joint| = |A_1| x |A_2| x ... x |A_N|. For 10 agents each with 5 actions, the joint action space has approximately 10 million elements. This exponential blowup makes centralized approaches intractable for large numbers of agents.

Approaches to scalability:

- Mean-field approximation: agents interact with the "average" of neighboring agents rather than with each individual, reducing the effective number of interactions. Mean Field Game theory provides a continuous-agent-population limit.
- Parameter sharing: all agents (or agents of the same type) share a single policy network, reducing the number of parameters from O(N) networks to O(1)
- Local communication and interaction graphs: agents only attend to nearby agents, reducing the effective problem size
- Hierarchical MARL: agents are organized into groups with a hierarchy of coordinators and workers
- Factored value functions that exploit independence structure in agent interactions

### Partial Observability

Real-world agents rarely observe the full global state. Each agent sees only a local observation, and optimal decentralized policies may require memory (conditioning on observation histories). This transforms the problem from planning in a fully observable game to planning in a Dec-POMDP, which is computationally much harder.

Recurrent architectures (LSTMs, GRUs) are commonly used to maintain belief states over observation histories. Attention mechanisms over observation histories and communication messages are becoming increasingly popular as transformer-based architectures enter the MARL space.

### Equilibrium Selection

Even when MARL algorithms converge, there may be multiple equilibria, and different runs may converge to different ones. In cooperative settings, agents can get stuck in suboptimal coordination equilibria. In competitive settings, the notion of "optimal" policy is inherently relative to the opponent's strategy. Nash equilibria may be non-unique, and some may be preferable to others. Choosing among equilibria remains an open problem.

---

## Applications

### Game AI

MARL has produced some of the most visible AI achievements:

- **AlphaStar (DeepMind, 2019):** Reached Grandmaster level in StarCraft II, a real-time strategy game with imperfect information, large action spaces, and long time horizons. Used population-based self-play with a league training process.
- **OpenAI Five (2019):** Defeated professional Dota 2 players using self-play with PPO for a team of five cooperating agents. Demonstrated that long-horizon cooperative strategy could emerge from RL.
- **Pluribus (Facebook AI, 2019):** Achieved superhuman performance in six-player no-limit Texas Hold'em poker using a combination of self-play and search.
- **Cicero (Meta, 2022):** Combined language models with strategic reasoning in the board game Diplomacy, a seven-player game requiring negotiation and alliance formation alongside strategic play.

### Robotics

Multi-robot coordination is a natural application of cooperative MARL:

- Warehouse logistics: fleets of autonomous robots coordinating pick-and-place operations while avoiding collisions
- Formation control: UAV swarms maintaining formations for surveillance, delivery, or agricultural monitoring
- Multi-robot manipulation: multiple robotic arms cooperating to manipulate objects too large or complex for a single arm
- Search and rescue: robots dividing a search area and communicating findings

Sim-to-real transfer is a major challenge in robotic MARL. Policies trained in simulation must handle the noise, latency, and model mismatch present in real hardware. Communication protocols learned in simulation may not transfer to real wireless channels with packet loss and delays.

### Traffic and Transportation

- Traffic signal control: treating each intersection as an agent that observes local traffic conditions and coordinates with neighbors. Cooperative MARL approaches have shown significant improvements over fixed-timing plans and even single-agent RL approaches.
- Autonomous vehicle coordination: multiple self-driving vehicles negotiating at intersections, merging on highways, or coordinating in parking lots without centralized control.
- Fleet management: ride-sharing platforms optimizing vehicle dispatch and routing with multiple vehicles operating simultaneously.

### Economics and Markets

- Market making and trading: multiple trading agents learning strategies in simulated or real financial markets
- Auction design: using MARL to study bidder behavior and design auction mechanisms that achieve desired properties (revenue, efficiency, fairness)
- Resource allocation: distributed allocation of shared resources (spectrum, compute, energy) among self-interested agents
- Mechanism design: using MARL simulations to test incentive structures before deployment

---

## Frameworks and Tools

### PettingZoo

PettingZoo is the standard multi-agent environment library, providing a unified API analogous to what Gymnasium (formerly OpenAI Gym) provides for single-agent RL. It supports both simultaneous and turn-based agent interactions.

```python
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
```

PettingZoo provides two API models:

- **AEC (Agent Environment Cycle):** Agents act sequentially in a defined order. Suitable for turn-based games and environments where agent ordering matters.
- **Parallel API:** All agents act simultaneously. Suitable for environments where agents select actions at the same time.

PettingZoo includes built-in environments:

- Multi-Agent Particle Environments (MPE): simple 2D physics environments for cooperative and competitive tasks
- Atari: multi-player Atari games
- Classic: board games and card games (chess, Go, poker)
- SISL: cooperative tasks like multi-agent pursuit and waterworld

### RLlib Multi-Agent

RLlib (part of Ray) provides the most comprehensive multi-agent RL framework for production use. It supports flexible policy-to-agent mappings, multiple algorithms, and distributed training.

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("multi_agent_env")
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
)

algo = config.build()
result = algo.train()
```

Key RLlib multi-agent capabilities:

- **Flexible policy mapping:** Map any agent to any policy. Agents can share a single policy, use heterogeneous policies, or use any combination.
- **Centralized critic support:** Built-in support for centralized value functions that condition on other agents' observations.
- **Multiple policies with different algorithms:** Different agents can use different RL algorithms (e.g., one PPO, one DQN).
- **Self-play integration:** Built-in support for self-play through policy versioning and opponent sampling from past checkpoints.
- **Distributed training:** Leverages Ray for scaling across multiple CPUs and GPUs.

### Other Frameworks

**EPyMARL:** A PyTorch-based framework focused on cooperative MARL research. Implements QMIX, VDN, MAPPO, MADDPG, and other algorithms with the StarCraft Multi-Agent Challenge (SMAC) as the default benchmark. Useful for reproducing cooperative MARL results.

**MARLlib:** A comprehensive MARL library that unifies many environments and algorithms under a single API. Supports 10+ environments and 18+ algorithms, making it valuable for systematic comparison across methods.

**OpenSpiel:** A DeepMind framework for game-theoretic research. Strong support for classical games, extensive-form games, and normal-form games. Includes implementations of Nash equilibrium solvers and game-theoretic MARL algorithms.

**SMAC and SMACv2 (StarCraft Multi-Agent Challenge):** While not a general framework, SMAC is the most widely used benchmark for cooperative MARL. It provides micromanagement scenarios in StarCraft II where teams of units must coordinate to defeat enemy teams. SMACv2 addresses overfitting concerns in the original SMAC by randomizing unit types and start positions.

---

## Practical Guidance

### Starting a Multi-Agent Project

1. **Define the interaction structure:** Is the problem cooperative, competitive, or mixed? How many agents are there? Do agents have homogeneous or heterogeneous roles?

2. **Start with strong baselines:** Independent PPO (IPPO) or MAPPO with parameter sharing is a surprisingly strong starting point for cooperative tasks. Do not jump to complex algorithms before trying these.

3. **Choose the right framework:** For research or benchmarking cooperative algorithms, use EPyMARL with SMAC. For production systems or custom environments, use RLlib. For environment development, use PettingZoo.

4. **Centralized critic matters:** Adding a centralized value function that conditions on the global state almost always improves performance over fully independent training, at minimal additional complexity.

5. **Parameter sharing as default:** For homogeneous agents, share a single policy network across all agents. Differentiate agents by including agent ID or role information in observations. This reduces the parameter count, improves sample efficiency, and often improves performance.

### Debugging Multi-Agent Systems

- Train with 2-3 agents first, even if the target is 20+. Many bugs are easier to diagnose at small scale.
- Visualize agent behavior, not just reward curves. Reward can increase while agents learn degenerate strategies.
- Check for lazy agent problems: in cooperative settings, some agents may learn to do nothing while others carry the team. Per-agent reward decomposition metrics help diagnose this.
- Monitor the entropy of each agent's policy. If one agent's entropy collapses to zero early in training, it has become deterministic too quickly and is likely stuck in a poor strategy.
- Compare against a random policy baseline and a fully centralized controller (single agent controlling all entities). The former checks that learning is happening, the latter provides an upper bound on coordination quality.

### Scaling Considerations

- Communication overhead grows quadratically with agent count in fully connected schemes. Use attention-based or local communication for 10+ agents.
- Replay buffer memory grows linearly with agent count in off-policy methods. Use shared replay buffers with appropriate multi-agent transition storage.
- Training wall-clock time often scales linearly with agent count even with parallelization, because environment steps require all agents to act.
- Consider hierarchical decomposition for 50+ agents: group agents into teams with intra-team coordination handled by one MARL instance and inter-team coordination by another.

---

## Summary of Algorithm Selection

```
Cooperative + Discrete Actions    --> QMIX, VDN, or MAPPO
Cooperative + Continuous Actions  --> MAPPO or MADDPG
Competitive + Zero-Sum            --> Self-play with PPO or SAC
Mixed Cooperative-Competitive     --> MADDPG or MAPPO with separate team critics
Large Agent Count (50+)           --> MAPPO with parameter sharing, mean-field methods
Simple Baseline                   --> Independent PPO (IPPO)
```

When in doubt, start with MAPPO. It is simple, well-understood, and competitive with more complex methods across a broad range of tasks.
