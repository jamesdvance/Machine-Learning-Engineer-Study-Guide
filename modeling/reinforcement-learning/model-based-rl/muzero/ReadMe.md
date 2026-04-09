# MuZero

## Summary

MuZero is a model-based reinforcement learning algorithm developed by DeepMind that learns to plan without being given the rules of the environment. Published in 2020 in Nature, MuZero represents the culmination of the AlphaGo lineage: where AlphaGo required human expert data and hand-coded game rules, and AlphaZero eliminated the need for human data but still required a perfect simulator of the environment, MuZero removes the final dependency by learning its own internal model of the environment dynamics. This learned model does not attempt to reconstruct raw observations (pixels, board states); instead, it learns to predict only the quantities that matter for planning -- rewards, values, and policies -- in a compact latent space.

The core architecture consists of three neural networks that work together. A representation function encodes the current observation into a hidden state. A dynamics function takes a hidden state and an action and produces the next hidden state along with a predicted reward. A prediction function takes a hidden state and outputs a policy distribution and a value estimate. These three functions are trained end-to-end from self-play data, where the training targets come from the actual outcomes of Monte Carlo Tree Search (MCTS) during games. The result is an agent that can match or exceed the performance of AlphaZero on board games (Go, chess, shogi) while also achieving state-of-the-art results on the Atari benchmark -- a domain with complex visual observations and unknown dynamics where AlphaZero cannot operate at all.

Key points to remember:

- MuZero learns a model of the environment rather than being given one, making it applicable to domains where the rules are unknown
- The learned model operates in a latent space and predicts planning-relevant quantities (reward, value, policy), not raw observations
- Three learned functions: representation (observation to hidden state), dynamics (hidden state + action to next hidden state + reward), prediction (hidden state to policy + value)
- Planning is performed via Monte Carlo Tree Search (MCTS) using the learned model
- Training uses self-play with unrolled model predictions matched against search-improved targets
- Achieves superhuman performance on Go, chess, shogi, and state-of-the-art on 57 Atari games with a single algorithm
- Evolution: AlphaGo (2016) -> AlphaGo Zero (2017) -> AlphaZero (2018) -> MuZero (2020)

When to use MuZero-style approaches:

- Domains where environment dynamics are unknown or too complex to specify
- Problems where planning (look-ahead search) would provide a significant advantage over reactive policies
- Settings with discrete action spaces or moderately sized continuous action spaces (with Sampled MuZero)
- When you have sufficient compute for self-play data generation and MCTS at inference time

When not to use MuZero-style approaches:

- Simple problems where model-free methods (DQN, PPO) already work well, since MuZero adds substantial complexity
- Real-time applications where MCTS latency at inference time is unacceptable
- Very large or continuous action spaces without modification (standard MuZero enumerates all actions in MCTS)
- When compute budget is limited -- MuZero requires significant resources for self-play and training

---

## The AlphaGo Lineage

Understanding MuZero requires understanding the progression of ideas that led to it. Each generation removed a dependency on human knowledge while maintaining or improving performance.

### AlphaGo (2016)

AlphaGo was the first program to defeat a professional human Go player. It combined deep neural networks with Monte Carlo Tree Search, using two networks: a policy network trained on millions of human expert games via supervised learning, and a value network trained via self-play RL. The MCTS used these networks to guide search, with the policy network selecting promising moves to explore and the value network evaluating board positions. Critically, AlphaGo required both human expert data for initialization and a perfect simulator of Go (the game rules) for MCTS rollouts and self-play.

### AlphaGo Zero and AlphaZero (2017-2018)

AlphaGo Zero eliminated the dependence on human expert data entirely, learning from scratch through pure self-play. It also simplified the architecture to a single neural network that outputs both a policy and a value estimate, and replaced the Monte Carlo rollouts in MCTS with the value network's evaluation. AlphaZero then generalized this approach to chess and shogi in addition to Go, demonstrating that the same algorithm could achieve superhuman play across multiple board games.

However, AlphaZero still required a perfect model of the environment -- the complete rules of the game -- to perform MCTS. During search, the algorithm simulates moves by actually applying them using the known game rules to produce the next state. This limits AlphaZero to domains where the transition dynamics are known and cheap to compute.

### MuZero (2020)

MuZero's key innovation is replacing the perfect simulator with a learned model. Instead of requiring the game rules to simulate what happens after taking an action, MuZero learns a dynamics function that predicts the next hidden state and reward given a current hidden state and action. This learned model is used inside MCTS in place of the real environment, enabling planning in domains where the rules are unknown.

The progression can be summarized as:

| Algorithm | Human Data | Known Rules | Domains |
|---|---|---|---|
| AlphaGo | Yes | Yes | Go |
| AlphaGo Zero | No | Yes | Go |
| AlphaZero | No | Yes | Go, Chess, Shogi |
| MuZero | No | No | Go, Chess, Shogi, Atari |

---

## Architecture: Three Learned Functions

MuZero's architecture is defined by three neural networks that together form the learned model. These operate entirely in a latent space -- the hidden states do not need to correspond to any recognizable representation of the environment.

### Representation Function: h

```
s_0 = h(o_1, ..., o_t)
```

The representation function takes the sequence of past observations (in practice, a stack of recent frames or the current board state) and maps it to an initial hidden state s_0. This is the bridge between the real environment and the learned latent space. For board games, the input is the current board position. For Atari, it is a stack of recent frames, similar to the input preprocessing used in DQN.

The representation function is only called once at the root of the search tree, when beginning to plan from the current real observation. All subsequent nodes in the search tree are produced by the dynamics function operating in latent space.

### Dynamics Function: g

```
r_k, s_k = g(s_{k-1}, a_k)
```

The dynamics function is the learned model of the environment. Given a hidden state s_{k-1} and an action a_k, it produces the next hidden state s_k and a predicted immediate reward r_k. This function plays the role that the game rules play in AlphaZero -- it simulates what happens when an action is taken.

Critically, the dynamics function does not need to predict observations (pixels, board states). It only needs to produce hidden states from which the prediction function can accurately estimate values and policies. This is a much easier learning problem than full observation prediction, and it is the central insight that makes MuZero work. The latent space is free to represent whatever information is most useful for planning, without being constrained to reconstruct raw sensory data.

### Prediction Function: f

```
p_k, v_k = f(s_k)
```

The prediction function takes a hidden state and produces two outputs: a policy vector p_k (probability distribution over actions) and a scalar value estimate v_k (expected cumulative reward from this state). This is architecturally identical to the policy-value head used in AlphaZero, except that it operates on learned hidden states rather than on game-rule-produced states.

### How They Work Together

During planning (MCTS), the process is:

1. Encode the current real observation into a hidden state using the representation function h
2. At each node in the search tree, use the dynamics function g to simulate taking an action and produce the next hidden state and reward
3. At each node, use the prediction function f to get a policy prior and value estimate to guide the search
4. After search completes, select the action with the highest visit count at the root

During execution, the agent observes the real environment, runs MCTS using its learned model, takes the action selected by search, and observes the next real state. The learned model is only used internally for planning -- the agent always acts in and receives feedback from the real environment.

---

## Monte Carlo Tree Search in MuZero

MCTS is the planning algorithm that uses the learned model to look ahead and select actions. MuZero's MCTS is structurally similar to AlphaZero's, but with the learned model replacing the game simulator.

### Search Procedure

Each MCTS iteration consists of three phases:

**Selection**: Starting from the root node (the current real observation encoded as s_0), traverse the tree by selecting actions according to the PUCT (Predictor Upper Confidence bounds applied to Trees) formula:

```
a_t = argmax_a [ Q(s, a) + c(s) * P(s, a) * sqrt(N(s)) / (1 + N(s, a)) ]
```

where Q(s, a) is the mean value of action a from state s, P(s, a) is the prior probability from the prediction function, N(s) is the visit count of the parent, N(s, a) is the visit count of the action, and c(s) is an exploration constant. This formula balances exploitation (high Q-values) with exploration (high prior probability, low visit count).

**Expansion**: When a leaf node is reached, expand it by applying the dynamics function g to produce the next hidden state and reward, then applying the prediction function f to get the prior policy and value estimate for the new node.

**Backup**: Propagate the value estimate back up the tree, updating the Q-values and visit counts of all nodes along the path from the leaf to the root. The backed-up value accounts for the intermediate rewards predicted by the dynamics function:

```
G_k = r_k + gamma * r_{k+1} + gamma^2 * r_{k+2} + ... + gamma^{l-k} * v_l
```

where l is the depth of the leaf node and v_l is its value estimate.

### Action Selection

After running a fixed number of MCTS simulations (typically 50-800 depending on the domain), the action is selected based on the visit counts at the root node. During self-play training, actions are sampled proportionally to visit counts (with a temperature parameter) to ensure exploration. During evaluation, the action with the highest visit count is selected greedily.

### Differences from AlphaZero's MCTS

The only structural difference is that MuZero uses the dynamics function g to transition between nodes, while AlphaZero uses the actual game rules. In MuZero, the hidden states in the search tree are latent representations, not real environment states. This means that the search tree lives entirely in the learned latent space, and errors in the dynamics function can compound as the search goes deeper. In practice, MuZero mitigates this by not searching excessively deep and by relying on the value function to evaluate positions rather than rolling out to terminal states.

---

## Training

MuZero is trained through self-play, similar to AlphaZero, but with a more complex training procedure that accounts for the learned model.

### Self-Play Data Generation

The training loop consists of two concurrent processes:

1. **Actors**: Multiple self-play actors generate games by running MCTS at each step using the current model. Each game produces a trajectory of observations, actions, rewards, MCTS policies, and MCTS value estimates. These trajectories are stored in a replay buffer.

2. **Learner**: A centralized learner samples trajectories from the replay buffer and updates the model parameters.

### Unrolled Training

The key innovation in MuZero's training is how the model is trained end-to-end through unrolled planning steps. For each sampled position in a trajectory, the training procedure:

1. Encodes the observation at time t into a hidden state s_0 using the representation function
2. Unrolls the dynamics function for K steps (typically K=5), applying it repeatedly with the actual actions taken in the game to produce a sequence of hidden states s_1, s_2, ..., s_K and predicted rewards r_1, ..., r_K
3. At each unrolled step k, applies the prediction function to get predicted policy p_k and value v_k

The training targets come from the actual game trajectory:

- **Policy target**: The MCTS search policy (visit count distribution) at time t+k. This is a richer training signal than the raw action taken, because the search policy encodes information about the relative quality of all actions, not just which one was selected.
- **Value target**: The n-step bootstrapped return from time t+k, computed as the sum of actual rewards for n steps plus the discounted MCTS value estimate at step t+k+n.
- **Reward target**: The actual reward observed at time t+k.

### Loss Function

The total loss for a training sample unrolled for K steps is:

```
L = sum_{k=0}^{K} [ L_policy(pi_{t+k}, p_k) + L_value(z_{t+k}, v_k) + L_reward(u_{t+k}, r_k) ] + c * ||theta||^2
```

where:
- L_policy is the cross-entropy loss between the MCTS search policy pi_{t+k} and the predicted policy p_k
- L_value is the cross-entropy loss (using a categorical representation of values) or MSE between the actual n-step return z_{t+k} and the predicted value v_k
- L_reward is the cross-entropy loss or MSE between the actual reward u_{t+k} and the predicted reward r_k
- c * ||theta||^2 is an L2 regularization term

### Why This Training Procedure Works

The training targets are key to MuZero's success. The MCTS search policies are better than the raw neural network policy because they incorporate look-ahead planning. By training the policy head to match the search output, the neural network improves, which in turn improves the search, creating a virtuous cycle. This is the same policy improvement mechanism used in AlphaZero.

For the model (dynamics function), training to match rewards, values, and policies at each unrolled step ensures that the model learns to produce hidden states that are useful for predicting planning-relevant quantities. The model does not need to be accurate in any pixel-level or state-reconstruction sense; it only needs to maintain enough information in the hidden state for the prediction function to produce accurate policies and values.

Gradients flow through the entire unrolled computation graph, from the loss at step K all the way back through the dynamics function and representation function. To prevent gradient explosion, hidden states are typically scaled (divided by their norm or a constant) between dynamics function applications.

---

## Results

### Board Games

On Go, chess, and shogi, MuZero matched the performance of AlphaZero despite not having access to the game rules during planning. This is a remarkable result: the learned model was accurate enough that planning with it was as effective as planning with a perfect simulator. In some cases, MuZero slightly exceeded AlphaZero's Elo rating, possibly because the learned representation could capture strategic abstractions that a raw board state cannot.

### Atari

On the Atari 2600 benchmark (57 games), MuZero achieved a new state of the art at the time of publication, surpassing all prior model-free and model-based methods. The mean and median human-normalized scores exceeded those of R2D2 (a strong model-free baseline) and SimPLe (a prior model-based method). MuZero was particularly strong on games requiring long-term planning and games with complex dynamics, where the look-ahead search provided the most benefit.

MuZero used 20 billion frames of training experience across the 57 games, with each game trained independently. The MCTS used 50 simulations per move in Atari (compared to 800 in board games), reflecting the tighter real-time constraints and the fact that Atari games have simpler decision structures than Go.

### Key Takeaway from Results

The most significant result is not any individual benchmark score but the generality: a single algorithm, with a single set of hyperparameters (modulo minor domain-specific preprocessing), achieved superhuman or state-of-the-art performance across both perfect-information board games and visually complex, partially observable video games. No prior algorithm had demonstrated this breadth.

---

## Sampled MuZero

Standard MuZero's MCTS enumerates all legal actions at each node in the search tree. This works for board games (Go has at most 361 moves) and Atari (typically 18 actions), but becomes intractable for large discrete action spaces or continuous action spaces.

**Sampled MuZero** (Hubert et al., 2021) extends MuZero to handle these cases by sampling a subset of actions at each node rather than enumerating all of them. The key ideas are:

- At each MCTS node, sample a fixed number of actions from the policy prior (or from a learned proposal distribution)
- Use importance weighting or policy correction to account for the fact that only a subset of actions was considered
- The search can now operate in continuous action spaces by treating each sampled action as a candidate

Sampled MuZero was demonstrated on continuous control tasks (DeepMind Control Suite) and showed competitive performance with model-free methods like D4PG and MPO, extending the MuZero framework to a much broader class of problems.

This variant is significant because it removes one of MuZero's main practical limitations. With Sampled MuZero, the framework is applicable to robotics, autonomous driving, and other domains with continuous action spaces.

---

## Connection to World Models

MuZero and World Models (Ha and Schmidhuber, 2018) represent two approaches to model-based RL with learned models, but with different design philosophies.

**World Models** learn a generative model of the environment that reconstructs observations. The architecture typically consists of a variational autoencoder (VAE) that compresses observations into a latent space and a recurrent neural network (MDN-RNN) that predicts future latent states. A simple controller is then trained entirely within the learned model -- the agent "dreams" in the learned world and transfers the learned policy to the real environment. The key characteristic is that the model explicitly tries to reconstruct or predict observations.

**MuZero** takes a different approach: it learns a model that predicts only the quantities needed for planning (rewards, values, policies) without attempting to reconstruct observations. The latent space is optimized for decision-making, not for perceptual fidelity.

The tradeoffs between these approaches:

| Aspect | MuZero | World Models |
|---|---|---|
| Latent space objective | Planning-relevant predictions | Observation reconstruction |
| Model complexity | Simpler (no decoder needed) | More complex (encoder + decoder + RNN) |
| Planning method | MCTS (explicit tree search) | Policy trained in dreamed environment |
| Interpretability | Low (latent states are opaque) | Higher (can decode latent states to images) |
| Sample efficiency | High with search | Depends on model quality |
| Scalability demonstrated | Board games + Atari | Initially simple environments, later scaled (DreamerV3) |

More recent work in the World Models lineage, particularly DreamerV2 and DreamerV3 (Hafner et al., 2023), has scaled World Models to complex environments including Atari, Minecraft, and continuous control, closing the gap with MuZero-style approaches. DreamerV3 in particular demonstrated strong results across diverse domains with a single set of hyperparameters, similar to MuZero's generality claim.

Both approaches validate the core idea that learning a model of the environment and using it for planning or imagination can yield significant improvements over pure model-free methods. The question of whether to predict observations or only planning-relevant quantities remains an active research topic.

---

## Practical Considerations

### Compute Requirements

MuZero is computationally expensive. The original paper used thousands of TPUs for self-play data generation and hundreds of TPUs for training. MCTS at each step requires dozens to hundreds of forward passes through the neural network, multiplied by every step in every self-play game. For practitioners without access to large-scale compute:

- **MuZero Unplugged** (Schrittwieser et al., 2021) adapts MuZero to work with fixed offline datasets, removing the need for self-play. This is useful when you have logged data but cannot interact with the environment.
- **EfficientZero** (Ye et al., 2021) dramatically improved MuZero's sample efficiency, achieving strong Atari performance with only 2 hours of real-time game experience (100K environment steps). It added self-supervised consistency losses and other modifications to extract more signal from fewer samples.

### Implementation Complexity

MuZero is significantly more complex to implement than standard model-free algorithms. The moving parts include:

- Three separate neural networks (representation, dynamics, prediction) that must be trained jointly
- MCTS implementation with UCB-based action selection, expansion, and backup
- Self-play infrastructure with parallel actors and a replay buffer
- Unrolled training with gradient flow through multiple dynamics function steps
- Priority-based replay sampling (reanalyze) for improved sample efficiency

Open-source implementations exist. The most notable is Google's **mctx** library (JAX-based MCTS) and community implementations like **muzero-general** (Python/PyTorch). These can serve as starting points, but expect significant engineering effort to adapt them to new domains.

### Hyperparameter Sensitivity

Key hyperparameters and their effects:

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Number of MCTS simulations | 50 (Atari), 800 (Go) | More simulations improve decision quality but increase compute |
| Unroll steps (K) | 5 | Number of dynamics function steps during training |
| Discount factor (gamma) | 0.997 (Atari), 1.0 (board games) | Board games are undiscounted; Atari uses near-1 discount |
| Replay buffer size | 10^5 to 10^6 games | Larger buffers help but require more memory |
| Temperature for action selection | 1.0 (early), 0.0 (late) | Controls exploration during self-play |
| Value target n-step | 10 (Atari), full game (board games) | Bootstrapping horizon for value targets |
| Learning rate | 0.001-0.01 with schedule | Cosine or step decay common |
| L2 regularization | 1e-4 | Standard weight decay |

### When to Choose MuZero over Alternatives

MuZero is the right choice when:

1. Planning matters -- the domain has complex decisions where looking ahead multiple steps is valuable
2. The environment dynamics are unknown or too expensive to simulate directly
3. You have sufficient compute for self-play and MCTS
4. The action space is tractable (discrete and not too large, or using Sampled MuZero for continuous)

For many practical RL applications, simpler model-free methods (PPO, SAC) or simpler model-based methods (Dreamer) will give you better performance per engineering-hour. MuZero's complexity is justified primarily in domains where the quality of decision-making matters greatly and where planning provides a substantial edge.

---

## Limitations

### Model Error Compounding

The learned dynamics function is imperfect. When MCTS searches many steps deep, errors in the dynamics function accumulate with each step. A small error in predicted hidden state at depth 1 is amplified by depth 5 or 10. MuZero mitigates this by keeping search depth moderate and relying on the value function to evaluate positions rather than searching to terminal states, but compounding model error remains a fundamental challenge.

### Latent Space Interpretability

MuZero's hidden states are opaque. Unlike World Models, where you can decode a latent state back to an image to see what the model "imagines," MuZero's latent representations have no direct interpretation. This makes debugging difficult. If the agent makes a bad decision, you cannot easily inspect the model's internal state to understand why.

### Scalability to Complex Environments

While MuZero demonstrated strong results on Atari and board games, these are still relatively constrained environments. Applying MuZero to 3D environments with complex physics, long time horizons, partial observability, and multi-agent dynamics remains an open challenge. The learned model needs to capture enough structure for planning to be useful, and in very complex environments, learning a sufficiently accurate model may be as hard as solving the task directly.

### Inference Latency

MCTS requires running the neural network many times per decision (once per simulation per tree expansion). For applications requiring real-time decisions (sub-10ms), such as real-time strategy games or robotic control at high frequencies, the latency of MCTS may be prohibitive. One mitigation is to distill the search-improved policy into a fast reactive network that does not require search at inference time.

### Discrete Action Space Assumption

Standard MuZero assumes a discrete and enumerable action space. While Sampled MuZero addresses this, the extension adds complexity and has not been as extensively validated as the original. For continuous control problems, model-free methods (SAC, TD3) or Dreamer-style world models are often simpler and equally effective choices.

---

## Key Terminology

- **Representation function (h)**: Encodes a real observation into a hidden state in the learned latent space.
- **Dynamics function (g)**: The learned model that predicts the next hidden state and reward given a current hidden state and action.
- **Prediction function (f)**: Predicts a policy and value from a hidden state.
- **PUCT**: Predictor Upper Confidence bounds applied to Trees. The UCB-variant used in MCTS to balance exploration and exploitation.
- **Unrolled training**: Training the model by applying the dynamics function for K steps and matching predictions at each step to actual game outcomes.
- **Reanalyze**: A technique where stored trajectories are periodically re-evaluated with the current (improved) model to generate fresher training targets.
- **Self-play**: The process of generating training data by having the agent play against itself (or interact with the environment autonomously).
