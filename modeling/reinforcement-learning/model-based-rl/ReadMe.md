# Model-Based Reinforcement Learning

## Summary

Model-based reinforcement learning is the family of RL methods that learn or leverage an explicit model of the environment -- the transition dynamics and reward function -- to improve decision-making. Instead of treating the environment as an opaque black box from which the agent collects reward signals (as model-free methods do), model-based RL builds an internal representation of how the world works and uses that representation to plan, generate synthetic experience, or compute policy gradients analytically. The result is dramatically better sample efficiency: agents can extract far more learning from each real-world interaction by "thinking" about consequences before acting.

The foundational idea is Sutton's Dyna architecture (1991), which showed that real experience can be augmented with simulated experience from a learned model within a single unified loop. Modern successors have scaled this idea to complex visual domains. World models (Ha and Schmidhuber, 2018; Hafner et al., 2020-2023) learn latent dynamics models from pixel observations and train policies entirely in imagination. MuZero (Schrittwieser et al., 2020) learns a model optimized for planning-relevant predictions and combines it with Monte Carlo Tree Search to achieve superhuman performance on board games and Atari without being given the rules. Together, these approaches demonstrate that learning a model of the environment is one of the most powerful strategies available when real-world interaction is expensive, dangerous, or slow.

Key points to remember:

- Model-based RL learns or uses an explicit model of environment dynamics (transitions and rewards) to improve policy learning
- The Dyna architecture is the conceptual foundation: interleave real experience, model learning, and planning in a single loop
- Sample efficiency is the primary advantage -- model-based methods can require 10-100x fewer environment interactions than model-free counterparts
- Planning with a learned model can take many forms: generating synthetic rollouts, backpropagating through differentiable dynamics, or performing tree search
- Model error is the central challenge -- small per-step prediction errors compound over multi-step rollouts, potentially leading agents to exploit model inaccuracies
- World models learn latent-space simulators and train policies in imagination via actor-critic methods
- MuZero learns a model that predicts only planning-relevant quantities (rewards, values, policies) and uses MCTS for decision-making
- The choice between model-based and model-free methods depends on the cost of environment interaction, the complexity of the dynamics, and the available compute budget

## What Makes a Method Model-Based

In the standard RL formulation, an agent interacts with an environment modeled as a Markov Decision Process (MDP). At each time step, the agent observes a state s_t, takes an action a_t, receives a reward r_t, and transitions to a new state s_{t+1}. The environment is characterized by:

```
Transition dynamics:  p(s_{t+1} | s_t, a_t)
Reward function:      r(s_t, a_t)
```

A model-free method learns a policy or value function directly from observed transitions without ever explicitly representing these functions. A model-based method either is given or learns approximations of these functions and uses them to improve decision-making.

The learned model can be used in several ways:

```
Uses of a Learned Environment Model:

1. Simulated experience   -- Generate synthetic transitions to augment real data (Dyna)
2. Policy optimization    -- Backpropagate through differentiable dynamics to compute
                            policy gradients analytically (Dreamer, SVG)
3. Tree search            -- Plan by searching over possible action sequences using
                            the model to predict outcomes (MuZero, AlphaZero)
4. Trajectory optimization-- Optimize action sequences by rolling out the model and
                            computing gradients w.r.t. actions (iLQR, MPPI)
5. Uncertainty estimation -- Use model ensembles to quantify uncertainty and guide
                            exploration or constrain planning
```

The common thread is that the model provides a mechanism for the agent to reason about consequences without executing actions in the real environment. This is fundamentally what distinguishes model-based from model-free approaches.

## The Dyna Architecture

The Dyna architecture, introduced by Richard Sutton in 1991, is the conceptual starting point for understanding model-based RL. Dyna unifies three processes into a single loop: acting in the real environment, learning a model from real experience, and planning by generating simulated experience from the model.

### The Dyna Loop

```
Dyna-Q Algorithm:

Initialize Q(s, a) arbitrarily
Initialize model M(s, a) -> (s', r)

Loop forever:
    (a) Observe current state s
    (b) Select action a using policy derived from Q (e.g., epsilon-greedy)
    (c) Execute a, observe reward r and next state s'
    (d) Direct RL update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    (e) Model learning:
        M(s, a) <- (s', r)    [store the observed transition]
    (f) Planning (repeat N times):
        Sample a previously visited state-action pair (s_sim, a_sim)
        (s'_sim, r_sim) <- M(s_sim, a_sim)    [simulate transition]
        Q(s_sim, a_sim) <- Q(s_sim, a_sim) + alpha * [r_sim + gamma * max_a' Q(s'_sim, a') - Q(s_sim, a_sim)]
```

The critical insight is step (f): after each real interaction, the agent performs N additional Q-learning updates using simulated experience from its model. If the model is accurate, these simulated updates are as valuable as real ones. The parameter N controls the ratio of planning to real interaction -- higher N means more learning per real step but also more reliance on model accuracy.

### Why Dyna Matters

Dyna established several principles that remain central to modern model-based RL:

**Experience amplification.** One real transition can generate many simulated transitions. If the model is accurate, the agent effectively multiplies its experience by a factor of N+1.

**Separation of concerns.** The model learning, value learning, and acting components are modular. The model can be updated independently, the value function can be improved through planning without further environment interaction, and the policy can be improved without re-collecting data.

**Incremental model improvement.** The model does not need to be perfect from the start. As the agent collects more data, the model improves, and planning becomes more effective.

In its original tabular form, Dyna-Q used a deterministic lookup table as its model. Modern descendants replace this with neural networks that can generalize across states, handle continuous spaces, and operate on high-dimensional observations like images.

## Model-Based vs Model-Free Tradeoffs

The decision between model-based and model-free approaches is one of the most consequential design choices in an RL system. Each approach has distinct strengths and weaknesses.

### Sample Efficiency

This is the primary motivation for model-based methods. Model-free algorithms like DQN, PPO, and SAC learn exclusively from real environment transitions. Every gradient update requires data collected from the actual environment. Model-based methods amplify real data by generating synthetic experience or by computing analytic gradients through a differentiable model.

Concrete comparisons from the literature:

- On DeepMind Control continuous control tasks, Dreamer (model-based) achieves expert performance in roughly 500K environment steps. SAC (model-free) typically requires 1-5M steps for comparable results.
- EfficientZero, a MuZero variant, achieves strong Atari performance with only 100K environment steps (roughly 2 hours of real-time play). Model-free methods require 10-200M steps for similar performance.
- In robotics, where each step involves physical hardware, the difference is often the distinction between feasible and infeasible.

### Asymptotic Performance

Model-free methods often achieve higher final performance given unlimited environment interaction. They do not suffer from model bias -- the systematic errors introduced by an imperfect model. When the agent trains long enough, model-free methods converge based on the true environment dynamics, while model-based methods converge based on the (possibly flawed) learned dynamics.

However, this advantage diminishes when the model is highly accurate or when the model is used judiciously (short rollouts, value bootstrapping).

### Computational Cost

Model-based methods trade environment interactions for compute. Learning a dynamics model, generating imagined rollouts, and performing planning all require GPU cycles. For environments where simulation is already fast and cheap (simple games, lightweight simulators), the computational overhead of a learned model may not be justified. For environments where interaction is slow or expensive (robotics, real-world systems), the trade is almost always worthwhile.

### Generalization and Transfer

A learned dynamics model captures environment structure that is independent of any particular reward function. If you change the task (reward) but not the environment (dynamics), the model transfers directly. This is a significant advantage for multi-task settings. A model-free policy trained for one reward function is generally useless for another, while a world model can be reused immediately.

### Summary of Tradeoffs

```
                    Model-Based                  Model-Free
                    +--------------------------+  +--------------------------+
Sample efficiency   | High                     |  | Low                      |
Asymptotic perf     | Limited by model accuracy|  | Can be higher            |
Compute per step    | Higher                   |  | Lower                    |
Robustness          | Subject to model error   |  | No model bias            |
Transfer potential  | High (model reusable)    |  | Low (policy is task-specific)|
Implementation      | More complex             |  | Simpler                  |
Best when           | Interactions are costly   |  | Simulation is cheap      |
                    +--------------------------+  +--------------------------+
```

## Sample Efficiency: The Core Advantage

Sample efficiency is worth exploring in more depth because it is the reason model-based RL exists as a field. The fundamental bottleneck in many RL applications is not compute but data -- specifically, the cost of collecting transitions from the real environment.

### Why Model-Free Methods Are Data-Hungry

Model-free methods must learn everything from direct experience. A Q-learning agent that has never seen the consequences of action A in state S cannot evaluate that action. Credit assignment is done entirely through reward propagation: the agent must experience a sequence of states leading to reward many times before it can reliably associate early actions with eventual outcomes.

### How Models Multiply Experience

A learned model provides three mechanisms for improved data efficiency:

**Synthetic rollouts.** Given a starting state from the replay buffer, the model can generate entire trajectories of synthetic experience. If the model is accurate, each synthetic transition is as informative as a real one. An agent can generate thousands of imagined transitions for every real one.

**Analytic gradients.** If the dynamics model is differentiable, the agent can backpropagate through the model to compute how changing the policy at time t affects rewards at time t+H. This provides a dense gradient signal that is far more informative than the scalar reward signal used in model-free policy gradient methods. Dreamer exploits this: it backpropagates through imagined rollouts of its latent dynamics model to update the actor.

**Structured exploration.** With a model, the agent can identify which parts of the state space are poorly understood (high model uncertainty) and direct exploration there. This is far more efficient than the undirected exploration (epsilon-greedy, entropy bonuses) used by most model-free methods.

### When the Advantage Matters Most

The sample efficiency advantage scales with the cost of real interaction:

- **Video games and simple simulators**: Real steps are essentially free. The model-based advantage is moderate and may not justify the added complexity.
- **High-fidelity simulators**: Steps take seconds to minutes (fluid dynamics, molecular simulation). Model-based methods provide meaningful speedups.
- **Real-world robotics**: Each step involves physical hardware, takes real time, and risks damage. Sample efficiency is transformative -- the difference between a task that takes 30 minutes of robot time and one that takes 30 hours.
- **Online systems**: In recommender systems or ad bidding, each interaction affects real users and revenue. Reducing the number of exploratory actions has direct business value.

## Planning with Learned Models

Planning -- using the model to look ahead and evaluate actions before committing -- is the distinguishing capability of model-based RL. There are several distinct approaches to planning with learned models.

### Shooting Methods (Open-Loop Planning)

The simplest approach is to generate multiple candidate action sequences, roll each one out through the model, and select the one with the highest predicted return. Cross-Entropy Method (CEM) and Model Predictive Path Integral (MPPI) are popular variants.

```
Cross-Entropy Method (CEM) Planning:

1. Initialize action distribution (e.g., Gaussian for each time step)
2. Repeat for K iterations:
   a. Sample N action sequences from distribution
   b. Roll each sequence out through the learned model
   c. Compute total predicted reward for each sequence
   d. Select top-M sequences (elites)
   e. Refit distribution to elites
3. Execute first action of best sequence
4. Re-plan at next time step (Model Predictive Control)
```

This approach is simple, parallelizable, and does not require the model to be differentiable. It works well for moderate planning horizons (10-30 steps) in continuous action spaces. PETS (Chua et al., 2018) combines CEM planning with probabilistic ensemble models and achieves strong sample efficiency on continuous control tasks.

### Backpropagation Through the Model

When the dynamics model is differentiable, the agent can compute gradients of predicted returns with respect to the policy parameters by unrolling the model and backpropagating through time. This is the approach used by Dreamer and related methods.

```
Dreamer-style planning:

1. Sample batch of initial states from replay buffer
2. Unroll dynamics model for H steps using current policy:
   For t = 1 to H:
       a_t = policy(s_t)
       s_{t+1}, r_t = dynamics_model(s_t, a_t)
3. Compute returns: R = sum of discounted r_t + bootstrapped value V(s_H)
4. Backpropagate dR/d(policy_params) through the entire unrolled graph
5. Update policy via gradient ascent
```

This is more sample-efficient than shooting methods (it uses gradient information rather than random sampling) but requires a differentiable model and is subject to vanishing or exploding gradients over long horizons.

### Monte Carlo Tree Search (MCTS)

MCTS builds a search tree by repeatedly simulating trajectories from the current state, using the model to predict transitions. It balances exploration (trying under-explored actions) with exploitation (favoring actions with high estimated value) using an upper confidence bound formula. After many simulations, the action with the most visits at the root is selected.

MuZero uses this approach with its learned dynamics model. MCTS is particularly well-suited to discrete action spaces and problems where precise per-decision planning is valuable (board games, complex strategic decisions).

### Model Predictive Control (MPC)

MPC is a planning framework borrowed from control theory. At each time step, the agent optimizes a sequence of actions over a fixed horizon using the model, executes only the first action, observes the real next state, and re-plans. This constant re-planning makes MPC robust to model errors because the agent never commits to a long action sequence based on potentially inaccurate model predictions.

```
MPC Loop:

Repeat at each real time step:
    1. Observe current real state s
    2. Optimize action sequence [a_1, ..., a_H] using the model
       (via CEM, gradient-based optimization, or MCTS)
    3. Execute only a_1 in the real environment
    4. Discard [a_2, ..., a_H] and re-plan from the new state
```

MPC is widely used in robotics and autonomous driving, where the model is updated continuously and re-planning at each step keeps the agent grounded in reality.

## Challenges: Model Error and Compounding

The central technical challenge in model-based RL is that learned models are imperfect, and their errors compound over multi-step predictions.

### Compounding Error

Consider a dynamics model with a small per-step prediction error epsilon. After one step, the predicted state is epsilon-close to the true state. After two steps, the error is approximately 2*epsilon (assuming the model operates on its own previous predictions rather than real states). After K steps, the error can grow as O(K * epsilon) in the best case and exponentially in the worst case, depending on the Lipschitz constant of the dynamics.

```
Compounding Error Illustration:

Step 1: predicted_state_1 = true_state_1 + error_1
Step 2: predicted_state_2 = model(predicted_state_1) + error_2
       = model(true_state_1 + error_1) + error_2
       ~ true_state_2 + (L * error_1 + error_2)      [L = Lipschitz constant]
Step K: error ~ O(L^K * epsilon)  in the worst case

If L > 1 (chaotic dynamics), errors grow exponentially.
If L < 1 (stable dynamics), errors stay bounded.
```

This means the effective planning horizon is limited by model accuracy. The more accurate the model, the further into the future the agent can reliably plan.

### Model Exploitation

A subtler problem is that the policy can learn to exploit systematic errors in the model. If the model consistently overestimates rewards in some region of state space, the policy will steer toward that region. The policy then performs well in the "dream" but poorly in the real environment. This is analogous to overfitting in supervised learning, except the agent is overfitting to the model rather than to a dataset.

### Mitigation Strategies

The field has developed several approaches to manage model error:

**Short rollout horizons.** Generate imagined trajectories of only 5-15 steps, then use a learned value function to estimate long-term returns beyond the horizon. This limits the horizon over which errors compound while still capturing multi-step consequences. Dreamer uses 15-step imagination horizons by default.

**Ensemble disagreement.** Train an ensemble of M dynamics models (typically 5-7) on the same data with different initializations. When the models disagree about a prediction, the agent is in an uncertain region of state space. This disagreement signal can be used to:
- Penalize the policy for visiting high-uncertainty states (conservative planning)
- Encourage exploration toward high-uncertainty states (optimistic exploration)
- Terminate rollouts early when uncertainty exceeds a threshold

**Model Predictive Control.** Re-plan at every step rather than committing to long action sequences. This limits the practical impact of model error because the agent corrects course after each real observation.

**Value function bootstrapping.** Instead of rolling the model out to the episode end, use a learned value function to estimate the return beyond the rollout horizon. The value function is trained on real returns and provides an unbiased (if potentially high-variance) estimate that does not suffer from model compounding error.

**Stochastic models.** Learn a distribution over next states rather than a point prediction. This captures the model's uncertainty about transitions and prevents the agent from over-committing to a single predicted future. The RSSM architecture used by Dreamer explicitly separates deterministic and stochastic state components for this reason.

**Regularization and early stopping.** Standard machine learning regularization techniques help prevent the dynamics model from overfitting to the training data, which would degrade its generalization to states visited by an evolving policy.

## Taxonomy of Model-Based Methods

Model-based RL is a broad field. It helps to organize the major approaches by how they use the model.

### Dyna-Style Methods

Learn a model and use it to generate synthetic transitions that augment the replay buffer. The policy is then trained using standard model-free algorithms on the combined real and synthetic data.

- **Dyna-Q** (Sutton, 1991): Original tabular version
- **MBPO** (Janner et al., 2019): Model-Based Policy Optimization. Uses an ensemble of neural network models to generate short rollouts, trains SAC on the augmented data. Achieves strong sample efficiency on continuous control.
- **STEVE** (Buckman et al., 2018): Stochastic Ensemble Value Expansion. Uses model ensembles to generate multi-step value targets with uncertainty-aware truncation.

### Backpropagation-Through-Model Methods

Learn a differentiable model and backpropagate policy gradients through imagined rollouts.

- **SVG** (Heess et al., 2015): Stochastic Value Gradients. Backpropagates through learned stochastic dynamics.
- **Dreamer family** (Hafner et al., 2020-2023): Learns a latent dynamics model (RSSM) and trains actor-critic via analytic gradients through imagination. Covered in detail in the [World Models](world-models/) child chapter.

### Search-Based Methods

Learn a model and use it for look-ahead search at decision time.

- **MuZero** (Schrittwieser et al., 2020): Learns representation, dynamics, and prediction functions. Plans via MCTS. Covered in detail in the [MuZero](muzero/) child chapter.
- **EfficientZero** (Ye et al., 2021): Extends MuZero with self-supervised consistency losses for improved sample efficiency.
- **Sampled MuZero** (Hubert et al., 2021): Extends MuZero to continuous action spaces by sampling actions in MCTS.

### Model Predictive Control Methods

Learn a model and use it for online trajectory optimization at each step.

- **PETS** (Chua et al., 2018): Probabilistic Ensemble Trajectory Sampling. Uses ensemble of probabilistic neural networks with CEM planning.
- **MBMF** (Nagabandi et al., 2018): Model-Based Model-Free. Uses a learned model for MPC, then fine-tunes with model-free methods.

### Hybrid Methods

Combine model-based components with model-free learning in various ways.

- **Imagination-Augmented Agents (I2A)** (Weber et al., 2017): Uses a learned model to generate imagined rollouts, encodes them into features, and feeds them to a model-free policy as additional context.
- **Value Prediction Networks (VPN)** (Oh et al., 2017): Learns to predict values along imagined trajectories, combining model learning with value prediction.

## Comparison: Model-Based vs Model-Free in Practice

For a mid-career ML engineer evaluating which approach to use, the decision often comes down to practical considerations.

### Choose Model-Based RL When

**Real-world interaction is expensive.** If each environment step costs money (cloud compute, physical robots, real users), model-based methods pay for themselves quickly through sample efficiency gains.

**The environment has learnable structure.** Environments governed by physics, game rules, or consistent dynamics are well-suited to model learning. The model can generalize from observed transitions to predict unseen ones.

**You need to adapt to changing tasks.** A learned dynamics model transfers across reward functions. If you anticipate changing objectives, a model-based approach amortizes the cost of environment modeling.

**Safety matters.** In robotics, healthcare, or autonomous driving, you cannot afford to let the agent explore dangerous actions in the real world. Planning in imagination lets the agent evaluate risky strategies safely.

### Choose Model-Free RL When

**Simulation is fast and unlimited.** If you have a fast simulator (video games, synthetic environments), the sample efficiency advantage of model-based methods diminishes and the added complexity is harder to justify.

**The dynamics are extremely complex or stochastic.** Highly chaotic environments, environments with many interacting agents, or environments with complex contact physics can be very hard to model accurately. A model-free method avoids this challenge entirely.

**You need simplicity.** Model-free algorithms like PPO and SAC are well-understood, widely implemented, and easier to debug. Model-based methods add significant engineering complexity (model training, planning infrastructure, managing the interaction between model accuracy and policy learning).

**Asymptotic performance is the priority.** If you have a large compute budget and care only about final performance, model-free methods with enough environment interaction can match or exceed model-based methods without the risk of model bias.

### The Hybrid Middle Ground

In practice, the boundary between model-based and model-free is often blurred. MBPO trains SAC (model-free) on data augmented by short model rollouts (model-based). Dreamer uses actor-critic updates (traditionally model-free) but computes them through imagined rollouts. MuZero distills search-improved policies into a neural network that can act without search. Many production systems use learned models for some aspects (exploration, safety checking, data augmentation) while relying on model-free methods for the core policy learning.

## Child Chapters

This chapter introduces the principles and tradeoffs of model-based RL. The child chapters dive deep into two of the most important modern instantiations:

**[World Models](world-models/)** covers the line of work from Ha and Schmidhuber's V-A-C architecture through the Dreamer family (V1, V2, V3). These methods learn latent dynamics models from pixel observations and train actor-critic policies entirely in imagination. The chapter covers the RSSM architecture, training in imagination vs. real environments, the progression of improvements from DreamerV1 through V3, and practical considerations for applying world models to continuous control, Atari, and robotics.

**[MuZero](muzero/)** covers DeepMind's approach of learning a model optimized for planning-relevant predictions (rewards, values, policies) rather than observation reconstruction. The chapter traces the AlphaGo lineage, explains the three learned functions (representation, dynamics, prediction), details how MCTS operates with a learned model, and covers practical variants including EfficientZero and Sampled MuZero.

## Key Terminology

- **Dynamics model / transition model**: A function that predicts the next state given the current state and action.
- **Reward model**: A function that predicts the immediate reward given a state and action.
- **Planning**: Using a model to evaluate actions or action sequences before executing them.
- **Imagination / dreaming**: Generating synthetic trajectories by rolling out a learned model.
- **Dyna**: An architecture that interleaves real experience collection, model learning, and model-based planning in a single loop.
- **Model Predictive Control (MPC)**: A planning strategy that optimizes over a fixed horizon at each step and executes only the first action.
- **Model exploitation**: When the policy learns to exploit inaccuracies in the learned model, performing well in simulation but poorly in reality.
- **Compounding error**: The accumulation of prediction errors over multi-step model rollouts.
- **Value equivalence**: The principle (used by MuZero) that a model only needs to be accurate with respect to value-relevant predictions, not full observation reconstruction.
- **Ensemble disagreement**: Using the variance across an ensemble of models as a proxy for epistemic uncertainty.
