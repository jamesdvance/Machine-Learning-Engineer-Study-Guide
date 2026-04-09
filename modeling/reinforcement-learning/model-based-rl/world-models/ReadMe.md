# World Models

## Summary

World models are learned simulators of an environment that allow a reinforcement learning agent to predict the consequences of its actions without interacting with the real world. Rather than learning purely from trial and error in the actual environment (as model-free methods do), an agent equipped with a world model can plan ahead by "imagining" future trajectories inside the learned model. This enables dramatically better sample efficiency, since the agent can generate unlimited synthetic experience from the model rather than relying solely on costly or dangerous real-world interactions.

The concept was formalized by Ha and Schmidhuber in their 2018 "World Models" paper, which introduced a Vision-Memory-Controller (V-A-C) architecture that compresses high-dimensional observations into a latent space, predicts future latent states with a recurrent memory, and trains a small controller entirely inside the learned model. This line of work has since evolved into the Dreamer family of algorithms (DreamerV1, V2, V3), which learn latent dynamics models end-to-end and train actor-critic policies by backpropagating through imagined trajectories. DeepMind's MuZero, a sibling approach, combines a learned dynamics model with Monte Carlo Tree Search to achieve superhuman performance in board games and Atari without any knowledge of the game rules.

Key points to remember:

- A world model learns the transition dynamics and reward function of an environment, enabling the agent to simulate experience internally
- The V-A-C architecture uses a Variational Autoencoder (vision), an RNN with a Mixture Density Network (memory), and a compact linear controller trained via evolution
- Dreamer and its successors learn a Recurrent State-Space Model (RSSM) in latent space and train policies by backpropagating through imagined rollouts
- Training "in imagination" means the agent can extract more learning from fewer real environment interactions, yielding strong sample efficiency
- Model error compounds over long rollouts, so world models must balance imagination horizon with prediction fidelity
- MuZero demonstrates a related but distinct approach where the learned model is optimized for planning quality rather than observation reconstruction
- World models are particularly valuable in domains where real-world interaction is expensive, slow, or dangerous (robotics, autonomous driving, healthcare)

## What Is a World Model

In the standard reinforcement learning formulation, an agent interacts with an environment by observing a state, taking an action, receiving a reward, and transitioning to a new state. A world model attempts to learn the functions that govern these transitions and rewards directly from data. Formally, given a state s_t and action a_t, the world model approximates:

```
s_{t+1} ~ p(s_{t+1} | s_t, a_t)     (transition model)
r_t     ~ p(r_t | s_t, a_t)           (reward model)
```

Once learned, these functions let the agent simulate what would happen if it took a particular sequence of actions, without actually executing them. This is analogous to how humans mentally rehearse scenarios before acting. A chess player imagines the consequences of a move before committing; a world model gives an RL agent the same capability.

The idea of learning environment dynamics is not new. Dyna-Q, introduced by Sutton in 1991, already combined real experience with simulated experience from a learned model. What distinguishes modern world models is that they operate on high-dimensional observations (images, point clouds), learn compact latent representations, and scale to complex continuous control tasks.

### Model-Based vs Model-Free

The distinction between model-based and model-free RL is fundamental. Model-free methods (DQN, PPO, SAC) learn a policy or value function directly from environment interactions. They are conceptually simple and can achieve strong asymptotic performance, but they often require millions of environment steps to learn. Model-based methods learn a model of the environment and use it for planning, policy optimization, or generating synthetic training data. The trade-off is clear:

| Aspect | Model-Free | Model-Based (World Models) |
|--------|-----------|---------------------------|
| Sample efficiency | Low (millions of steps) | High (orders of magnitude fewer) |
| Asymptotic performance | Often higher | Can be limited by model accuracy |
| Computational cost per step | Lower | Higher (model learning + planning) |
| Robustness | No model bias | Subject to model error |
| Applicability | General | Especially valuable when interactions are costly |

World models sit squarely in the model-based camp but distinguish themselves by learning everything from raw observations (typically pixels) rather than relying on hand-crafted simulators or known physics.

## The V-A-C Architecture (Ha and Schmidhuber, 2018)

The "World Models" paper by Ha and Schmidhuber introduced a modular architecture with three distinct components, often referred to as V-A-C: Vision (V), Memory (M, but "A" for the autoregressive/memory component in some formulations), and Controller (C).

### Vision Model (V) -- Variational Autoencoder

The vision component is a convolutional Variational Autoencoder (VAE) that compresses each high-dimensional observation (e.g., a 64x64 RGB image) into a compact latent vector z_t. The VAE learns to encode observations into a distribution over latent codes and decode them back, ensuring that the latent space is smooth and structured.

```
Encoder: q(z_t | x_t)     maps observation x_t to latent z_t
Decoder: p(x_t | z_t)     reconstructs observation from latent z_t
```

The latent vector z_t (typically 32 or 64 dimensions) captures the essential spatial features of the observation while discarding irrelevant details. This compression is critical because it makes the downstream dynamics model tractable.

### Memory Model (M) -- MDN-RNN

The memory component is a Recurrent Neural Network (specifically an LSTM) augmented with a Mixture Density Network (MDN) output layer. It takes the current latent vector z_t and action a_t as input and predicts the distribution of the next latent vector:

```
p(z_{t+1} | a_t, z_t, h_t) = MDN-RNN(a_t, z_t, h_t)
```

The MDN output represents the predicted next latent state as a mixture of Gaussians, capturing the inherent uncertainty and multimodality of environment transitions. The hidden state h_t of the RNN serves as a compressed memory of the entire history of observations and actions.

This is the actual "world model" in the strict sense -- it predicts how the latent state evolves in response to actions. By unrolling this model forward in time, the agent can simulate entire trajectories in latent space without touching the real environment.

### Controller (C) -- Compact Policy

The controller is deliberately kept simple: a single linear layer that maps the concatenation of z_t and h_t to an action:

```
a_t = W_c [z_t, h_t] + b_c
```

The key insight is that if V and M have learned a good representation of the world, the controller's job is trivial. A simple linear mapping suffices because all the complexity has been absorbed by the world model. The controller is trained using Covariance Matrix Adaptation Evolution Strategy (CMA-ES), a derivative-free optimization method. This avoids the need for backpropagation through the entire system.

### Training Pipeline

The V-A-C architecture is trained in three sequential stages:

1. **Data collection**: A random policy interacts with the environment, collecting a dataset of observation sequences.
2. **Train V**: The VAE is trained on the collected observations to learn latent encodings.
3. **Train M**: The MDN-RNN is trained on sequences of (z_t, a_t, z_{t+1}) to predict future latent states.
4. **Train C**: The controller is trained entirely inside the learned world model (i.e., in imagination), using CMA-ES to maximize cumulative reward over imagined rollouts.

The remarkable result was that a controller trained entirely in the "dream" of the MDN-RNN, without any further access to the real environment, could solve the VizDoom "Take Cover" task and achieve competitive performance on CarRacing.

## Latent Space Dynamics Models

The core technical idea underlying modern world models is learning dynamics in a compact latent space rather than in the raw observation space. This offers several advantages:

**Computational tractability.** Predicting the next 64x64x3 image pixel by pixel requires modeling 12,288 dimensions. Predicting the next 200-dimensional latent vector is orders of magnitude cheaper.

**Abstraction.** Latent representations can capture the task-relevant structure of observations while discarding irrelevant details (lighting variations, background textures). This makes dynamics prediction easier and more generalizable.

**Structured representations.** Methods like the RSSM (discussed below) explicitly separate deterministic and stochastic components of the latent state, providing better modeling of both predictable physics and inherent randomness.

### The Recurrent State-Space Model (RSSM)

The RSSM, introduced by Hafner et al. in the PlaNet paper (2019) and refined in the Dreamer line of work, is the most influential latent dynamics architecture for world models. It maintains two types of latent state:

- **Deterministic state h_t**: Updated by a GRU (or similar recurrent cell), this captures the predictable, deterministic aspects of the dynamics.
- **Stochastic state s_t**: Sampled from a learned prior, this captures the inherent randomness of the environment.

The full RSSM consists of four components:

```
Deterministic state model:   h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
Stochastic prior:            p(s_t | h_t)
Stochastic posterior:        q(s_t | h_t, o_t)
Observation model:           p(o_t | h_t, s_t)
Reward model:                p(r_t | h_t, s_t)
```

During training, the posterior q uses the actual observation o_t to infer a better stochastic state. During imagination (planning), only the prior p is used since no real observations are available. The KL divergence between prior and posterior is minimized, encouraging the prior to become a good predictor of the stochastic state.

## The Dreamer Family

### Dreamer (V1, 2020)

Dreamer (Hafner et al., 2020) was the first algorithm to successfully learn long-horizon behaviors from images by backpropagating analytic value gradients through imagined latent trajectories. Its key contributions:

1. **Latent imagination**: Using the RSSM, Dreamer generates imagined trajectories entirely in latent space. Starting from a real state, it rolls out the dynamics model for H steps using the current policy.

2. **Actor-critic in imagination**: Rather than using derivative-free optimization like CMA-ES, Dreamer trains an actor (policy) and critic (value function) on the imagined trajectories. The actor is updated by backpropagating through the dynamics model (a form of analytic gradients, similar in spirit to backpropagation through time).

3. **Lambda returns**: The value targets use generalized lambda-returns that blend short-horizon model predictions (more accurate) with long-horizon value estimates (less accurate but unbiased).

The imagined rollout and training loop looks like:

```
For each training step:
    1. Observe real environment, add transition to replay buffer
    2. Sample batch of sequences from replay buffer
    3. Update world model (RSSM) on real data
    4. Generate imagined trajectories from learned model
    5. Compute lambda-returns using imagined rewards and critic
    6. Update actor to maximize lambda-returns (via backprop through model)
    7. Update critic to predict lambda-returns
```

### DreamerV2 (2021)

DreamerV2 introduced several improvements that allowed it to match or exceed model-free methods on the full Atari benchmark (55 games), a first for a world model agent:

- **Discrete latent states**: The stochastic component uses categorical distributions (32 categories with 32 classes each) rather than Gaussian. This provides a more expressive and stable representation.
- **KL balancing**: Rather than symmetrically minimizing KL(posterior || prior), DreamerV2 balances the gradient to primarily update the prior to match the posterior (rather than collapsing the posterior to match a bad prior). This prevents the common failure mode where the model learns to ignore the stochastic state entirely.
- **Straight-through gradients**: For the discrete latent variables, straight-through estimators allow backpropagation through the sampling operation.

### DreamerV3 (2023)

DreamerV3 aimed for a single set of hyperparameters that works across diverse domains, from Atari to continuous control to Minecraft:

- **Symlog predictions**: All predictions (rewards, values, reconstructions) use a symmetric logarithmic transform that handles varying scales without domain-specific tuning.
- **Unimix categoricals**: A small uniform mixture (1%) is added to the categorical distribution to prevent exact zeros that can cause numerical issues.
- **Free bits**: A minimum KL threshold prevents posterior collapse while allowing the model to use the stochastic state.
- **Percentile-based return normalization**: Instead of fixed normalization, returns are normalized based on running percentile estimates.

DreamerV3 demonstrated strong performance across seven diverse benchmarks without modifying any hyperparameters, including learning to collect diamonds in Minecraft from scratch using only pixel observations.

## Training in Imagination vs Real Environment

The central advantage of world models is the ability to separate environment interaction from policy learning. In a model-free algorithm like PPO, every gradient update requires new environment experience. With a world model, the process is decoupled:

**Real environment interaction** happens infrequently and is used solely to train the world model and ground it in reality. The agent collects transitions, stores them in a replay buffer, and updates the dynamics model, reward model, and representation.

**Imagination-based training** happens frequently and cheaply. The agent generates thousands of imagined trajectories from the world model and uses them to update the policy and value function. Since this happens entirely in latent space (not pixel space), it is fast -- a single GPU can generate millions of imagined time steps.

This decoupling has profound implications:

- **Sample efficiency**: The agent extracts far more learning per real environment step. Where PPO might need 200 million frames on Atari, DreamerV2 achieves comparable performance with 200 million frames but uses each frame far more effectively through imagination.
- **Safety**: In domains like robotics, the agent can explore dangerous strategies in imagination without physical consequences.
- **Parallelism**: Imagination is embarrassingly parallel. Multiple rollouts can be generated simultaneously on GPU hardware.
- **Credit assignment**: Because the dynamics model is differentiable, gradients can flow from future rewards back through the model to inform current actions. This provides a richer learning signal than the scalar reward used in model-free methods.

### Balancing Real and Imagined Experience

A key design decision is the ratio of real-to-imagined experience. Too little real data and the world model becomes inaccurate; too much reliance on imagination and the agent exploits model errors. Dreamer addresses this through:

- Regular world model updates on fresh real data
- Relatively short imagination horizons (15 steps by default) to limit error compounding
- Starting imagined rollouts from states visited in the real environment (grounding)

## Sample Efficiency

Sample efficiency is often the primary motivation for world models. Consider concrete numbers from the literature:

- **Dreamer on DMControl**: Achieves expert-level performance on many continuous control tasks in 500K environment steps. Model-free methods like SAC often require 1-5M steps for comparable performance.
- **DreamerV2 on Atari**: Achieves competitive performance within 200M frames, which sounds like a lot but represents far more efficient learning per frame than model-free baselines that need 200M frames without imagination.
- **DreamerV3 on Minecraft**: Learns to collect diamonds (a task requiring long sequential reasoning) from pixels, something model-free methods have not achieved from scratch.

The sample efficiency gain scales with the cost of environment interaction. In simulated environments where steps are cheap, the advantage is moderate. In robotics where each step involves real hardware, the advantage is transformative. A robot arm that can learn a manipulation task from 30 minutes of real interaction (plus extensive imagination) rather than 30 hours represents a practical enabling capability.

## Challenges

### Model Error Compounding

The most fundamental challenge is that small per-step prediction errors compound over multi-step rollouts. If the dynamics model has a 1% error per step, after 50 steps the imagined state may have diverged significantly from reality. The agent can then learn to exploit these model inaccuracies, converging on policies that perform well in the model's "dream" but fail in the real environment.

Mitigation strategies:

- **Short imagination horizons**: Dreamer uses 15-step rollouts by default, short enough that errors remain manageable.
- **Ensemble disagreement**: Train multiple dynamics models and measure their disagreement as an uncertainty estimate. Penalize the agent for visiting high-disagreement (uncertain) states.
- **Scheduled imagination length**: Start with very short horizons and gradually increase as the model improves.
- **Value function bootstrapping**: Use the critic to estimate long-term returns rather than relying on very long model rollouts.

### Partial Observability

Most real-world tasks are partially observable -- the agent's observation does not contain all the information needed to predict the future. A single camera image does not reveal velocities, hidden objects, or the state of distant parts of the environment.

World models handle partial observability through the recurrent state h_t, which acts as a belief state summarizing the history of observations. The RSSM architecture is explicitly designed for this: the deterministic state integrates information over time, while the stochastic state captures the remaining uncertainty. However, if the partial observability is severe, the model may still struggle to make accurate predictions.

### Representation Learning Difficulties

The quality of the world model depends heavily on the learned latent representation. If the encoder fails to capture task-relevant features (e.g., the position of a small but important object), the dynamics model cannot predict the relevant dynamics and the agent will fail.

- **Reconstruction-based objectives** (used by Dreamer) require the model to reconstruct all aspects of the observation, wasting capacity on task-irrelevant details.
- **Contrastive or self-predictive objectives** (used by some alternatives) focus on task-relevant features but may miss unexpected important details.

DreamerV3 mitigates this by combining reconstruction loss with reward prediction and KL regularization, creating a multi-objective learning signal.

### Computational Overhead

While world models reduce real environment interactions, they increase computational requirements. Training the dynamics model, generating imagined rollouts, and performing backpropagation through the model all require significant GPU compute. For environments where simulation is already fast and cheap (e.g., simple games), the added computation may not be worthwhile.

## Connection to MuZero

MuZero (Schrittwieser et al., 2020) is a closely related approach that also learns an environment model, but with fundamentally different design choices:

| Aspect | World Models (Dreamer) | MuZero |
|--------|----------------------|--------|
| Model predicts | Observations (or latent states with observation reconstruction) | Rewards, values, and policies only |
| Planning method | Backpropagation through model | Monte Carlo Tree Search (MCTS) |
| Representation loss | Reconstruction + KL + reward | Value equivalence (prediction of value-relevant quantities) |
| Domains | Continuous control, Atari, diverse tasks | Board games, Atari |
| Policy training | Actor-critic with analytic gradients | Distillation from MCTS policy |

MuZero's key insight is that the learned model does not need to reconstruct observations at all. It only needs to predict the quantities relevant for planning: rewards, value estimates, and action probabilities. This "value equivalence" principle produces a model optimized for decision-making rather than pixel prediction. In board games (Go, chess, shogi), MuZero matches AlphaZero's performance without knowing the game rules.

However, MuZero's reliance on MCTS makes it most natural for discrete action spaces and tree-structured planning, while Dreamer's gradient-based approach extends more naturally to continuous control.

Both approaches demonstrate the same core principle: learning a dynamics model from data and using it to plan or train policies in imagination.

## Comparison with Model-Free Methods

When choosing between world models and model-free approaches, consider:

**Choose world models when:**
- Sample efficiency matters (expensive or slow environments)
- The environment has learnable, structured dynamics
- You need to plan multi-step sequences
- Safety constraints prevent extensive real-world exploration
- Transfer across related tasks is desirable (the world model transfers even if the reward changes)

**Choose model-free when:**
- Simulation is fast and unlimited (e.g., video games)
- The dynamics are extremely complex or chaotic (hard to model)
- Simplicity of implementation is a priority
- Asymptotic performance is more important than sample efficiency
- The observation space is very high-dimensional with limited useful structure

In practice, many modern systems are hybrid. A world model can generate synthetic data to augment a model-free learner, or a model-free policy can be fine-tuned after initial world model training.

## Broader Applications and Future Directions

World models extend well beyond the traditional RL benchmarks:

**Robotics**: World models are especially compelling for real-world robotics, where every interaction risks hardware damage. Recent work has demonstrated sim-to-real transfer where a world model is trained in simulation and refined with a small amount of real robot data.

**Autonomous driving**: Predicting how the environment (other vehicles, pedestrians) evolves in response to the ego vehicle's actions is fundamentally a world modeling problem. Companies are increasingly using learned world models for scenario generation and planning.

**Video generation and prediction**: Architectures like Sora from OpenAI and Genie from DeepMind are essentially world models for visual data, trained to predict how video frames evolve. The connection between video prediction and RL world models is becoming increasingly clear.

**Large language models as world models**: There is growing interest in whether LLMs implicitly learn world models through next-token prediction on text describing the world. This connects to debates about whether language models truly "understand" or merely pattern-match.

## Key Takeaways

1. **World models learn to simulate the environment**, converting the RL problem from costly real-world trial-and-error into cheap imagination-based optimization.

2. **Latent space dynamics are essential**. Operating in a compact, learned representation makes dynamics prediction tractable and enables gradient-based planning.

3. **The V-A-C architecture** established the paradigm: compress observations (VAE), predict dynamics (RNN), and train a simple controller in imagination.

4. **Dreamer and its successors** refined this into a practical, high-performance framework using the RSSM, actor-critic training through imagined rollouts, and careful design choices for representation learning.

5. **Sample efficiency is the primary win**. World models shine when real environment interaction is expensive, dangerous, or slow.

6. **Model error compounding is the primary challenge**. Short imagination horizons, ensemble uncertainty, and value bootstrapping help mitigate this.

7. **MuZero shows an alternative design point**, optimizing the model for planning quality rather than observation reconstruction.

8. **The field is converging** with video prediction, generative models, and foundation models, suggesting world models will play an increasingly central role in AI systems.
