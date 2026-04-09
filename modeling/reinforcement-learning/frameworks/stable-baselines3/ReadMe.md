# Stable Baselines3

## Summary

Stable Baselines3 (SB3) is the most widely adopted library for single-agent reinforcement learning in PyTorch. Maintained by Antonin Raffin and collaborators at the German Aerospace Center (DLR), it provides clean, well-tested implementations of core RL algorithms behind a consistent, high-level API. The library descends from the original Stable Baselines (a fork of OpenAI Baselines that cleaned up and unified the TensorFlow code) and was rewritten from scratch in PyTorch to take advantage of modern deep learning tooling. Published in the Journal of Machine Learning Research in 2021, SB3 has become the de facto standard for practitioners who need reliable, off-the-shelf RL training with minimal boilerplate.

SB3 covers the most important algorithm families: on-policy policy gradient methods (PPO, A2C), off-policy actor-critic methods for continuous control (SAC, TD3, DDPG), value-based methods for discrete actions (DQN), and goal-conditioned replay (HER). Its companion library SB3-Contrib extends this with experimental algorithms such as Recurrent PPO, TQC, and Maskable PPO, while the RL Baselines3 Zoo provides a full training framework with pre-tuned hyperparameters and Optuna-based hyperparameter optimization.

Key points to remember:

- SB3 provides production-quality PyTorch implementations of PPO, A2C, DQN, SAC, TD3, DDPG, and HER, all behind a unified API where model.learn() trains and model.predict() produces actions
- The library enforces a strict separation between the stable core (SB3) and experimental features (SB3-Contrib), keeping the main package compact and reliable
- All algorithms work with Gymnasium environments and support Box, Discrete, MultiDiscrete, and MultiBinary action spaces, as well as dictionary observation spaces
- VecEnv wrappers allow parallel environment execution, observation normalization, frame stacking, and other preprocessing without modifying environment code
- Callbacks provide hooks into the training loop for evaluation, early stopping, checkpointing, and custom logging to TensorBoard or Weights and Biases
- The RL Baselines3 Zoo integrates Optuna for automated hyperparameter search and ships pre-tuned hyperparameters for dozens of standard environments
- SB3 targets single-agent, single-machine training; for distributed multi-agent or cluster-scale workloads, RLlib is the better choice

When to use SB3:

- Prototyping and benchmarking RL algorithms on standard Gymnasium environments
- Training single-agent policies for robotics simulation, game playing, or control tasks
- Research that requires a reliable, well-documented baseline implementation
- Production systems where you need a clean save/load workflow and deterministic inference
- Projects where PyTorch ecosystem integration (custom networks, mixed precision, ONNX export) matters

When not to use SB3:

- Multi-agent reinforcement learning (look at PettingZoo with SuperSuit, or RLlib)
- Distributed training across a cluster of machines (RLlib or custom infrastructure)
- When you need full algorithmic transparency and single-file implementations for pedagogical purposes (CleanRL)
- Offline RL from logged datasets (use d3rlpy or Decision Transformer)

---

## Architecture and Design Philosophy

### The Stable Baselines Lineage

OpenAI Baselines was one of the first comprehensive RL libraries, providing reference implementations of major algorithms. However, the codebase was notoriously difficult to work with: inconsistent interfaces, poor documentation, and tightly coupled components made it hard to extend or debug. Stable Baselines (v1/v2) forked the project and unified the API, added documentation, and improved code quality, but remained tied to TensorFlow 1.x.

Stable Baselines3 was a ground-up rewrite in PyTorch. This was not merely a porting exercise. The authors redesigned the class hierarchy, introduced type hints throughout, enforced PEP 8 compliance, and achieved high test coverage. The result is a codebase that is straightforward to read, extend, and contribute to. For a mid-career ML engineer accustomed to PyTorch workflows, SB3 code feels natural rather than framework-imposed.

### Core Design Principles

SB3 is built around several principles that distinguish it from other RL libraries:

**Consistency over flexibility.** Every algorithm inherits from a common base class and exposes the same methods: learn(), predict(), save(), load(), set_env(), and get_env(). Switching from PPO to SAC requires changing a single import and constructor call. This consistency extends to hyperparameter names: learning_rate, batch_size, gamma, and n_steps mean the same thing across all algorithms.

**Separation of concerns.** The policy network (which maps observations to actions), the algorithm (which defines the loss function and update rule), and the environment interface (Gymnasium) are cleanly separated. You can swap custom network architectures into any algorithm without touching the training loop. You can change environments without modifying policy code.

**Stability over novelty.** The core SB3 package only includes algorithms that are well-understood, thoroughly tested, and widely used. Newer or experimental algorithms live in SB3-Contrib. This means that when you use an algorithm from the main package, you can trust that it has been validated against published benchmarks.

---

## Supported Algorithms

### On-Policy Methods

**PPO (Proximal Policy Optimization).** The default choice for most RL tasks. PPO uses a clipped surrogate objective to constrain policy updates, achieving trust-region-like stability with first-order optimization. SB3's PPO implementation supports both MLP and CNN policies, Generalized Advantage Estimation (GAE), entropy bonus, value function clipping, and gradient norm clipping. It runs naturally with vectorized environments for parallel data collection.

**A2C (Advantage Actor-Critic).** The synchronous variant of A3C. A2C collects a fixed number of steps from each vectorized environment, computes advantages, and performs a single gradient update. It is simpler and faster per update than PPO but generally less stable. A2C is a good choice when wall-clock speed matters more than sample efficiency, or when you want a lightweight baseline.

### Off-Policy Methods

**SAC (Soft Actor-Critic).** An entropy-regularized actor-critic method for continuous action spaces. SAC maximizes both expected return and policy entropy, which encourages exploration and makes training robust to hyperparameter choices. SB3's SAC implementation includes automatic entropy coefficient tuning, twin Q-networks, and replay buffer management. SAC is often the best off-policy choice for continuous control tasks.

**TD3 (Twin Delayed DDPG).** An improvement over DDPG that addresses function approximation errors through clipped double Q-learning, delayed policy updates, and target policy smoothing. TD3 is deterministic (no entropy term) and tends to produce sharper, more exploitative policies than SAC. It works well in environments where exploration is less of a challenge.

**DDPG (Deep Deterministic Policy Gradient).** The original continuous-action off-policy actor-critic. Largely superseded by TD3 and SAC, DDPG remains in SB3 for historical completeness and for environments where its specific characteristics are desired. In practice, TD3 or SAC should be preferred for new projects.

**DQN (Deep Q-Network).** The foundational value-based method for discrete action spaces. SB3's DQN includes double DQN, dueling network architecture support, prioritized experience replay, and target network updates. DQN is the default choice when your action space is discrete and you want off-policy sample efficiency.

### Goal-Conditioned Replay

**HER (Hindsight Experience Replay).** Not an algorithm per se, but a replay strategy that can be combined with any off-policy algorithm (SAC, TD3, DDPG, DQN). HER addresses sparse reward problems by retroactively relabeling failed trajectories with achieved goals, dramatically improving sample efficiency in goal-conditioned tasks like robotic manipulation. In SB3, HER is implemented as a replay buffer wrapper that can be passed to any compatible algorithm.

### SB3-Contrib Algorithms

SB3-Contrib extends the core library with algorithms that are useful but either newer, less widely tested, or more specialized:

- **Recurrent PPO (PPO-LSTM):** PPO with LSTM-based policies for partially observable environments where the agent needs memory of past observations.
- **TQC (Truncated Quantile Critics):** A distributional RL extension of SAC that uses quantile regression for value estimation, often outperforming SAC on complex continuous control benchmarks.
- **QR-DQN (Quantile Regression DQN):** A distributional variant of DQN that models the full return distribution rather than just the expected value.
- **Maskable PPO:** PPO with action masking support for environments where only a subset of actions is valid at each step, common in board games, scheduling, and combinatorial optimization.
- **CrossQ:** A recent method that eliminates the need for target networks in off-policy learning, reducing hyperparameter sensitivity.
- **ARS (Augmented Random Search):** A simple, derivative-free optimization method that can be surprisingly effective for continuous control tasks.
- **TRPO (Trust Region Policy Optimization):** The predecessor to PPO, included for completeness and for research comparisons.

---

## The Unified API

### Model Creation and Training

SB3's API is designed so that the same three-line pattern works across all algorithms:

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=100_000)
```

The first argument is the policy type. "MlpPolicy" uses a fully connected network. "CnnPolicy" uses a convolutional network for image observations. "MultiInputPolicy" handles dictionary observation spaces. The second argument is the environment, which can be a Gymnasium environment instance, an environment ID string, or a VecEnv.

All algorithm-specific hyperparameters are passed as keyword arguments to the constructor:

```python
model = PPO(
    "MlpPolicy",
    "CartPole-v1",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)
```

For off-policy algorithms, you configure the replay buffer size, learning starts, and update frequency:

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    "Pendulum-v1",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    verbose=1,
)
model.learn(total_timesteps=500_000)
```

### Prediction and Inference

After training, model.predict() returns actions given observations:

```python
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

The deterministic parameter controls whether the policy samples from its action distribution (stochastic, for exploration) or returns the mode/mean (deterministic, for evaluation and deployment). The _states return value carries recurrent hidden states for LSTM policies and is None for feedforward policies.

### Saving and Loading

Models are saved and loaded with single method calls:

```python
model.save("ppo_cartpole")
loaded_model = PPO.load("ppo_cartpole")
```

This serializes the policy network weights, optimizer state, and all hyperparameters. For off-policy algorithms, the replay buffer can be saved separately:

```python
model.save_replay_buffer("sac_replay_buffer")
model.load_replay_buffer("sac_replay_buffer")
```

When loading a model for continued training, you must re-attach an environment:

```python
loaded_model = PPO.load("ppo_cartpole")
loaded_model.set_env(env)
loaded_model.learn(total_timesteps=50_000)
```

For deployment without SB3 as a dependency, you can export the policy to ONNX or extract the PyTorch module directly and run inference with pure PyTorch.

---

## Custom Environments with Gymnasium

SB3 works exclusively with the Gymnasium API (the maintained fork of OpenAI Gym). Any environment that implements the Gymnasium interface can be used with SB3 without modification.

### Environment Structure

A minimal custom environment looks like this:

```python
import gymnasium as gym
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.astype(np.float32), {}

    def step(self, action):
        # Environment dynamics here
        reward = 1.0
        terminated = False
        truncated = False
        return self.state.astype(np.float32), reward, terminated, truncated, {}
```

SB3 includes a check_env() utility that validates your environment against the Gymnasium specification, catching common mistakes like incorrect observation shapes, wrong dtypes, or missing return values:

```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv()
check_env(env)  # Raises warnings or errors for API violations
```

### Dictionary Observation Spaces

For environments with multiple observation modalities (images plus proprioceptive state, for example), SB3 supports dictionary observation spaces:

```python
self.observation_space = gym.spaces.Dict({
    "image": gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
    "vector": gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
})
```

Use "MultiInputPolicy" to handle these spaces. SB3 automatically creates separate network branches for each observation key and concatenates their outputs.

---

## Vectorized Environments

Vectorized environments (VecEnv) are SB3's mechanism for running multiple environment instances in parallel. This is critical for on-policy methods like PPO that need large batches of experience, and beneficial for off-policy methods that can fill their replay buffers faster.

### VecEnv Types

**SubprocVecEnv** runs each environment in a separate process, providing true parallelism at the cost of inter-process communication overhead:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _init

env = SubprocVecEnv([make_env("CartPole-v1", i) for i in range(8)])
```

**DummyVecEnv** runs all environments sequentially in the same process. It has no parallelism but avoids serialization overhead, making it faster for environments that are very cheap to step:

```python
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
```

### VecEnv Wrappers

SB3 provides a suite of wrappers that operate on vectorized environments:

- **VecNormalize:** Running normalization of observations and rewards. Essential for environments with unbounded or poorly scaled observation spaces. Saves and loads normalization statistics alongside the model.
- **VecFrameStack:** Stacks consecutive observation frames, a standard technique for Atari and other vision-based environments where temporal information matters.
- **VecTransposeImage:** Converts image observations from HWC (height, width, channels) to CHW format expected by PyTorch convolutional layers.
- **VecMonitor:** Records episode statistics (rewards, lengths) for logging and evaluation.
- **VecCheckNan:** Detects NaN values in observations and rewards, useful for debugging unstable environments.

A typical wrapper chain for an Atari environment:

```python
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper

env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=8)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)
```

---

## Callbacks, Logging, and Evaluation

### Callbacks

Callbacks are the primary extension point for SB3's training loop. They allow you to execute custom code at specific points during training without modifying the algorithm implementation. SB3 provides several built-in callbacks and a base class for custom ones.

**EvalCallback** periodically evaluates the agent on a separate evaluation environment and optionally saves the best model:

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_env = gym.make("CartPole-v1")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model.learn(total_timesteps=500_000, callback=eval_callback)
```

**CheckpointCallback** saves the model at regular intervals:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./logs/checkpoints",
    name_prefix="ppo_cartpole",
)
```

**StopTrainingOnRewardThreshold** terminates training early when the agent reaches a target performance, typically combined with EvalCallback:

```python
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=495)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=10_000,
)
```

**CallbackList** composes multiple callbacks:

```python
from stable_baselines3.common.callbacks import CallbackList

callback = CallbackList([eval_callback, checkpoint_callback])
model.learn(total_timesteps=500_000, callback=callback)
```

For custom behavior, subclass BaseCallback and override _on_step(), _on_rollout_start(), _on_rollout_end(), or _on_training_end():

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Access self.model, self.logger, self.num_timesteps, self.locals
        if self.num_timesteps % 10_000 == 0:
            self.logger.record("custom/metric", some_value)
        return True  # Return False to stop training
```

### Logging

SB3 logs training metrics to stdout by default and supports TensorBoard, CSV, and custom loggers:

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1", tensorboard_log="./tb_logs/")
model.learn(total_timesteps=100_000, tb_log_name="ppo_run_1")
```

Then visualize with:

```
tensorboard --logdir ./tb_logs/
```

Standard logged metrics include episode reward, episode length, policy loss, value loss, entropy, explained variance, learning rate, and clip fraction (for PPO). You can log custom metrics through the callback system using self.logger.record().

For integration with Weights and Biases, wrap the training with wandb's SB3 integration or log metrics from a custom callback.

### Evaluation Utilities

Beyond EvalCallback, SB3 provides the evaluate_policy() function for one-off evaluation:

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=100, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

---

## Hyperparameter Tuning with RL Zoo and Optuna

### RL Baselines3 Zoo

The RL Baselines3 Zoo is a companion training framework that wraps SB3 with a command-line interface for training, evaluation, hyperparameter tuning, and benchmarking. It ships with pre-tuned hyperparameter configurations for dozens of standard environments across all SB3 algorithms. These serve as strong starting points and save significant experimentation time.

Training with RL Zoo:

```bash
python -m rl_zoo3.train --algo ppo --env CartPole-v1 --tensorboard-log ./tb_logs/
```

Evaluating a trained agent:

```bash
python -m rl_zoo3.enjoy --algo ppo --env CartPole-v1 --folder logs/
```

### Optuna Integration

RL Zoo integrates Optuna for automated hyperparameter optimization. You define a search space in a YAML configuration file, and RL Zoo runs Optuna trials with pruning and parallel evaluation:

```bash
python -m rl_zoo3.train --algo ppo --env CartPole-v1 \
    -optimize --n-trials 100 --n-jobs 4 \
    --sampler tpe --pruner median
```

The search space is defined in a hyperparameter YAML file:

```yaml
CartPole-v1:
  n_timesteps: 100000
  policy: MlpPolicy
  n_steps:
    type: categorical
    categories: [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  gamma:
    type: float
    low: 0.9
    high: 0.9999
    log: true
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  ent_coef:
    type: float
    low: 0.00000001
    high: 0.1
    log: true
  clip_range:
    type: categorical
    categories: [0.1, 0.2, 0.3, 0.4]
  gae_lambda:
    type: float
    low: 0.8
    high: 1.0
  batch_size:
    type: categorical
    categories: [8, 16, 32, 64, 128, 256, 512]
```

Optuna's TPE sampler and median pruner work well for RL hyperparameter search, where individual trials are expensive. The pruner terminates poorly performing trials early based on intermediate evaluation results, saving significant compute.

### Practical Hyperparameter Tuning Advice

Hyperparameter sensitivity varies significantly across RL algorithms. Some practical guidelines:

- **PPO** is relatively robust. Start with the defaults (learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2, gae_lambda=0.95). The most impactful parameters to tune are n_steps, learning_rate, and the network architecture.
- **SAC** is also forgiving. The automatic entropy tuning eliminates one of the most sensitive hyperparameters. Focus on learning_rate, buffer_size, and batch_size.
- **DQN** requires more care. The exploration schedule (epsilon decay), target network update frequency, and learning rate interact in subtle ways.
- **TD3** and **DDPG** are the most sensitive to hyperparameters. Use the RL Zoo defaults as starting points and tune from there.

For all algorithms, the observation and reward scaling can matter as much as the algorithm hyperparameters. VecNormalize is often the single most impactful change you can make.

---

## Custom Network Architectures

SB3 allows you to customize the policy and value network architectures through the policy_kwargs parameter:

```python
model = PPO(
    "MlpPolicy",
    "CartPole-v1",
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    ),
)
```

The net_arch parameter accepts a dictionary with "pi" (policy) and "vf" (value function) keys, each specifying the hidden layer sizes. For shared layers followed by separate heads, use a list with a dict at the end:

```python
policy_kwargs = dict(
    net_arch=[256, dict(pi=[128, 64], vf=[128, 64])],
    activation_fn=torch.nn.Tanh,
)
```

For more complex architectures (attention layers, residual connections, custom feature extractors for image observations), you can subclass the feature extractor:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        # Compute shape by doing a forward pass
        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None]
            ).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    ),
)
```

---

## Comparison with Other RL Frameworks

### SB3 vs. RLlib

RLlib (part of the Ray ecosystem) is the primary alternative for production RL workloads. The two libraries serve different niches:

**Scale.** RLlib is designed for distributed training across clusters of machines. It leverages Ray for parallelism and can scale to hundreds of workers. SB3 is single-machine, using Python multiprocessing (SubprocVecEnv) for parallelism. If your workload requires multi-node distributed training, RLlib is the clear choice.

**Multi-agent support.** RLlib has first-class support for multi-agent environments with independent, cooperative, and competitive learning. SB3 is strictly single-agent. For multi-agent tasks, you either use RLlib or combine SB3 with PettingZoo through wrappers.

**Complexity.** RLlib is a large, complex framework with a steep learning curve. Its configuration system, training API, and abstraction layers add significant overhead for simple tasks. SB3 prioritizes simplicity: three lines of code to train a PPO agent. For single-agent experiments on a single machine, SB3 is dramatically faster to iterate with.

**Algorithm correctness.** SB3's implementations are carefully validated against published results and have high test coverage. RLlib covers more algorithms but has historically been more prone to subtle implementation bugs, in part due to the complexity of its abstraction layers.

**Ecosystem.** RLlib benefits from integration with the broader Ray ecosystem (Ray Tune for hyperparameter search, Ray Serve for deployment). SB3 integrates with the PyTorch ecosystem directly and uses Optuna through RL Zoo.

Choose SB3 when you want to prototype quickly, need reliable implementations of core algorithms, and are working on a single machine. Choose RLlib when you need distributed training, multi-agent support, or integration with the Ray ecosystem.

### SB3 vs. CleanRL

CleanRL takes the opposite design philosophy from SB3. Instead of providing a library with reusable abstractions, CleanRL implements each algorithm in a single, self-contained Python file with no hidden dependencies between modules.

**Transparency.** CleanRL is designed for researchers who want to see exactly what happens at every step. There are no base classes, no callback abstractions, no wrappers -- just a flat script. This makes it excellent for learning, debugging, and modifying algorithms. SB3's abstractions are helpful for users but make it harder to trace exactly what happens during a training step.

**Reproducibility.** CleanRL provides tracked experiment results for every algorithm on standard benchmarks, with detailed documentation of implementation details and differences from published papers. SB3 validates against benchmarks but does not ship a centralized benchmark suite.

**Usability.** SB3 wins for practical use. Its reusable components, consistent API, save/load workflow, callback system, and VecEnv wrappers save substantial engineering effort. CleanRL requires you to build all of this infrastructure yourself or copy-paste from example files.

**Scope.** CleanRL covers a broader set of algorithms (including multi-agent and environment-specific implementations) as single-file implementations. SB3 covers fewer algorithms but wraps them in production-ready interfaces.

Choose CleanRL when you want to understand an algorithm's implementation in complete detail, when you need to make substantial modifications to the training loop, or for educational purposes. Choose SB3 when you need a reliable tool that handles the engineering for you.

### SB3 vs. SBX (Stable Baselines Jax)

SBX is a JAX-based sibling of SB3 that provides the same API but uses JAX for computation. SBX can achieve roughly 20x faster training through JAX's JIT compilation and XLA optimization. However, it covers fewer algorithms and has a smaller community. Consider SBX when training speed is critical and the algorithms you need are supported.

---

## Production Considerations

### Deterministic Evaluation

For reproducible evaluation, seed everything:

```python
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

model = PPO("MlpPolicy", "CartPole-v1", seed=42)
```

Use deterministic=True in model.predict() for deployment to avoid stochastic action selection.

### Model Export

For deployment in environments where SB3 is not available, extract the policy network and export it:

```python
# Extract and save as standard PyTorch
policy = model.policy
torch.save(policy.state_dict(), "policy_weights.pt")

# Or export to ONNX
dummy_input = torch.randn(1, *env.observation_space.shape)
torch.onnx.export(policy, dummy_input, "policy.onnx")
```

### Monitoring Training Health

Watch these metrics during training to detect problems:

- **Explained variance:** Should be positive and ideally above 0.5 for on-policy methods. Near-zero or negative values indicate the value function is not learning.
- **Policy loss magnitude:** Sudden spikes indicate instability. For PPO, also monitor clip fraction -- values consistently above 0.3 suggest the learning rate is too high.
- **Entropy:** Should decrease gradually. A sudden collapse to near-zero means the policy has become deterministic prematurely, likely due to insufficient entropy bonus.
- **Episode reward trend:** The primary metric. Look for monotonic improvement with variance. Sudden drops followed by no recovery indicate catastrophic forgetting.

---

## Putting It All Together

A complete training pipeline with SB3 typically looks like this:

```python
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create vectorized training environment
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _init

train_env = SubprocVecEnv([make_env("HalfCheetah-v4", i) for i in range(8)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
train_env = VecMonitor(train_env)

# 2. Create separate evaluation environment
eval_env = SubprocVecEnv([make_env("HalfCheetah-v4", i + 100) for i in range(4)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# 3. Configure model
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    ),
    tensorboard_log="./tb_logs/",
    seed=42,
    verbose=1,
)

# 4. Set up callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./logs/checkpoints",
)
callback = CallbackList([eval_callback, checkpoint_callback])

# 5. Train
model.learn(total_timesteps=2_000_000, callback=callback)

# 6. Save final model and normalization stats
model.save("ppo_halfcheetah_final")
train_env.save("vecnormalize_stats.pkl")

# 7. Load and evaluate
loaded_model = PPO.load("ppo_halfcheetah_final")
loaded_env = VecNormalize.load("vecnormalize_stats.pkl", eval_env)
mean_reward, std_reward = evaluate_policy(loaded_model, loaded_env, n_eval_episodes=100)
print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")
```

This pattern -- vectorized environments with normalization, periodic evaluation with best model saving, checkpointing, TensorBoard logging, and a clean save/load workflow -- covers the vast majority of single-agent RL projects. SB3 makes each of these steps straightforward while leaving room for customization through callbacks, custom network architectures, and environment wrappers.
