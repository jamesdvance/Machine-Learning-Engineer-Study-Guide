# CleanRL

## Summary

CleanRL is an open-source deep reinforcement learning library that provides high-quality, single-file implementations of RL algorithms. Unlike modular frameworks such as Stable Baselines3 or RLlib, CleanRL puts every detail of an algorithm variant into one standalone file with no hidden abstractions, no base classes to inherit from, and no multi-file dependency chains. The project was created by Shengyi Huang and published in the Journal of Machine Learning Research in 2022. It is designed for learning, understanding, and prototyping RL algorithms rather than deploying them into production systems.

Key points to remember:

- Each algorithm variant lives in a single self-contained file, typically 300 to 500 lines of code
- CleanRL is not meant to be imported as a library; you clone the repo and read or modify individual files directly
- Supports 13+ algorithms across policy gradient, value-based, and actor-critic families (PPO, DQN, C51, DDPG, TD3, SAC, PPG, RND, and others)
- Works with diverse environments including Classic Control, Atari, MuJoCo, Procgen, PettingZoo, and Isaac Gym
- Integrated with Weights and Biases for experiment tracking, metric logging, and video capture via a simple --track flag
- The Open RL Benchmark project provides reproducible W&B reports comparing CleanRL against other libraries
- Best suited for understanding algorithm internals, prototyping research ideas, and educational purposes
- Not designed for production deployment; choose Stable Baselines3 or RLlib for that

## Philosophy: Single-File Implementations with No Abstractions

The central design principle of CleanRL is radical simplicity. In most RL libraries, an algorithm like PPO is spread across multiple files: a base agent class, a policy network module, a rollout buffer, an update function, environment wrappers, and configuration objects. Understanding how everything fits together requires tracing execution across many layers of abstraction. CleanRL rejects this entirely.

In CleanRL, `ppo_atari.py` is a single file of roughly 340 lines that contains everything needed to train PPO on Atari games: argument parsing, environment setup, neural network definition, rollout collection, advantage estimation, and the PPO update loop. There are no imports from internal CleanRL modules. The only dependencies are standard libraries like PyTorch, Gymnasium, and NumPy. If you want to understand how PPO works with Atari preprocessing, you read one file from top to bottom.

This approach has concrete consequences. There is no code reuse between algorithm files. The DQN implementation duplicates environment setup code that also appears in the PPO implementation. The network architectures are defined inline rather than pulled from a shared module. This is intentional. The project trades code reuse for clarity, operating on the principle that understanding an algorithm should never require navigating to a different file.

Each variant of an algorithm gets its own file. PPO alone has over twelve variants: `ppo.py` for classic control, `ppo_atari.py` for Atari games, `ppo_continuous_action.py` for continuous action spaces, `ppo_atari_lstm.py` for recurrent policies, `ppo_atari_envpool.py` for high-performance training, and JAX-based versions for hardware acceleration. These are not parameterized versions of a single implementation. They are independent files that you can read, modify, and run without worrying about breaking other variants.

## Why CleanRL Exists

RL algorithm implementations in production-grade libraries are notoriously difficult to understand. A researcher trying to verify how generalized advantage estimation (GAE) interacts with value function clipping in PPO might need to trace through five or six files in a modular library, mentally reconstructing the data flow across abstract interfaces. Implementation details that matter enormously for performance -- things like observation normalization, reward scaling, orthogonal initialization, and advantage normalization -- are often buried in base classes or configuration defaults that are easy to miss.

CleanRL was created to solve this problem. The project originated from Shengyi Huang's work at Costa Huang's research group, with the explicit goal of producing RL implementations that a researcher could read and fully understand in a single sitting. The JMLR paper accompanying the release documented how specific implementation details in PPO can cause large performance differences, and how those details are often obscured in modular codebases.

The library also addresses reproducibility. Because each file is self-contained, reproducing a result means running a single script with specified arguments. There are no hidden configuration files, no framework-level defaults that change between versions, and no implicit behavior from base classes. The full specification of what happens during training is visible in the file you execute.

## Supported Algorithms

CleanRL implements algorithms across the major families of deep reinforcement learning.

### Policy Gradient Methods

**Proximal Policy Optimization (PPO)** is the most extensively covered algorithm, with variants for discrete actions (Classic Control, Atari), continuous actions (MuJoCo), recurrent policies (LSTM-based), multi-GPU training, multi-agent settings (PettingZoo), and JAX-based implementations. The Atari variant with EnvPool achieves a 3-4x speedup over standard Gymnasium environments.

**Phasic Policy Gradient (PPG)** is implemented for Procgen environments, which test generalization to procedurally generated levels.

### Value-Based Methods

**Deep Q-Network (DQN)** covers both classic control and Atari environments, with PyTorch and JAX variants.

**Categorical DQN (C51)** implements distributional reinforcement learning, where the agent learns a distribution over returns rather than a single expected value.

**Rainbow** combines multiple improvements to DQN (prioritized replay, distributional values, noisy networks) in a single implementation.

### Actor-Critic Methods

**Deep Deterministic Policy Gradient (DDPG)** targets continuous action spaces with deterministic policy learning, available in both PyTorch and JAX.

**Twin Delayed DDPG (TD3)** addresses overestimation bias in DDPG through clipped double Q-learning, delayed policy updates, and target policy smoothing.

**Soft Actor-Critic (SAC)** implements maximum entropy reinforcement learning for both continuous control and Atari environments.

### Exploration and Specialized Methods

**Random Network Distillation (RND)** provides curiosity-driven exploration for hard exploration problems.

**QDagger** combines DQN with imitation learning for Atari environments.

**Parallel Q Network (PQN)** and **Robust Policy Optimization (RPO)** round out the collection with more recent algorithmic contributions.

## Supported Environments

CleanRL implementations are benchmarked and tested across a range of environment suites:

- **Classic Control** (CartPole, LunarLander, MountainCar): Simple environments useful for debugging and quick iteration
- **Atari 2600** (34+ games via Arcade Learning Environment): The standard benchmark for discrete-action RL, with proper preprocessing (frame stacking, grayscale conversion, frame skipping)
- **MuJoCo** (HalfCheetah, Hopper, Walker2d, Ant, Humanoid): Continuous control physics simulation, the standard benchmark for actor-critic methods
- **Procgen** (16 procedurally generated games): Tests generalization ability since train and test levels differ
- **PettingZoo**: Multi-agent environments including cooperative and competitive Atari games
- **Isaac Gym**: GPU-accelerated physics simulation for robotics tasks
- **Memory Gym**: Environments that require memory and recurrent policies
- **EnvPool**: High-performance C++-based environment implementations that replace standard Gymnasium wrappers for Atari and Classic Control, providing significant speedups

## Weights and Biases Integration

CleanRL has first-class integration with Weights and Biases (W&B) for experiment tracking. Every implementation accepts a `--track` flag that enables logging to W&B with no additional code changes:

```bash
# Login to W&B (one-time setup)
wandb login

# Run PPO on Atari with full tracking
python cleanrl/ppo_atari.py \
    --env-id BreakoutNoFrameskip-v4 \
    --track \
    --wandb-project-name my-rl-experiments
```

When tracking is enabled, CleanRL logs:

- **Training metrics**: episodic returns, episode lengths, learning rate, loss components (policy loss, value loss, entropy), explained variance, and clip fractions
- **Hyperparameters**: every command-line argument is recorded, ensuring full reproducibility
- **Videos**: periodic recordings of the agent's gameplay, viewable directly in the W&B dashboard
- **System metrics**: GPU utilization, memory usage, and throughput (steps per second)
- **Dependencies**: the exact package versions used for the run

The Open RL Benchmark project (benchmark.cleanrl.dev) takes this further by publishing interactive W&B reports that compare CleanRL implementations against Stable Baselines3, OpenAI Baselines, and jaxrl across standard benchmarks. These reports let you verify that a CleanRL implementation matches the performance of reference implementations, and they serve as a shared resource for the RL community to compare results.

The W&B integration is implemented inline in each file, typically in 10 to 15 lines of code. There is no separate tracking module. This means you can see exactly what is being logged and modify it without hunting through a logging framework.

## Installation and Usage

CleanRL requires Python 3.7.1 through 3.10 and recommends the `uv` package manager:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl
uv pip install .
```

Optional dependencies are installed separately based on what environments you need:

```bash
# For Atari environments
uv pip install ".[atari]"

# For MuJoCo environments
uv pip install ".[mujoco]"

# For Procgen environments
uv pip install ".[procgen]"

# For JAX-based implementations
uv pip install ".[jax]"
```

Running an algorithm is a single command:

```bash
# Train PPO on CartPole
python cleanrl/ppo.py --seed 1 --env-id CartPole-v1 --total-timesteps 50000

# Train DQN on Atari Breakout
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4 --total-timesteps 10000000

# Train SAC on MuJoCo HalfCheetah
python cleanrl/sac_continuous_action.py --env-id HalfCheetah-v4
```

Every implementation uses argparse for configuration, so you can see all available options with `--help`:

```bash
python cleanrl/ppo_atari.py --help
```

TensorBoard logging is enabled by default. After training, visualize results with:

```bash
tensorboard --logdir runs
```

## Using CleanRL for Learning

CleanRL is one of the best resources available for understanding how RL algorithms actually work in practice. The recommended approach for learning is:

1. **Start with the paper or a conceptual explanation** of the algorithm you want to understand. For PPO, read the original paper by Schulman et al. or a tutorial that covers the theory.

2. **Read the simplest variant file from top to bottom.** For PPO, start with `ppo.py` (Classic Control), not `ppo_atari.py`. The Classic Control version has fewer environment-specific details and lets you focus on the algorithm itself.

3. **Trace the data flow.** Identify where observations are collected, how advantages are computed, how the policy and value networks are updated, and how the clipping mechanism works. In CleanRL, all of this is in one file, so you can follow it linearly.

4. **Run the code and examine the logs.** Use TensorBoard or W&B to watch training curves. Modify hyperparameters and observe how they affect training. Change the clipping epsilon, the number of epochs, or the GAE lambda and see what happens.

5. **Compare variants.** Once you understand the Classic Control version, read the Atari version to see what changes: convolutional networks, frame stacking, different hyperparameters, and reward clipping. Then read the continuous action version to see how the policy distribution changes from categorical to Gaussian.

This approach works because CleanRL does not hide anything behind abstractions. When you read the code, you are reading the actual algorithm, not a framework's interpretation of it.

## Using CleanRL as a Research Starting Point

CleanRL is well-suited as a foundation for RL research prototyping. The workflow is:

1. **Identify the closest existing variant** to your research idea. If you want to modify PPO's objective function, start with `ppo.py` or `ppo_atari.py`.

2. **Copy the file** and rename it. Since each file is self-contained, your copy is immediately runnable and has no dependencies on other CleanRL files.

3. **Make your modifications directly.** Add a new loss term, change the network architecture, modify the rollout collection, or add a new exploration mechanism. Because the file is flat and has no abstractions, you know exactly where to make changes.

4. **Use the W&B integration** to track your experiments and compare against the baseline. The Open RL Benchmark provides reference numbers you can compare against.

This workflow is faster for small-to-medium modifications than working with a modular framework, because you never need to understand or subclass an abstract interface. The tradeoff is that if your modification needs to be applied across many algorithm variants, you will need to modify each file independently.

## Comparison with Stable Baselines3

Stable Baselines3 (SB3) is a modular, well-tested RL library designed for ease of use and reliable results. The two libraries serve fundamentally different purposes.

**Architecture**: SB3 uses a class hierarchy with base classes for on-policy and off-policy algorithms, separate modules for policies, replay buffers, and feature extractors. CleanRL uses flat, single-file implementations with no inheritance.

**API design**: SB3 provides a scikit-learn-style API where you instantiate a model, call `.learn()`, and call `.predict()`. CleanRL has no API; you run a script.

```python
# Stable Baselines3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=50000)
```

```bash
# CleanRL
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000
```

**Customization**: In SB3, customization happens through callbacks, custom policies, and subclassing. You work within the framework's extension points. In CleanRL, you edit the algorithm file directly. SB3 is easier for standard modifications (custom reward shaping, different network sizes). CleanRL is easier for non-standard modifications (changing the loss function, modifying the rollout logic, adding auxiliary tasks).

**Production readiness**: SB3 supports model saving and loading, evaluation utilities, and integration with standard deployment tools. CleanRL does not provide these features.

**When to choose SB3**: You want to train a standard RL algorithm with minimal code, you need model saving and loading, you want callback-based customization, or you are deploying a trained agent.

**When to choose CleanRL**: You want to understand how the algorithm works, you need to make non-standard modifications to the training loop, or you are prototyping a research idea.

## Comparison with RLlib

RLlib is a scalable, distributed RL library built on Ray. It targets large-scale training and production deployment. The gap between RLlib and CleanRL is even wider than between SB3 and CleanRL.

**Scale**: RLlib is designed for distributed training across many machines. It handles parallel environment execution, distributed replay buffers, and multi-GPU policy optimization. CleanRL runs on a single machine, though some variants support multi-GPU training.

**Complexity**: RLlib has a large codebase with extensive abstractions for policies, trainers, execution plans, and resource management. Understanding how a specific algorithm works in RLlib requires navigating many layers. CleanRL lets you read the entire algorithm in one file.

**Configuration**: RLlib uses a configuration system with hundreds of parameters. CleanRL uses command-line arguments, typically 15 to 25 per algorithm.

**Ecosystem**: RLlib integrates with Ray Tune for hyperparameter search, Ray Serve for model serving, and Ray's distributed computing primitives. CleanRL integrates with W&B for tracking and TensorBoard for visualization.

**When to choose RLlib**: You need distributed training, you are deploying RL at scale, you need hyperparameter tuning across many runs, or you are working in a production environment that already uses Ray.

**When to choose CleanRL**: You want to understand the algorithm, you are doing single-machine research, or you need fine-grained control over every aspect of training.

## When to Choose CleanRL

CleanRL is the right choice in specific scenarios:

**Learning RL algorithms**: If your goal is to understand how PPO, SAC, or DQN actually work at the implementation level, CleanRL is the best available resource. The single-file structure means you can read the complete algorithm without navigating a framework.

**Prototyping research ideas**: When you want to test a modification to an existing algorithm, CleanRL lets you copy a file and start editing immediately. There are no framework constraints on what you can change.

**Reproducing results**: The combination of single-file implementations, explicit hyperparameters, and W&B integration makes it straightforward to reproduce published results or verify that your understanding of an algorithm is correct.

**Teaching**: CleanRL is widely used in university courses and tutorials because students can read the entire implementation and understand what every line does.

CleanRL is not the right choice when:

**You need production deployment**: There is no model serving, no model saving/loading API, no checkpoint management, and no deployment tooling. Use SB3 or RLlib instead.

**You need distributed training at scale**: CleanRL runs on a single machine. For multi-node training, use RLlib.

**You want a high-level API**: If you want to train a standard algorithm with three lines of code and do not care about implementation details, use SB3.

**You are building a product**: CleanRL is a research and education tool. It is not designed for or tested as infrastructure that other code depends on.

## Implementation Details That Matter

One of CleanRL's contributions is documenting implementation details that significantly affect performance but are often undocumented in other libraries. The JMLR paper and the codebase highlight several:

- **Advantage normalization**: Normalizing advantages to have zero mean and unit variance within each minibatch can substantially affect PPO performance. This is a single line of code but is not always present or documented in other implementations.

- **Orthogonal initialization**: Initializing network weights with orthogonal matrices and specific gain values (sqrt(2) for hidden layers, 0.01 for the policy head, 1.0 for the value head) follows the OpenAI Baselines convention and affects training stability.

- **Value function clipping**: Clipping the value function update in PPO is mentioned in the original paper but its effect is debated. CleanRL implements it explicitly so you can enable or disable it.

- **Learning rate annealing**: Linearly decaying the learning rate over the course of training is used in most competitive PPO implementations but is not part of the core algorithm.

- **Global gradient clipping**: Clipping the global norm of gradients (typically to 0.5) prevents large updates that destabilize training.

These details are all visible in the code because nothing is hidden behind a base class. A researcher can see exactly which details are present, understand their effect, and modify them as needed.

## Project Structure

The CleanRL repository is organized as follows:

```
cleanrl/
    ppo.py                    # PPO for Classic Control
    ppo_atari.py              # PPO for Atari
    ppo_continuous_action.py  # PPO for continuous actions
    ppo_atari_lstm.py         # PPO with LSTM for Atari
    ppo_atari_envpool.py      # PPO with EnvPool (fast)
    dqn.py                    # DQN for Classic Control
    dqn_atari.py              # DQN for Atari
    c51.py                    # C51 for Classic Control
    c51_atari.py              # C51 for Atari
    ddpg_continuous_action.py # DDPG for continuous actions
    td3_continuous_action.py  # TD3 for continuous actions
    sac_continuous_action.py  # SAC for continuous actions
    sac_atari.py              # SAC for Atari
    ppg_procgen.py            # PPG for Procgen
    rnd_ppo.py                # RND with PPO
    ...
    cleanrl_utils/            # Minimal utilities (evals, benchmarking)
```

The `cleanrl_utils` directory contains lightweight utilities for evaluation and benchmarking, but the algorithm implementations do not depend on them. Each `.py` file in the `cleanrl/` directory is an independent, runnable implementation.

## Further Reading

- CleanRL Documentation: https://docs.cleanrl.dev
- GitHub Repository: https://github.com/vwxyzjn/cleanrl
- JMLR Paper: https://www.jmlr.org/papers/v23/21-1342.html
- ArXiv Preprint: https://arxiv.org/abs/2111.08819
- Open RL Benchmark: https://benchmark.cleanrl.dev
