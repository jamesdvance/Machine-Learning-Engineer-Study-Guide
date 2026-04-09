# Reinforcement Learning Frameworks

## Summary

Reinforcement learning frameworks provide the software infrastructure needed to train RL agents: algorithm implementations, environment interfaces, experience collection pipelines, and logging utilities. Unlike supervised learning where a single framework like PyTorch or TensorFlow dominates the training loop, RL requires orchestrating interactions between agents and environments, managing replay buffers, computing advantage estimates, and often parallelizing across many environment instances. This orchestration complexity has produced a diverse ecosystem of frameworks, each making different tradeoffs between ease of use, transparency, scalability, and production readiness.

The three most important RL frameworks for a practicing ML engineer are Stable Baselines3, RLlib, and CleanRL. They span a spectrum from high-level abstraction to radical transparency, and from single-machine simplicity to cluster-scale distributed training. Understanding their strengths, limitations, and intended audiences is essential for choosing the right tool for a given problem.

- Stable Baselines3 (SB3) provides production-quality PyTorch implementations of core algorithms behind a consistent, high-level API. It is the default choice for single-agent, single-machine RL workloads where reliability and rapid iteration matter most.
- RLlib, built on the Ray distributed computing framework, is designed for scalable, distributed training and is the standard choice for multi-agent RL, cluster-scale training, and production deployments that require fault tolerance and serving infrastructure.
- CleanRL implements each algorithm in a single, self-contained file with no abstractions. It is the best tool for understanding algorithm internals, prototyping research modifications, and educational purposes.

Beyond these three, the broader ecosystem includes Gymnasium (the universal environment interface), environment suites like MuJoCo and Atari, and additional frameworks such as TF-Agents, Tianshou, and rlax that serve specific niches. This chapter provides an overview of the landscape, compares the major frameworks across key dimensions, and offers guidance for selecting the right framework for your project.

---

## The RL Framework Landscape

### Why RL Needs Specialized Frameworks

In supervised learning, the training loop is straightforward: load a batch of data, compute a forward pass, calculate a loss, and backpropagate. The framework (PyTorch, TensorFlow) handles gradient computation and parameter updates. RL training is fundamentally more complex because the data generation process is coupled to the model being trained. The agent must interact with an environment, collect experience, and use that experience to update its policy, which then changes the distribution of future experience. This creates a loop that requires managing environment instances, rollout buffers, exploration strategies, and synchronization between data collection and gradient updates.

Additionally, RL algorithms differ from each other more dramatically than supervised learning optimizers do. On-policy methods like PPO collect fresh experience for every update and discard it afterward. Off-policy methods like SAC store experience in replay buffers and reuse it across many updates. Model-based methods learn a dynamics model and plan within it. Multi-agent methods must coordinate multiple policies interacting in a shared environment. Each of these patterns requires different infrastructure, which is why RL frameworks tend to be more opinionated and complex than supervised learning frameworks.

### The Three Pillars

The three frameworks covered in this guide represent three distinct design philosophies that together cover the vast majority of RL use cases:

**Stable Baselines3** prioritizes consistency and usability. Every algorithm exposes the same API: construct a model, call `learn()`, call `predict()`. Switching from PPO to SAC requires changing a single import and constructor. The library enforces stability over novelty -- only well-tested algorithms make it into the core package. SB3 is the right starting point for most single-agent RL projects because it minimizes the gap between having an idea and getting a trained agent.

**RLlib** prioritizes scale and flexibility. Built on Ray, it can distribute rollout collection and gradient computation across hundreds of machines. Multi-agent RL is a first-class feature, not an afterthought. The tradeoff is complexity: RLlib has a steep learning curve, a heavy dependency footprint, and an abstraction-heavy codebase that can be difficult to debug. It is the right choice when your problem outgrows a single machine or involves multiple interacting agents.

**CleanRL** prioritizes transparency and understanding. Each algorithm is a single self-contained file of 300 to 500 lines with no base classes, no hidden dependencies, and no framework abstractions. You read the file top to bottom and see exactly what happens. CleanRL is the right choice when you need to deeply understand an algorithm, modify its internals for research, or teach RL to others.

### Other Notable Frameworks

Several other frameworks occupy important niches in the RL ecosystem:

**Gymnasium (formerly OpenAI Gym)** is not a training framework but the universal environment interface that all major RL frameworks depend on. Gymnasium defines the API contract between environments and agents: `reset()` returns an initial observation, `step(action)` returns the next observation, reward, termination signal, and info dict. Any environment that implements this interface works with any Gymnasium-compatible framework. Gymnasium is maintained by the Farama Foundation, which took over stewardship from OpenAI in 2022.

**TF-Agents** is Google's TensorFlow-based RL library. It provides modular, well-tested implementations of common algorithms (DQN, DDPG, TD3, SAC, PPO, REINFORCE) within the TensorFlow ecosystem. TF-Agents is a reasonable choice for teams deeply invested in TensorFlow, but its adoption has declined as the RL community has consolidated around PyTorch. Development activity has slowed compared to PyTorch-based alternatives.

**Tianshou** is a PyTorch-based RL library that occupies a middle ground between SB3's high-level API and CleanRL's transparency. It provides modular components (policies, collectors, trainers, replay buffers) that can be composed flexibly, with relatively clean code that is easier to read than RLlib's but more structured than CleanRL's. Tianshou supports a broad set of algorithms and has good multi-agent support through integration with PettingZoo. It is a strong alternative to SB3 for engineers who want more control over the training pipeline without the weight of RLlib.

**rlax** is a JAX-based library from DeepMind that provides low-level RL building blocks rather than complete algorithms. It includes functions for computing TD errors, policy gradient losses, distributional RL operations, and other mathematical primitives. rlax is designed for researchers who want to compose their own algorithms from vetted components rather than use pre-built implementations. It pairs well with JAX's functional programming model and hardware acceleration but requires significant engineering to build a complete training system.

**SBX (Stable Baselines Jax)** is a JAX-based sibling of SB3 that mirrors its API but uses JAX for computation. SBX can achieve roughly 20x faster training through JIT compilation and XLA optimization, making it attractive when training speed is a bottleneck and the algorithms you need are supported.

---

## The Role of Environments

RL frameworks are only half the picture. The other half is the environment -- the system the agent interacts with. The quality, performance, and interface of environments directly impact training speed, algorithm selection, and framework choice.

### Gymnasium: The Universal Interface

Gymnasium provides the standard API that all major RL frameworks expect. A Gymnasium environment exposes an observation space, an action space, a `reset()` method, and a `step()` method. This standardization means you can develop an environment once and use it with SB3, RLlib, CleanRL, or any other Gymnasium-compatible framework without modification.

Gymnasium also ships a collection of built-in environments for benchmarking: Classic Control tasks (CartPole, MountainCar, Pendulum, Acrobot, LunarLander), toy text environments, and Box2D physics simulations. These are useful for sanity-checking algorithm implementations and for rapid prototyping.

### Atari (Arcade Learning Environment)

The Arcade Learning Environment (ALE) provides RL agents access to over 50 Atari 2600 games through Gymnasium-compatible wrappers. Atari games are the standard benchmark for discrete-action RL algorithms. They involve high-dimensional visual observations (210x160 RGB frames), require temporal reasoning (frame stacking), and span a wide range of difficulty levels from near-trivial (Pong) to extremely hard exploration problems (Montezuma's Revenge).

Standard Atari preprocessing -- frame skipping, grayscale conversion, 84x84 resizing, frame stacking, reward clipping, and episodic life handling -- is critical for performance and is handled by environment wrappers provided by each framework. SB3 includes `AtariWrapper` and `make_atari_env` utilities. CleanRL implements preprocessing inline in each Atari-specific file. RLlib handles it through its connector and preprocessor system.

### MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) is a physics simulator for continuous control tasks. Standard MuJoCo benchmarks include HalfCheetah, Hopper, Walker2d, Ant, and Humanoid -- articulated bodies that must learn locomotion through continuous torque control. These environments are the standard benchmark for actor-critic methods (SAC, TD3, PPO with continuous actions). MuJoCo became open-source in 2022 after DeepMind acquired it, and it is now available through Gymnasium's built-in environment suite. For GPU-accelerated physics, NVIDIA's Isaac Gym provides MuJoCo-like tasks that can run thousands of parallel environments on a single GPU.

### PettingZoo

PettingZoo is the multi-agent counterpart to Gymnasium. It defines an API for multi-agent environments where multiple agents interact in a shared world. PettingZoo supports both simultaneous-action environments (all agents act at once) and turn-based environments (agents act sequentially). It includes environment suites for multi-agent Atari, classic board games, and multi-particle environments.

RLlib has native support for PettingZoo environments through its multi-agent API. SB3 does not natively support multi-agent settings but can be used with PettingZoo through wrapper libraries like SuperSuit that convert multi-agent environments into parallel single-agent environments. CleanRL has specific multi-agent PPO implementations for PettingZoo.

### EnvPool

EnvPool is a high-performance environment execution engine written in C++ that provides Gymnasium-compatible interfaces for Atari, Classic Control, and other environments. By batching environment operations and eliminating Python overhead, EnvPool can achieve 3-4x speedups over standard Gymnasium environments. CleanRL provides EnvPool-specific algorithm variants (e.g., `ppo_atari_envpool.py`), and both SB3 and RLlib can use EnvPool environments through their vectorized environment interfaces.

---

## Comparing the Three Frameworks

### Complexity and Learning Curve

SB3 has the lowest barrier to entry. A working training script is three lines of code. The API is consistent across all algorithms, documentation is extensive, and error messages are informative. An ML engineer familiar with PyTorch and scikit-learn patterns can be productive with SB3 within an hour.

CleanRL has a different kind of simplicity. There is no API to learn -- you read a script and run it. But understanding what the script does requires RL knowledge. The learning curve is in the algorithms, not the framework. For someone who already understands PPO conceptually, reading CleanRL's `ppo.py` and understanding every line might take an afternoon.

RLlib has the steepest learning curve. The AlgorithmConfig builder pattern, the distinction between EnvRunners and Learners, the RLModule API, the ConnectorV2 pipeline, and the interaction with Ray's distributed runtime all require significant study. Expect days to weeks to become productive with RLlib, depending on your familiarity with Ray and distributed systems. The payoff is access to capabilities that the other frameworks do not provide.

### Scale

SB3 is single-machine only. Parallelism comes from vectorized environments (SubprocVecEnv) running in separate processes on the same machine. This is sufficient for most single-agent problems but hits a ceiling when the environment is computationally expensive or when you need to scale data collection beyond what one machine can handle.

CleanRL is also single-machine by default, though some implementations support multi-GPU training. Like SB3, it uses vectorized environments for parallelism.

RLlib is the only framework in this comparison with native distributed training. It can spread rollout collection across hundreds of CPU workers, run gradient computation on multiple GPUs, and distribute replay buffers across machines. The same training script that runs locally scales to a cluster with configuration changes. For problems that require hundreds or thousands of parallel environments -- large-scale robotics simulation, game AI training, or population-based training -- RLlib is the only viable option among these three.

### Algorithm Coverage

RLlib has the broadest algorithm selection with over 30 algorithms spanning on-policy, off-policy, model-based, offline, and multi-agent methods. It includes distributed-specific algorithms like IMPALA and APPO that are not available in the other frameworks.

SB3 covers 7 core algorithms (PPO, A2C, DQN, SAC, TD3, DDPG, HER) with additional algorithms in SB3-Contrib (Recurrent PPO, TQC, QR-DQN, Maskable PPO, CrossQ, ARS, TRPO). The selection is smaller but each implementation is thoroughly tested and validated.

CleanRL covers 13 or more algorithm variants. Its breadth comes from having environment-specific implementations: PPO alone has over twelve files for different environment types, action spaces, and hardware backends. The total number of distinct algorithms is comparable to SB3, but the variant-specific implementations mean you get optimized code for each setting.

### Multi-Agent Support

RLlib is the clear leader. Multi-agent RL is woven into its architecture with support for heterogeneous policies, self-play, centralized training with decentralized execution, and population-based training. The multi-agent API allows different agents to use different algorithms, observation spaces, and action spaces within a single training run.

SB3 does not support multi-agent RL natively. You can use it with PettingZoo through wrapper libraries, but this is limited to independent training of agents.

CleanRL has specific multi-agent implementations (PPO for PettingZoo) but these are individual scripts, not a general multi-agent framework.

### Production Readiness

RLlib has the most complete production story. Ray Serve provides model serving with autoscaling and request batching. Ray's checkpointing infrastructure enables fault-tolerant training with recovery from failures. Offline RL capabilities allow training from logged production data. Integration with Ray Tune provides hyperparameter optimization at scale.

SB3 provides solid single-machine production utilities: model save/load, ONNX export, evaluation callbacks, TensorBoard logging, and Optuna integration through RL Zoo. For single-agent deployments that do not require distributed infrastructure, SB3's production capabilities are sufficient and simpler to operate.

CleanRL is explicitly not designed for production. There is no model saving/loading API, no checkpoint management, no serving infrastructure, and no deployment tooling. Trained models exist only as PyTorch state dicts within the script's runtime.

### Customization Model

SB3 provides structured extension points: custom network architectures through `policy_kwargs`, custom training behavior through callbacks, custom environments through the Gymnasium interface, and custom preprocessing through VecEnv wrappers. You work within the framework's boundaries.

RLlib provides deeper customization through custom RLModules, custom Learners, custom ConnectorV2 pipelines, and custom environments. The extension points are more powerful but also more complex, requiring familiarity with RLlib's internal architecture.

CleanRL provides the most radical form of customization: you edit the file. There are no extension points because there is no framework. If you want to change how advantages are computed, you find the three lines that compute advantages and change them. This is maximally flexible but does not scale to modifications you want to apply across many algorithm variants.

---

## Decision Framework for Choosing a Framework

The following decision process covers the most common scenarios an ML engineer will encounter:

### Start with the Problem Requirements

**Is your problem multi-agent?** If multiple agents interact in a shared environment, RLlib is the primary choice. Its multi-agent API handles the complexity of routing observations to policies, managing multiple replay buffers, and supporting heterogeneous agent configurations. Alternatives exist (PettingZoo wrappers with SB3, CleanRL multi-agent scripts) but they are limited compared to RLlib's native support.

**Does training need to scale beyond a single machine?** If you need distributed rollout collection across a cluster, RLlib is the only mainstream option. SB3 and CleanRL are single-machine frameworks. If your environment is computationally expensive (high-fidelity physics simulation, complex game worlds) and a single machine cannot generate experience fast enough, you need RLlib.

**Do you need offline RL?** If you are training from logged production data without live environment interaction, RLlib provides CQL, MARWIL, and BC with native dataset integration. SB3 and CleanRL do not have built-in offline RL support. Alternatively, consider specialized offline RL libraries like d3rlpy.

**Is your goal to understand an algorithm?** If you are studying how PPO, SAC, or DQN works at the implementation level -- for learning, teaching, or research -- CleanRL is the best choice. Its single-file implementations with no abstractions let you trace every computation from observation to gradient update.

**Are you prototyping a research modification?** If you need to change something fundamental about an algorithm (a new loss term, a different exploration strategy, a modified rollout collection scheme), CleanRL is the fastest starting point. Copy a file, modify it, and run it. For modifications that fit within SB3's extension points (custom networks, custom callbacks, custom wrappers), SB3 may be more convenient because you retain its engineering infrastructure.

**Is this a standard single-agent problem on a single machine?** For the common case of training PPO, SAC, DQN, or another standard algorithm on a Gymnasium environment using one machine, SB3 is the default choice. Its reliable implementations, consistent API, hyperparameter tuning integration, and save/load workflow minimize engineering overhead.

### Consider Team and Ecosystem Factors

**Is your team already using Ray?** If your ML infrastructure uses Ray Data, Ray Tune, or Ray Serve, RLlib integrates naturally and avoids introducing separate tooling. The overhead of learning RLlib is lower when the team already understands Ray's execution model.

**What is the team's RL experience level?** For teams new to RL, SB3's low barrier to entry is valuable. For teams with strong RL backgrounds, CleanRL's transparency or RLlib's power may be more appropriate.

**Will you need to migrate later?** Teams often start with SB3 or CleanRL for prototyping and migrate to RLlib when they need scale. This migration is nontrivial because the APIs, configuration patterns, and execution models are substantially different. If you anticipate needing distributed training eventually, starting with RLlib may save migration costs, but only if the upfront complexity is justified.

### Quick Reference

| Scenario | Recommended Framework |
|---|---|
| Standard single-agent training on one machine | Stable Baselines3 |
| Multi-agent RL | RLlib |
| Distributed training across a cluster | RLlib |
| Offline RL from logged data | RLlib |
| Understanding algorithm implementations | CleanRL |
| Research prototyping with algorithm modifications | CleanRL |
| Production deployment with serving infrastructure | RLlib |
| Rapid benchmarking with pre-tuned hyperparameters | Stable Baselines3 (RL Zoo) |
| Teaching or coursework | CleanRL |
| Already using Ray ecosystem | RLlib |

---

## Common Patterns Across Frameworks

Despite their differences, the three frameworks share patterns that reflect the structure of RL training itself.

### Environment Vectorization

All three frameworks use vectorized environments -- running multiple environment instances in parallel -- to increase data throughput. SB3 uses SubprocVecEnv and DummyVecEnv wrappers. RLlib runs multiple environments per EnvRunner through the `num_envs_per_env_runner` config. CleanRL uses Gymnasium's `SyncVectorEnv` or EnvPool for batched execution. The principle is the same: RL algorithms need large amounts of experience data, and running environments in parallel is the most direct way to get it.

### Observation and Reward Preprocessing

Proper observation normalization and reward scaling are critical for RL performance and are often more impactful than algorithm choice or hyperparameter tuning. SB3 provides `VecNormalize` for running normalization of observations and rewards. RLlib handles this through its ConnectorV2 pipeline. CleanRL implements normalization inline where needed. Regardless of framework, always consider whether your observations and rewards need normalization, especially for environments with unbounded or poorly scaled signals.

### Experiment Tracking

All three frameworks support TensorBoard logging. SB3 accepts a `tensorboard_log` parameter. CleanRL writes to TensorBoard by default. RLlib logs training results that can be directed to TensorBoard. Additionally, CleanRL has first-class Weights and Biases integration through its `--track` flag, SB3 supports W&B through callbacks, and RLlib integrates with W&B through Ray Tune's logger system.

### Hyperparameter Sensitivity

RL algorithms are notoriously sensitive to hyperparameters, and the right settings vary significantly across environments. SB3 addresses this through the RL Baselines3 Zoo, which ships pre-tuned hyperparameters for dozens of environments and integrates Optuna for automated search. RLlib uses Ray Tune for distributed hyperparameter optimization. CleanRL documents its hyperparameters explicitly in each file and publishes benchmark results through the Open RL Benchmark.

---

## A Practical Workflow

For an ML engineer starting a new RL project, a reasonable workflow is:

1. **Define the problem and environment.** Implement your environment following the Gymnasium API. Use SB3's `check_env()` utility to validate it. If your problem is multi-agent, implement it using PettingZoo or RLlib's MultiAgentEnv.

2. **Prototype with SB3 or CleanRL.** For single-agent problems, use SB3 to quickly test whether standard algorithms can learn a reasonable policy. Train PPO first (it is the most robust general-purpose algorithm), then try SAC for continuous control or DQN for discrete actions. If you need to understand or modify the algorithm, use CleanRL as a reference or starting point.

3. **Tune hyperparameters.** Use RL Zoo's Optuna integration for SB3 or Ray Tune for RLlib. Start with published hyperparameters for similar environments and adjust from there. Pay particular attention to observation normalization, learning rate, and batch size.

4. **Scale if needed.** If single-machine training is too slow or your problem requires multiple agents, migrate to RLlib. Expect this migration to require rewriting environment interfaces, model definitions, and training scripts.

5. **Deploy.** For single-agent deployments, export the SB3 model to ONNX or pure PyTorch. For production systems needing serving infrastructure, use RLlib with Ray Serve.

---

## Child Chapters

For detailed coverage of each framework, including architecture, API usage, supported algorithms, and practical examples, see:

- [Stable Baselines3](./stable-baselines3/ReadMe.md) -- Production-quality implementations of core RL algorithms with a unified, high-level API
- [RLlib](./rllib/ReadMe.md) -- Scalable, distributed RL training with native multi-agent support, built on Ray
- [CleanRL](./cleanrl/ReadMe.md) -- Single-file, no-abstraction algorithm implementations for learning, research, and prototyping
