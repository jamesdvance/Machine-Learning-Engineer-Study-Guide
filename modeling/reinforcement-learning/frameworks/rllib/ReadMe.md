# RLlib

## Summary

RLlib is Ray's open-source library for reinforcement learning, designed from the ground up for scalable, distributed training. Originally developed at UC Berkeley's RISELab alongside the Ray distributed computing framework, RLlib has matured into the most feature-complete RL library available for production workloads. It provides a unified API across dozens of algorithms, native support for multi-agent reinforcement learning, and seamless scaling from a laptop to a thousand-node cluster. For ML engineers who need to move beyond single-process RL experiments and deploy policies in real production systems, RLlib is the standard choice.

RLlib sits in a unique position in the RL ecosystem. While libraries like Stable Baselines3 prioritize simplicity and CleanRL prioritizes transparency, RLlib prioritizes scale and flexibility. It is built on Ray, which handles distributed execution, fault tolerance, and resource management, allowing RLlib to focus purely on RL abstractions. This architecture means that the same training script that runs on a single machine can scale to hundreds of workers with minimal code changes. The tradeoff is complexity: RLlib has a steeper learning curve and heavier dependency footprint than its simpler counterparts.

Key points to remember:

- RLlib is built on Ray and leverages Ray's actor model for distributed rollout workers, replay buffers, and learner processes
- It supports over 30 algorithms out of the box, including PPO, SAC, DQN, IMPALA, APPO, Dreamer, and CQL for offline RL
- Multi-agent RL is a first-class feature, not a bolt-on, with support for independent, cooperative, and competitive agent configurations sharing a single environment
- The Algorithm Configuration API (AlgorithmConfig) provides a builder pattern for configuring every aspect of training, from environment to model to resources
- RLlib supports custom models, custom policies, custom environments, and custom training loops through well-defined extension points
- Offline RL and batch inference are natively supported, enabling training from logged data and deploying policies without a live environment
- Integration with Ray Tune enables hyperparameter optimization and experiment management alongside distributed training
- RLlib underwent a major API overhaul (the "new stack") starting in Ray 2.x, modernizing its internals around RLModules and Learner APIs

When to use RLlib:

- Multi-agent RL problems where multiple policies interact in a shared environment
- Large-scale training requiring distributed rollout collection across many CPU/GPU workers
- Production deployments needing serving infrastructure, checkpointing, and fault tolerance
- Projects already using the Ray ecosystem (Ray Serve, Ray Data, Ray Tune)
- Offline RL from logged production data
- When you need a wide algorithm selection without switching libraries

When not to use RLlib:

- Simple single-agent experiments where Stable Baselines3 or CleanRL would suffice with less overhead
- When you need to deeply understand and modify every line of the algorithm (CleanRL is better for this)
- Lightweight research prototyping where dependency weight matters
- When your team is unfamiliar with Ray and the learning curve cost is not justified by scale requirements

---

## Architecture: Built on Ray

### The Ray Foundation

Understanding RLlib requires understanding Ray. Ray is a general-purpose distributed computing framework that provides three core primitives: tasks (stateless remote functions), actors (stateful remote classes), and objects (shared immutable data in a distributed object store). RLlib maps RL concepts onto these primitives. Rollout workers are Ray actors that hold a copy of the policy and collect experience from environments. The learner is a Ray actor (or group of actors) that receives experience and updates model weights. Replay buffers, if used, are also Ray actors with their own memory.

This actor-based architecture provides several benefits. Each component runs in its own process, potentially on its own machine, with Ray handling serialization, scheduling, and fault recovery. If a rollout worker dies, Ray can restart it automatically. If the cluster grows, new workers can be added without stopping training. The distributed object store enables zero-copy data sharing between components on the same node, reducing the overhead of moving large batches of experience data.

### Component Architecture

RLlib's architecture consists of several major components:

**Algorithm**: The top-level orchestrator. It owns the training loop, coordinates workers, manages checkpointing, and exposes the configuration API. Each algorithm (PPO, SAC, DQN, etc.) subclasses Algorithm and implements its specific training logic.

**EnvRunner (formerly RolloutWorker)**: Responsible for stepping through environments and collecting experience. Each EnvRunner holds a local copy of the policy and one or more environment instances. In a typical setup, you might have 16 or 64 EnvRunners spread across a cluster, each collecting trajectories in parallel.

**Learner**: Introduced in the new stack (Ray 2.x+), the Learner is responsible for computing loss and updating model weights. It can run on GPU while EnvRunners run on CPU, enabling efficient resource utilization. Multiple Learners can be used for data-parallel training across GPUs.

**RLModule**: The new abstraction for the neural network model. RLModules define the forward passes for inference and training, replacing the older ModelV2 API. They are framework-native (PyTorch or TensorFlow) and can be composed into MultiRLModules for multi-agent settings.

**ConnectorV2**: Preprocessing and postprocessing pipelines that sit between environments and policies. Connectors handle observation normalization, action clipping, reward shaping, and agent-to-module mapping in multi-agent settings.

### Execution Flow

A typical RLlib training iteration proceeds as follows:

1. The Algorithm dispatches weight updates to all EnvRunners
2. Each EnvRunner collects a configurable number of timesteps or episodes from its local environment(s), running policy inference locally
3. Collected experience (observations, actions, rewards, dones) is returned to the Algorithm
4. For on-policy algorithms (PPO, APPO), experience is sent directly to the Learner for gradient computation
5. For off-policy algorithms (SAC, DQN), experience is stored in a replay buffer actor, from which the Learner samples minibatches
6. The Learner computes gradients and updates the model weights
7. Updated weights are broadcast back to EnvRunners for the next iteration

This pipeline is fully asynchronous when using algorithms like IMPALA or APPO, where EnvRunners do not wait for weight updates before collecting more experience.

---

## Supported Algorithms

RLlib provides one of the broadest algorithm selections of any RL library. The algorithms span the major families of RL approaches:

### On-Policy

- **PPO (Proximal Policy Optimization)**: The default choice for most problems. RLlib's PPO supports distributed rollout collection, GPU-accelerated training, and multi-agent configurations.
- **APPO (Asynchronous PPO)**: A variant where rollout workers do not synchronize with the learner. Workers continuously collect data with potentially stale weights, increasing throughput at the cost of some sample efficiency. Well-suited for environments with variable step times.
- **IMPALA (Importance Weighted Actor-Learner Architecture)**: DeepMind's distributed architecture using V-trace off-policy corrections. Designed for extremely high throughput with many parallel actors.

### Off-Policy

- **DQN (Deep Q-Network)**: With support for Double DQN, Dueling DQN, prioritized experience replay, and n-step returns, all configurable through the API.
- **SAC (Soft Actor-Critic)**: Maximum entropy RL for continuous action spaces. Includes automatic entropy coefficient tuning.
- **TD3 (Twin Delayed DDPG)**: Deterministic policy gradient method with twin critics and delayed policy updates.

### Model-Based

- **DreamerV3**: A model-based algorithm that learns a world model and trains a policy within the learned model. Available as a community-contributed algorithm.

### Offline RL

- **CQL (Conservative Q-Learning)**: Penalizes Q-values for out-of-distribution actions, enabling learning from static datasets.
- **MARWIL (Monotonic Advantage Re-Weighted Imitation Learning)**: Imitation learning from offline data weighted by advantage estimates.
- **BC (Behavior Cloning)**: Supervised learning baseline for offline RL.

### Multi-Agent Specific

- **Parameter sharing**: Multiple agents share a single policy, with agent IDs as additional input features.
- **Independent learners**: Each agent has its own algorithm instance treating other agents as part of the environment.
- **Centralized critic**: Agents have independent policies but a centralized value function that conditions on global state.

---

## Multi-Agent Reinforcement Learning

Multi-agent RL (MARL) is RLlib's most significant differentiator. While other libraries require significant custom engineering to support multiple agents, RLlib treats multi-agent as a core abstraction woven throughout its architecture.

### The Multi-Agent API

In RLlib's multi-agent setup, a single environment instance can contain multiple agents, each identified by a string agent ID. At each timestep, the environment returns a dictionary mapping agent IDs to their observations, rewards, and done flags. The framework routes observations from each agent to its assigned policy, collects actions, and returns them as a dictionary back to the environment.

The mapping from agents to policies is configured through a policy mapping function:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("my_multi_agent_env")
    .multi_agent(
        policies={"policy_attacker", "policy_defender"},
        policy_mapping_fn=lambda agent_id, episode, **kwargs:
            "policy_attacker" if agent_id.startswith("attacker") else "policy_defender",
    )
)
```

This design enables several common patterns:

**Heterogeneous policies**: Different agent types (e.g., predator and prey, buyer and seller) each have their own policy with their own observation and action spaces, model architecture, and algorithm configuration.

**Self-play**: An agent trains against copies of itself. RLlib supports maintaining a pool of historical policy checkpoints and sampling opponents from this pool, which is essential for competitive games.

**Centralized training with decentralized execution (CTDE)**: During training, a centralized critic can observe the full state of all agents. During deployment, each agent uses only its local policy with local observations.

**Population-based training**: Multiple policies in the same environment can be evolved independently, with periodic evaluation tournaments determining which policies survive.

### Why Multi-Agent Matters

Many real-world RL problems are inherently multi-agent: traffic signal coordination, robotic swarm control, market making with multiple competing agents, network routing with distributed controllers, and game AI with teams of cooperating units. Trying to shoehorn these into single-agent frameworks requires either training each agent independently (ignoring the non-stationarity caused by other learning agents) or treating the joint action space as a single massive action (which scales exponentially with agent count). RLlib's native multi-agent support handles both the engineering complexity and the algorithmic challenges of MARL.

---

## Custom Models and Policies

### RLModules (New Stack)

The RLModule API, introduced in Ray 2.x, is RLlib's modern approach to defining custom models. An RLModule is a PyTorch (or TensorFlow) nn.Module subclass with a specific interface:

```python
from ray.rllib.core.rl_module.rl_module import RLModule

class MyCustomModule(RLModule):
    def setup(self):
        # Define layers, networks, etc.
        self.encoder = nn.Sequential(...)
        self.policy_head = nn.Linear(...)
        self.value_head = nn.Linear(...)

    def _forward_inference(self, batch):
        # Used during environment interaction (no exploration noise)
        ...

    def _forward_exploration(self, batch):
        # Used during training rollouts (with exploration)
        ...

    def _forward_train(self, batch):
        # Used during loss computation (returns all quantities needed for loss)
        ...
```

The three forward methods separate inference, exploration, and training concerns. This separation allows different behavior at each stage without conditional branching inside the model. For example, _forward_exploration might add entropy-based exploration while _forward_inference returns deterministic actions.

For multi-agent settings, multiple RLModules are composed into a MultiRLModule, which routes inputs to the appropriate sub-module based on agent-to-module mapping.

### Custom Policies

Beyond custom models, RLlib allows fully custom policies that define their own loss functions, update rules, and postprocessing. This is useful for implementing novel algorithms that do not fit RLlib's existing algorithm templates. In the new stack, custom training logic is implemented through custom Learner subclasses:

```python
from ray.rllib.core.learner.learner import Learner

class MyCustomLearner(Learner):
    def compute_loss_for_module(self, module_id, config, batch, fwd_out):
        # Define custom loss computation
        ...
```

### Custom Environments

RLlib supports any environment that follows the Gymnasium (formerly OpenAI Gym) API. For multi-agent environments, it uses its own MultiAgentEnv base class. Environments can be registered by name or passed directly:

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MyTradingEnv(MultiAgentEnv):
    def __init__(self, config):
        self.agents = ["market_maker_0", "market_maker_1"]
        ...

    def reset(self, seed=None, options=None):
        ...

    def step(self, action_dict):
        # action_dict maps agent_id -> action
        # Returns obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
        ...
```

External environments are also supported through RLlib's external API, which allows environments running in separate processes or even separate machines to communicate with RLlib through a client-server protocol.

---

## Offline RL Capabilities

Offline RL (also called batch RL) trains policies from previously collected data without live environment interaction. This is critical for domains where online exploration is expensive or dangerous: healthcare, autonomous driving, industrial control, and recommendation systems.

### Dataset Ingestion

RLlib integrates with Ray Data for reading offline datasets. Data can be stored in Parquet, JSON, or custom formats. The framework handles sharding data across Learner workers and shuffling:

```python
config = (
    PPOConfig()
    .offline_data(
        input_="dataset",
        input_config={"paths": "/path/to/experience_data/"},
    )
)
```

### Offline Algorithms

RLlib ships CQL, MARWIL, and BC for offline training. CQL is particularly important because it addresses the distributional shift problem inherent in offline RL: naively applying off-policy algorithms to static data tends to overestimate Q-values for state-action pairs not represented in the data, leading to poor policies. CQL adds a regularizer that penalizes high Q-values for out-of-distribution actions.

### Mixed Online/Offline

RLlib also supports hybrid settings where an agent is pretrained on offline data and then fine-tuned with online interaction. This is configured by switching the input source from a dataset to a live environment after initial training, or by mixing offline data with online experience in the replay buffer.

---

## Production Deployment Considerations

### Checkpointing and Recovery

RLlib integrates with Ray's checkpointing infrastructure. Checkpoints capture the full training state: model weights, optimizer state, replay buffer contents, and training metrics. Training can be resumed from any checkpoint, and checkpoints can be stored on local disk, S3, GCS, or HDFS.

```python
# Save a checkpoint
checkpoint_dir = algorithm.save()

# Restore from checkpoint
algorithm.restore(checkpoint_dir)

# Export just the policy for serving
algorithm.get_policy().export_model("/path/to/exported_model")
```

### Serving with Ray Serve

For inference in production, trained policies can be deployed using Ray Serve, Ray's model serving framework. This provides autoscaling, request batching, and multi-model composition:

```python
from ray import serve

@serve.deployment
class RLPolicyServer:
    def __init__(self, checkpoint_path):
        from ray.rllib.algorithms.ppo import PPO
        self.algorithm = PPO.from_checkpoint(checkpoint_path)

    async def __call__(self, request):
        obs = await request.json()
        action = self.algorithm.compute_single_action(obs)
        return {"action": action.tolist()}
```

### Scaling Considerations

When scaling RLlib training, several considerations apply:

**CPU vs GPU allocation**: EnvRunners are typically CPU-bound (running environments and inference), while Learners are GPU-bound (computing gradients). A common configuration is many CPU-only EnvRunners with one or a few GPU Learners.

**Network bandwidth**: In distributed settings, experience data and weight updates flow between workers. For environments with large observations (images, point clouds), this can become a bottleneck. Observation compression and weight delta broadcasting can mitigate this.

**Environment throughput**: If environments are slow (complex simulators, external systems), adding more EnvRunners is the primary scaling lever. If environments are fast, the learner GPU becomes the bottleneck.

**Memory management**: Replay buffers for off-policy algorithms can consume significant memory. RLlib supports distributed replay buffers that shard data across multiple machines, as well as prioritized replay with configurable capacity limits.

### Fault Tolerance

Because RLlib runs on Ray, it inherits Ray's fault tolerance features. If an EnvRunner crashes (common with complex simulators), Ray restarts it automatically. Training continues with the remaining workers while the failed worker recovers. The Algorithm periodically syncs weights to all workers, so a restarted worker quickly catches up. For longer training runs, periodic checkpointing to durable storage ensures that a full cluster failure does not lose significant progress.

---

## Comparison with Stable Baselines3 and CleanRL

RLlib, Stable Baselines3 (SB3), and CleanRL represent three distinct philosophies for RL libraries. Understanding their tradeoffs is essential for choosing the right tool.

### Stable Baselines3

SB3 is designed for ease of use. It provides well-tested, single-process implementations of core algorithms (PPO, SAC, DQN, TD3, A2C, HER) with a clean, consistent API. Training a policy is often three lines of code. SB3 excels at rapid prototyping, benchmarking, and educational use.

However, SB3 is fundamentally single-process. It uses vectorized environments (SubprocVecEnv) for parallelism, but all training happens in one process on one machine. There is no built-in distributed training, no multi-agent support, and limited customization of the training loop. For problems that fit on a single machine with standard algorithms and single-agent environments, SB3 is often the best choice due to its simplicity.

### CleanRL

CleanRL takes a radical approach: each algorithm is implemented in a single, self-contained Python file with no abstraction layers. The entire PPO implementation, including environment setup, network definition, training loop, and logging, is in one file of approximately 300 lines. This makes algorithms completely transparent and easy to modify for research.

CleanRL is ideal when you need to deeply understand an algorithm, modify its internals for a research contribution, or reproduce results exactly. It is not designed for production use, multi-agent settings, or distributed training. It sacrifices reusability and scale for clarity.

### Comparison Table

| Feature | RLlib | Stable Baselines3 | CleanRL |
|---|---|---|---|
| Distributed training | Native (Ray) | No | No |
| Multi-agent RL | First-class support | No | No |
| Algorithm count | 30+ | 7 core | 10+ variants |
| Custom models | RLModule API | Policy kwargs | Edit the file |
| Offline RL | CQL, MARWIL, BC | No native support | No |
| Hyperparameter tuning | Ray Tune integration | Optuna integration | Manual |
| Serving/deployment | Ray Serve | Manual export | Manual export |
| Learning curve | Steep | Low | Low |
| Dependency weight | Heavy (Ray ecosystem) | Moderate | Minimal |
| Single-file readability | No | No | Yes |
| Framework support | PyTorch, TF | PyTorch | PyTorch, JAX |

### Migration Considerations

Teams often start with SB3 or CleanRL for initial experiments and migrate to RLlib when they need scale. This migration is nontrivial because RLlib's API, configuration patterns, and execution model are substantially different. Key friction points include: adapting custom environments to RLlib's API expectations, rewriting custom model architectures into the RLModule framework, and understanding the distributed execution model for debugging. Plan for this migration cost when choosing a starting library.

---

## When to Choose RLlib

### Choose RLlib When:

**You need distributed training at scale.** If your environment is computationally expensive (physics simulators, large game worlds) and you need to parallelize rollout collection across many machines, RLlib is the only mainstream RL library with native distributed execution. The same code scales from one worker to hundreds.

**Your problem involves multiple agents.** If you have cooperative agents, competing agents, mixed cooperative-competitive settings, or need self-play, RLlib's multi-agent API saves months of custom engineering compared to building multi-agent training on top of single-agent libraries.

**You are deploying to production.** RLlib's integration with Ray Serve, its checkpointing infrastructure, fault tolerance, and resource management make it suitable for production RL systems where reliability and operability matter.

**You are already in the Ray ecosystem.** If your ML pipeline uses Ray Data for data processing, Ray Tune for hyperparameter optimization, or Ray Serve for model serving, RLlib integrates naturally and avoids introducing separate infrastructure.

**You need offline RL.** If you have logged interaction data and want to train policies without live environment access, RLlib's offline RL algorithms and dataset integration provide a supported path.

**You need algorithm breadth.** If your project may require switching between PPO, SAC, IMPALA, CQL, and other algorithms during development, RLlib's unified API makes this a configuration change rather than a library change.

### Do Not Choose RLlib When:

**Simplicity is the priority.** For course projects, quick experiments, or problems where SB3's algorithms are sufficient, RLlib's complexity is not justified.

**You need to modify algorithm internals deeply.** If your research contribution involves changing the core training loop of an algorithm, CleanRL's transparent implementations are easier to work with than navigating RLlib's abstraction layers.

**You are resource-constrained.** RLlib's dependency on Ray means a heavier installation, longer startup times, and more memory overhead even for simple experiments.

**Your problem is simple enough for DQN on a laptop.** Not every RL problem needs distributed infrastructure. Using RLlib for a CartPole experiment is like using Kubernetes to deploy a static website.

---

## Practical Tips

**Start with the AlgorithmConfig API.** Every aspect of RLlib is configured through the builder-pattern AlgorithmConfig. Spend time understanding the config groups: .environment(), .training(), .resources(), .env_runners(), .multi_agent(), .offline_data(), .evaluation().

**Use the new stack.** As of Ray 2.x, RLlib is migrating from its old API (ModelV2, Policy, RolloutWorker) to a new API (RLModule, Learner, EnvRunner). New projects should use the new stack exclusively. The old stack is in maintenance mode.

**Debugging distributed issues.** When something goes wrong in distributed training, use Ray Dashboard (a web UI that ships with Ray) to inspect actor states, resource usage, and error logs. Set num_env_runners=0 to run everything in the local process for easier debugging, then scale out once the logic is correct.

**Vectorized environments matter.** Even within a single EnvRunner, you can run multiple environment instances using the num_envs_per_env_runner config. This amortizes Python overhead and increases throughput, especially for fast environments.

**Profile before scaling.** Use RLlib's built-in timing metrics (reported in training results) to identify whether your bottleneck is environment stepping, inference, or gradient computation. Scale the appropriate component rather than uniformly adding resources.

**Checkpoint frequently.** RL training is notoriously unstable. Configure periodic checkpointing and keep multiple checkpoints so you can roll back to a good policy if performance degrades.
