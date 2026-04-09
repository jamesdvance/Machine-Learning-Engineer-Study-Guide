# RLlib Resources

## Foundational Papers

- [Ray: A Distributed Framework for Emerging AI Applications (Moritz et al., 2018)](https://arxiv.org/abs/1712.05889) - The Ray framework paper, foundational to understanding RLlib's distributed architecture.
- [RLlib: Abstractions for Distributed Reinforcement Learning (Liang et al., 2018)](https://arxiv.org/abs/1712.09381) - The original RLlib paper describing its architecture and scalable RL abstractions.
- [Scalable Multi-Agent Reinforcement Learning (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275) - MADDPG, foundational work on multi-agent RL relevant to RLlib's MARL capabilities.

## Official Documentation

- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html) - Official RLlib documentation including API reference, algorithm catalog, and tutorials.
- [RLlib New Stack Migration Guide](https://docs.ray.io/en/latest/rllib/rllib-new-api-stack.html) - Guide for migrating from old API (ModelV2/Policy) to the new RLModule/Learner API.
- [RLlib Algorithms Overview](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html) - Complete catalog of supported algorithms with configuration options.
- [RLlib Multi-Agent Documentation](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) - Guide to multi-agent environment setup and policy mapping.

## Offline RL References

- [Conservative Q-Learning for Offline Reinforcement Learning (Kumar et al., 2020)](https://arxiv.org/abs/2006.04779) - CQL paper, one of RLlib's supported offline RL algorithms.
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems (Levine et al., 2020)](https://arxiv.org/abs/2005.01643) - Comprehensive survey of offline RL, relevant to RLlib's offline capabilities.

## Distributed RL

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (Espeholt et al., 2018)](https://arxiv.org/abs/1802.01561) - The IMPALA algorithm implemented in RLlib for high-throughput distributed training.
- [Massively Parallel Methods for Deep Reinforcement Learning (Nair et al., 2015)](https://arxiv.org/abs/1507.04296) - Early work on distributed RL architectures that influenced RLlib's design.

## Tutorials and Examples

- [RLlib Examples Repository](https://github.com/ray-project/ray/tree/master/rllib/examples) - Official collection of RLlib example scripts covering algorithms, custom models, multi-agent, and offline RL.
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html) - Documentation for deploying trained RLlib policies as scalable inference services.

## Sibling Framework Comparisons

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/) - Single-process RL library optimized for simplicity and ease of use.
- [CleanRL GitHub Repository](https://github.com/vwxyzjn/cleanrl) - Single-file RL implementations prioritizing transparency and reproducibility.
