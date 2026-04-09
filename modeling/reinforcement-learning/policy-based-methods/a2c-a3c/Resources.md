# A2C/A3C Resources

## Foundational Papers

- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783) - The original A3C paper introducing asynchronous parallel workers for on-policy deep RL.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - Introduces GAE, the advantage estimation method used in modern A2C and PPO implementations.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO, the direct successor to A2C that adds the clipped surrogate objective.
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation (Sutton et al., 2000)](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) - Foundational policy gradient theorem that underlies all actor-critic methods.

## Tutorials and Guides

- [Hugging Face Deep RL Course - Actor-Critic Methods](https://huggingface.co/learn/deep-rl-course/unit6/introduction) - Interactive course covering A2C with code examples and theory.
- [OpenAI Spinning Up - Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) - Covers the actor-critic foundation that A2C builds on.
- [CleanRL A2C Implementation](https://github.com/vwxyzjn/cleanrl) - Single-file, well-documented A2C implementation with benchmarks.
- [The 37 Implementation Details of Proximal Policy Optimization (Huang et al., 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Covers many implementation details shared between A2C and PPO.

## Reference Implementations

- [Stable Baselines3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) - Production-quality A2C with documentation, benchmarks, and hyperparameter guidance.
- [OpenAI Baselines A2C](https://github.com/openai/baselines/tree/master/baselines/a2c) - Reference A2C implementation from OpenAI.
- [PyTorch RL Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning) - Official PyTorch examples including actor-critic implementations.
