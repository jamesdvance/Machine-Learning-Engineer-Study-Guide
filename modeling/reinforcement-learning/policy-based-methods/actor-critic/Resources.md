# Actor-Critic Resources

## Foundational Papers

- [Policy Gradient Methods for Reinforcement Learning with Function Approximation (Sutton et al., 2000)](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) - Introduces the policy gradient theorem that underpins all actor-critic methods.
- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783) - The A3C paper that brought deep actor-critic methods to prominence.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - Introduces GAE, the standard advantage estimator used in modern actor-critic algorithms.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO, the most widely used actor-critic algorithm in practice.
- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) - TRPO, which constrains policy updates using a KL divergence trust region.

## Tutorials and Guides

- [Actor-Critic Methods - Mastering Reinforcement Learning](https://gibberblot.github.io/rl-notes/single-agent/actor-critic.html) - Detailed notes on actor-critic theory and derivations.
- [Sutton and Barto, Section 13.5: Actor-Critic Methods](http://incompleteideas.net/book/ebook/node66.html) - The canonical textbook treatment of actor-critic methods.
- [Playing CartPole with the Actor-Critic Method (TensorFlow)](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic) - Official TensorFlow tutorial implementing actor-critic from scratch.
- [Actor-Critic Algorithm in Reinforcement Learning - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/) - Accessible introduction to the algorithm with code examples.

## Reference Implementations

- [CleanRL A2C Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c.py) - Single-file, well-documented A2C implementation in PyTorch.
- [Stable Baselines3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) - Production-quality A2C with documentation and benchmarks.
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) - Production-quality PPO, the most commonly used actor-critic variant.
