# Resources

## Papers

- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) - The original TRPO paper.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - PPO, the simplified successor to TRPO.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - GAE, the advantage estimation method used by TRPO and PPO.
- [Approximately Optimal Approximate Reinforcement Learning (Kakade and Langford, 2002)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) - The conservative policy iteration theory that TRPO builds on.
- [Natural Gradient Works Efficiently in Learning (Amari, 1998)](https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning) - The natural gradient foundation underlying TRPO.

## Reference Implementations

- [OpenAI Spinning Up - TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html) - Clear explanation and implementation of TRPO with pedagogical code.
- [Stable Baselines3 Contrib - TRPO](https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html) - Production-quality TRPO implementation in PyTorch.
- [CleanRL - PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) - Single-file PPO implementation useful for comparison with TRPO.

## Blog Posts and Tutorials

- [Lilian Weng - Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) - Comprehensive overview covering REINFORCE, TRPO, and PPO with derivations.
- [Jonathan Hui - RL: Trust Region Policy Optimization (TRPO) Explained](https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9) - Step-by-step walkthrough of the TRPO derivation.
