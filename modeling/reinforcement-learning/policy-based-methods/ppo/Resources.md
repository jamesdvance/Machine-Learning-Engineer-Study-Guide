# PPO Resources

## Foundational Papers

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - The original PPO paper introducing both clipped and penalty variants.
- [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) - TRPO, the predecessor to PPO that introduced trust region methods for policy optimization.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) - The GAE paper, essential for understanding PPO's advantage computation.
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) - InstructGPT paper describing PPO for RLHF.

## Implementation References and Analysis

- [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO (Engstrom et al., 2020)](https://arxiv.org/abs/2005.12729) - Analysis of implementation details that significantly affect PPO performance.
- [The 37 Implementation Details of Proximal Policy Optimization (Huang et al., 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Comprehensive catalog of PPO implementation details and their impact.
- [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) - Single-file, well-documented reference implementation.
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) - Production-quality PPO with documentation and benchmarks.

## Tutorials and Courses

- [Hugging Face Deep RL Course - PPO](https://huggingface.co/learn/deep-rl-course/unit8/introduction) - Interactive course covering PPO with code examples.
- [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - OpenAI's educational resource with clear PPO explanation and implementation.

## RLHF and LLM Alignment

- [TRL (Transformer Reinforcement Learning) Library](https://github.com/huggingface/trl) - Hugging Face library implementing PPO for LLM fine-tuning.
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) - DPO as an alternative to PPO for RLHF.
- [DeepSeek-R1 (DeepSeek AI, 2025)](https://arxiv.org/abs/2501.12948) - Introduces GRPO as an alternative to PPO for reasoning model training.
