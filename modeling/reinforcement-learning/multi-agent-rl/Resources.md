# Multi-Agent Reinforcement Learning Resources

## Foundational Papers

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275) - The MADDPG paper introducing centralized training with decentralized execution for continuous action spaces.
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (Rashid et al., 2018)](https://arxiv.org/abs/1803.11485) - QMIX value decomposition for cooperative MARL.
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (Yu et al., 2022)](https://arxiv.org/abs/2103.01955) - MAPPO paper demonstrating that well-tuned PPO is a strong multi-agent baseline.
- [Value-Decomposition Networks for Cooperative Multi-Agent Learning (Sunehag et al., 2017)](https://arxiv.org/abs/1706.05296) - VDN, the additive value decomposition predecessor to QMIX.
- [Counterfactual Multi-Agent Policy Gradients (Foerster et al., 2018)](https://arxiv.org/abs/1705.08926) - COMA, introducing counterfactual baselines for multi-agent credit assignment.
- [Learning to Communicate with Deep Multi-Agent Reinforcement Learning (Foerster et al., 2016)](https://arxiv.org/abs/1605.06676) - DIAL, one of the first works on learned communication in MARL.
- [Learning Multiagent Communication with Backpropagation (Sukhbaatar et al., 2016)](https://arxiv.org/abs/1605.07736) - CommNet, continuous communication learned end-to-end.

## Landmark Application Papers

- [Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning (Vinyals et al., 2019)](https://www.nature.com/articles/s41586-019-1724-z) - AlphaStar reaching Grandmaster in StarCraft II with league training and population-based self-play.
- [Dota 2 with Large Scale Deep Reinforcement Learning (Berner et al., 2019)](https://arxiv.org/abs/1912.06680) - OpenAI Five defeating professional Dota 2 players using self-play with PPO.
- [Superhuman AI for Multiplayer Poker (Brown and Sandholm, 2019)](https://www.science.org/doi/10.1126/science.aay2400) - Pluribus achieving superhuman six-player poker through self-play and search.
- [Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning (Meta FAIR, 2022)](https://www.science.org/doi/10.1126/science.ade9097) - Cicero combining LLMs with game-theoretic reasoning for Diplomacy.

## Surveys and Tutorials

- [A Survey of Multi-Agent Reinforcement Learning with Communication (Zhu et al., 2022)](https://arxiv.org/abs/2203.08975) - Comprehensive survey of communication in MARL.
- [Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms (Zhang et al., 2021)](https://arxiv.org/abs/1911.10635) - Broad theoretical survey of MARL.
- [An Introduction to Multi-Agent Reinforcement Learning and Review of its Driving Applications (Gronauer and Diepold, 2022)](https://arxiv.org/abs/2104.08601) - Accessible introduction with application focus.

## Frameworks and Libraries

- [PettingZoo](https://pettingzoo.farama.org/) - Standard multi-agent environment library with unified API for simultaneous and turn-based interactions.
- [RLlib Multi-Agent](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) - Production-grade multi-agent RL framework with flexible policy mapping and distributed training.
- [EPyMARL](https://github.com/uoe-agents/epymarl) - PyTorch framework for cooperative MARL research with QMIX, VDN, MAPPO, and SMAC support.
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) - DeepMind framework for game-theoretic research and MARL.
- [MARLlib](https://github.com/Replicable-MARL/MARLlib) - Comprehensive MARL library unifying multiple environments and algorithms.
- [SMAC: The StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac) - Standard benchmark for cooperative MARL.
