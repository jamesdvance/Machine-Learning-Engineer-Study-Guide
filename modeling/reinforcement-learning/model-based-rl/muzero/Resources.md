# MuZero Resources

## Foundational Papers

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265) - The original MuZero paper published in Nature.
- [Mastering the Game of Go with Deep Neural Networks and Tree Search (Silver et al., 2016)](https://www.nature.com/articles/nature16961) - AlphaGo, the predecessor that combined deep learning with MCTS using human expert data.
- [Mastering the Game of Go without Human Knowledge (Silver et al., 2017)](https://www.nature.com/articles/nature24270) - AlphaGo Zero, removing the dependency on human data.
- [A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play (Silver et al., 2018)](https://arxiv.org/abs/1712.01815) - AlphaZero, generalizing to multiple board games.
- [Online and Offline Reinforcement Learning by Planning with a Learned Model (Schrittwieser et al., 2021)](https://arxiv.org/abs/2104.06294) - MuZero Unplugged, adapting MuZero for offline RL settings.
- [Learning and Planning in Complex Action Spaces (Hubert et al., 2021)](https://arxiv.org/abs/2104.06303) - Sampled MuZero, extending MuZero to continuous and large discrete action spaces.
- [Mastering Atari Games with Limited Data (Ye et al., 2021)](https://arxiv.org/abs/2111.00210) - EfficientZero, achieving strong performance with 100x less data than MuZero.

## Related Work

- [World Models (Ha and Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122) - An alternative model-based RL approach that learns a generative environment model.
- [Mastering Diverse Domains through World Models (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) - DreamerV3, scaling the world model approach to diverse domains.

## Tutorials and Guides

- [DeepMind MuZero Blog Post](https://www.deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules) - Official DeepMind blog post explaining MuZero at a high level.
- [MuZero General (GitHub)](https://github.com/werner-duvaud/muzero-general) - Community PyTorch reimplementation of MuZero for experimentation.
- [Google DeepMind mctx (GitHub)](https://github.com/google-deepmind/mctx) - JAX library for MCTS, including MuZero-style search.

## Reference Implementations

- [MuZero General](https://github.com/werner-duvaud/muzero-general) - Well-documented Python/PyTorch implementation supporting multiple environments.
- [EfficientZero (GitHub)](https://github.com/YeWR/EfficientZero) - Official implementation of EfficientZero with MuZero as the base algorithm.
