# Policy-Based Methods

## Summary

Policy-based methods are a family of reinforcement learning algorithms that directly parameterize and optimize a policy -- a mapping from states to a probability distribution over actions -- rather than learning a value function and deriving a policy from it. Where value-based methods like Q-Learning and DQN answer "how valuable is each action?" and select the argmax, policy-based methods answer "what is the probability of taking each action?" and optimize those probabilities directly via gradient ascent on expected cumulative reward. This direct approach resolves several fundamental limitations of value-based methods: it naturally handles continuous action spaces, produces stochastic policies when stochasticity is optimal, and provides smoother optimization landscapes where small parameter changes cause small behavioral changes rather than abrupt policy switches.

The theoretical foundation of all policy-based methods is the policy gradient theorem (Sutton et al., 1999), which expresses the gradient of expected return as an expectation over trajectories sampled from the current policy. This gradient can be estimated from experience without requiring a model of the environment, making policy gradient methods model-free. The gradient estimator involves multiplying the log-probability of actions taken by a signal indicating how good those actions were -- the return, the advantage, or some other estimate of action quality. The policy gradient theorem is what enables these methods to optimize policies end-to-end using standard automatic differentiation and stochastic gradient ascent.

The evolution of policy-based methods traces a clear path from simple but impractical to sophisticated and production-ready. REINFORCE (Williams, 1992) is the simplest implementation of the policy gradient theorem, using full Monte Carlo returns to weight gradient updates. It is conceptually clean and theoretically unbiased, but its high variance makes it impractical for all but the simplest problems. Actor-Critic methods address this by introducing a learned value function (the critic) that provides lower-variance advantage estimates through bootstrapping, at the cost of some bias. A2C and A3C (Mnih et al., 2016) scaled actor-critic methods to complex environments by using parallel workers and n-step returns. TRPO (Schulman et al., 2015) added a trust region constraint based on KL divergence to prevent catastrophically large policy updates, providing monotonic improvement guarantees. PPO (Schulman et al., 2017) achieved similar stability through a much simpler clipped surrogate objective, becoming the dominant algorithm in modern deep RL and the standard choice for RLHF in large language model alignment.

Key points to remember:

- Policy-based methods parameterize the policy directly (e.g., with a neural network) and optimize it via gradient ascent on expected return
- The policy gradient theorem is the shared theoretical foundation: it expresses the performance gradient as an expectation that can be estimated from sampled trajectories without a model of the environment
- Policy-based methods naturally handle continuous action spaces, stochastic policies, and large or structured action spaces where argmax-based approaches fail
- The central challenge is variance reduction: raw policy gradient estimates are unbiased but noisy, requiring techniques like baselines, bootstrapping, and advantage normalization to be practical
- The field evolved from REINFORCE (high variance, simple) through Actor-Critic (reduced variance, some bias) to A2C/A3C (parallel training), TRPO (trust region stability), and PPO (simple clipped objective, production-ready)
- Nearly all modern deep RL systems for continuous control, game playing, and LLM alignment use policy-based methods, specifically actor-critic variants like PPO

---

## Why Policy-Based Methods Exist

### Limitations of Value-Based Methods

Value-based methods like Q-Learning and DQN learn an action-value function Q(s, a) and derive a policy by selecting the action with the highest Q-value: pi(s) = argmax_a Q(s, a). This approach works well for problems with small, discrete action spaces, but it encounters three fundamental limitations:

**Continuous action spaces.** The argmax operation requires evaluating Q(s, a) for every possible action and selecting the best one. When the action space is continuous -- for example, the torques applied to each joint of a robotic arm, or the throttle and steering angle of a vehicle -- there are infinitely many possible actions. You cannot enumerate them, and the argmax has no closed-form solution for a general neural network Q-function. Methods like DDPG and TD3 work around this by learning a separate actor network, but at that point they have become actor-critic methods, not pure value-based methods.

**Deterministic policies only.** The argmax always produces the same action in a given state, yielding a deterministic policy. There are settings where a stochastic policy is genuinely better: partially observable environments where different actions should be taken with different probabilities because the agent cannot distinguish between states that require different responses; multi-agent games where a deterministic policy can be exploited by an adversary; and exploration, where randomization is essential for discovering rewarding behaviors. Value-based methods require ad-hoc exploration strategies like epsilon-greedy, which are effective but unprincipled.

**Sensitivity to value estimation errors.** When two actions have similar Q-values, small estimation errors can cause the argmax to consistently select the wrong action. The policy flips entirely based on which action happens to have a slightly higher estimated value. This discontinuity means that small changes in the Q-function can cause large, abrupt changes in behavior, making training unstable.

### What Policy-Based Methods Offer

Policy gradient methods address all three limitations by directly parameterizing the policy as a probability distribution pi_theta(a | s):

- **Continuous action spaces**: The policy network outputs the parameters of a distribution (e.g., the mean and standard deviation of a Gaussian), and actions are sampled from it. No argmax is needed.
- **Stochastic policies**: The policy is inherently a distribution over actions, so stochasticity is built in. The optimal level of stochasticity emerges from the optimization.
- **Smooth optimization**: Small changes to the policy parameters theta cause small changes in the action probabilities, leading to smooth, gradual behavioral changes rather than abrupt switches.

Additionally, policy-based methods offer convergence guarantees that value-based methods with function approximation lack. Under mild conditions, policy gradient methods converge to a local optimum of the expected return. Value-based methods with function approximation can diverge -- the "deadly triad" of function approximation, bootstrapping, and off-policy learning can cause Q-values to grow without bound.

---

## The Policy Gradient Theorem

The policy gradient theorem is the mathematical foundation on which every algorithm in this chapter is built. It provides a way to compute the gradient of expected return with respect to the policy parameters, using only quantities that can be estimated from sampled experience.

### Setup

Consider an agent with a parameterized policy pi_theta(a | s) interacting with an environment. A trajectory tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T) is a sequence of states, actions, and rewards. The objective is to maximize the expected discounted return:

```
J(theta) = E_{tau ~ pi_theta} [ sum_{t=0}^{T} gamma^t * r_t ]
```

We need grad_theta J(theta) to perform gradient ascent.

### The Theorem

The policy gradient theorem (Sutton et al., 1999) states:

```
grad_theta J(theta) = E_{tau ~ pi_theta} [ sum_{t=0}^{T} grad_theta log pi_theta(a_t | s_t) * Psi_t ]
```

where Psi_t is a signal indicating the quality of action a_t. Different choices of Psi_t yield different algorithms:

- Psi_t = G_t (the full return from time t onward): This gives REINFORCE.
- Psi_t = G_t - V(s_t) (the Monte Carlo advantage): This gives REINFORCE with a baseline.
- Psi_t = r_t + gamma * V(s_{t+1}) - V(s_t) (the TD error): This gives one-step actor-critic.
- Psi_t = A_t^GAE (Generalized Advantage Estimation): This gives PPO and modern actor-critic methods.

### Why This Works: The Log-Probability Trick

The key mathematical identity that makes the policy gradient theorem practical is the log-probability trick (also called the score function estimator or REINFORCE trick):

```
grad_theta p_theta(x) = p_theta(x) * grad_theta log p_theta(x)
```

This converts a gradient of an expectation into an expectation of a gradient:

```
grad_theta E_{x ~ p_theta}[f(x)] = E_{x ~ p_theta}[ grad_theta log p_theta(x) * f(x) ]
```

The right-hand side can be estimated from samples: draw trajectories from the current policy, compute the log-probability gradients (via automatic differentiation through the policy network), multiply by the return or advantage, and average. Crucially, the environment dynamics (transition probabilities) do not appear in the gradient expression. They affect which trajectories are sampled, but they do not need to be known or differentiated through. This is what makes policy gradient methods model-free.

### Practical Significance

The policy gradient theorem tells us that to improve the policy, we should:

1. Increase the probability of actions that led to better-than-expected outcomes (positive advantage)
2. Decrease the probability of actions that led to worse-than-expected outcomes (negative advantage)
3. Leave unchanged the probability of actions that performed about as expected (near-zero advantage)

The magnitude of the change is proportional to how much better or worse the action was. This is an intuitive and principled update rule, and all the algorithms in this chapter implement it with different choices for estimating the advantage and different mechanisms for controlling the update magnitude.

---

## Variance Reduction: The Central Challenge

The policy gradient theorem provides an unbiased gradient estimator, but the raw estimator has extremely high variance. Variance reduction is the dominant concern in the design of policy gradient algorithms, and the evolution from REINFORCE to PPO can be understood almost entirely as a sequence of variance reduction innovations.

### Why Variance Is High

The gradient estimator multiplies log-probability gradients by returns or advantages. The return G_t is a sum of many random variables -- all future rewards from time t to the end of the episode. Each future reward depends on a stochastic action selection and a stochastic environment transition. For an episode of length T, the return at time 0 depends on roughly T random variables, and the variance of the sum grows roughly linearly with T. Two trajectories starting from the same state can yield wildly different returns.

This variance means that individual gradient estimates are very noisy. Many trajectories are needed before the gradient estimate points consistently in the direction of improvement. The practical consequences are slow convergence, high sample complexity, and sensitivity to hyperparameters like the learning rate.

### Baseline Subtraction

The most fundamental variance reduction technique is subtracting a baseline b(s) from the return:

```
grad_theta J(theta) = E [ sum_t grad_theta log pi_theta(a_t | s_t) * (G_t - b(s_t)) ]
```

Any baseline that depends only on the state (not the action) preserves unbiasedness. The optimal baseline is approximately the state-value function V(s), which gives the advantage A(s, a) = G_t - V(s_t). This centers the gradient signal around zero: actions better than average get positive weight, actions worse than average get negative weight. Without a baseline, if all returns are positive (which is common), every action gets reinforced -- just by different amounts -- and the useful relative signal is buried under a large common-mode signal.

In practice, V(s) is approximated by a learned neural network (the critic), giving rise to the actor-critic architecture.

### Bootstrapping and Temporal-Difference Estimation

Instead of using the full Monte Carlo return G_t (which depends on all future randomness), we can replace part of the return with a value function estimate:

- One-step TD: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
- N-step return: A_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n}) - V(s_t)
- GAE: A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

Bootstrapping replaces future randomness with a value estimate. This introduces bias (the value estimate is imperfect) but dramatically reduces variance (fewer random variables in the estimate). The bias-variance tradeoff is controlled by how many real rewards are used before bootstrapping: more real rewards means less bias but more variance.

Generalized Advantage Estimation (GAE), introduced by Schulman et al. in 2016, provides a continuous interpolation between one-step TD (lambda = 0) and full Monte Carlo (lambda = 1) through the parameter lambda. With the typical setting of lambda = 0.95, GAE provides a practical sweet spot that works well across most environments. GAE is the standard advantage estimator in PPO and most modern actor-critic implementations.

### Other Variance Reduction Techniques

- **Advantage normalization**: Normalizing the advantage estimates to zero mean and unit variance within each minibatch stabilizes gradient magnitudes across updates. This is simple but surprisingly impactful.
- **Batch updates**: Averaging gradients over multiple trajectories or parallel environments reduces variance by a factor of 1/N.
- **Entropy regularization**: Adding an entropy bonus to the objective encourages the policy to remain stochastic, which smooths the optimization landscape and prevents premature convergence.
- **Reward normalization**: Scaling rewards or returns to consistent magnitudes prevents the gradient from being dominated by reward scale rather than relative action quality.

---

## The Evolution of Policy-Based Methods

The development of policy-based methods follows a clear trajectory, with each algorithm addressing a specific limitation of its predecessor. Understanding this progression is essential for knowing when to use each method and why.

### REINFORCE: The Foundation

REINFORCE (Williams, 1992) is the simplest implementation of the policy gradient theorem. It collects complete episodes, computes Monte Carlo returns, and updates the policy to reinforce actions that led to high returns. The gradient estimate is unbiased, but its high variance makes REINFORCE impractical for anything beyond toy environments.

REINFORCE introduced the key concepts: parameterized policies, the log-probability trick, gradient ascent on expected return, and the idea of using complete episode returns as the learning signal. These concepts persist in every subsequent algorithm.

The limitation that drove the next development: high variance from Monte Carlo returns and the inability to learn from incomplete episodes.

See the [REINFORCE](./reinforce/ReadMe.md) chapter for full details, derivations, and implementation.

### Actor-Critic: Variance Reduction Through Bootstrapping

Actor-Critic methods (Konda and Tsitsiklis, 2000) address REINFORCE's variance problem by introducing a learned value function (the critic) that participates in return estimation through bootstrapping. Instead of using the full Monte Carlo return, the critic provides a TD-based advantage estimate:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

This TD error is a one-step estimate of the advantage. It depends on only one reward rather than all future rewards, dramatically reducing variance at the cost of some bias from the imperfect value function.

The actor-critic architecture -- a policy network (actor) trained with advantage-weighted policy gradients, and a value network (critic) trained with temporal-difference or regression losses -- became the template for all subsequent algorithms. The two networks can share a backbone or be entirely separate.

The limitation that drove the next development: single-environment data collection was slow, and there was no mechanism to prevent the policy from changing too much in a single update.

See the [Actor-Critic](./actor-critic/ReadMe.md) chapter for architecture variants, GAE, and implementation.

### A2C/A3C: Scaling with Parallel Workers

A3C (Asynchronous Advantage Actor-Critic, Mnih et al., 2016) scaled actor-critic to complex environments by running multiple workers in parallel, each interacting with its own copy of the environment. The parallel workers decorrelate training data without needing a replay buffer (unlike DQN), enabling on-policy learning at scale. Each worker computes gradients and asynchronously applies them to a shared global network.

A2C (the synchronous variant) replaced asynchronous updates with synchronized batch updates, which turned out to be simpler, more GPU-friendly, and equally effective. Modern implementations use vectorized environments rather than multi-threading.

A2C/A3C also standardized the use of n-step returns and entropy regularization as practical components of actor-critic training.

The limitation that drove the next development: no mechanism to control how much the policy changes at each update. A large gradient step could catastrophically collapse performance, and recovery could take many iterations.

See the [A2C/A3C](./a2c-a3c/ReadMe.md) chapter for asynchronous vs. synchronous training and implementation.

### TRPO: Principled Stability Through Trust Regions

TRPO (Trust Region Policy Optimization, Schulman et al., 2015) addressed the step-size problem by framing each policy update as a constrained optimization: maximize the expected advantage of the new policy subject to a KL divergence constraint between old and new policies. This constraint defines a trust region in policy space, and updates are only accepted if they stay within it. The result is a monotonic improvement guarantee -- each update is provably no worse than the previous policy, up to approximation error.

TRPO computes the natural gradient direction (which accounts for the curvature of policy space) using conjugate gradient with Fisher-vector products, then performs a line search to find a step that satisfies the KL constraint. This second-order optimization is theoretically elegant but computationally expensive and complex to implement.

The limitation that drove the next development: TRPO's implementation complexity. Conjugate gradient solvers, Fisher-vector products, and line searches are difficult to implement correctly, hard to parallelize on GPUs, and computationally expensive relative to first-order methods.

See the [TRPO](./trpo/ReadMe.md) chapter for the constrained optimization formulation, natural gradients, and comparison with PPO.

### PPO: Simple, Stable, and Dominant

PPO (Proximal Policy Optimization, Schulman et al., 2017) achieves trust-region-like stability through a much simpler mechanism: a clipped surrogate objective. Instead of constraining the KL divergence with a second-order optimizer, PPO clips the importance sampling ratio between old and new policies:

```
L^CLIP = E_t [ min( r_t * A_t, clip(r_t, 1 - epsilon, 1 + epsilon) * A_t ) ]
```

This prevents the policy from changing too much at each step, providing a soft trust region. The clipping is per-sample, adaptive, and requires only first-order optimization (standard SGD or Adam). PPO also enables multiple epochs of minibatch updates on the same batch of experience, improving sample efficiency compared to A2C.

PPO has become the dominant algorithm in modern RL. It is the standard choice for continuous control, game playing, robotics, and the RLHF stage of large language model alignment (InstructGPT, ChatGPT, Claude, Llama). Its combination of simplicity, stability, and strong performance across diverse tasks has made it the default starting point for most RL projects.

See the [PPO](./ppo/ReadMe.md) chapter for the clipped objective, implementation details, and RLHF applications.

---

## Comparison with Value-Based Methods

Understanding when to use policy-based versus value-based methods is a critical practical decision. The two families have complementary strengths.

| Aspect | Policy-Based Methods | Value-Based Methods (DQN family) |
|---|---|---|
| What is learned | Policy pi_theta(a given s) directly | Action-value function Q(s, a) |
| Policy derivation | Explicit parameterized distribution | Implicit via argmax of Q-values |
| Action spaces | Discrete and continuous | Discrete only |
| Policy type | Stochastic (natural) or deterministic | Deterministic (argmax) with ad-hoc exploration |
| Sample usage | Typically on-policy (discard after use) | Off-policy (replay buffer, reuse data) |
| Sample efficiency | Lower (on-policy methods discard data) | Higher (replay buffer enables data reuse) |
| Exploration | Built-in via stochastic policy | Requires explicit strategy (epsilon-greedy) |
| Optimization stability | Smooth (small param changes cause small policy changes) | Can be discontinuous (argmax switches) |
| Convergence | Local optimum of expected return | Can diverge with function approximation |
| Variance | Higher (policy gradient estimates are noisy) | Lower (TD targets are less noisy) |
| Computational cost | Moderate (first-order methods) | Moderate (replay buffer overhead) |

### When to Use Policy-Based Methods

- **Continuous action spaces**: This is the strongest argument. DQN and its variants cannot handle continuous actions without significant modification. Policy-based methods handle them natively by outputting distribution parameters.
- **Stochastic policies needed**: In partially observable environments, adversarial settings, or problems where the optimal policy is genuinely stochastic, policy-based methods represent this naturally.
- **Large or structured action spaces**: When the action space is very large (e.g., combinatorial actions), policy networks can generalize across actions, while Q-networks must evaluate each action independently.
- **Smooth behavioral requirements**: In applications like robotics or control, gradual policy changes are safer and more desirable than abrupt switches.
- **LLM alignment and RLHF**: PPO is the standard algorithm for fine-tuning language models with human feedback.

### When to Use Value-Based Methods

- **Small discrete action spaces**: When the action space is manageable (e.g., a few dozen actions), DQN variants are simpler, better understood, and more sample-efficient due to replay buffers.
- **Sample efficiency is critical**: Off-policy methods with replay buffers can reuse data many times, while on-policy policy gradient methods discard data after one use.
- **Offline RL**: When you only have a fixed dataset of logged interactions and cannot collect new experience, value-based offline methods (CQL, IQL) are more mature than policy-based offline methods.

### In Practice: The Actor-Critic Middle Ground

Most modern deep RL systems use actor-critic methods, which combine a parameterized policy (the actor, from policy-based methods) with a learned value function (the critic, from value-based methods). This hybrid architecture gets the benefits of both families: the actor handles continuous actions and stochastic policies, while the critic provides low-variance feedback for training the actor. PPO, SAC, TD3, and DDPG are all actor-critic methods. Pure REINFORCE and pure DQN are primarily of pedagogical interest.

---

## When to Use Each Policy-Based Method

For practitioners choosing between the methods covered in this chapter, here is a practical decision guide:

**Start with PPO.** For the vast majority of problems -- discrete actions, continuous actions, single environments, parallel environments, on-policy training -- PPO is the best starting point. It is the most robust, most widely supported, and most extensively tested policy gradient algorithm. Major RL libraries (Stable Baselines3, RLlib, CleanRL, TorchRL) all provide well-tested PPO implementations.

**Use A2C for learning.** If you are implementing an actor-critic algorithm from scratch to understand the mechanics, A2C is simpler than PPO and teaches the core concepts (advantage estimation, entropy regularization, parallel data collection) without the added complexity of the clipped objective and multiple epochs.

**Use REINFORCE for education.** REINFORCE is the clearest implementation of the policy gradient theorem. Use it to understand the fundamentals -- the log-probability trick, Monte Carlo returns, baseline subtraction -- before moving to more practical algorithms.

**Consider TRPO for safety-critical applications.** When you need the strongest guarantees on policy stability and can afford the implementation complexity, TRPO provides a principled trust region constraint. In most other settings, PPO achieves comparable results with far less effort.

**Consider off-policy actor-critic methods (SAC, TD3) when sample efficiency matters.** PPO and the other on-policy methods in this chapter discard data after each update. If environment interactions are expensive and you need maximum sample efficiency, off-policy methods with replay buffers are more appropriate. These are covered in separate chapters.

---

## Summary of the Method Family

```
Policy Gradient Methods
|
|-- REINFORCE (Monte Carlo returns, no critic, high variance)
|       Foundational algorithm. Unbiased but impractical for complex tasks.
|
|-- Actor-Critic (bootstrapped advantage, learned critic)
|   |   Reduces variance via TD-based advantage estimation.
|   |
|   |-- A2C / A3C (parallel workers, n-step returns, entropy bonus)
|   |       Scales actor-critic to complex environments.
|   |
|   |-- TRPO (KL-constrained trust region, natural gradient)
|   |       Principled stability via constrained optimization.
|   |
|   |-- PPO (clipped surrogate objective, multiple epochs)
|           Simple, stable, dominant. The practical standard.
```

Each step in this progression addresses a specific failure mode of the previous method. REINFORCE has high variance, so Actor-Critic introduces bootstrapping. Actor-Critic has no mechanism to prevent destructive updates, so TRPO adds a trust region constraint. TRPO is too complex to implement, so PPO replaces it with a simple clipped objective. Understanding this progression -- and the specific problems each method solves -- is what allows a practitioner to make informed algorithmic choices rather than defaulting to PPO for everything.

---

## Child Chapters

- [REINFORCE](./reinforce/ReadMe.md) -- Monte Carlo policy gradient, the log-probability trick, baseline subtraction
- [Actor-Critic](./actor-critic/ReadMe.md) -- TD-based advantage estimation, GAE, shared and separate architectures
- [A2C/A3C](./a2c-a3c/ReadMe.md) -- Asynchronous and synchronous parallel training, n-step returns, entropy regularization
- [TRPO](./trpo/ReadMe.md) -- Trust region constraint, KL divergence, natural gradient, conjugate gradient
- [PPO](./ppo/ReadMe.md) -- Clipped surrogate objective, multiple epochs, RLHF applications
