# TRPO (Trust Region Policy Optimization)

## Summary

Trust Region Policy Optimization (TRPO) is a policy gradient algorithm introduced by Schulman et al. in 2015 that provides a principled approach to updating policy parameters with guaranteed monotonic improvement. The core problem TRPO addresses is a fundamental instability in vanilla policy gradient methods: a single large update can catastrophically destroy a good policy, and there is no reliable way to choose a learning rate that avoids this. TRPO solves this by framing policy optimization as a constrained optimization problem, where each update must keep the new policy "close" to the old policy as measured by KL divergence. This constraint defines a trust region in policy space, and the algorithm only takes steps within that region, ensuring that the theoretical performance bound is respected at every iteration.

TRPO was a landmark contribution because it bridged the gap between the theoretical guarantees of conservative policy iteration and practical deep reinforcement learning. Before TRPO, practitioners relied on ad-hoc tricks like small learning rates, gradient clipping, and reward scaling to prevent policy collapse during training. TRPO replaced these heuristics with a mathematically grounded procedure. However, TRPO's implementation complexity -- involving conjugate gradient solvers, Fisher vector products, and line searches -- made it difficult to use in practice. This directly motivated the development of Proximal Policy Optimization (PPO), which achieves similar stability through a much simpler clipped surrogate objective. PPO has largely replaced TRPO in most applied settings, but understanding TRPO is essential for grasping why PPO works and when the simpler approximation might fall short.

Key points to remember:

- TRPO maximizes a surrogate objective (expected advantage under the new policy) subject to a KL divergence constraint between old and new policies
- The KL constraint provides a monotonic improvement guarantee: each update is provably no worse than the previous policy (up to approximation error)
- The constrained optimization is solved approximately using conjugate gradient to compute the natural gradient direction, followed by a line search with backtracking
- TRPO is closely related to natural gradient methods; the KL constraint induces the Fisher information matrix as the metric for the parameter update
- PPO was designed as a first-order approximation to TRPO that drops the conjugate gradient machinery in favor of a clipped objective or adaptive KL penalty
- TRPO works with both discrete and continuous action spaces and is compatible with actor-critic architectures
- Implementation is significantly more complex than PPO, REINFORCE, or A2C/A3C, which is the primary reason for its reduced adoption

When to use TRPO:

- When you need stronger theoretical guarantees on policy improvement than PPO provides
- Research settings where principled optimization is more important than implementation simplicity
- Problems where PPO's clipping heuristic leads to suboptimal behavior
- As a reference implementation to validate that a simpler method like PPO is performing adequately

When not to use TRPO:

- Most practical applications where PPO achieves comparable results with far less code and compute
- Large-scale distributed training where TRPO's second-order computations are a bottleneck
- Rapid prototyping or experimentation where development speed matters
- When using very large policy networks, where the conjugate gradient step becomes expensive

---

## The Motivation: Why Vanilla Policy Gradients Are Fragile

### The Step Size Problem

In vanilla policy gradient methods like REINFORCE, the update rule is:

```
theta_{k+1} = theta_k + alpha * grad J(theta_k)
```

where J(theta) is the expected cumulative reward and alpha is the learning rate. The gradient grad J(theta) tells you the direction that improves performance, but it says nothing about how far you should step in that direction. This is a serious problem in policy optimization for two reasons:

1. **Policy sensitivity to parameters**: Small changes in theta can cause large changes in the policy's behavior. A step that looks small in parameter space can dramatically alter the probability distribution over actions, causing the agent to suddenly take very different actions and collapse its performance.

2. **Non-stationarity of the objective landscape**: The objective J(theta) depends on the data distribution induced by the policy. When the policy changes, the data distribution changes too, which means the landscape itself shifts under the optimizer's feet. A step size that was safe for one update may be catastrophic for the next.

Practitioners addressed this with small learning rates, but this made training extremely slow. Gradient clipping helped prevent the worst blow-ups but offered no guarantees. What was needed was a principled way to limit how much the policy itself (not just the parameters) changes at each step.

### Kakade and Langford's Conservative Policy Iteration

The theoretical foundation for TRPO comes from work by Kakade and Langford (2002) on conservative policy iteration. They showed that if you have an estimate of the advantage function A_pi(s, a) under the current policy pi, you can construct a lower bound on the performance of a new policy pi':

```
J(pi') >= J(pi) + sum_s rho_pi(s) * sum_a pi'(a|s) * A_pi(s, a) - C * D_KL_max(pi, pi')
```

where rho_pi(s) is the discounted state visitation frequency under pi, and D_KL_max is the maximum KL divergence between the two policies across all states. The constant C depends on the discount factor and the maximum advantage.

This bound says: the new policy's performance is at least as good as the old policy's performance plus the expected advantage of the new actions, minus a penalty proportional to how different the new policy is from the old one. If you keep the KL divergence small enough, the expected advantage term will dominate the penalty term, and improvement is guaranteed.

---

## The TRPO Objective

### The Surrogate Objective

TRPO formulates policy optimization as a constrained problem. Define the surrogate objective (also called the surrogate advantage or the policy ratio objective):

```
L(theta) = E_t [ (pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)) * A_t ]
```

where pi_theta is the new policy, pi_theta_old is the policy used to collect the data, and A_t is the advantage estimate at timestep t. The ratio pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t) is often written as r_t(theta) and is called the importance sampling ratio. When theta = theta_old, this ratio equals 1 and L(theta) equals the expected advantage, which is zero.

The surrogate objective L(theta) is a first-order approximation to the actual performance difference J(pi_theta) - J(pi_theta_old). It is accurate in the neighborhood of theta_old but can become arbitrarily inaccurate as theta moves far from theta_old. This is precisely why the trust region constraint is needed.

### The Constrained Optimization Problem

TRPO solves:

```
maximize    L(theta)
   theta

subject to  E_t [ D_KL( pi_theta_old(.|s_t) || pi_theta(.|s_t) ) ] <= delta
```

where delta is a hyperparameter (typically 0.01) that defines the size of the trust region. The constraint uses the mean KL divergence across states visited under the old policy, which is a practical relaxation of the maximum KL divergence in the theoretical bound.

In words: find the parameters theta that maximize the expected advantage of the new policy over the old policy, subject to the constraint that the new policy does not deviate too far from the old policy as measured by KL divergence.

### Why KL Divergence?

KL divergence is the natural measure of distance between probability distributions for several reasons:

- It directly measures how different the two policies' action distributions are, regardless of how the parameters are structured
- It is invariant to reparameterization of the policy (a linear transformation of the parameters does not change the KL divergence between the resulting distributions)
- It connects to the Fisher information matrix, which provides the natural gradient -- the steepest ascent direction in distribution space rather than parameter space
- The monotonic improvement bound is stated in terms of KL divergence, so constraining KL directly maps onto the theoretical guarantee

---

## Solving the Constrained Problem

### Second-Order Approximation

The constrained optimization problem cannot be solved exactly in closed form for neural network policies. TRPO uses a sequence of approximations:

1. **Linearize the objective**: Approximate L(theta) by its first-order Taylor expansion around theta_old:

```
L(theta) ~ g^T (theta - theta_old)
```

where g = grad_theta L(theta)|_{theta=theta_old} is the policy gradient.

2. **Quadraticize the constraint**: Approximate the KL divergence by its second-order Taylor expansion around theta_old:

```
D_KL(theta_old || theta) ~ (1/2) (theta - theta_old)^T F (theta - theta_old)
```

where F is the Fisher information matrix (the Hessian of the KL divergence at theta_old). The first-order term vanishes because the KL divergence is zero and has zero gradient at theta = theta_old.

With these approximations, the problem becomes:

```
maximize    g^T (theta - theta_old)

subject to  (1/2) (theta - theta_old)^T F (theta - theta_old) <= delta
```

This is a quadratically-constrained linear program, which has an analytical solution.

### The Analytical Solution

Using the method of Lagrange multipliers, the optimal update direction is:

```
theta - theta_old = sqrt(2 * delta / (g^T F^{-1} g)) * F^{-1} g
```

The direction F^{-1} g is the natural gradient. The scalar factor sqrt(2 * delta / (g^T F^{-1} g)) sets the step size to exactly satisfy the KL constraint with equality.

### The Fisher Information Matrix

The Fisher information matrix F is defined as:

```
F = E_t [ grad_theta log pi_theta(a_t|s_t) * (grad_theta log pi_theta(a_t|s_t))^T ]
```

evaluated at theta = theta_old. It is the covariance of the score function (the gradient of the log-policy). The Fisher matrix captures the curvature of the policy space: directions in which the policy distribution changes rapidly have large entries in F, while directions in which the policy is insensitive have small entries.

For a neural network with millions of parameters, F is a matrix with potentially trillions of entries. Explicitly forming and inverting F is completely infeasible. This is where conjugate gradient comes in.

---

## Conjugate Gradient and Fisher Vector Products

### Avoiding Explicit Matrix Construction

TRPO never computes or stores the full Fisher matrix. Instead, it uses the conjugate gradient (CG) algorithm to approximately solve:

```
F x = g
```

for x = F^{-1} g, where g is the policy gradient vector. The conjugate gradient method is an iterative algorithm that only requires the ability to compute matrix-vector products Fv for arbitrary vectors v. Each CG iteration requires one such product, and typically 10-20 iterations suffice for a good approximation.

### Computing Fisher Vector Products

The Fisher-vector product Fv can be computed efficiently using automatic differentiation. The key identity is:

```
Fv = grad_theta [ (grad_theta D_KL)^T v ]
```

This is a "Hessian-vector product" of the KL divergence, computed by:

1. Compute the gradient of the mean KL divergence: g_kl = grad_theta D_KL(theta_old || theta)|_{theta=theta_old}
2. Compute the dot product g_kl^T v (a scalar)
3. Differentiate this scalar with respect to theta to get the Fisher-vector product

In PyTorch or TensorFlow, this amounts to two backward passes. The first pass computes g_kl, and the second pass differentiates the dot product g_kl^T v. This avoids ever forming the full matrix and runs in time proportional to a single gradient computation.

### The Conjugate Gradient Procedure

```
Initialize x_0 = 0, r_0 = g, d_0 = g
For i = 0, 1, ..., K-1:
    z = F * d_i                          (Fisher vector product)
    alpha = (r_i^T r_i) / (d_i^T z)
    x_{i+1} = x_i + alpha * d_i
    r_{i+1} = r_i - alpha * z
    beta = (r_{i+1}^T r_{i+1}) / (r_i^T r_i)
    d_{i+1} = r_{i+1} + beta * d_i
Return x_K (approximate solution to F^{-1} g)
```

Typically K = 10 iterations are used. The cost is K Fisher-vector products, each roughly equivalent to two gradient computations. So the total cost of the CG step is about 20 gradient computations, which is the main computational overhead compared to first-order methods.

---

## Line Search with Backtracking

### Why a Line Search Is Needed

The solution from conjugate gradient gives the update direction, but the linear and quadratic approximations used to derive it are only locally accurate. Taking the full step may:

- Violate the KL constraint (because the quadratic approximation of KL underestimates the true KL for large steps)
- Decrease the actual surrogate objective (because the linear approximation overestimates the true objective far from theta_old)

TRPO therefore performs a backtracking line search after computing the update direction. Starting from the full step size, it repeatedly halves the step until two conditions are satisfied:

1. The true mean KL divergence (computed on the sampled data) is within the trust region: D_KL <= delta
2. The true surrogate objective L(theta) actually improves

### The Line Search Procedure

```
Compute search direction: s = F^{-1} g  (via conjugate gradient)
Compute full step size: beta = sqrt(2 * delta / (s^T F s))
For j = 0, 1, 2, ..., max_backtracks:
    theta_new = theta_old + (0.5^j) * beta * s
    Compute L(theta_new) and D_KL(theta_old || theta_new) on the batch
    If L(theta_new) > 0 and D_KL <= delta:
        Accept theta_new
        Break
If no step accepted:
    Keep theta_old (no update this iteration)
```

The backtracking typically terminates within 1-3 iterations. Occasionally, no valid step is found, and the parameters remain unchanged. This is a feature, not a bug: it means the trust region is too small to make progress in the computed direction, and the algorithm conservatively stays put rather than risking a bad update.

---

## Connection to Natural Gradient Methods

### What Is the Natural Gradient?

The standard gradient grad_theta J(theta) gives the steepest ascent direction in Euclidean parameter space. But parameter space is not the right geometry for probability distributions. A step of length epsilon in parameter space can produce wildly different changes in the policy distribution depending on the region of parameter space.

The natural gradient, introduced by Amari (1998), uses the Fisher information matrix as a Riemannian metric on the parameter space. The natural gradient direction is:

```
F^{-1} grad_theta J(theta)
```

This gives the steepest ascent direction in the space of distributions, not parameters. It is invariant to reparameterization: if you transform your parameters (e.g., switch from Cartesian to polar coordinates for a 2D Gaussian policy), the natural gradient direction in distribution space remains the same.

### TRPO as Natural Gradient with Adaptive Step Size

TRPO can be understood as natural gradient ascent with an adaptive step size determined by the trust region constraint. The update direction F^{-1} g is the natural gradient, and the step size sqrt(2 * delta / (g^T F^{-1} g)) is chosen to exactly satisfy the KL constraint. This means:

- Unlike fixed-learning-rate natural gradient, TRPO automatically adjusts how far it steps based on the local curvature of the policy space
- In flat regions of policy space (where the Fisher information is small), TRPO takes larger steps in parameter space
- In highly curved regions (where small parameter changes cause large distribution changes), TRPO takes smaller steps

This adaptive behavior is a significant advantage over both vanilla policy gradients (which use the wrong geometry) and fixed-step-size natural gradient (which may overshoot or undershoot).

---

## The TRPO Algorithm End to End

Putting all the pieces together, the complete TRPO algorithm for one iteration is:

```
1. Run the current policy pi_theta_old in the environment to collect a batch
   of trajectories {(s_t, a_t, r_t)}

2. Compute advantage estimates A_t for each timestep (typically using
   Generalized Advantage Estimation / GAE with a learned value function baseline)

3. Compute the policy gradient:
   g = grad_theta L(theta)|_{theta=theta_old}
   where L(theta) = E_t [ r_t(theta) * A_t ] and r_t(theta) is the importance
   sampling ratio

4. Use conjugate gradient (with Fisher vector products) to approximately solve:
   F x = g
   obtaining x ~ F^{-1} g

5. Compute the step size:
   beta = sqrt(2 * delta / (x^T F x))

6. Perform backtracking line search:
   For j = 0, 1, 2, ...
       theta_new = theta_old + (0.5^j) * beta * x
       If L(theta_new) improves and D_KL(theta_old, theta_new) <= delta:
           Accept theta_new; break

7. Update the value function baseline (fit V(s) to the observed returns
   using standard regression)

8. Repeat from step 1
```

### Generalized Advantage Estimation (GAE)

TRPO typically uses GAE (Schulman et al., 2016) to compute advantage estimates. GAE provides a smooth tradeoff between bias and variance through a parameter lambda:

```
A_t^GAE = sum_{l=0}^{infinity} (gamma * lambda)^l * delta_t^V

where delta_t^V = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD residual
```

When lambda = 1, GAE reduces to the Monte Carlo advantage (high variance, zero bias). When lambda = 0, it reduces to the one-step TD advantage (low variance, high bias). Values of lambda = 0.95-0.99 are typical. GAE is not specific to TRPO; it is also used by PPO and other advantage-based methods.

---

## Comparison with PPO

### How PPO Simplifies TRPO

Proximal Policy Optimization (PPO), also by Schulman et al. (2017), was explicitly designed as a simpler alternative to TRPO. PPO drops the constrained optimization machinery entirely and instead modifies the objective function to discourage large policy updates. There are two main PPO variants:

**PPO-Clip** (the dominant variant):

```
L_CLIP(theta) = E_t [ min( r_t(theta) * A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t ) ]
```

The clip function limits the importance sampling ratio to [1 - epsilon, 1 + epsilon] (typically epsilon = 0.2). If the ratio tries to move outside this range, the gradient is zeroed out. This is a crude but effective approximation to the trust region: it prevents the policy from changing too much, but it does so through a hard clip on the ratio rather than a principled KL constraint.

**PPO-Penalty** (less commonly used):

```
L_KL(theta) = E_t [ r_t(theta) * A_t - beta * D_KL(pi_theta_old || pi_theta) ]
```

This adds a KL penalty directly to the objective, with beta adapted during training to target a desired KL divergence. This is closer to TRPO in spirit but still avoids the conjugate gradient step.

### What PPO Gains Over TRPO

| Aspect | TRPO | PPO-Clip |
|---|---|---|
| Gradient computation | Second-order (Fisher-vector products via CG) | First-order only (standard backprop) |
| Lines of code | Hundreds (CG solver, line search, FVP) | Tens (standard SGD loop with clipped loss) |
| Computation per update | ~20x a single gradient (K CG iterations) | ~1x a single gradient |
| Multiple epochs per batch | Typically one pass (constraint harder to satisfy after) | Multiple epochs (3-10) per batch of data |
| Hyperparameter sensitivity | delta (trust region size) is fairly robust | epsilon (clip range) requires some tuning but is also fairly robust |
| Distributed training | Difficult (CG is sequential and hard to parallelize) | Easy (standard minibatch SGD) |
| GPU utilization | Poor (CG loop is compute-bound, not data-parallel) | Excellent (standard forward/backward pass) |

### What PPO Loses Compared to TRPO

PPO's clipping mechanism is a heuristic approximation. There are cases where it can behave differently from a true trust region:

1. **No hard KL guarantee**: PPO clips the ratio, but the actual KL divergence between old and new policies is not explicitly constrained. In practice, the KL often stays reasonable, but there is no formal bound.

2. **Asymmetric clipping effects**: The clip function treats increases and decreases in the ratio differently depending on the sign of the advantage. This can create subtle biases, particularly when advantage estimates are noisy.

3. **Multiple epochs can drift**: PPO runs multiple SGD epochs on the same batch. By the later epochs, the policy may have drifted enough that the importance sampling ratio is stale. TRPO's single-step approach avoids this issue.

4. **Reduced exploration in some settings**: Some empirical studies have found that PPO's clipping can overly constrain exploration, particularly in environments with sparse rewards or deceptive gradients.

### Why PPO Replaced TRPO in Practice

Despite the theoretical advantages of TRPO, PPO became the dominant algorithm for several practical reasons:

- **Simplicity**: PPO can be implemented in under 100 lines of core code. TRPO requires a conjugate gradient solver, Fisher-vector product computation, and a line search, each of which introduces potential for bugs.
- **Performance**: On the vast majority of benchmarks (MuJoCo locomotion, Atari, robotics), PPO matches or comes close to TRPO's performance. The cases where TRPO meaningfully outperforms PPO are rare.
- **Scalability**: PPO is trivially parallelizable and works well with GPU batching. TRPO's conjugate gradient loop is inherently sequential and does not benefit from GPU parallelism in the same way.
- **Ecosystem support**: All major RL libraries (Stable Baselines3, RLlib, CleanRL, TorchRL) provide well-tested PPO implementations. TRPO implementations are less common and less maintained.
- **Flexibility with architectures**: PPO works seamlessly with any differentiable policy architecture, including transformers and large networks. TRPO's Fisher-vector products can become memory-intensive for very large models.

---

## When TRPO Might Still Be Preferred

Despite PPO's dominance, there are scenarios where TRPO offers meaningful advantages:

1. **Safety-critical applications**: In domains like robotic surgery, autonomous driving, or industrial control, the stronger guarantee on policy stability may justify the implementation overhead. A policy that suddenly takes dangerous actions due to an optimization artifact is unacceptable in these settings.

2. **Theoretical research**: When studying the properties of policy optimization algorithms, TRPO provides a cleaner object of analysis. Its behavior is more predictable and easier to reason about mathematically.

3. **Debugging and validation**: Running TRPO alongside PPO can help diagnose whether PPO's clipping is causing issues. If TRPO significantly outperforms PPO on a given problem, the clipping heuristic may be too aggressive or too loose.

4. **Environments with highly sensitive dynamics**: Some environments have the property that small changes in action distributions lead to very different trajectories (chaotic dynamics, contact-rich manipulation). In these cases, the precise KL control of TRPO may provide more stable learning than PPO's approximate constraint.

5. **Small policy networks**: When the policy network is small (a few thousand parameters), the overhead of the conjugate gradient step is negligible, and TRPO's advantages come essentially for free.

---

## Implementation Complexity Breakdown

To give a concrete sense of what implementing TRPO entails versus PPO, here is a comparison of the major components:

### Components Shared by TRPO and PPO

- Environment interaction loop and trajectory collection
- Advantage estimation (GAE)
- Value function fitting (regression on returns)
- Policy network definition
- Importance sampling ratio computation

### Components Unique to TRPO

- **Fisher-vector product function**: Requires computing the gradient of the KL divergence, taking its dot product with a vector, and differentiating again. This uses second-order automatic differentiation, which not all frameworks support cleanly.

- **Conjugate gradient solver**: An iterative linear algebra routine that must be implemented correctly to avoid numerical issues. Preconditioning, early termination criteria, and damping all require careful tuning.

- **Line search**: After computing the update direction, a backtracking procedure that evaluates the true surrogate objective and KL divergence at each candidate step size. This requires additional forward passes through the policy network on the full batch.

- **Damping**: A small constant (typically 0.1) is added to the diagonal of the Fisher matrix (equivalently, added to each Fv product as Fv + damping * v) to ensure numerical stability. Without damping, the conjugate gradient can produce directions that are nearly orthogonal to the gradient, leading to negligible or harmful updates.

### Components Unique to PPO

- **Clipped surrogate loss**: A one-line modification to the standard policy gradient loss. This is trivially implemented using any deep learning framework's min and clamp/clip operations.

The asymmetry is stark. PPO replaces several hundred lines of numerical linear algebra with a single line of loss modification.

---

## Connection to Actor-Critic, A2C/A3C, and PPO

TRPO belongs to the family of policy gradient methods and shares significant DNA with its siblings in the actor-critic family.

### The Policy Gradient Family Tree

```
Policy Gradient Methods
|
|-- REINFORCE (vanilla policy gradient, no baseline, high variance)
|
|-- Actor-Critic (policy gradient with a learned value function baseline)
|   |
|   |-- A2C (synchronous advantage actor-critic)
|   |-- A3C (asynchronous advantage actor-critic)
|   |
|   |-- TRPO (actor-critic with trust region constraint)
|   |   |
|   |   |-- PPO (simplified TRPO with clipped surrogate)
```

### Relationship to Actor-Critic

TRPO is an actor-critic method. The "actor" is the policy network pi_theta(a|s), and the "critic" is the value function V_phi(s) used to compute advantage estimates. The key difference from basic actor-critic methods is how the actor is updated:

- **Basic actor-critic / A2C**: The actor is updated with standard gradient ascent on the policy gradient objective. The step size is a fixed learning rate.
- **A3C**: Same as A2C but with asynchronous parallel workers. Multiple copies of the actor-critic run in parallel, computing gradients and applying them to a shared parameter server.
- **TRPO**: The actor is updated using the trust region constrained optimization described above. The critic is updated using standard regression (MSE loss between V_phi(s) and the observed returns).
- **PPO**: The actor is updated using the clipped surrogate objective with standard SGD. Like TRPO, the critic is trained via regression.

### What Changed from A2C to TRPO

A2C and A3C use the standard policy gradient estimator:

```
grad J(theta) = E_t [ grad_theta log pi_theta(a_t|s_t) * A_t ]
```

and update with:

```
theta <- theta + alpha * grad J(theta)
```

TRPO replaces this simple gradient step with the full constrained optimization procedure. The gradient signal is the same (the advantage-weighted policy gradient), but TRPO uses it more carefully by:

1. Transforming the gradient into the natural gradient direction (accounting for the curvature of policy space)
2. Scaling the step to satisfy the KL constraint (ensuring the policy does not change too much)
3. Verifying the step with a line search (ensuring actual improvement)

### What Changed from TRPO to PPO

PPO keeps TRPO's idea of limiting how much the policy changes but replaces the mechanism:

- TRPO constrains D_KL <= delta and uses second-order optimization to satisfy this constraint
- PPO clips the importance sampling ratio to [1 - epsilon, 1 + epsilon] and uses first-order optimization

Both methods use the same data collection procedure, the same advantage estimation, and the same value function fitting. The only difference is the actor update step.

---

## Pseudocode: TRPO with GAE

```
Hyperparameters: delta (trust region size, e.g., 0.01)
                 gamma (discount factor, e.g., 0.99)
                 lambda (GAE parameter, e.g., 0.97)
                 K (CG iterations, e.g., 10)
                 damping (CG damping, e.g., 0.1)
                 max_backtracks (e.g., 10)

Initialize policy parameters theta, value function parameters phi

For iteration = 1, 2, ...:

    # Collect trajectories
    Run policy pi_theta in the environment for N timesteps
    Store {s_t, a_t, r_t, s_{t+1}, done_t} for all timesteps

    # Compute advantages using GAE
    For each timestep t (in reverse order):
        delta_t = r_t + gamma * (1 - done_t) * V_phi(s_{t+1}) - V_phi(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    Normalize advantages: A = (A - mean(A)) / (std(A) + 1e-8)

    # Compute returns for value function fitting
    R_t = A_t + V_phi(s_t)

    # Policy update via TRPO
    Compute policy gradient: g = grad_theta (1/N) sum_t r_t(theta) * A_t
    Use CG to solve: F x = g  (approximately, with K iterations and damping)
    Compute step size: beta = sqrt(2 * delta / (x^T F x))
    Backtracking line search:
        For j = 0, ..., max_backtracks:
            theta_new = theta + (0.5^j) * beta * x
            If surrogate improves and mean KL <= delta:
                theta <- theta_new
                Break

    # Value function update
    Fit V_phi to targets R_t using SGD (multiple epochs of minibatch updates)
```

---

## Key Hyperparameters

| Hyperparameter | Typical Value | Role |
|---|---|---|
| delta (trust region size) | 0.01 | Maximum mean KL divergence per update. Larger values allow bigger policy changes but risk instability. |
| gamma (discount factor) | 0.99 | Standard RL discount factor for return computation. |
| lambda (GAE parameter) | 0.95 - 0.98 | Bias-variance tradeoff in advantage estimation. |
| CG iterations (K) | 10 - 20 | Number of conjugate gradient steps. More iterations give a better approximation of the natural gradient. |
| CG damping | 0.1 | Regularization added to Fisher-vector products for numerical stability. |
| max_backtracks | 10 - 15 | Maximum number of line search halvings before giving up. |
| Batch size (N) | 2048 - 8192 | Number of environment timesteps per update. Larger batches reduce variance in gradient estimates. |

The trust region size delta is the most important TRPO-specific hyperparameter. A value of 0.01 is nearly universal in the literature and works well across a wide range of tasks. Unlike learning rates in first-order methods, delta has a clear interpretation (the maximum KL divergence), which makes it easier to set and more transferable across problems.

---

## Common Pitfalls

1. **Forgetting the damping term**: Without damping in the Fisher-vector product, the conjugate gradient solution can be numerically unstable, producing near-zero or extremely large updates. Always add a small damping constant (e.g., 0.1) to the Fv computation.

2. **Computing KL in the wrong direction**: The KL divergence is asymmetric. TRPO constrains D_KL(pi_theta_old || pi_theta), not the reverse. Using the wrong direction changes the geometry of the constraint and can lead to different (and worse) behavior.

3. **Not normalizing advantages**: Unnormalized advantages can have arbitrary scale, which interacts poorly with the trust region constraint. Always normalize advantages to zero mean and unit variance before computing the surrogate objective.

4. **Too few CG iterations**: With K = 1 or 2, the conjugate gradient solution is a poor approximation to the natural gradient. This can cause TRPO to behave little better than vanilla policy gradient. Use at least 10 iterations.

5. **Reusing stale data**: TRPO is an on-policy algorithm. Each batch of data should be collected with the current policy and used for exactly one update. Reusing data from previous iterations violates the importance sampling assumptions.

---

## Historical Context and Impact

TRPO was published in 2015 by John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel at Berkeley. It was a turning point in deep reinforcement learning because it showed that policy gradient methods could be made reliable enough for complex continuous control tasks (locomotion in MuJoCo, simulated robotics) without the extensive reward engineering and hyperparameter tuning that had been necessary before.

The direct lineage from TRPO to PPO (published two years later by Schulman et al. in 2017) is one of the clearest examples in machine learning of a theoretically principled method inspiring a simpler practical algorithm. PPO kept the insight (constrain how much the policy changes) but dropped the mechanism (second-order optimization). The fact that PPO works nearly as well as TRPO in most settings is itself a finding about the structure of policy optimization: the precise form of the constraint matters less than having some constraint at all.

TRPO also influenced algorithms beyond PPO. Natural policy gradient methods, maximum a posteriori policy optimization (MPO), and various constrained RL algorithms all draw on the trust region idea. The connection between KL constraints and natural gradients that TRPO made explicit has become a standard tool in the RL researcher's toolkit.
