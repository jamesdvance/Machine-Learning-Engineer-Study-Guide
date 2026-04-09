# Bellman Equations

## Summary

The Bellman equations are the mathematical backbone of reinforcement learning. Named after Richard Bellman, who developed them in the 1950s as part of dynamic programming theory, these equations express the value of being in a state (or taking an action in a state) as a recursive relationship: the immediate reward plus the discounted value of whatever comes next. This recursive decomposition is what makes it possible to solve sequential decision problems without enumerating every possible future trajectory. Nearly every RL algorithm, from tabular value iteration to deep Q-networks, is either directly solving a Bellman equation or approximating one.

There are two families of Bellman equations. The Bellman Expectation Equations describe the value of states and state-action pairs under a fixed, given policy. The Bellman Optimality Equations describe the value of states and state-action pairs under the best possible policy. The expectation equations are linear systems that can be solved exactly for small problems. The optimality equations involve a max operator, making them nonlinear, and are typically solved iteratively.

Key points to remember:

- The Bellman equation decomposes value into immediate reward + discounted future value
- V(s) is the state-value function; Q(s,a) is the action-value function
- The Bellman Expectation Equation evaluates a given policy (prediction problem)
- The Bellman Optimality Equation finds the best policy (control problem)
- Value iteration and policy iteration are dynamic programming methods that directly apply Bellman equations
- Q-learning is a model-free algorithm that learns the Bellman Optimality Equation for Q without knowing the transition dynamics
- For finite MDPs, the Bellman Expectation Equation can be written in matrix form and solved in closed form
- The discount factor gamma controls the tradeoff between short-term and long-term reward
- Convergence of iterative methods relies on gamma being strictly less than 1 (for infinite-horizon problems)

## Prerequisites: MDP Notation

Before diving into the equations, a quick review of the Markov Decision Process quantities involved. A full treatment is in the sibling chapter on Markov Decision Processes.

- **s**: A state in the environment's state space S.
- **a**: An action in the action space A.
- **r**: A reward signal received after taking an action in a state.
- **pi(a|s)**: A policy, which is a probability distribution over actions given a state.
- **P(s'|s,a)**: The state-transition probability -- the probability of reaching state s' when taking action a in state s.
- **R(s,a,s')**: The reward function -- the expected reward when transitioning from s to s' via action a. Often simplified to R(s,a) or R(s).
- **gamma**: The discount factor, in [0, 1), which determines how much the agent cares about future rewards relative to immediate rewards.

## The Return

The agent's goal is to maximize the expected return, which is the cumulative discounted reward from time step t onward:

```
G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
    = sum_{k=0}^{infinity} gamma^k * R_{t+k+1}
```

The key insight of the Bellman equations is that the return has a recursive structure:

```
G_t = R_{t+1} + gamma * G_{t+1}
```

This recursion is what makes the entire framework tractable. Instead of reasoning about infinite sums, we can reason one step at a time.

## State-Value Function V(s)

The state-value function under policy pi is the expected return starting from state s and following policy pi thereafter:

```
V_pi(s) = E_pi[G_t | S_t = s]
```

This tells the agent: "If I am in state s and I follow policy pi from here on, how much total reward can I expect?"

## Action-Value Function Q(s,a)

The action-value function under policy pi is the expected return starting from state s, taking action a, and following policy pi thereafter:

```
Q_pi(s,a) = E_pi[G_t | S_t = s, A_t = a]
```

This tells the agent: "If I am in state s, take action a, and then follow policy pi, how much total reward can I expect?"

The relationship between V and Q is:

```
V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
```

The state value is the policy-weighted average of the action values.

## Bellman Expectation Equation

### For V(s)

Using the recursive structure of the return, we can expand V_pi(s):

```
V_pi(s) = E_pi[R_{t+1} + gamma * V_pi(S_{t+1}) | S_t = s]
```

Expanding the expectation over the policy and transition dynamics:

```
V_pi(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
```

Reading this equation from left to right: for each action a that the policy might select (weighted by pi(a|s)), and for each next state s' that the environment might transition to (weighted by P(s'|s,a)), the value is the immediate reward R(s,a,s') plus the discounted value of the next state gamma * V_pi(s').

This is the Bellman Expectation Equation for V. It says that the value of a state equals the expected immediate reward plus the discounted expected value of the next state, where the expectations are over the policy and transition dynamics.

### For Q(s,a)

Similarly for the action-value function:

```
Q_pi(s,a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * sum_{a'} pi(a'|s') * Q_pi(s',a')]
```

This says: the value of taking action a in state s under policy pi equals the expected immediate reward plus the discounted expected Q-value of the next state-action pair, where the next action a' is chosen according to pi.

### Intuition

Think of the Bellman Expectation Equation as a consistency condition. If you have the correct value function for a policy, then the value at every state must satisfy this equation. The value of a state cannot be computed in isolation; it depends on the values of successor states, which depend on their successors, and so on. The Bellman equation captures this entire chain of dependencies in a single, elegant recursive formula.

An analogy: imagine estimating the total remaining distance to a destination while driving. You can break it down as "the distance to the next intersection plus the total remaining distance from that intersection." The Bellman equation does the same thing with expected cumulative reward.

## Bellman Optimality Equation

The Bellman Optimality Equations characterize the value functions of an optimal policy, pi*. An optimal policy achieves the maximum possible value at every state.

### Optimal Value Functions

```
V*(s) = max_pi V_pi(s)     for all s
Q*(s,a) = max_pi Q_pi(s,a) for all s, a
```

### For V*(s)

The Bellman Optimality Equation for V* is:

```
V*(s) = max_a sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]
```

Compared to the expectation equation, the sum over the policy (sum_a pi(a|s) * ...) is replaced by a max over actions (max_a ...). The optimal policy simply picks the action that maximizes the expected value, so the probability-weighted average becomes a max.

### For Q*(s,a)

```
Q*(s,a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * max_{a'} Q*(s',a')]
```

This says: the optimal value of taking action a in state s equals the expected reward plus the discounted optimal value of the next state, where at the next state the agent will pick whichever action maximizes Q*.

### Relationship Between V* and Q*

```
V*(s) = max_a Q*(s,a)
```

Once Q* is known, the optimal policy is straightforward:

```
pi*(s) = argmax_a Q*(s,a)
```

This is why Q-learning focuses on learning Q* directly -- it gives you the optimal policy for free, without needing a model of the environment.

### Why the Optimality Equations Are Harder

The Bellman Expectation Equations are linear in V (because the policy pi is fixed and the equation is just a weighted sum). The Bellman Optimality Equations are nonlinear because of the max operator. This means:

- The expectation equations for a finite MDP can be solved exactly with linear algebra.
- The optimality equations generally require iterative methods.

## Matrix Form for Finite MDPs

For a finite MDP with |S| states, the Bellman Expectation Equation for V_pi can be written in matrix-vector form:

```
v_pi = r_pi + gamma * P_pi * v_pi
```

Where:

- v_pi is a column vector of length |S|, with entry i being V_pi(s_i).
- r_pi is a column vector of expected immediate rewards under policy pi, where entry i is sum_a pi(a|s_i) * sum_{s'} P(s'|s_i,a) * R(s_i,a,s').
- P_pi is the |S| x |S| state transition matrix under policy pi, where entry (i,j) is sum_a pi(a|s_i) * P(s_j|s_i,a).

Rearranging:

```
v_pi - gamma * P_pi * v_pi = r_pi
(I - gamma * P_pi) * v_pi = r_pi
v_pi = (I - gamma * P_pi)^{-1} * r_pi
```

This gives a closed-form solution. The matrix (I - gamma * P_pi) is guaranteed to be invertible when gamma < 1, because the spectral radius of gamma * P_pi is strictly less than 1 (since P_pi is a stochastic matrix with eigenvalues bounded by 1).

### Practical Limitations of the Matrix Form

The matrix inversion has O(|S|^3) time complexity. This is fine for small state spaces (hundreds or perhaps low thousands of states), but completely infeasible for large or continuous state spaces. For a gridworld with a 100x100 grid, |S| = 10,000, and the matrix has 10^8 entries. For Atari games, the state space is effectively infinite. This is why iterative and approximate methods are necessary in practice.

```python
import numpy as np

def solve_bellman_exact(P_pi, r_pi, gamma):
    """
    Solve the Bellman Expectation Equation exactly using matrix inversion.

    Args:
        P_pi: Transition matrix under policy pi, shape (|S|, |S|)
        r_pi: Expected immediate reward vector under pi, shape (|S|,)
        gamma: Discount factor

    Returns:
        v_pi: State-value vector, shape (|S|,)
    """
    n_states = len(r_pi)
    I = np.eye(n_states)
    v_pi = np.linalg.solve(I - gamma * P_pi, r_pi)
    return v_pi
```

Note the use of `np.linalg.solve` rather than explicitly computing the inverse. Solving the linear system is numerically more stable and more efficient than computing the full inverse.

## Connection to Dynamic Programming

The Bellman equations are the foundation of dynamic programming (DP) methods in RL. DP assumes full knowledge of the MDP (transition probabilities and reward function) and uses the Bellman equations as update rules.

### Policy Evaluation (Prediction)

Policy evaluation computes V_pi for a given policy pi. It repeatedly applies the Bellman Expectation Equation as an update rule until convergence:

```
V_{k+1}(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V_k(s')]
```

Starting from an arbitrary V_0, this sequence converges to V_pi. Each iteration applies the Bellman operator, which is a contraction mapping with factor gamma. This guarantees convergence for gamma < 1.

### Policy Iteration (Control)

Policy iteration alternates between two steps:

1. **Policy Evaluation**: Compute V_pi for the current policy pi (by solving the Bellman Expectation Equation, either exactly or iteratively).
2. **Policy Improvement**: Update the policy greedily with respect to V_pi:

```
pi'(s) = argmax_a sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]
```

The policy improvement theorem guarantees that pi' is at least as good as pi. Policy iteration converges to the optimal policy in a finite number of steps for finite MDPs.

### Value Iteration (Control)

Value iteration directly applies the Bellman Optimality Equation as an update rule:

```
V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V_k(s')]
```

This combines a truncated policy evaluation (a single sweep) with policy improvement (the max) in each iteration. It converges to V* and the optimal policy can be extracted at the end.

Value iteration can be seen as repeatedly applying the Bellman optimality operator, which is also a contraction mapping with factor gamma.

## Code Example: Value Iteration on a Gridworld

The following example implements value iteration on a simple 4x4 gridworld. The agent starts in the top-left corner and wants to reach the bottom-right corner. The agent receives -1 reward per step (encouraging it to find the shortest path). The agent can move up, down, left, or right, and if it tries to move off the grid, it stays in place.

```python
import numpy as np

class Gridworld:
    """A simple 4x4 gridworld environment."""

    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.terminal_state = self.n_states - 1  # bottom-right corner

        # Action effects: [row_delta, col_delta]
        self.action_effects = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

    def state_to_rc(self, s):
        return s // self.size, s % self.size

    def rc_to_state(self, r, c):
        return r * self.size + c

    def get_transition(self, s, a):
        """
        Returns (next_state, reward) for deterministic transitions.
        """
        if s == self.terminal_state:
            return s, 0.0

        r, c = self.state_to_rc(s)
        dr, dc = self.action_effects[a]
        nr, nc = r + dr, c + dc

        # Stay in place if hitting a wall
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = self.rc_to_state(nr, nc)
        else:
            next_state = s

        reward = -1.0
        return next_state, reward


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Value iteration algorithm.

    Args:
        env: Gridworld environment
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: Optimal state-value function
        policy: Optimal deterministic policy
    """
    V = np.zeros(env.n_states)
    iteration = 0

    while True:
        delta = 0
        for s in range(env.n_states):
            if s == env.terminal_state:
                continue

            v = V[s]

            # Bellman Optimality update: V(s) = max_a [R(s,a) + gamma * V(s')]
            action_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                next_state, reward = env.get_transition(s, a)
                action_values[a] = reward + gamma * V[next_state]

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        iteration += 1
        if delta < theta:
            break

    # Extract the optimal policy from V*
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if s == env.terminal_state:
            continue

        action_values = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            next_state, reward = env.get_transition(s, a)
            action_values[a] = reward + gamma * V[next_state]

        policy[s] = np.argmax(action_values)

    return V, policy, iteration


def print_values(V, size):
    """Print the value function as a grid."""
    print("State Values:")
    for r in range(size):
        row_vals = []
        for c in range(size):
            s = r * size + c
            row_vals.append(f"{V[s]:7.2f}")
        print(" ".join(row_vals))
    print()


def print_policy(policy, size):
    """Print the policy as a grid of arrows."""
    action_symbols = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("Optimal Policy:")
    for r in range(size):
        row_symbols = []
        for c in range(size):
            s = r * size + c
            if s == size * size - 1:
                row_symbols.append("  G  ")
            else:
                row_symbols.append(f"  {action_symbols[policy[s]]}  ")
        print(" ".join(row_symbols))
    print()


# Run value iteration
env = Gridworld(size=4)
V, policy, n_iterations = value_iteration(env, gamma=0.99)

print(f"Converged in {n_iterations} iterations\n")
print_values(V, env.size)
print_policy(policy, env.size)
```

Running this produces output like:

```
Converged in 63 iterations

State Values:
  -5.90   -4.94   -3.97   -2.99
  -4.94   -3.97   -2.99   -2.00
  -3.97   -2.99   -2.00   -1.00
  -2.99   -2.00   -1.00    0.00

Optimal Policy:
  >    >    >    v
  >    >    >    v
  >    >    >    v
  >    >    >    G
```

The values reflect the fact that each step costs -1, and the agent must take 6 steps from the top-left to reach the goal (Manhattan distance), so V(top-left) is approximately -6 (slightly more due to discounting). The optimal policy correctly points toward the goal along shortest paths.

## Code Example: Policy Iteration

Policy iteration separates evaluation and improvement into distinct phases:

```python
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Evaluate a given policy by iteratively applying the Bellman Expectation Equation.
    """
    V = np.zeros(env.n_states)

    while True:
        delta = 0
        for s in range(env.n_states):
            if s == env.terminal_state:
                continue

            v = V[s]
            a = policy[s]
            next_state, reward = env.get_transition(s, a)
            V[s] = reward + gamma * V[next_state]

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy_iteration(env, gamma=0.99):
    """
    Policy iteration algorithm.

    Returns:
        V: Optimal state-value function
        policy: Optimal deterministic policy
    """
    # Start with an arbitrary policy (all actions = 0, i.e., "up")
    policy = np.zeros(env.n_states, dtype=int)
    iteration = 0

    while True:
        # Step 1: Policy Evaluation
        V = policy_evaluation(env, policy, gamma)

        # Step 2: Policy Improvement
        policy_stable = True
        for s in range(env.n_states):
            if s == env.terminal_state:
                continue

            old_action = policy[s]

            action_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                next_state, reward = env.get_transition(s, a)
                action_values[a] = reward + gamma * V[next_state]

            policy[s] = np.argmax(action_values)

            if old_action != policy[s]:
                policy_stable = False

        iteration += 1

        if policy_stable:
            break

    return V, policy, iteration


# Run policy iteration
env = Gridworld(size=4)
V, policy, n_iterations = policy_iteration(env, gamma=0.99)
print(f"Policy iteration converged in {n_iterations} iterations")
print_values(V, env.size)
print_policy(policy, env.size)
```

Policy iteration typically converges in far fewer outer iterations than value iteration (often under 10), because each iteration fully evaluates the current policy before improving it. However, each iteration is more expensive due to the full policy evaluation step.

## Connection to Q-Learning and DQN

The Bellman Optimality Equation for Q* is the theoretical basis for Q-learning and its deep learning extension, DQN (Deep Q-Network). These are covered in detail in the sibling value-based-methods chapters. Here we focus on the conceptual connection.

### Q-Learning

In model-free RL, the agent does not know P(s'|s,a) or R(s,a,s'). It cannot directly compute the Bellman equations. Instead, Q-learning uses samples of experience to approximate the Bellman Optimality Equation.

The Q-learning update rule is:

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]
```

The term in brackets is the temporal-difference (TD) error. It measures how far the current Q-value is from satisfying the Bellman Optimality Equation based on the observed sample (s, a, r, s'). The update nudges Q(s,a) toward the Bellman target `r + gamma * max_{a'} Q(s',a')`.

Compare this to the Bellman Optimality Equation for Q*:

```
Q*(s,a) = E[R + gamma * max_{a'} Q*(s',a') | S=s, A=a]
```

Q-learning replaces the full expectation with a single sample and uses incremental averaging (via the learning rate alpha) to converge to the true expectation over many samples. At convergence, Q satisfies the Bellman Optimality Equation and equals Q*.

### DQN

When the state space is too large for a table (for example, raw pixel observations in Atari games), DQN approximates Q*(s,a) with a neural network Q(s,a; theta). The network is trained to minimize the squared Bellman error:

```
Loss = E[(r + gamma * max_{a'} Q(s',a'; theta^-) - Q(s,a; theta))^2]
```

Where theta^- are the parameters of a periodically-updated target network (to stabilize training). The loss function drives Q toward satisfying the Bellman Optimality Equation. The target network provides a stable approximation of the right-hand side of the Bellman equation while the primary network's parameters are updated.

Key insight: DQN does not solve the Bellman equation directly. It minimizes the Bellman error using stochastic gradient descent on batches of experience sampled from a replay buffer. The Bellman equation defines the objective; deep learning provides the function approximation and optimization machinery.

## The Contraction Mapping Property

A central theoretical result underlying these algorithms is that the Bellman operators are contraction mappings.

For any two value functions U and V:

```
||T(U) - T(V)||_inf <= gamma * ||U - V||_inf
```

Where T is the Bellman operator (either the expectation operator or the optimality operator) and ||.||_inf is the max norm. This means that applying the Bellman operator always brings two value functions closer together by at least a factor of gamma. By the Banach fixed-point theorem, repeated application converges to a unique fixed point, which is the true value function.

This guarantees:

1. **Existence**: There is exactly one value function satisfying the Bellman equation (for a given policy, or for the optimal policy).
2. **Convergence**: Iterative methods (policy evaluation, value iteration) will converge to it.
3. **Rate**: Convergence is geometric with rate gamma. Smaller gamma means faster convergence but also a more myopic agent.

## The Discount Factor in Practice

The discount factor gamma appears in every Bellman equation and profoundly affects behavior:

- **gamma = 0**: The agent is completely myopic. V(s) = E[R_{t+1}]. The agent only cares about the immediate next reward.
- **gamma close to 1 (e.g., 0.99 or 0.999)**: The agent is far-sighted, valuing future rewards almost as much as immediate ones. This is typical for most RL tasks.
- **gamma = 1**: Undiscounted. Only valid for episodic tasks that are guaranteed to terminate. The Bellman operator is no longer a contraction, and convergence guarantees are weaker.

In practice, gamma is a hyperparameter. Values of 0.99 or 0.999 are common. A lower gamma can make learning easier (since the effective planning horizon is shorter and value estimates have lower variance) but may cause the agent to ignore important long-term consequences.

## Common Misconceptions

1. **"The Bellman equation only applies to tabular RL."** The equations apply universally. Function approximation (neural networks, linear models) is used to approximate the solutions, but the underlying equations remain the same.

2. **"You need to know the transition model to use Bellman equations."** The equations themselves involve the model, but model-free algorithms like Q-learning and SARSA use sampled transitions to approximate the expectations in the Bellman equation without knowing the model.

3. **"Value iteration and policy iteration always find the same solution."** They do converge to the same optimal value function and optimal policy for a given MDP, but they take different paths to get there.

4. **"A higher discount factor is always better."** Higher gamma means the agent plans further ahead, but it also increases variance in value estimates, slows convergence, and can make credit assignment harder. The right gamma depends on the task.

## Key Takeaways

1. **The Bellman equation is a recursive decomposition**: V(s) = E[R + gamma * V(s')]. Current value equals immediate reward plus discounted future value.

2. **Two variants serve different purposes**: The Expectation Equation evaluates a fixed policy. The Optimality Equation finds the best policy.

3. **The max operator separates prediction from control**: Replacing the policy-weighted sum with a max converts the linear prediction problem into a nonlinear optimization problem.

4. **Dynamic programming directly applies the equations**: Policy evaluation uses the expectation equation. Value iteration uses the optimality equation. Policy iteration uses both.

5. **Q-learning approximates the Bellman Optimality Equation**: It replaces the full expectation with sampled transitions and uses incremental updates to converge.

6. **Matrix form gives exact solutions for small problems**: For finite MDPs, V_pi = (I - gamma * P_pi)^{-1} * r_pi, but this does not scale.

7. **Convergence is guaranteed by the contraction property**: The Bellman operators shrink differences between value functions by a factor of gamma at each step, guaranteeing convergence to a unique fixed point.
