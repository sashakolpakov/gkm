# Free Energy Objective: Detailed Explanation

## Overview

This document explains the free energy formulation used in GKM (Gödel-Kolmogorov Machine), based on the paper "Loss-Complexity Landscape and Model Structure Functions" (arXiv:2507.13543).

## What is Free Energy?

The **free energy** F(λ) is a functional from statistical physics that balances two competing objectives:

```
F(λ) = λ·C_agent + L_train
```

Where:
- **λ** (lambda): Regularization parameter controlling the trade-off
- **C_agent**: Complexity of the agent's code (degrees of freedom)
- **L_train**: Training loss (error rate on training problems)
- **L_test**: Test loss (reported separately to measure generalization)

## Key Insight: Lambda Sweep, Not Optimization

### Standard Hyperparameter Optimization:
```python
# λ is optimized as a hyperparameter
best_lambda = optimize(lambda: evaluate(model, lambda))
```

### GKM Approach (Following Structure-Functions Paper):
```python
# λ is SWEPT to explore the loss-complexity landscape
for lambda_val in [0.0, 0.1, 0.2, ..., 0.5]:
    # For each λ, use TPE to find best agent
    best_agent = tpe_search(lambda_val)
    free_energy = lambda_val * complexity(best_agent) + train_loss(best_agent)

# Return the (λ, agent) pair with minimum F(λ)
```

## Why Sweep λ Instead of Optimizing It?

### 1. **Explore the Loss-Complexity Landscape**

Different λ values reveal different optimal model structures:
- **λ = 0.0**: No regularization → most complex agent that fits training data
- **λ = 0.5**: Strong regularization → simplest agent that still solves problems
- **λ = 0.1-0.3**: Sweet spot → balanced agents

### 2. **Phase Transitions**

The optimal complexity C*(λ) changes **discontinuously** at certain λ values:

```
λ = 0.0  → C* = 100 (very complex)
λ = 0.05 → C* = 85
λ = 0.1  → C* = 72  ← Phase transition
λ = 0.15 → C* = 45
λ = 0.2  → C* = 40  ← Another transition
```

These jumps correspond to the agent discovering fundamentally different strategies.

### 3. **Legendre-Fenchel Duality**

The sweep traces out the **structure function** C*(L):

```
F(λ) = inf_C [λ·C + L_train(C)]
```

This gives the Pareto-optimal frontier of complexity vs. performance.

### 4. **Robustness**

Sweeping is more robust than gradient-based λ optimization:
- No local minima in λ space
- Explores the full range of trade-offs
- Matches the physics intuition

## Implementation Details

### The Lambda Sweep

```python
def run(self):
    # Sweep λ values (not optimized!)
    lambda_values = np.linspace(0.0, 0.5, 6)  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for lambda_val in lambda_values:
        # For this fixed λ, use TPE to find best agent modification
        for iteration in range(iters_per_lambda):
            # TPE suggests: modification_goal, target_complexity
            params = tpe.suggest()

            # Generate modified agent
            agent = modify_agent(parent, params['modification_goal'])

            # Evaluate: F(λ) = λ·C_agent + L_train
            complexity = compute_complexity(agent.code)
            train_loss = evaluate(agent, train_problems)
            free_energy = lambda_val * complexity + train_loss

            # TPE learns which modifications work for this λ
            tpe.observe(params, free_energy)

    # Return best (λ, agent) across ALL sweeps
    return best_agent
```

### Train Loss vs Test Loss

**Important**: The free energy uses **training loss**, not test loss:

```python
# Free energy (what we minimize)
F(λ) = λ·C_agent + L_train

# Test loss (what we report)
L_test = evaluate(agent, test_problems)
```

**Why?**
- **F(λ) with L_train**: Guides the search, balances fitting training data vs. complexity
- **L_test**: Measures if we're overfitting (L_test >> L_train means overfitting)

### Complexity as Degrees of Freedom

For the agent's code:

```python
def compute_agent_complexity(agent_code: str) -> float:
    lines = count_non_comment_lines(agent_code)
    assignments = count_assignments(agent_code)
    control_flow = count_if_for_while(agent_code)
    functions = count_function_defs(agent_code)
    classes = count_classes(agent_code)

    # Weighted sum (degrees of freedom)
    complexity = (lines + assignments + control_flow * 2 +
                 functions * 3 + classes * 5)
    return complexity
```

This approximates the agent's **effective model capacity**.

## Example Output

```
======================================================================
SELF-MODIFYING AGENT (TPE + Free Energy)
======================================================================
Objective: minimize F(λ) = λ·C_agent + L_train

Lambda sweep: 6 values from 0.00 to 0.50
Iterations per λ: 3 (total: 18)

======================================================================
LAMBDA 1/6: λ = 0.0000
======================================================================
[  0] λ=0.000 C_agent= 64.0 L_train=1.0000 L_test=1.0000 F(λ)=1.0000
[  1] λ=0.000 C_agent= 85.0 L_train=0.8333 L_test=0.6667 F(λ)=0.8333
✓ New best! F(λ)=0.8333

======================================================================
LAMBDA 2/6: λ = 0.1000
======================================================================
[  3] λ=0.100 C_agent= 72.0 L_train=0.6667 L_test=0.5000 F(λ)=7.8667
✓ New best! F(λ)=7.8667

...

======================================================================
OPTIMIZATION COMPLETE
======================================================================
Best: agent_0012 (gen 8)
  λ*: 0.1000
  C_agent: 72.0
  Train loss: 0.6667
  Test loss: 0.5000
  F(λ*): 7.8667
======================================================================
```

**Interpretation:**
- Swept 6 λ values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- Best found at λ* = 0.1
- Agent has 72 degrees of freedom
- Train loss: 0.67 (solving 1/3 of training problems)
- Test loss: 0.50 (solving 1/2 of test problems) - **generalizes well!**

## Stochastic Evaluation for Speed

To handle large benchmarks efficiently:

```python
agent = SelfImprovingAgent(
    eval_subset_size=2,      # Use only 2 test cases per problem during search
    stochastic_eval=True,    # Randomly sample different cases each iteration
)
```

**Benefits:**
- **Speed**: 2x-5x faster evaluations during search
- **Prevents overfitting**: Different test cases each time
- **Final evaluation**: Full test set used for best agent

## Connection to Statistical Physics

### Partition Function

The free energy comes from the partition function:

```
Z(λ) = Σ_C exp(-λ·C - L(C))
F(λ) = -log Z(λ)
```

### Phase Diagram

Different λ values correspond to different "phases":
- **λ → 0**: High temperature → complex, flexible agents
- **λ → ∞**: Low temperature → simple, rigid agents
- **λ moderate**: Phase coexistence → optimal trade-off

## Comparison to Standard ML

| Aspect | Standard Regularization | GKM Free Energy |
|--------|------------------------|-----------------|
| Objective | L_train + λ·C | λ·C + L_train |
| λ treatment | Fixed or grid search | **Swept systematically** |
| Loss used | Training loss | Training loss (test reported separately) |
| Complexity | Penalty term | Physical quantity (DoF) |
| Search | Single model | **Landscape exploration** |
| Output | One model | Optimal (λ*, C*, agent*) + phase diagram |

## Customization

### Adjust Lambda Range

```python
# In agent.py, run() method:
lambda_values = np.linspace(0.0, 1.0, 11)  # Wider range, finer resolution
```

### Change Iterations Per Lambda

```python
agent = SelfImprovingAgent(
    lambda_points=11,       # More λ values
    iters_per_lambda=5,     # More iterations per λ
)
# Total: 11 × 5 = 55 evaluations
```

### Modify Complexity Metric

```python
def _compute_agent_complexity(self, agent_code: str) -> float:
    # Custom complexity: just count lines
    return len([l for l in agent_code.split('\n') if l.strip()])
```

## References

1. **Paper**: Kolpakov, "Loss-Complexity Landscape and Model Structure Functions", [arXiv:2507.13543](https://arxiv.org/abs/2507.13543)
2. **Code**: [structure-functions repo](https://github.com/sashakolpakov/structure-functions)
3. **Hyperopt/TPE**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)
4. **Darwin-Gödel Machine**: [dgm repo](https://github.com/jennyzzt/dgm)

## Mathematical Appendix

### Structure Function

The structure function S(L) gives the minimum complexity to achieve training loss ≤ L:

```
S(L) = min{C : L_train(C) ≤ L}
```

### Free Energy as Dual

The free energy is the Legendre-Fenchel dual:

```
F(λ) = inf_C [λ·C + L_train(C)]
```

### Optimal Complexity

For each λ, the optimal complexity C*(λ) satisfies:

```
dL_train/dC = -λ
```

This is the tangent to the structure function curve S(L).

### Phase Transition Condition

A phase transition occurs when:

```
d²L_train/dC² → ∞
```

The agent discovers a fundamentally new strategy.
