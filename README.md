# GKM: Gödel-Kolmogorov Machine

A self-improving AI agent that modifies its own source code to optimize a free energy functional, combining:

- **Gödel Machine**: Self-modifying agent that rewrites its own code
- **Kolmogorov Complexity**: Regularization via code complexity
- **Free Energy Principle**: F(λ) = λ·C_agent + L_train from [structure-functions paper](https://arxiv.org/abs/2507.13543)
- **TPE Search**: Tree-structured Parzen Estimator guides the search over λ values

Unlike the original [Darwin-Gödel Machine](https://github.com/jennyzzt/dgm), GKM uses principled Bayesian optimization (TPE) to sweep over regularization strengths λ, balancing agent complexity against training performance.

## Quick Start

### Prerequisites

Install Ollama (local LLM runtime):

**macOS:**
```bash
brew install ollama
ollama serve &
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Install Dependencies

```bash
pip install hyperopt numpy openai
```

### Pull a Model

```bash
# Start with 4B model (good balance)
ollama pull gemma3:4b

# Or try more powerful models:
ollama pull llama3        # 8B, better at reasoning
ollama pull codellama     # 7B+, code-specialized
ollama pull qwen2.5-coder # Best for code
```

### Run

```bash
python agent.py
```

With debug output:
```bash
python agent.py --debug
```

## How It Works

### Free Energy Optimization

The agent minimizes **F(λ) = λ·C_agent + L_train** where:
- **λ**: Regularization strength (swept from 0.0 to 0.5)
- **C_agent**: Complexity of the agent's code (lines, functions, control flow)
- **L_train**: Training loss (error rate on training problems)
- **L_test**: Test loss (reported separately to measure generalization)

### Lambda Sweep Strategy

Unlike traditional hyperparameter optimization, we **sweep λ** like in the structure-functions paper:

1. **For each λ value** (e.g., 0.0, 0.1, 0.2, ..., 0.5):
   - Use TPE to find the best agent modification for this λ
   - Evaluate: F(λ) = λ·C_agent + L_train
2. **Return**: The (λ, agent) pair with minimum F(λ) across all sweeps

This explores the **loss-complexity landscape** and finds the optimal regularization strength.

### Self-Modification Process

Each iteration:
1. **TPE suggests** a modification goal (e.g., "add few-shot examples", "use chain-of-thought")
2. **LLM generates** improved agent code based on template prompts
3. **Agent is evaluated** on algorithmic problems (edit distance, regex matching, etc.)
4. **TPE learns** which modifications reduce F(λ)

The agent evolves better prompting strategies, error handling, and code generation logic.

## Example Output

```
======================================================================
SELF-MODIFYING AGENT (TPE + Free Energy)
======================================================================
Objective: minimize F(λ) = λ·C_agent + L_train
LLM: ollama / gemma3:4b
Strategy: Sweep λ, use TPE for agent modifications

Lambda sweep: 6 values from 0.00 to 0.50
Iterations per λ: 3 (total: 18)

======================================================================
LAMBDA 1/6: λ = 0.0000
======================================================================

  [*] First iteration: using SEED agent
✓ New best! F(λ)=1.0000 [λ=0.000, C_agent=64.0, L_test=1.0000]
[  0] λ=0.000 C_agent= 64.0 L_train=1.0000 L_test=1.0000 F(λ)=1.0000

  ✓ Code validated successfully
[  1] λ=0.000 C_agent= 85.0 L_train=0.8333 L_test=0.6667 F(λ)=0.8333
✓ New best! F(λ)=0.8333 [λ=0.000, C_agent=85.0, L_test=0.6667]

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

## Configuration

Edit `agent.py` `main()` to customize:

```python
agent = SelfImprovingAgent(
    model="gemma3:4b",          # Model to use
    lambda_points=6,             # Number of λ values to sweep
    iters_per_lambda=3,          # TPE iterations per λ
    eval_subset_size=2,          # Test cases per problem (for speed)
    stochastic_eval=True,        # Random sampling prevents overfitting
)
```

**Performance Tuning:**
- **Faster**: `lambda_points=4, iters_per_lambda=2` (8 total evals)
- **More thorough**: `lambda_points=11, iters_per_lambda=5` (55 total evals)
- **Disable subsampling**: `eval_subset_size=None` (use all test cases)

## Benchmark

The agent is evaluated on **LeetCode Hard** algorithmic problems:

**Training Set:**
- Edit distance (dynamic programming)
- Longest increasing subsequence (DP)
- Trapping rain water (two pointers)

**Test Set:**
- Regular expression matching (DP)
- Word ladder (BFS/graph search)
- Max sliding window (deque)

These problems require sophisticated reasoning and algorithm knowledge, making them a strong test of the agent's self-improvement capabilities.

## Files

```
.
├── agent.py                    # Main implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── old/                        # Previous iterations (archived)
└── output/                     # Results (created on run)
    ├── agent_history.jsonl     # All evaluations
    └── best_agent.py           # Best agent found
```

## Theory

### Free Energy F(λ) = λ·C + L_train

From Kolpakov's [structure-functions paper](https://arxiv.org/abs/2507.13543):

The free energy balances:
- **Training performance** (L_train): How well the agent solves problems
- **Complexity cost** (C_agent): How much code/logic the agent uses
- **Regularization** (λ): Controls the trade-off

**Key insight**: λ is not optimized by TPE—it's **swept** to explore the loss-complexity landscape, like phase transitions in statistical physics.

### Why Sweep λ Instead of Optimizing It?

1. **Explore the landscape**: Different λ values reveal different optimal complexities
2. **Phase transitions**: The optimal model structure changes discontinuously with λ
3. **Legendre-Fenchel duality**: The sweep traces out the structure function C*(L)
4. **Avoid local minima**: Sweeping is more robust than gradient-based λ optimization

This follows the original paper's methodology rather than treating λ as just another hyperparameter.

## Comparison to Darwin-Gödel Machine

| Feature | Darwin-Gödel Machine | GKM (this project) |
|---------|---------------------|-------------------|
| **Self-modification** | ✓ Agent modifies own code | ✓ Agent modifies own code |
| **Search algorithm** | Evolutionary | TPE (Bayesian) |
| **Objective** | Accuracy only | F(λ) = λ·C + L_train |
| **Regularization** | None | Kolmogorov complexity |
| **Lambda** | N/A | Swept (not optimized) |
| **Theory** | Empirical | Statistical physics |
| **Infrastructure** | Docker, SWE-bench | Pure Python, algorithmic problems |
| **Complexity** | ~1000+ lines | ~700 lines |

## References

1. **Paper**: Kolpakov, "Loss-Complexity Landscape and Model Structure Functions", [arXiv:2507.13543](https://arxiv.org/abs/2507.13543)
2. **Code**: [structure-functions repo](https://github.com/sashakolpakov/structure-functions)
3. **Darwin-Gödel Machine**: [dgm repo](https://github.com/jennyzzt/dgm)
4. **Hyperopt/TPE**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)

## License

MIT
