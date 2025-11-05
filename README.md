# GKM: Gödel-Kolmogorov Machine

A self-improving AI agent that modifies its own source code to optimize a free energy functional, combining:

- **Gödel Machine**: Self-modifying agent that rewrites its own code
- **Kolmogorov Complexity**: Regularization via code complexity
- **Free Energy Principle**: F(λ) = λ·C_agent + (L_train + L_val)/2
- **TPE Search**: Tree-structured Parzen Estimator guides code modifications

Unlike the original [Darwin-Gödel Machine](https://github.com/jennyzzt/dgm), GKM uses principled free energy optimization with lambda sweeps to balance agent complexity against performance.

## Quick Start

### Prerequisites

**Anthropic API** (recommended - fastest):
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or use **Ollama** (local):
```bash
brew install ollama  # macOS
ollama serve &
ollama pull qwen2.5-coder
```

### Install Dependencies

```bash
pip install -r requirements.txt
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

The agent minimizes **F(λ) = λ·C_agent + (L_train + L_val)/2** where:
- **λ**: Regularization strength (swept from 0.0 to 0.5)
- **C_agent**: Complexity of agent code (lines, control flow, functions)
- **L_train**: Training loss (error rate on 9 training problems)
- **L_val**: Validation loss (error rate on 5 validation problems)
- **L_test**: Test loss (held out, reported at end)

### Lambda Sweep Strategy

1. **For each λ value** (e.g., 0.0, 0.056, 0.111, ..., 0.5):
   - Use TPE to find best agent modifications
   - Evaluate: F(λ) = λ·C_agent + (L_train + L_val)/2
2. **Return**: The (λ, agent) pair with minimum F(λ)

This explores the **loss-complexity landscape** to find optimal regularization.

### Self-Modification Process

Each iteration:
1. **TPE suggests** modification goal (e.g., "add few-shot examples")
2. **Meta-LLM generates** improved prompting strategy
3. **Agent generates** solutions for all problems in one batched call
4. **Evaluation** measures success rate with detailed OK/FAIL output
5. **TPE learns** which modifications reduce F(λ)

The agent evolves only its `build_prompt()` method - the scaffold is fixed.

## Features

### Batched Solution Generation
- **1 LLM call** generates solutions for all problems (not N sequential calls)
- Dramatically faster than naive approach

### Progress Tracking
- **tqdm progress bars** show solution generation and evaluation progress
- **OK/FAIL output** for each problem with detailed error messages
- **Real-time feedback** on which problems pass/fail

### Smart Test Dispatch
- Automatically handles different input formats:
  - Single arguments: `max_subarray_sum([1, 2, 3])`
  - Multiple arguments: `two_sum([1, 2, 3], 5)`
  - Multiple primitives: `is_anagram('listen', 'silent')`

### Comprehensive Logging
- **agent_evolution.log** tracks all modifications and evaluations
- Debug compilation failures and LLM generation issues
- Trace which modifications succeed vs fail

## Configuration

Edit `agent.py` `main()`:

```python
agent = SelfImprovingAgent(
    llm_backend="anthropic",              # "anthropic", "ollama", "openai"
    model="claude-3-5-haiku-20241022",    # Fast and cheap for testing
    max_iterations=60,                     # Total budget (split across λ values)
    lambda_points=10,                      # Fine-grained λ sweep
    iters_per_lambda=6,                    # TPE iterations per λ
    eval_subset_size=2,                    # Use 2 test cases per problem (faster)
    stochastic_eval=True,                  # Random subset prevents overfitting
    debug=False                            # Set True for verbose output
)
```

**LLM Backends:**
- **anthropic**: Requires `ANTHROPIC_API_KEY` (fastest, recommended)
- **ollama**: Requires local `ollama serve` or cloud API key
- **openai**: Requires `OPENAI_API_KEY`

**Performance Tuning:**
- **Faster**: `lambda_points=4, iters_per_lambda=2` (8 total evals)
- **More thorough**: `lambda_points=15, iters_per_lambda=10` (150 total evals)
- **Full evaluation**: `eval_subset_size=None` (use all test cases)

## Benchmark

The agent is evaluated on **19 LeetCode Easy/Medium problems**:

**Training Set (9 problems):**
- is_palindrome_string, sum_of_list, find_max, two_sum
- valid_parentheses, merge_sorted_arrays, remove_duplicates
- best_time_to_buy_sell_stock, contains_duplicate

**Validation Set (5 problems):**
- count_vowels, fibonacci, longest_common_prefix
- reverse_linked_list, palindrome_number

**Test Set (5 problems):**
- is_anagram, factorial, max_subarray_sum
- search_insert_position, climbing_stairs

## Example Output

```
======================================================================
SELF-MODIFYING AGENT (TPE + Free Energy)
======================================================================
Objective: minimize F(λ) = λ·C_agent + (L_train + L_val) / 2
LLM: anthropic / claude-3-5-haiku-20241022

Lambda sweep: 10 values from 0.00 to 0.50
Iterations per λ: 6 (total: 60)

======================================================================
LAMBDA 1/10: λ = 0.0000
======================================================================

  Evaluating agent on 14 problems (train+val)...

  Training set evaluation:
    Evaluating: 100%|████████████| 9/9
    ✓ is_palindrome_string: OK (4/4 tests passed)
    ✓ sum_of_list: OK (4/4 tests passed)
    ✗ two_sum: FAIL (3/4 tests passed) - test 4: expected [1, 3], got [2, 3]
    Summary: 8/9 problems passed

  Validation set evaluation:
    Evaluating: 100%|████████████| 5/5
    ✓ count_vowels: OK (4/4 tests passed)
    ✓ fibonacci: OK (4/4 tests passed)
    Summary: 5/5 problems passed

✓ New best! F(λ)=0.3167 [λ=0.000, C_agent=169.0, L_val=0.0000]
[  0] λ=0.000 C_agent=169.0 L_train=0.1111 L_val=0.0000 F(λ)=0.0556
         Goal: SEED agent (unmodified)

...
```

## Files

```
.
├── agent.py                    # Main implementation
├── scaffold.py                 # Fixed agent infrastructure
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── FREE_ENERGY_EXPLANATION.md  # Theory deep-dive
├── agent_evolution.log         # Detailed logs (created on run)
└── output/                     # Results (created on run)
    ├── agent_history.jsonl     # All evaluations
    └── best_agent_code.py      # Best agent found
```

## Debugging

**Check logs:**
```bash
tail -f agent_evolution.log          # Watch in real-time
grep "ERROR" agent_evolution.log      # Find errors
grep "Compilation failed" agent_evolution.log  # Syntax issues
```

**Common issues:**
- **No functions found**: LLM not following delimiter format
- **Compilation failed**: Syntax errors in generated code
- **TypeError in tests**: Input dispatch logic issue (check test case format)

## Architecture

**Fixed Scaffold** (`scaffold.py`):
- Solution generation (batched LLM calls)
- Function extraction (multiple fallback strategies)
- Code parsing and validation

**Modifiable Strategy** (evolved by meta-LLM):
- `build_prompt()` method only
- Prompting techniques, examples, instructions
- Chain-of-thought, few-shot learning, etc.

**Meta-Optimizer** (`agent.py`):
- TPE-guided search over modifications
- Lambda sweep for free energy landscape
- Evaluation and logging infrastructure

## Theory

See [FREE_ENERGY_EXPLANATION.md](FREE_ENERGY_EXPLANATION.md) for details on:
- Why F(λ) = λ·C + L balances complexity vs performance
- Why we sweep λ instead of optimizing it
- Connection to statistical physics and phase transitions
- Legendre-Fenchel duality and structure functions

## References

1. Kolpakov, "Loss-Complexity Landscape and Model Structure Functions", [arXiv:2507.13543](https://arxiv.org/abs/2507.13543)
2. [structure-functions repo](https://github.com/sashakolpakov/structure-functions)
3. [Darwin-Gödel Machine](https://github.com/jennyzzt/dgm)
4. Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)

## License

MIT
