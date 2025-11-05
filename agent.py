"""
Self-Modifying Agent with TPE + Free Energy

Combines Darwin-Gödel Machine + Structure Functions:
- Agent modifies its own source code (like DGM)
- Free energy objective: F(λ) = λ·C_agent + L_train (test loss measures generalization)
- TPE (Parzen Tree Estimator) guides the search
- λ is optimized, not fixed
"""

import json
import os
import time
import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('agent_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentVersion:
    """A version of the self-modifying agent"""
    agent_id: str
    generation: int
    parent_id: Optional[str]
    agent_code: str

    train_loss: float = 999.0
    val_loss: float = 999.0
    test_loss: float = 999.0
    agent_complexity: float = 100.0
    lambda_val: float = 0.5
    free_energy: float = 999.0

    modification_description: str = ""
    timestamp: float = 0.0


class LLMClient:
    """LLM client"""
    def __init__(self, backend="ollama", model="gemma3:1b", api_key=None):
        self.backend = backend
        self.model = model

        if backend == "ollama":
            from openai import OpenAI
            import os

            # Check if using cloud model (ends with -cloud)
            if model.endswith("-cloud"):
                # Use ollama.com API for cloud models
                ollama_api_key = api_key or os.getenv("OLLAMA_API_KEY")
                if not ollama_api_key:
                    raise ValueError(
                        "Cloud model requires OLLAMA_API_KEY environment variable.\n"
                        "Get your key from: https://ollama.com/settings/keys"
                    )
                # Ollama cloud uses OpenAI-compatible endpoint at /v1
                self.client = OpenAI(
                    base_url="https://ollama.com/v1",
                    api_key=ollama_api_key
                )
            else:
                # Use local Ollama server
                self.client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
        elif backend == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        elif backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        if self.backend == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content


class Benchmark:
    """LeetCode Easy/Medium problems with train/val/test split"""

    @staticmethod
    def get_problems():
        return {
            # TRAIN SET (9 problems) - used for optimization
            'train': [
                {
                    'id': 'is_palindrome_string',
                    'description': 'Check if string is a palindrome (ignoring spaces, case)',
                    'signature': 'def is_palindrome_string(s: str) -> bool:',
                    'test_cases': [
                        {'input': 'racecar', 'expected': True},
                        {'input': 'hello', 'expected': False},
                        {'input': 'A man a plan a canal Panama', 'expected': True},
                        {'input': '', 'expected': True},
                    ]
                },
                {
                    'id': 'sum_of_list',
                    'description': 'Return sum of all integers in list',
                    'signature': 'def sum_of_list(nums: list) -> int:',
                    'test_cases': [
                        {'input': [1, 2, 3, 4], 'expected': 10},
                        {'input': [], 'expected': 0},
                        {'input': [-1, -2, 3], 'expected': 0},
                        {'input': [100], 'expected': 100},
                    ]
                },
                {
                    'id': 'find_max',
                    'description': 'Find maximum element in non-empty list',
                    'signature': 'def find_max(nums: list) -> int:',
                    'test_cases': [
                        {'input': [1, 5, 3, 9, 2], 'expected': 9},
                        {'input': [1], 'expected': 1},
                        {'input': [-5, -1, -10], 'expected': -1},
                        {'input': [0, 0, 0], 'expected': 0},
                    ]
                },
                {
                    'id': 'two_sum',
                    'description': 'Find two indices in array that sum to target. Return indices [i, j] where i < j.',
                    'signature': 'def two_sum(nums: list, target: int) -> list:',
                    'test_cases': [
                        {'input': [[2, 7, 11, 15], 9], 'expected': [0, 1]},
                        {'input': [[3, 2, 4], 6], 'expected': [1, 2]},
                        {'input': [[3, 3], 6], 'expected': [0, 1]},
                        {'input': [[1, 5, 3, 7], 10], 'expected': [1, 3]},
                    ]
                },
                {
                    'id': 'valid_parentheses',
                    'description': 'Check if string with brackets ()[]{}  is valid (properly nested and closed)',
                    'signature': 'def valid_parentheses(s: str) -> bool:',
                    'test_cases': [
                        {'input': '()', 'expected': True},
                        {'input': '()[]{}', 'expected': True},
                        {'input': '(]', 'expected': False},
                        {'input': '([)]', 'expected': False},
                        {'input': '{[]}', 'expected': True},
                        {'input': '', 'expected': True},
                    ]
                },
                {
                    'id': 'merge_sorted_arrays',
                    'description': 'Merge two sorted arrays into one sorted array',
                    'signature': 'def merge_sorted_arrays(arr1: list, arr2: list) -> list:',
                    'test_cases': [
                        {'input': [[1, 3, 5], [2, 4, 6]], 'expected': [1, 2, 3, 4, 5, 6]},
                        {'input': [[1, 2, 3], []], 'expected': [1, 2, 3]},
                        {'input': [[], [1, 2]], 'expected': [1, 2]},
                        {'input': [[1, 5, 9], [2, 3, 10]], 'expected': [1, 2, 3, 5, 9, 10]},
                    ]
                },
                {
                    'id': 'remove_duplicates',
                    'description': 'Remove duplicates from sorted array in-place, return new length',
                    'signature': 'def remove_duplicates(nums: list) -> int:',
                    'test_cases': [
                        {'input': [1, 1, 2], 'expected': 2},
                        {'input': [0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 'expected': 5},
                        {'input': [], 'expected': 0},
                        {'input': [1], 'expected': 1},
                    ]
                },
                {
                    'id': 'best_time_to_buy_sell_stock',
                    'description': 'Find max profit from buying and selling stock once (buy before sell)',
                    'signature': 'def best_time_to_buy_sell_stock(prices: list) -> int:',
                    'test_cases': [
                        {'input': [7, 1, 5, 3, 6, 4], 'expected': 5},  # buy at 1, sell at 6
                        {'input': [7, 6, 4, 3, 1], 'expected': 0},  # no profit
                        {'input': [2, 4, 1], 'expected': 2},
                        {'input': [1, 2], 'expected': 1},
                    ]
                },
                {
                    'id': 'contains_duplicate',
                    'description': 'Return True if any value appears at least twice in array',
                    'signature': 'def contains_duplicate(nums: list) -> bool:',
                    'test_cases': [
                        {'input': [1, 2, 3, 1], 'expected': True},
                        {'input': [1, 2, 3, 4], 'expected': False},
                        {'input': [1, 1, 1, 3, 3, 4, 3, 2, 4, 2], 'expected': True},
                        {'input': [], 'expected': False},
                    ]
                },
            ],
            # VALIDATION SET (5 problems) - used during search for cross-validation
            'val': [
                {
                    'id': 'count_vowels',
                    'description': 'Count number of vowels (a,e,i,o,u) in string (case insensitive)',
                    'signature': 'def count_vowels(s: str) -> int:',
                    'test_cases': [
                        {'input': 'hello', 'expected': 2},
                        {'input': 'AEIOU', 'expected': 5},
                        {'input': 'xyz', 'expected': 0},
                        {'input': '', 'expected': 0},
                    ]
                },
                {
                    'id': 'fibonacci',
                    'description': 'Return nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1)',
                    'signature': 'def fibonacci(n: int) -> int:',
                    'test_cases': [
                        {'input': 0, 'expected': 0},
                        {'input': 1, 'expected': 1},
                        {'input': 5, 'expected': 5},
                        {'input': 10, 'expected': 55},
                    ]
                },
                {
                    'id': 'longest_common_prefix',
                    'description': 'Find longest common prefix string among array of strings',
                    'signature': 'def longest_common_prefix(strs: list) -> str:',
                    'test_cases': [
                        {'input': [['flower', 'flow', 'flight']], 'expected': 'fl'},
                        {'input': [['dog', 'racecar', 'car']], 'expected': ''},
                        {'input': [['interspecies', 'interstellar', 'interstate']], 'expected': 'inters'},
                        {'input': [['alone']], 'expected': 'alone'},
                    ]
                },
                {
                    'id': 'reverse_linked_list',
                    'description': 'Reverse a singly linked list represented as array [1,2,3] -> [3,2,1]',
                    'signature': 'def reverse_linked_list(nums: list) -> list:',
                    'test_cases': [
                        {'input': [1, 2, 3, 4, 5], 'expected': [5, 4, 3, 2, 1]},
                        {'input': [1, 2], 'expected': [2, 1]},
                        {'input': [], 'expected': []},
                        {'input': [1], 'expected': [1]},
                    ]
                },
                {
                    'id': 'palindrome_number',
                    'description': 'Return True if integer is palindrome (reads same backward)',
                    'signature': 'def palindrome_number(x: int) -> bool:',
                    'test_cases': [
                        {'input': 121, 'expected': True},
                        {'input': -121, 'expected': False},
                        {'input': 10, 'expected': False},
                        {'input': 0, 'expected': True},
                    ]
                },
            ],
            # TEST SET (8 problems) - held out for final evaluation (includes harder problems)
            'test': [
                {
                    'id': 'is_anagram',
                    'description': 'Check if two strings are anagrams (same letters, different order)',
                    'signature': 'def is_anagram(s1: str, s2: str) -> bool:',
                    'test_cases': [
                        {'input': ['listen', 'silent'], 'expected': True},
                        {'input': ['hello', 'world'], 'expected': False},
                        {'input': ['', ''], 'expected': True},
                        {'input': ['a', 'a'], 'expected': True},
                    ]
                },
                {
                    'id': 'factorial',
                    'description': 'Calculate factorial of non-negative integer n (n!)',
                    'signature': 'def factorial(n: int) -> int:',
                    'test_cases': [
                        {'input': 0, 'expected': 1},
                        {'input': 1, 'expected': 1},
                        {'input': 5, 'expected': 120},
                        {'input': 7, 'expected': 5040},
                    ]
                },
                {
                    'id': 'max_subarray_sum',
                    'description': 'Find maximum sum of contiguous subarray (Kadane\'s algorithm)',
                    'signature': 'def max_subarray_sum(nums: list) -> int:',
                    'test_cases': [
                        {'input': [-2, 1, -3, 4, -1, 2, 1, -5, 4], 'expected': 6},  # [4,-1,2,1]
                        {'input': [1], 'expected': 1},
                        {'input': [5, 4, -1, 7, 8], 'expected': 23},
                        {'input': [-1, -2, -3], 'expected': -1},
                    ]
                },
                {
                    'id': 'search_insert_position',
                    'description': 'Find index where target would be inserted in sorted array',
                    'signature': 'def search_insert_position(nums: list, target: int) -> int:',
                    'test_cases': [
                        {'input': [[1, 3, 5, 6], 5], 'expected': 2},
                        {'input': [[1, 3, 5, 6], 2], 'expected': 1},
                        {'input': [[1, 3, 5, 6], 7], 'expected': 4},
                        {'input': [[1, 3, 5, 6], 0], 'expected': 0},
                    ]
                },
                {
                    'id': 'climbing_stairs',
                    'description': 'Number of distinct ways to climb n stairs (taking 1 or 2 steps at a time)',
                    'signature': 'def climbing_stairs(n: int) -> int:',
                    'test_cases': [
                        {'input': 2, 'expected': 2},  # 1+1, 2
                        {'input': 3, 'expected': 3},  # 1+1+1, 1+2, 2+1
                        {'input': 4, 'expected': 5},
                        {'input': 5, 'expected': 8},
                    ]
                },
                # HARDER PROBLEMS BELOW
                {
                    'id': 'longest_increasing_subsequence',
                    'description': 'Find length of longest strictly increasing subsequence (LIS) - requires DP',
                    'signature': 'def longest_increasing_subsequence(nums: list) -> int:',
                    'test_cases': [
                        {'input': [10, 9, 2, 5, 3, 7, 101, 18], 'expected': 4},  # [2,3,7,101] or [2,3,7,18]
                        {'input': [0, 1, 0, 3, 2, 3], 'expected': 4},  # [0,1,2,3]
                        {'input': [7, 7, 7, 7, 7, 7, 7], 'expected': 1},
                        {'input': [1, 3, 6, 7, 9, 4, 10, 5, 6], 'expected': 6},  # [1,3,4,5,6 and more]
                    ]
                },
                {
                    'id': 'coin_change',
                    'description': 'Minimum number of coins to make amount (coins can be reused). Return -1 if impossible.',
                    'signature': 'def coin_change(coins: list, amount: int) -> int:',
                    'test_cases': [
                        {'input': [[1, 2, 5], 11], 'expected': 3},  # 5+5+1
                        {'input': [[2], 3], 'expected': -1},  # impossible
                        {'input': [[1], 0], 'expected': 0},
                        {'input': [[1, 2, 5], 100], 'expected': 20},  # 20 coins of 5
                    ]
                },
                {
                    'id': 'group_anagrams',
                    'description': 'Group strings that are anagrams of each other. Return list of groups (any order).',
                    'signature': 'def group_anagrams(strs: list) -> list:',
                    'test_cases': [
                        {'input': [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
                         'expected': [['bat'], ['nat', 'tan'], ['ate', 'eat', 'tea']]},
                        {'input': [['']], 'expected': [['']]},
                        {'input': [['a']], 'expected': [['a']]},
                    ]
                },
            ]
        }


# ==============================================================================
# SEED AGENT - Scaffold + Seed Prompting Strategy
# ==============================================================================

# Import the fixed scaffold
from scaffold import AGENT_SCAFFOLD

# SEED PROMPTING STRATEGY - This is what the meta-LLM modifies
SEED_PROMPT_STRATEGY = '''
    def build_prompt(self, problem: dict) -> str:
        """
        Build the prompt for the LLM (MODIFIABLE).
        """
        prompt = f"""Write a Python function to solve this problem:

{problem['description']}

Function signature: {problem['signature']}

Put your solution between these delimiters:
[FUNCTION_START]
[FUNCTION_END]

Example:
[FUNCTION_START]
def add(x: int, y: int) -> int:
    return x + y
[FUNCTION_END]"""
        return prompt
'''

SEED_AGENT_CODE = AGENT_SCAFFOLD + SEED_PROMPT_STRATEGY

# ==============================================================================


class SelfImprovingAgent:
    """
    Self-improving agent with realistic setup.
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        model: str = "gemma3:1b",
        api_key: Optional[str] = None,
        max_iterations: int = 15,
        output_dir: str = "./output",
        train_test_split: float = 0.5,  # 3 train, 3 test
        debug: bool = False,
        lambda_points: int = 6,  # Number of λ values to sweep
        iters_per_lambda: Optional[int] = None,  # If None, auto-compute from max_iterations
        eval_subset_size: Optional[int] = None,  # Use subset of test cases (None = all)
        stochastic_eval: bool = False  # Randomly sample eval subset each iteration
    ):
        self.llm = LLMClient(backend=llm_backend, model=model, api_key=api_key)
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.debug = debug
        self.lambda_points = lambda_points
        self.iters_per_lambda = iters_per_lambda
        self.eval_subset_size = eval_subset_size
        self.stochastic_eval = stochastic_eval

        os.makedirs(output_dir, exist_ok=True)

        self.trials = Trials()
        self.best_agent = None
        self.agent_history: List[AgentVersion] = []
        self.current_lambda = 0.0  # Will be set during lambda sweep

        # Load benchmark with train/val/test split
        problems = Benchmark.get_problems()
        self.train_problems = problems['train']
        self.val_problems = problems['val']
        self.test_problems = problems['test']

        print(f"Loaded {len(self.train_problems) + len(self.val_problems) + len(self.test_problems)} problems:")
        print(f"  Train ({len(self.train_problems)}): {[p['id'] for p in self.train_problems]}")
        print(f"  Val ({len(self.val_problems)}): {[p['id'] for p in self.val_problems]}")
        print(f"  Test ({len(self.test_problems)}): {[p['id'] for p in self.test_problems]}")

        if self.eval_subset_size is not None:
            print(f"\nEvaluation mode: Using {self.eval_subset_size} test cases per problem during search")
            print(f"  (Full evaluation on best agent at end)")
            if self.stochastic_eval:
                print(f"  Stochastic: Random subset each iteration")
            else:
                print(f"  Deterministic: First {self.eval_subset_size} cases")
        print()

        # Seed agent
        self.seed_agent = AgentVersion(
            agent_id="agent_0000_seed",
            generation=0,
            parent_id=None,
            agent_code=SEED_AGENT_CODE,
            modification_description="Initial seed agent",
            timestamp=time.time()
        )

        # Record seed complexity for normalization
        self.seed_complexity = self._compute_agent_complexity(SEED_AGENT_CODE)
        logger.info(f"Seed agent complexity: {self.seed_complexity}")

    def modify_agent_code(
        self,
        parent_agent: AgentVersion,
        modification_goal: str,
        target_complexity: Optional[float] = None
    ) -> AgentVersion:
        """Use LLM to modify agent's source code"""

        agent_id = f"agent_{len(self.agent_history):04d}"
        logger.info(f"Starting modification for {agent_id}")
        logger.info(f"Goal: {modification_goal}")
        logger.info(f"Target complexity: {target_complexity}")

        # Build evaluation context from parent agent's performance
        eval_context = ""
        if parent_agent.train_loss < 999.0:
            eval_context = f"""
PARENT AGENT PERFORMANCE:
- Training loss: {parent_agent.train_loss:.2%} of problems failed
- Validation loss: {parent_agent.val_loss:.2%} of problems failed
- Complexity: {parent_agent.agent_complexity:.1f} degrees of freedom
- Free energy F(λ): {parent_agent.free_energy:.2f}

ANALYSIS:
"""
            if parent_agent.train_loss > 0.5:
                eval_context += "- The agent is failing on more than half of the training problems.\n"
                eval_context += "- Focus on improving the PROMPT to help the LLM generate correct code.\n"
            elif parent_agent.train_loss > 0.3:
                eval_context += "- The agent is moderately successful but can improve.\n"
                eval_context += "- Consider adding more guidance or examples to the PROMPT.\n"
            else:
                eval_context += "- The agent is performing well on training problems.\n"
                eval_context += "- Focus on making it more general or more efficient.\n"

            if parent_agent.val_loss > parent_agent.train_loss + 0.2:
                eval_context += "- OVERFITTING WARNING: Validation loss is much higher than training loss.\n"
                eval_context += "- Make the prompts more general, avoid hardcoding specific examples.\n"

        # Complexity hint
        complexity_hint = ""
        if target_complexity is not None:
            current_complexity = self._compute_agent_complexity(parent_agent.agent_code)
            if target_complexity < current_complexity * 0.7:
                complexity_hint = "\nIMPORTANT: Simplify the code - make it MORE CONCISE."
            elif target_complexity > current_complexity * 1.3:
                complexity_hint = "\nIMPORTANT: Add more sophistication - make it MORE CAPABLE."

        # CRITICAL CONSTRAINTS that apply to ALL modifications
        critical_constraints = f"""
CRITICAL: You are ONLY modifying the build_prompt() method.

THE FIXED SCAFFOLD (DO NOT MODIFY):
{AGENT_SCAFFOLD}

YOUR TASK: Make SMALL, INCREMENTAL improvements to build_prompt().

RULES:
1. Return ONLY the build_prompt(self, problem: dict) -> str method
2. The method must return a string (the prompt)
3. Make SMALL changes - don't completely rewrite the prompt
4. Keep the prompt CONCISE - verbose prompts confuse the LLM
5. The LLM API has only: self.llm.generate(prompt, max_tokens=N)
6. DO NOT implement problem solutions as methods
7. Test your changes mentally - will they actually help the LLM generate correct code?

IMPORTANT: Simpler is often better. Don't over-engineer the prompt.

CORRECT OUTPUT FORMAT:
    def build_prompt(self, problem: dict) -> str:
        # Small improvement to prompting strategy
        prompt = f"..."
        return prompt
"""

        # Use simple, concrete prompt templates for weak models
        # IMPORTANT: Use custom delimiters to avoid conflicts with code content
        if "few-shot" in modification_goal.lower() or "examples" in modification_goal.lower():
            prompt = f"""Make a SMALL improvement to build_prompt() by adding ONE concrete example.

CURRENT build_prompt() METHOD:
```python
{parent_agent.agent_code.split('def build_prompt')[1] if 'def build_prompt' in parent_agent.agent_code else SEED_PROMPT_STRATEGY}
```
{eval_context}

GOAL: Add just ONE simple, clear example showing how to format the output.
{complexity_hint}

Keep it SHORT and CLEAR. Don't add verbose explanations.

{critical_constraints}

IMPORTANT: Wrap ONLY the build_prompt() method with these delimiters:
<|BEGIN_CODE|>
    def build_prompt(self, problem: dict) -> str:
        # Small improvement here
        return prompt
<|END_CODE|>"""

        elif "chain-of-thought" in modification_goal.lower():
            prompt = f"""Make a SMALL improvement to build_prompt() by adding brief reasoning guidance.

CURRENT build_prompt() METHOD:
```python
{parent_agent.agent_code.split('def build_prompt')[1] if 'def build_prompt' in parent_agent.agent_code else SEED_PROMPT_STRATEGY}
```
{eval_context}

GOAL: Add 1-2 sentences asking the LLM to think before coding.
{complexity_hint}

Keep it BRIEF. Long explanations confuse the LLM.

{critical_constraints}

IMPORTANT: Wrap ONLY the build_prompt() method with these delimiters:
<|BEGIN_CODE|>
    def build_prompt(self, problem: dict) -> str:
        # Small improvement here
        return prompt
<|END_CODE|>"""

        else:
            # Default: improve based on parent
            prompt = f"""Make a SMALL, focused improvement to build_prompt().

CURRENT build_prompt() METHOD:
```python
{parent_agent.agent_code.split('def build_prompt')[1] if 'def build_prompt' in parent_agent.agent_code else SEED_PROMPT_STRATEGY}
```
{eval_context}

GOAL: {modification_goal}
{complexity_hint}

Make ONE small change. Keep the prompt CONCISE and CLEAR.

{critical_constraints}

IMPORTANT: Wrap ONLY the build_prompt() method with these delimiters:
<|BEGIN_CODE|>
    def build_prompt(self, problem: dict) -> str:
        # Small improvement here
        return prompt
<|END_CODE|>"""

        # Generate code, then fix errors iteratively
        improved_code = None
        max_retries = 3
        current_code = None
        error_feedback = None

        for attempt in range(max_retries):
            try:
                # Build prompt with error feedback if available
                if error_feedback is None:
                    # First attempt: normal generation
                    generation_prompt = prompt
                else:
                    # Subsequent attempts: ask LLM to fix the error
                    generation_prompt = f"""The previous code had an error. Fix it!

BROKEN CODE:
```python
{current_code}
```

ERROR: {error_feedback}

Generate FIXED code. It must have:
- __init__(self, llm_client) method
- solve_problems(self, problems: list) -> str method

Return ONLY the complete corrected Python code for the SolvingAgent class."""

                generated = self.llm.generate(generation_prompt, max_tokens=2000)

                logger.info(f"LLM generated code (attempt {attempt+1}/{max_retries})")
                logger.debug(f"LLM output: {generated[:500]}...")

                if self.debug:
                    print(f"\n{'='*70}")
                    print(f"DEBUG: LLM Output (attempt {attempt+1}):")
                    print(f"{'='*70}")
                    print(generated[:2000] + ("..." if len(generated) > 2000 else ""))
                    print(f"{'='*70}\n")

                # Extract ONLY the build_prompt() method
                import re

                # Strategy 1: Look for custom delimiters first
                if "<|BEGIN_CODE|>" in generated and "<|END_CODE|>" in generated:
                    start = generated.find("<|BEGIN_CODE|>") + len("<|BEGIN_CODE|>")
                    end = generated.find("<|END_CODE|>")
                    build_prompt_method = generated[start:end].strip()

                # Strategy 2: Fall back to markdown code blocks
                elif "```" in generated:
                    match = re.search(r'```(?:python)?\s*\n(.*)```', generated, re.DOTALL)
                    if match:
                        build_prompt_method = match.group(1).strip()
                    else:
                        build_prompt_method = generated.strip()

                # Strategy 3: Find def build_prompt directly
                elif "def build_prompt" in generated:
                    build_prompt_method = generated[generated.find("def build_prompt"):].strip()

                else:
                    build_prompt_method = generated.strip()

                # Ensure it starts with "def build_prompt"
                if not build_prompt_method.strip().startswith("def build_prompt"):
                    # Try to extract just the method
                    if "def build_prompt" in build_prompt_method:
                        idx = build_prompt_method.find("def build_prompt")
                        build_prompt_method = build_prompt_method[idx:]
                    else:
                        raise ValueError("Generated code does not contain build_prompt() method")

                # Reconstruct full agent: SCAFFOLD + build_prompt() method
                code = AGENT_SCAFFOLD + build_prompt_method

                # VALIDATE: Check if code compiles
                compile(code, '<string>', 'exec')

                # Success! Use this code
                improved_code = code
                logger.info(f"Code validation successful for {agent_id} (attempt {attempt+1})")
                print(f"  ✓ Code validated successfully" + (f" after {attempt+1} attempts" if attempt > 0 else ""))
                break

            except SyntaxError as e:
                error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
                logger.warning(f"Compilation failed for {agent_id} (attempt {attempt+1}): {error_msg}")
                print(f"  [!] Attempt {attempt+1}/{max_retries}: {error_msg}")

                if self.debug and 'code' in locals():
                    lines = code.split('\n')
                    print(f"  [DEBUG] Code around line {e.lineno}:")
                    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
                        marker = " >>> " if i == e.lineno-1 else "     "
                        print(f"  {marker}{i+1:3d}: {repr(lines[i])}")

                current_code = code
                error_feedback = error_msg

                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate valid code for {agent_id} after {max_retries} attempts, falling back to SEED")
                    print(f"  [!] Could not fix errors, using SEED agent as fallback")
                    improved_code = SEED_AGENT_CODE

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"Error in {agent_id} (attempt {attempt+1}): {error_msg}")
                print(f"  [!] Attempt {attempt+1}/{max_retries}: {error_msg}")
                current_code = code if 'code' in locals() else parent_agent.agent_code
                error_feedback = error_msg

                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate valid code for {agent_id} after {max_retries} attempts, falling back to SEED")
                    print(f"  [!] Could not fix errors, using SEED agent as fallback")
                    improved_code = SEED_AGENT_CODE

        return AgentVersion(
            agent_id=agent_id,
            generation=parent_agent.generation + 1,
            parent_id=parent_agent.agent_id,
            agent_code=improved_code,
            modification_description=modification_goal,
            timestamp=time.time()
        )

    def evaluate_agent(
        self,
        agent: AgentVersion,
        lambda_val: float,
        use_subset: bool = True
    ) -> AgentVersion:
        """Evaluate agent on benchmark"""
        logger.info(f"Evaluating {agent.agent_id} (λ={lambda_val:.3f}, subset={use_subset})")
        try:
            # Execute agent code
            namespace = {'llm_client': self.llm}
            exec(agent.agent_code, namespace)

            if 'SolvingAgent' not in namespace:
                raise ValueError("No SolvingAgent class found")

            # Create agent instance
            agent_instance = namespace['SolvingAgent'](self.llm)

            # Measure AGENT complexity
            agent_complexity = self._compute_agent_complexity(agent.agent_code)

            # During search: only evaluate train+val (faster!)
            # Final eval: evaluate all including test
            if use_subset:
                # FAST PATH: Only generate solutions for train+val during search
                search_problems = self.train_problems + self.val_problems
                print(f"\n  Evaluating agent on {len(search_problems)} problems (train+val)...")
                solution_code = agent_instance.solve_problems(search_problems)

                # Compile solutions ONCE
                # Fix any literal \n in the response (LLM returns \n as text)
                solution_code = solution_code.replace(r'\n', '\n')

                namespace = {}
                try:
                    exec(solution_code, namespace)
                except Exception as e:
                    print(f"  [!] Code compilation failed: {e}")
                    print(f"  [DEBUG] Code preview:")
                    print(solution_code[:500])
                    raise

                # DEBUG: Show what functions were found
                found_funcs = [k for k in namespace.keys() if not k.startswith('_')]
                if len(found_funcs) == 0:
                    print(f"  [!] No functions found!")
                    print(f"  [DEBUG] Code: {repr(solution_code[:300])}")

                print(f"\n  Training set evaluation:")
                train_loss = self._compute_loss(namespace, self.train_problems, use_subset=True, verbose=True)

                print(f"\n  Validation set evaluation:")
                val_loss = self._compute_loss(namespace, self.val_problems, use_subset=True, verbose=True)

                test_loss = -1.0  # Not computed during search (held out)
            else:
                # FULL EVAL: Generate solutions for all problems
                all_problems = self.train_problems + self.val_problems + self.test_problems
                print(f"\n  Final evaluation on {len(all_problems)} problems (train+val+test)...")
                solution_code = agent_instance.solve_problems(all_problems)

                # Compile solutions ONCE
                # Fix any literal \n in the response (LLM returns \n as text)
                solution_code = solution_code.replace(r'\n', '\n')

                namespace = {}
                try:
                    exec(solution_code, namespace)
                except Exception as e:
                    print(f"  [!] Code compilation failed: {e}")
                    print(f"  [DEBUG] Code preview:")
                    print(solution_code[:500])
                    raise

                # DEBUG: Show what functions were found
                found_funcs = [k for k in namespace.keys() if not k.startswith('_')]
                if len(found_funcs) == 0:
                    print(f"  [!] No functions found!")
                    print(f"  [DEBUG] Code: {repr(solution_code[:300])}")

                print(f"\n  Training set evaluation:")
                train_loss = self._compute_loss(namespace, self.train_problems, use_subset=False, verbose=True)

                print(f"\n  Validation set evaluation:")
                val_loss = self._compute_loss(namespace, self.val_problems, use_subset=False, verbose=True)

                print(f"\n  Test set evaluation:")
                test_loss = self._compute_loss(namespace, self.test_problems, use_subset=False, verbose=True)

            # FREE ENERGY: Use train+val loss for optimization (cross-validation)
            # F(λ) = λ·sigmoid(C/C_seed - 1) + (L_train + L_val) / 2
            # Sigmoid squashes complexity change to [0, 1] range to match loss scale
            # - C < C_seed → negative input → sigmoid < 0.5 (simpler agent)
            # - C = C_seed → sigmoid(0) = 0.5 (same complexity)
            # - C > C_seed → positive input → sigmoid > 0.5 (more complex agent)
            # Test loss is held out for final evaluation only
            complexity_ratio = agent_complexity / self.seed_complexity
            complexity_penalty = 1 / (1 + np.exp(-(complexity_ratio - 1)))  # sigmoid(C/C_seed - 1)
            free_energy = lambda_val * complexity_penalty + (train_loss + val_loss) / 2

            # Update
            agent.train_loss = train_loss
            agent.val_loss = val_loss
            agent.test_loss = test_loss
            agent.agent_complexity = agent_complexity
            agent.lambda_val = lambda_val
            agent.free_energy = free_energy

            logger.info(f"Evaluation complete for {agent.agent_id}: F(λ)={free_energy:.4f}, "
                       f"train={train_loss:.4f}, val={val_loss:.4f}, test={test_loss:.4f}")

        except Exception as e:
            logger.error(f"Evaluation failed for {agent.agent_id}: {e}")
            print(f"  [!] Evaluation failed: {e}")
            agent.train_loss = 999.0
            agent.val_loss = 999.0
            agent.test_loss = 999.0
            agent.agent_complexity = self._compute_agent_complexity(agent.agent_code)
            agent.lambda_val = lambda_val
            agent.free_energy = 999.0 + lambda_val * agent.agent_complexity

        return agent

    def _compute_agent_complexity(self, agent_code: str) -> float:
        """Complexity of AGENT code"""
        lines = len([l for l in agent_code.split('\n')
                    if l.strip() and not l.strip().startswith('#')])
        assignments = len(re.findall(r'\b\w+\s*=', agent_code))
        control_flow = len(re.findall(r'\b(if|for|while|elif|try|except)\b', agent_code))
        functions = len(re.findall(r'\bdef\s+\w+', agent_code))
        classes = len(re.findall(r'\bclass\s+\w+', agent_code))

        complexity = (lines + assignments + control_flow * 2 +
                     functions * 3 + classes * 5)
        return float(max(1, complexity))

    def _get_eval_subset(self, problems: List[Dict]) -> List[Dict]:
        """Get subset of test cases for faster evaluation during search"""
        if self.eval_subset_size is None:
            return problems  # Use all test cases

        # Subsample test cases from each problem
        import random
        subsampled = []
        for problem in problems:
            test_cases = problem['test_cases']

            if self.stochastic_eval:
                # Randomly sample subset each time
                n_cases = min(self.eval_subset_size, len(test_cases))
                sampled_cases = random.sample(test_cases, n_cases)
            else:
                # Always use first N test cases (deterministic)
                sampled_cases = test_cases[:self.eval_subset_size]

            subsampled.append({
                **problem,
                'test_cases': sampled_cases
            })

        return subsampled

    def _compute_loss(self, namespace: dict, problems: List[Dict], use_subset: bool = True, verbose: bool = True) -> float:
        """Error rate on problems"""
        from tqdm import tqdm

        # Use subset during search, full set for final evaluation
        if use_subset:
            problems = self._get_eval_subset(problems)

        total_tests = 0
        failed_tests = 0
        problem_results = []

        # Progress bar for problems
        pbar = tqdm(problems, desc="    Evaluating", disable=not verbose,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

        for problem in pbar:
            problem_failed = 0
            problem_total = 0

            if problem['id'] not in namespace:
                problem_failed = len(problem['test_cases'])
                problem_total = len(problem['test_cases'])
                failed_tests += problem_failed
                total_tests += problem_total
                problem_results.append((problem['id'], False, "MISSING"))
                if verbose:
                    pbar.write(f"    ✗ {problem['id']}: FAIL (function not found)")
                continue

            func = namespace[problem['id']]
            test_passed = True
            error_msg = None

            for test_case in problem['test_cases']:
                problem_total += 1
                total_tests += 1
                try:
                    inp = test_case['input']
                    # Smart dispatch: handle different input formats
                    if isinstance(inp, list):
                        if any(isinstance(x, list) for x in inp):
                            # Contains nested lists -> multiple args
                            result = func(*inp)
                        elif len(inp) > 1 and all(isinstance(x, (str, int, float, bool)) for x in inp):
                            # Multiple primitives -> try unpacking as multiple args first
                            try:
                                result = func(*inp)
                            except TypeError:
                                # If that fails, pass as single list
                                result = func(inp)
                        else:
                            # Single list arg
                            result = func(inp)
                    else:
                        result = func(inp)

                    if result != test_case['expected']:
                        problem_failed += 1
                        failed_tests += 1
                        test_passed = False
                        if error_msg is None:
                            error_msg = f"wrong output: got {result}, expected {test_case['expected']}"
                except Exception as e:
                    problem_failed += 1
                    failed_tests += 1
                    test_passed = False
                    if error_msg is None:
                        error_msg = str(e)

            problem_results.append((problem['id'], test_passed, error_msg))
            if verbose:
                if test_passed:
                    pbar.write(f"    ✓ {problem['id']}: OK ({problem_total}/{problem_total} tests passed)")
                else:
                    pbar.write(f"    ✗ {problem['id']}: FAIL ({problem_total - problem_failed}/{problem_total} tests passed) - {error_msg}")

        pbar.close()

        if verbose:
            passed_count = sum(1 for _, passed, _ in problem_results if passed)
            print(f"    Summary: {passed_count}/{len(problems)} problems passed")

        return failed_tests / max(1, total_tests)

    def objective_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """TPE objective (for current λ)"""
        iteration = len(self.agent_history)

        # Lambda is fixed for this sweep iteration
        lambda_val = self.current_lambda
        modification_goal = params['modification_goal']
        target_complexity = params.get('target_complexity', None)

        # Get parent
        parent = self.best_agent if self.best_agent else self.seed_agent

        # First iteration: just use SEED agent directly (don't modify yet)
        if iteration == 0:
            print(f"  [*] First iteration: using SEED agent")
            modified_agent = AgentVersion(
                agent_id=f"agent_{iteration:04d}",
                generation=0,
                parent_id=None,
                agent_code=SEED_AGENT_CODE,
                modification_description="SEED agent (unmodified)",
                timestamp=time.time()
            )
        else:
            # SELF-MODIFICATION
            modified_agent = self.modify_agent_code(parent, modification_goal, target_complexity)

        # Evaluate
        evaluated_agent = self.evaluate_agent(modified_agent, lambda_val)

        # Save
        self.agent_history.append(evaluated_agent)
        self._save_agent(evaluated_agent)

        # Update best
        if self.best_agent is None or evaluated_agent.free_energy < self.best_agent.free_energy:
            self.best_agent = evaluated_agent
            self._save_best_agent(evaluated_agent)
            c_ratio = evaluated_agent.agent_complexity / self.seed_complexity
            c_penalty = 1 / (1 + np.exp(-(c_ratio - 1)))
            if evaluated_agent.test_loss >= 0:
                # Full eval - show test loss
                print(f"✓ New best! F(λ)={evaluated_agent.free_energy:.4f} "
                      f"[λ={evaluated_agent.lambda_val:.3f}, σ(C)={c_penalty:.3f}, "
                      f"L_val={evaluated_agent.val_loss:.4f}, L_test={evaluated_agent.test_loss:.4f}]")
            else:
                # During search - don't show test loss
                print(f"✓ New best! F(λ)={evaluated_agent.free_energy:.4f} "
                      f"[λ={evaluated_agent.lambda_val:.3f}, σ(C)={c_penalty:.3f}, "
                      f"L_val={evaluated_agent.val_loss:.4f}]")

        # During search: don't show test loss (held out). After search: show it.
        c_ratio = evaluated_agent.agent_complexity / self.seed_complexity
        c_penalty = 1 / (1 + np.exp(-(c_ratio - 1)))
        if evaluated_agent.test_loss >= 0:
            # Full evaluation - show all losses
            print(f"[{iteration:3d}] λ={lambda_val:.3f} σ(C)={c_penalty:.3f} "
                  f"L_train={evaluated_agent.train_loss:.4f} L_val={evaluated_agent.val_loss:.4f} "
                  f"L_test={evaluated_agent.test_loss:.4f} F(λ)={evaluated_agent.free_energy:.4f}")
        else:
            # During search - only show train+val (test held out)
            print(f"[{iteration:3d}] λ={lambda_val:.3f} σ(C)={c_penalty:.3f} "
                  f"L_train={evaluated_agent.train_loss:.4f} L_val={evaluated_agent.val_loss:.4f} "
                  f"F(λ)={evaluated_agent.free_energy:.4f}")
        print(f"         Goal: {evaluated_agent.modification_description}")

        return {
            'loss': evaluated_agent.free_energy,
            'status': STATUS_OK,
        }

    def run(self):
        """Run TPE-guided self-improvement with λ sweep (like structure-functions paper)"""
        print("="*70)
        print("SELF-MODIFYING AGENT (TPE + Free Energy)")
        print("="*70)
        print(f"Objective: minimize F(λ) = λ·C_agent + (L_train + L_val) / 2")
        print(f"LLM: {self.llm.backend} / {self.llm.model}")
        print(f"Strategy: Sweep λ, use TPE for agent modifications")
        print(f"Max iterations per λ: {self.max_iterations}")
        print("="*70 + "\n")

        # Lambda sweep (like in structure-functions paper!)
        # We try different regularization strengths
        lambda_values = np.linspace(0.0, 0.5, self.lambda_points)

        # Budget iterations across lambda values
        if self.iters_per_lambda is None:
            # Auto-compute: split max_iterations across lambda values
            iters_per_lambda = max(2, self.max_iterations // len(lambda_values))
        else:
            iters_per_lambda = self.iters_per_lambda

        total_iters = iters_per_lambda * len(lambda_values)

        print(f"Lambda sweep: {len(lambda_values)} values from {lambda_values[0]:.2f} to {lambda_values[-1]:.2f}")
        print(f"Iterations per λ: {iters_per_lambda} (total: {total_iters})")
        print(f"Note: Fewer λ points = faster but coarser search\n")

        # For each λ, use TPE to find best agent
        for lambda_idx, lambda_val in enumerate(lambda_values):
            print(f"{'='*70}")
            print(f"LAMBDA {lambda_idx+1}/{len(lambda_values)}: λ = {lambda_val:.4f}")
            print(f"{'='*70}\n")

            # Seed RNG for this λ sweep so all agents see the same test cases
            # (makes comparisons fair within each λ sweep)
            if self.stochastic_eval:
                import random
                random.seed(lambda_idx * 42)  # Different seed per λ, but consistent within λ

            # Store current lambda for objective function
            self.current_lambda = lambda_val

            # TPE search space (no lambda here! just agent modifications)
            search_space = {
                'modification_goal': hp.choice('modification_goal', [
                    'Improve the LLM prompts to generate better code',
                    'Add few-shot examples to prompts',
                    'Add chain-of-thought reasoning',
                    'Improve error handling and fallbacks',
                    'Add self-correction or verification',
                    'Optimize the problem-solving strategy',
                ]),
                'target_complexity': hp.quniform('target_complexity', 30, 150, 10),
            }

            # Run TPE for this λ
            trials = Trials()
            fmin(
                fn=self.objective_function,
                space=search_space,
                algo=tpe.suggest,
                max_evals=iters_per_lambda,
                trials=trials,
                verbose=False
            )

            print()  # Spacing between lambda values

        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        if self.best_agent:
            # Re-evaluate best agent on FULL test set if we used subsets during search
            if self.eval_subset_size is not None:
                print(f"Re-evaluating best agent on full test set...")
                self.best_agent = self.evaluate_agent(
                    self.best_agent,
                    self.best_agent.lambda_val,
                    use_subset=False  # Full evaluation
                )

            c_ratio = self.best_agent.agent_complexity / self.seed_complexity
            c_penalty = 1 / (1 + np.exp(-(c_ratio - 1)))
            print(f"Best: {self.best_agent.agent_id} (gen {self.best_agent.generation})")
            print(f"  λ*: {self.best_agent.lambda_val:.4f}")
            print(f"  C_agent: {self.best_agent.agent_complexity:.1f}")
            print(f"  C/C_seed: {c_ratio:.2f}")
            print(f"  σ(C): {c_penalty:.3f}")
            print(f"  Train loss: {self.best_agent.train_loss:.4f}")
            print(f"  Test loss: {self.best_agent.test_loss:.4f}")
            print(f"  F(λ*): {self.best_agent.free_energy:.4f}")
        print(f"{'='*70}\n")

        return self.best_agent

    def _save_agent(self, agent: AgentVersion):
        filepath = os.path.join(self.output_dir, "agent_history.jsonl")
        with open(filepath, 'a') as f:
            json.dump(asdict(agent), f)
            f.write('\n')

    def _save_best_agent(self, agent: AgentVersion):
        filepath = os.path.join(self.output_dir, "best_agent_code.py")
        with open(filepath, 'w') as f:
            f.write(f"# Best: {agent.agent_id} (gen {agent.generation})\n")
            f.write(f"# F(λ): {agent.free_energy:.4f}\n")
            f.write(f"# C_agent: {agent.agent_complexity:.1f}\n")
            f.write(f"# L_test: {agent.test_loss:.4f}\n\n")
            f.write(agent.agent_code)


def main():
    import sys
    debug = "--debug" in sys.argv

    # Configuration: 10 lambda points × 6 iters each = 60 total evaluations
    # Lambda sweep: [0.0, 0.056, 0.111, 0.167, 0.222, 0.278, 0.333, 0.389, 0.444, 0.5]
    # Fine-grained sweep to capture the phase transition / elbow point
    #
    # Why finer sweep?
    # - The elbow in the loss-complexity curve reveals the optimal regularization
    # - Too coarse (3 points) → miss the phase transition entirely
    # - Too fine (20 points) → not enough iterations per λ to find good agents
    # - 10 points × 6 iters = good balance for finding the elbow
    #
    # Philosophy: Use relatively weak LLM but evolve good agent behavior
    # - Weak LLM alone: poor coding performance
    # - Weak LLM + evolved prompting: can solve non-trivial problems!
    #
    # Models to try:
    #
    # ANTHROPIC (requires ANTHROPIC_API_KEY) - FASTEST, RECOMMENDED:
    # - claude-3-5-sonnet-20241022: Latest Sonnet (best for coding)
    # - claude-3-5-haiku-20241022: Haiku (faster, cheaper)
    #
    # OLLAMA CLOUD (requires OLLAMA_API_KEY from ollama.com/settings/keys) - SLOWER:
    # - minimax-m2-cloud: MiniMax M2 model
    # - gpt-oss:120b-cloud: 120B general purpose
    # - qwen3-coder:480b-cloud: Alibaba's 480B coding specialist
    # - deepseek-v3.1:671b-cloud: 671B reasoning/coding
    # Note: Cloud models must include the exact tag with -cloud suffix
    agent = SelfImprovingAgent(
        llm_backend="anthropic",  # Use Anthropic API directly (MUCH faster than Ollama)
        model="claude-3-5-haiku-20241022",  # Fast and cheap - good for testing evolution!
        max_iterations=60,  # Total budget (will be split across lambda values)
        output_dir="./output",
        debug=debug,
        lambda_points=10,  # Fine-grained sweep to capture phase transitions
        iters_per_lambda=6,  # Enough iterations to find good agents per λ
        eval_subset_size=2,  # Use 2 test cases per problem during search (stochastic!)
        stochastic_eval=True,  # Random subset each iter → prevents overfitting
    )

    best = agent.run()

    if best and best.test_loss < 0.9:
        print("\nEVOLVED AGENT CODE:")
        print("="*70)
        print(best.agent_code[:500] + "...")


if __name__ == "__main__":
    main()
