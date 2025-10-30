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
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL


@dataclass
class AgentVersion:
    """A version of the self-modifying agent"""
    agent_id: str
    generation: int
    parent_id: Optional[str]
    agent_code: str

    train_loss: float = 999.0
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
    """Very hard algorithmic problems - require deep reasoning"""

    @staticmethod
    def get_problems():
        return [
            # Train set - hard algorithmic problems
            {
                'id': 'edit_distance',
                'description': 'Compute minimum edit distance (Levenshtein) between two strings using insertions, deletions, substitutions',
                'signature': 'def edit_distance(s1: str, s2: str) -> int:',
                'test_cases': [
                    {'input': ['horse', 'ros'], 'expected': 3},
                    {'input': ['intention', 'execution'], 'expected': 5},
                    {'input': ['', 'abc'], 'expected': 3},
                    {'input': ['abc', 'abc'], 'expected': 0},
                ]
            },
            {
                'id': 'longest_increasing_subsequence',
                'description': 'Find length of longest strictly increasing subsequence',
                'signature': 'def longest_increasing_subsequence(nums: list) -> int:',
                'test_cases': [
                    {'input': [10,9,2,5,3,7,101,18], 'expected': 4},  # [2,3,7,101]
                    {'input': [0,1,0,3,2,3], 'expected': 4},
                    {'input': [7,7,7,7,7,7,7], 'expected': 1},
                    {'input': [1,3,6,7,9,4,10,5,6], 'expected': 6},
                ]
            },
            {
                'id': 'trapping_rain_water',
                'description': 'Compute total trapped rainwater given elevation map array',
                'signature': 'def trapping_rain_water(heights: list) -> int:',
                'test_cases': [
                    {'input': [0,1,0,2,1,0,1,3,2,1,2,1], 'expected': 6},
                    {'input': [4,2,0,3,2,5], 'expected': 9},
                    {'input': [1,2,3,4], 'expected': 0},
                    {'input': [4,3,2,1], 'expected': 0},
                ]
            },
            # Test set - extremely hard problems
            {
                'id': 'regular_expression_matching',
                'description': 'Implement regex matching with . (any char) and * (zero or more of prev char)',
                'signature': 'def regular_expression_matching(s: str, p: str) -> bool:',
                'test_cases': [
                    {'input': ['aa', 'a'], 'expected': False},
                    {'input': ['aa', 'a*'], 'expected': True},
                    {'input': ['ab', '.*'], 'expected': True},
                    {'input': ['mississippi', 'mis*is*p*.'], 'expected': False},
                ]
            },
            {
                'id': 'word_ladder_length',
                'description': 'Min transformations from begin to end word, changing one letter at a time (all intermediate must be in wordList)',
                'signature': 'def word_ladder_length(beginWord: str, endWord: str, wordList: list) -> int:',
                'test_cases': [
                    {'input': ['hit', 'cog', ['hot','dot','dog','lot','log','cog']], 'expected': 5},
                    {'input': ['hit', 'cog', ['hot','dot','dog','lot','log']], 'expected': 0},
                    {'input': ['a', 'c', ['a','b','c']], 'expected': 2},
                ]
            },
            {
                'id': 'max_sliding_window',
                'description': 'Return max value in each sliding window of size k',
                'signature': 'def max_sliding_window(nums: list, k: int) -> list:',
                'test_cases': [
                    {'input': [[1,3,-1,-3,5,3,6,7], 3], 'expected': [3,3,5,5,6,7]},
                    {'input': [[1], 1], 'expected': [1]},
                    {'input': [[1,-1], 1], 'expected': [1,-1]},
                    {'input': [[9,11], 2], 'expected': [11]},
                ]
            },
        ]


# ==============================================================================
# REALISTIC SEED AGENT - Has structure but needs improvement
# ==============================================================================

SEED_AGENT_CODE = '''
class SolvingAgent:
    """
    Weak baseline agent with naive prompting.

    This needs significant improvement:
    - Prompt is too terse
    - No examples
    - No guidance on edge cases
    - No error handling hints
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def solve_problem(self, problem: dict) -> str:
        """Naive prompt - just asks for code directly"""
        # Very basic prompt - needs improvement!
        prompt = f"Write: {problem['signature']}\\nTask: {problem['description']}"

        code = self.llm.generate(prompt, max_tokens=300)

        # Minimal cleanup
        if "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            if code.startswith("python"):
                code = code[6:].strip()

        return code

    def solve_problems(self, problems: list) -> str:
        """Solve all problems"""
        solutions = []
        for problem in problems:
            try:
                solution = self.solve_problem(problem)
                solutions.append(solution)
            except:
                solutions.append(f"# Error on {problem['id']}")
        return "\\n\\n".join(solutions)
'''

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

        # Load benchmark
        all_problems = Benchmark.get_problems()
        split_idx = max(1, int(len(all_problems) * train_test_split))
        self.train_problems = all_problems[:split_idx]
        self.test_problems = all_problems[split_idx:]

        print(f"Loaded {len(all_problems)} problems:")
        print(f"  Train ({len(self.train_problems)}): {[p['id'] for p in self.train_problems]}")
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

    def modify_agent_code(
        self,
        parent_agent: AgentVersion,
        modification_goal: str,
        target_complexity: Optional[float] = None
    ) -> AgentVersion:
        """Use LLM to modify agent's source code"""

        agent_id = f"agent_{len(self.agent_history):04d}"

        # Complexity hint
        complexity_hint = ""
        if target_complexity is not None:
            current_complexity = self._compute_agent_complexity(parent_agent.agent_code)
            if target_complexity < current_complexity * 0.7:
                complexity_hint = "\nIMPORTANT: Simplify the code - make it MORE CONCISE."
            elif target_complexity > current_complexity * 1.3:
                complexity_hint = "\nIMPORTANT: Add more sophistication - make it MORE CAPABLE."

        # Use simple, concrete prompt templates for weak models
        if "few-shot" in modification_goal.lower() or "examples" in modification_goal.lower():
            prompt = f"""Generate a SolvingAgent class that uses few-shot examples in prompts.

Example template:
```python
class SolvingAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def solve_problem(self, problem: dict) -> str:
        # Add examples to the prompt
        prompt = f"Solve this problem step by step with examples.\\n\\n"
        prompt += f"Problem: {{problem['description']}}\\n"
        prompt += f"Signature: {{problem['signature']}}\\n\\n"
        prompt += "Example: For input [1,2,3], output should be...\\n\\n"
        prompt += "Now write the Python function:"

        code = self.llm.generate(prompt, max_tokens=500)
        # Clean code...
        return code

    def solve_problems(self, problems: list) -> str:
        return "\\n\\n".join([self.solve_problem(p) for p in problems])
```

Generate similar code that adds helpful examples to prompts."""

        elif "chain-of-thought" in modification_goal.lower():
            prompt = f"""Generate a SolvingAgent that uses chain-of-thought prompting.

Template:
```python
class SolvingAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def solve_problem(self, problem: dict) -> str:
        prompt = f"Let's solve this step by step:\\n\\n"
        prompt += f"Problem: {{problem['description']}}\\n"
        prompt += f"Signature: {{problem['signature']}}\\n\\n"
        prompt += "Step 1: Understand the problem\\n"
        prompt += "Step 2: Plan the approach\\n"
        prompt += "Step 3: Write the code:\\n"

        code = self.llm.generate(prompt, max_tokens=500)
        return code

    def solve_problems(self, problems: list) -> str:
        return "\\n\\n".join([self.solve_problem(p) for p in problems])
```

Generate similar code with step-by-step reasoning."""

        else:
            # Default: simple improvement
            prompt = f"""Generate a better SolvingAgent class that improves the prompt quality.

Current approach is too simple. Make the prompt MORE DETAILED and CLEARER.

Template:
```python
class SolvingAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def solve_problem(self, problem: dict) -> str:
        # Better, more detailed prompt
        prompt = f"Write a Python function:\\n"
        prompt += f"Function: {{problem['signature']}}\\n"
        prompt += f"Task: {{problem['description']}}\\n\\n"
        prompt += "Return only the function code, no explanations.\\n"

        code = self.llm.generate(prompt, max_tokens=400)

        # Clean up code
        if "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    def solve_problems(self, problems: list) -> str:
        solutions = []
        for p in problems:
            try:
                solutions.append(self.solve_problem(p))
            except:
                solutions.append("# Error")
        return "\\n\\n".join(solutions)
```

Generate improved code with better prompts."""

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

                if self.debug:
                    print(f"\n{'='*70}")
                    print(f"DEBUG: LLM Output (attempt {attempt+1}):")
                    print(f"{'='*70}")
                    print(generated[:2000] + ("..." if len(generated) > 2000 else ""))
                    print(f"{'='*70}\n")

                # Extract code (be careful with ``` inside strings!)
                # Use GREEDY match to get everything until LAST ```
                import re

                # Match from opening ``` to the LAST closing ``` (greedy)
                # This handles both ```python and ``` openings
                match = re.search(r'```(?:python)?\s*\n(.*)```', generated, re.DOTALL)

                if match:
                    code = match.group(1).strip()
                else:
                    code = generated.strip()

                if not code.strip().startswith("class SolvingAgent"):
                    code = "class SolvingAgent:\n" + code

                # VALIDATE: Check if code compiles
                compile(code, '<string>', 'exec')

                # Success! Use this code
                improved_code = code
                print(f"  ✓ Code validated successfully" + (f" after {attempt+1} attempts" if attempt > 0 else ""))
                break

            except SyntaxError as e:
                error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
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
                    print(f"  [!] Could not fix errors, using SEED agent as fallback")
                    improved_code = SEED_AGENT_CODE

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  [!] Attempt {attempt+1}/{max_retries}: {error_msg}")
                current_code = code if 'code' in locals() else parent_agent.agent_code
                error_feedback = error_msg

                if attempt == max_retries - 1:
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
        try:
            # Execute agent code
            namespace = {'llm_client': self.llm}
            exec(agent.agent_code, namespace)

            if 'SolvingAgent' not in namespace:
                raise ValueError("No SolvingAgent class found")

            # Create agent instance
            agent_instance = namespace['SolvingAgent'](self.llm)

            # Generate solutions for all problems
            all_problems = self.train_problems + self.test_problems
            solution_code = agent_instance.solve_problems(all_problems)

            # Measure AGENT complexity
            agent_complexity = self._compute_agent_complexity(agent.agent_code)

            # Compute losses (use subset during search, full set for final eval)
            train_loss = self._compute_loss(solution_code, self.train_problems, use_subset=use_subset)
            test_loss = self._compute_loss(solution_code, self.test_problems, use_subset=use_subset)

            # FREE ENERGY (use train loss, like in structure-functions paper!)
            # F(λ) = λ·C + L_train
            # Test loss is just for measuring generalization
            free_energy = lambda_val * agent_complexity + train_loss

            # Update
            agent.train_loss = train_loss
            agent.test_loss = test_loss
            agent.agent_complexity = agent_complexity
            agent.lambda_val = lambda_val
            agent.free_energy = free_energy

        except Exception as e:
            print(f"  [!] Evaluation failed: {e}")
            agent.train_loss = 999.0
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

    def _compute_loss(self, solution_code: str, problems: List[Dict], use_subset: bool = True) -> float:
        """Error rate on problems"""
        # Use subset during search, full set for final evaluation
        if use_subset:
            problems = self._get_eval_subset(problems)

        total_tests = 0
        failed_tests = 0

        namespace = {}
        try:
            exec(solution_code, namespace)
        except:
            return 1.0

        for problem in problems:
            if problem['id'] not in namespace:
                failed_tests += len(problem['test_cases'])
                total_tests += len(problem['test_cases'])
                continue

            func = namespace[problem['id']]

            for test_case in problem['test_cases']:
                total_tests += 1
                try:
                    inp = test_case['input']
                    if isinstance(inp, list) and len(inp) > 1:
                        result = func(*inp)
                    else:
                        result = func(inp)

                    if result != test_case['expected']:
                        failed_tests += 1
                except Exception as e:
                    failed_tests += 1

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
            print(f"✓ New best! F(λ)={evaluated_agent.free_energy:.4f} "
                  f"[λ={evaluated_agent.lambda_val:.3f}, C_agent={evaluated_agent.agent_complexity:.1f}, "
                  f"L_test={evaluated_agent.test_loss:.4f}]")

        print(f"[{iteration:3d}] λ={lambda_val:.3f} C_agent={evaluated_agent.agent_complexity:5.1f} "
              f"L_train={evaluated_agent.train_loss:.4f} L_test={evaluated_agent.test_loss:.4f} "
              f"F(λ)={evaluated_agent.free_energy:.4f}")

        return {
            'loss': evaluated_agent.free_energy,
            'status': STATUS_OK,
        }

    def run(self):
        """Run TPE-guided self-improvement with λ sweep (like structure-functions paper)"""
        print("="*70)
        print("SELF-MODIFYING AGENT (TPE + Free Energy)")
        print("="*70)
        print(f"Objective: minimize F(λ) = λ·C_agent + L_train")
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

            print(f"Best: {self.best_agent.agent_id} (gen {self.best_agent.generation})")
            print(f"  λ*: {self.best_agent.lambda_val:.4f}")
            print(f"  C_agent: {self.best_agent.agent_complexity:.1f}")
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

    # Default: 6 lambda points × 3 iters each = 18 total evaluations
    # For faster testing, reduce lambda_points (e.g., 4) or iters_per_lambda (e.g., 2)
    #
    # Models to try (requires ollama pull <model>):
    # - gemma3:1b: Fast but very weak (1B params) - struggles with hard problems
    # - gemma3:4b: Good middle ground (4B params) - can solve some problems
    # - llama3 or llama3.1: Better balance (8B params) - recommended
    # - codellama: Code-specialized (7B+ params)
    # - qwen2.5-coder: Best for code (7B+ params)
    agent = SelfImprovingAgent(
        llm_backend="ollama",
        model="gemma3:4b",  # Middle ground - 4x more capable than 1b!
        max_iterations=18,  # Total budget (will be split across lambda values)
        output_dir="./output",
        debug=debug,
        lambda_points=6,  # Number of λ values to sweep [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        iters_per_lambda=3,  # TPE iterations per λ (higher = better but slower)
        eval_subset_size=2,  # Use only 2 test cases per problem during search (faster!)
        stochastic_eval=True,  # True = random subset each iter (prevents overfitting)
    )

    best = agent.run()

    if best and best.test_loss < 0.9:
        print("\nEVOLVED AGENT CODE:")
        print("="*70)
        print(best.agent_code[:500] + "...")


if __name__ == "__main__":
    main()
