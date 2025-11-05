"""
Fixed scaffold for the SolvingAgent.

This file contains the infrastructure that should NEVER be modified by the meta-LLM.
Only the build_prompt() method in the modifiable agent gets evolved.
"""

AGENT_SCAFFOLD = '''
import json

class SolvingAgent:
    """Agent that uses an LLM to solve coding problems"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def solve_problem(self, problem: dict) -> str:
        """Generate solution code for a problem"""
        # Build prompt using the modifiable strategy
        prompt = self.build_prompt(problem)

        # Call LLM (FIXED: only method available)
        response = self.llm.generate(prompt, max_tokens=500)

        # Extract function (FIXED: parsing logic)
        return self._extract_function(response)

    def _extract_function(self, response: str) -> str:
        """Extract function code from LLM response (FIXED)"""
        # Try delimiters first
        if "[FUNCTION_START]" in response and "[FUNCTION_END]" in response:
            start = response.find("[FUNCTION_START]") + len("[FUNCTION_START]")
            end = response.find("[FUNCTION_END]")
            code = response[start:end].strip()
            if code:
                return code

        # Fallback: markdown code block
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                if "def " in part:
                    code = part
                    if code.startswith("python"):
                        code = code[6:].strip()
                    return code.strip()

        # Last resort: extract first def statement
        if "def " in response:
            lines = response.split('\\\\n')
            code_lines = []
            in_function = False
            indent_level = 0

            for line in lines:
                if 'def ' in line and '(' in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif in_function:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() == '' or current_indent > indent_level:
                        code_lines.append(line)
                    else:
                        break

            if code_lines:
                return '\\\\n'.join(code_lines)

        return "# Failed to extract function"

    def solve_problems(self, problems: list) -> str:
        """Solve all problems in a single batched LLM call"""
        # Build one big batch prompt with all problems
        batch_prompt = "Solve these Python programming problems. Return each solution with the exact function name specified.\\\\n\\\\n"

        for i, problem in enumerate(problems, 1):
            batch_prompt += f"--- PROBLEM {i} ---\\\\n"
            batch_prompt += f"Function name: {problem['id']}\\\\n"
            batch_prompt += f"Description: {problem['description']}\\\\n"
            batch_prompt += f"Signature: {problem['signature']}\\\\n\\\\n"

        batch_prompt += "\\\\nReturn ALL functions with format [FUNCTION_START]\\\\ndef function_name(...): ...\\\\n[FUNCTION_END]"

        response = self.llm.generate(batch_prompt, max_tokens=4000)

        # Fix literal backslash-n in response (LLM returns \\\\n as text, convert to actual newlines)
        response = response.replace('\\\\\\\\n', '\\\\n')

        # Extract all functions from the batch response
        solutions = []
        remaining = response
        for problem in problems:
            # Try to find this problem's solution
            solution = self._extract_function(remaining)
            solutions.append(solution)
            # Remove extracted part to find next function
            if "[FUNCTION_START]" in remaining and "[FUNCTION_END]" in remaining:
                end_idx = remaining.find("[FUNCTION_END]") + len("[FUNCTION_END]")
                remaining = remaining[end_idx:]

        return "\\\\n\\\\n".join(solutions)

    # ========== MODIFIABLE PART BELOW ==========
    # Only the build_prompt() method should be modified by meta-LLM:
'''
