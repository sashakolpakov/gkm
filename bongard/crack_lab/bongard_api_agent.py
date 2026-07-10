"""Lean Messages-API proposer for the Bongard crack (the gkm_api_agent rung).

The next rung DOWN the proposer ladder from the headless Claude Code agent:
no tools, no shell -- just a Messages-API conversation loop. The model sees
the 12 panels as images plus the current shared library, and must reply with
a complete new `predicates.py`; the LOOP (not the model) runs the verifier
and feeds the RESULT line back. Iterate until solved or budget.

Two prompt variants for the stage-1.5 A/B (bongard_crack_plan.md Section 9):

  current        straight to predicates (implicit description)
  describe_first mandatory human-like panel descriptions + a candidate
                 one-sentence rule BEFORE any code (language as an
                 inductive-bias channel); descriptions are logged

The boundary holds: descriptions are hypothesis generation and articulation
only. The verified object remains deterministic p_*(panel) code; no VLM call
ever sits inside a predicate.

Factory `api_propose(variant)` returns a propose_fn compatible with
`bongard_legs.run`, so the whole orchestration (admission, marginal C, WIP,
taint, git checkpoints, infra-failure guardrails) is reused unchanged.
"""
from __future__ import annotations

import base64
import contextlib
import glob
import os
import re
import signal
import sys
import time
from typing import Callable, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_arena as A

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
KEY_FILE = os.path.join(REPO_ROOT, "ANTHROPIC_API_KEY.env.local")

MODEL_MAP = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}

DESCRIBE_FIRST_INSTRUCTION = """\
BEFORE writing any code, look at the twelve panels and write, in plain
language: (1) a one-line human description of EACH panel (pos_0..5,
neg_0..5), and (2) a candidate ONE-SENTENCE rule that separates the sides,
as a human would state it. Begin that section with `DESCRIPTIONS:`. Only
then implement the property your sentence names as predicate code."""

REPLY_FORMAT = """\
Reply with the COMPLETE new content of `predicates.py` in a single
```python fenced block (module-level pure functions `p_<name>(panel) ->
float | bool` over a 128x128 uint8 array, ink=1; numpy/math/scipy only;
deterministic; no file or network access). Keep and reuse existing library
predicates unless they are broken -- reused predicates are free, new code
is charged. Use ASCII only, including comments and string literals. I will
run the verifier and report its RESULT line back."""


def load_api_key(path: str = KEY_FILE) -> str:
    text = open(path).read().strip()
    return text.split("=", 1)[1].strip() if "=" in text.split("\n")[0] and text.startswith("ANTHROPIC") else text


def extract_code(reply: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    return blocks[-1] if blocks else None


def _panel_blocks(pdir: str) -> List[dict]:
    blocks: List[dict] = []
    for path in sorted(glob.glob(os.path.join(pdir, "*.png"))):
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        blocks.append({"type": "text", "text": os.path.basename(path) + ":"})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": data}})
    return blocks


def _verify_ws(ws: str) -> A.VerifyResult:
    oid = open(os.path.join(ws, "current_problem.txt")).read().strip()
    pdir = os.path.join(ws, oid)
    pos = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "pos_*.npy")))]
    neg = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "neg_*.npy")))]
    problem = A.Problem("current", "?", "?", pos, neg)
    try:
        preds = A.load_predicates(os.path.join(ws, "predicates.py"))
    except Exception as exc:
        return A.VerifyResult(False, 0.0, 0.0, f"LOAD_ERROR:{exc}", 0.0, 0, 36)
    return A.verify(preds, problem)


class APICallTimeout(TimeoutError):
    """Raised when one model call exceeds the harness hard wall."""


@contextlib.contextmanager
def _hard_timeout(seconds: float):
    def _raise_timeout(signum, frame):
        raise APICallTimeout(f"API call timed out after {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def api_propose(variant: str = "current", max_turns: int = 8,
                max_tokens: int = 8000, per_call_timeout: float = 90.0,
                client_factory: Callable = None,
                verbose: bool = True) -> Callable[[str, str, str, int], Optional[str]]:
    """Build a propose_fn for bongard_legs.run.

    `client_factory` is injectable for offline tests; default builds a real
    anthropic.Anthropic client with the repo-local key."""
    assert variant in ("current", "describe_first")

    def propose(task: str, ws: str, model: str, minutes: int) -> Optional[str]:
        if client_factory is not None:
            client = client_factory()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=load_api_key())
        model_id = MODEL_MAP.get(model, model)
        oid = open(os.path.join(ws, "current_problem.txt")).read().strip()
        lib_path = os.path.join(ws, "predicates.py")
        library = open(lib_path).read() if os.path.exists(lib_path) else ""

        intro = [{"type": "text", "text": task}]
        intro += _panel_blocks(os.path.join(ws, oid))
        parts = ["Current shared library `predicates.py`:\n```python\n"
                 + library + "\n```"]
        if variant == "describe_first":
            parts.append(DESCRIBE_FIRST_INSTRUCTION)
        parts.append(REPLY_FORMAT)
        intro.append({"type": "text", "text": "\n\n".join(parts)})

        messages = [{"role": "user", "content": intro}]
        transcript: List[str] = []
        deadline = time.time() + minutes * 60
        for turn in range(max_turns):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                timeout = max(1.0, min(per_call_timeout, remaining))
                with _hard_timeout(timeout + 5.0):
                    reply = client.messages.create(
                        model=model_id, max_tokens=max_tokens,
                        messages=messages, timeout=timeout)
                text = "".join(b.text for b in reply.content
                               if getattr(b, "type", "") == "text")
            except Exception as exc:
                transcript.append(f"API failure: {exc}")
                break
            transcript.append(text)
            if variant == "describe_first" and "DESCRIPTIONS:" in text:
                with open(os.path.join(ws, f"descriptions_{oid}.md"), "a") as f:
                    f.write(text.split("```")[0] + "\n---\n")
            code = extract_code(text)
            if code is None:
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content":
                              "No ```python block found. " + REPLY_FORMAT}]
                continue
            with open(lib_path, "w") as f:
                f.write(code)
            result = _verify_ws(ws)
            transcript.append(result.result_line())
            if verbose:
                print(f"  api-turn {turn + 1}: {result.result_line()}")
            if result.solved:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content":
                          result.result_line() + "\nNot solved yet -- "
                          "improve the predicates and reply with the full "
                          "new predicates.py again."}]
        full = "\n\n".join(transcript)
        with open(os.path.join(ws, f"api_transcript_{oid}.md"), "a") as f:
            f.write(full + "\n\n=====\n\n")
        return full

    return propose
