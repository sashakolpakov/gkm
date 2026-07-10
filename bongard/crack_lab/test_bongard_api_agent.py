"""Offline tests for the Messages-API proposer loop (fake client, no
network). The witness predicate lives only in tests."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_api_agent as API
import bongard_arena as B
from test_bongard_arena import two_vs_one_problem

WITNESS_REPLY = """DESCRIPTIONS:
pos panels each show two separate figures; neg panels show one.
Candidate rule: a panel is positive iff it contains two drawn objects.

```python
def p_ink(panel):
    return float(panel.sum())
```"""


class FakeClient:
    """Replays canned replies; records how many calls were made."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        text = self.replies.pop(0)
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])


def _workspace(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    problem = two_vs_one_problem()
    B.write_panels(ws, problem, "problem_00")
    with open(os.path.join(ws, "current_problem.txt"), "w") as f:
        f.write("problem_00")
    with open(os.path.join(ws, "predicates.py"), "w") as f:
        f.write("# empty library\n")
    return ws


def test_extract_code_takes_last_fence():
    text = "```python\nx=1\n```\nprose\n```python\ny=2\n```"
    assert API.extract_code(text) == "y=2\n"
    assert API.extract_code("no code here") is None


def test_loop_solves_and_stops(tmp_path):
    ws = _workspace(tmp_path)
    client = FakeClient([WITNESS_REPLY, "SHOULD NEVER BE REQUESTED"])
    propose = API.api_propose("describe_first",
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this Bongard problem.", ws, "sonnet", 5)
    assert client.calls == 1  # solved on turn 1 -> loop stops
    assert "solved=True" in transcript
    assert "p_ink" in open(os.path.join(ws, "predicates.py")).read()
    # describe-first descriptions logged
    assert os.path.exists(os.path.join(ws, "descriptions_problem_00.md"))
    # full transcript persisted for WIP snapshots
    assert os.path.exists(os.path.join(ws, "api_transcript_problem_00.md"))


def test_loop_feeds_result_back_and_iterates(tmp_path):
    ws = _workspace(tmp_path)
    bad = "```python\ndef p_zero(panel):\n    return 0.0\n```"
    client = FakeClient([bad, WITNESS_REPLY])
    propose = API.api_propose("current",
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this.", ws, "sonnet", 5)
    assert client.calls == 2
    assert "solved=False" in transcript and "solved=True" in transcript


def test_api_failure_surfaces_for_infra_markers(tmp_path):
    ws = _workspace(tmp_path)

    class Exploding:
        def __init__(self):
            self.messages = types.SimpleNamespace(create=self._boom)

        def _boom(self, **kwargs):
            raise RuntimeError("rate limit exceeded, retry later")

    propose = API.api_propose("current",
                              client_factory=Exploding, verbose=False)
    transcript = propose("Solve this.", ws, "sonnet", 5)
    assert "rate limit" in transcript.lower()  # bongard_legs markers catch it
