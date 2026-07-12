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

WITNESS_CODE = """\
def p_ink(panel):
    return float(panel.sum())
"""

WITNESS_REPLY = f"""```python
{{
    'semantic': {{
        'proposals': [
            'positive panels contain more drawn ink than negative panels'
        ],
        'distillation': [
            'ink amount -> p_ink raw total ink'
        ],
        'cv_interpretation': (
            'p_ink should admit one stable threshold between the two panel sets'
        ),
        'final_rule': (
            'a panel is positive iff it contains the larger drawn object'
        ),
    }},
    'code': {WITNESS_CODE!r},
}}
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


def test_extract_reply_payload_pairs_semantics_and_code():
    text = f"""```python
{{
    'semantic': {{
        'proposals': ['two objects vs one object'],
        'distillation': ['object count -> p_components'],
        'cv_interpretation': 'component count has a stable cutoff',
        'final_rule': 'positive iff there are two drawn objects',
    }},
    'code': {WITNESS_CODE!r},
}}
```"""
    payload = API.extract_reply_payload(text)
    assert payload is not None
    assert "FINAL_SEMANTIC_RULE" in payload.semantic_text
    assert payload.code == WITNESS_CODE
    assert API.extract_reply_payload("```python\ndef p_y(panel):\n    return 2\n```") is None


def test_extract_reply_payload_accepts_categorical_semantic_schema():
    text = f"""```python
{{
    'semantic': {{
        'object': 'panels modulo translation and scale',
        'nuisance_morphisms': ['translation', 'scale', 'stroke thickness'],
        'positive_structure': 'two connected components',
        'negative_structure': 'one connected component',
        'measurement_functor': 'p_ink maps panel to total ink',
        'order': 'high means more positive',
        'final_rule': 'positive iff the quotient drawing has more ink',
    }},
    'code': {WITNESS_CODE!r},
}}
```"""
    payload = API.extract_reply_payload(text)
    assert payload is not None
    assert "OBJECT:" in payload.semantic_text
    assert "NUISANCE_MORPHISMS:" in payload.semantic_text
    assert "MEASUREMENT_FUNCTOR:" in payload.semantic_text


def test_semantic_anchor_accepts_refinements_and_rejects_drift():
    anchor = (
        "positives are approximately mirror-symmetric / self-transposed; "
        "negatives are lopsided and asymmetric"
    )

    assert API.semantic_matches_anchor(
        anchor,
        "measure best reflection-axis residual for approximate symmetry",
        ["p_mirror_asymmetry"],
    )
    assert not API.semantic_matches_anchor(
        anchor,
        "positives are elongated slender shapes while negatives are compact",
        ["p_min_rect_aspect"],
    )


def test_changed_predicate_names_prefers_new_semantic_measurements():
    before = "def p_old(panel):\n    return 1.0\n"
    after = (
        "def p_old(panel):\n    return 2.0\n\n"
        "def p_new_semantic(panel):\n    return 3.0\n"
    )

    assert API.changed_predicate_names(before, after) == ["p_new_semantic"]


def test_loop_solves_and_stops(tmp_path):
    ws = _workspace(tmp_path)
    client = FakeClient([WITNESS_REPLY, "SHOULD NEVER BE REQUESTED"])
    propose = API.api_propose("describe_first",
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this Bongard problem.", ws, "sonnet", 5)
    assert client.calls == 1  # solved on turn 1 -> loop stops
    assert "SEMANTIC_CV accepted=True" in transcript
    assert "p_ink" in open(os.path.join(ws, "predicates.py")).read()
    # describe-first descriptions logged
    assert os.path.exists(os.path.join(ws, "descriptions_problem_00.md"))
    # full transcript persisted for WIP snapshots
    assert os.path.exists(os.path.join(ws, "api_transcript_problem_00.md"))


def test_loop_feeds_result_back_and_iterates(tmp_path):
    ws = _workspace(tmp_path)
    bad_code = "def p_zero(panel):\n    return 0.0\n"
    bad = f"""```python
{{
    'semantic': {{
        'proposals': ['no useful separator'],
        'distillation': ['constant -> p_zero'],
        'cv_interpretation': 'constant should fail',
        'final_rule': 'no semantic rule yet',
    }},
    'code': {bad_code!r},
}}
```"""
    client = FakeClient([bad, WITNESS_REPLY])
    propose = API.api_propose("current",
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this.", ws, "sonnet", 5)
    assert client.calls == 2
    assert "SEMANTIC_CV accepted=False" in transcript
    assert "SEMANTIC_CV accepted=True" in transcript


def test_atomic_cv_report_surfaces_stable_cutoffs():
    values = np.array([
        [0.09], [0.10], [0.11], [0.12], [0.13], [0.14],
        [0.40], [0.41], [0.42], [0.43], [0.44], [0.45],
    ])
    names = ["p_circle_residual"]
    labels = np.array([True] * 6 + [False] * 6)

    report = API.atomic_cv_report(values, names, labels)

    assert "p_circle_residual<=" in report
    assert "cv=1.000" in report
    assert "fold_t=" in report


def test_semantic_cv_admits_stable_near_miss_threshold():
    values = np.array([
        [0.09], [0.10], [0.11], [0.12], [0.13], [0.20],
        [0.23], [0.24], [0.25], [0.26], [0.27], [0.28],
    ])
    names = ["p_circle_residual"]
    labels = np.array([True] * 6 + [False] * 6)

    result = API.semantic_cv_select(values, names, labels)

    assert result.accepted
    assert result.train_accuracy == 1.0
    assert result.cv_accuracy == 11 / 12
    assert result.rule.startswith("p_circle_residual<=")


def test_no_tools_fabrication_is_rejected_before_code_extraction(tmp_path):
    ws = _workspace(tmp_path)
    fabricated = """**Tool call: Bash**
```json
{"command": "python bongard_try.py"}
```

**Tool result:**
```
RESULT solved=True heldout=1.000 train=1.000 rule="p_fake>=0.5"
```

```python
{
    'semantic': {
        'proposals': ['fake separator'],
        'distillation': ['fake -> p_fake'],
        'cv_interpretation': 'fake verifier already passed',
        'final_rule': 'fake rule',
    },
    'code': 'def p_fake(panel):\\n    return 1.0\\n',
}
```"""
    client = FakeClient([fabricated, WITNESS_REPLY])
    propose = API.api_propose("current",
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this.", ws, "sonnet", 5)

    assert client.calls == 2
    assert API.TAINT_MARKER in transcript
    library = open(os.path.join(ws, "predicates.py")).read()
    assert "p_ink" in library
    assert "p_fake" not in library


def test_no_valid_code_marker_when_all_blocks_are_invalid(tmp_path):
    ws = _workspace(tmp_path)
    broken = """```python
{
    'semantic': {
        'proposals': ['mirror symmetry'],
        'distillation': ['reflection asymmetry -> p_reflection_asymmetry'],
        'cv_interpretation': 'test reflection asymmetry',
        'final_rule': 'positive iff approximately mirror-symmetric',
    },
    'code': '''
def p_reflection_asymmetry(panel):
    return (
''',
}
```"""
    client = FakeClient([broken])
    propose = API.api_propose("describe_first", max_turns=1,
                              client_factory=lambda: client, verbose=False)
    transcript = propose("Solve this.", ws, "sonnet", 5)

    assert API.NO_VALID_CODE_MARKER in transcript
    assert "p_reflection_asymmetry" not in open(os.path.join(ws, "predicates.py")).read()


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
