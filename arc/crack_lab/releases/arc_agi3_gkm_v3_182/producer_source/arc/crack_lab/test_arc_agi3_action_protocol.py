"""Differential contract test for the public ARC-AGI-3 action grammar."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import audit_action_protocol as Audit
import arc_agi3_arena_rpc as Rpc
import arc_agi3_contiguous_supervisor as Supervisor
import arc_agi3_proposer_worker as Proposer
import arc_agi3_release_gate as Release


def _source_function(
    filename: str,
    function_name: str,
    *,
    globals_: dict[str, object] | None = None,
) -> Callable:
    """Compile one exact function from a sealed source without its imports."""

    source = Path(__file__).with_name(filename)
    tree = ast.parse(source.read_bytes(), filename=str(source))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(matches) == 1
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {} if globals_ is None else dict(globals_)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[function_name]


ACQUISITION_VALIDATOR = _source_function(
    "gkm_legs.py",
    "_valid_replay_action",
    globals_={"A": SimpleNamespace(FRAME_SIDE=64)},
)
SCORECARD_DECODER = _source_function(
    "replay_scorecard.py",
    "decode_action",
)


def _accepts_decode(action: object) -> bool:
    try:
        SCORECARD_DECODER(action)
    except ValueError:
        return False
    return True


def _accepts_rpc(action: object) -> bool:
    try:
        Rpc._normalize_action(action)
    except Rpc.ArenaRpcContractError:
        return False
    return True


VALIDATORS: tuple[tuple[str, Callable[[object], bool]], ...] = (
    ("acquisition", ACQUISITION_VALIDATOR),
    ("scorecard", _accepts_decode),
    ("supervisor", Supervisor._valid_replay_action),
    ("proposer", Proposer._valid_action),
    ("release", Release._valid_action),
    ("rpc", _accepts_rpc),
    ("audit", lambda action: Audit.action_error(action) is None),
)

VALID_JSON_TOKENS = (
    1,
    2,
    3,
    4,
    5,
    7,
    [6, 0, 0],
    [6, 63, 63],
    [6, 17, 42],
)

INVALID_JSON_TOKENS = (
    0,
    6,
    8,
    True,
    1.0,
    "1",
    None,
    [],
    [6],
    [6, 1],
    [6, 1, 2, 3],
    [5, 1, 2],
    [6, True, 2],
    [6, 1.0, 2],
    [6, -1, 2],
    [6, 64, 2],
    [6, 2, -1],
    [6, 2, 64],
)


def test_all_layers_accept_exactly_the_same_public_json_action_tokens():
    for expected, tokens in ((True, VALID_JSON_TOKENS), (False, INVALID_JSON_TOKENS)):
        for token in tokens:
            outcomes = {
                name: accepts(token)
                for name, accepts in VALIDATORS
            }
            assert set(outcomes.values()) == {expected}, (
                f"action grammar divergence for {token!r}: {outcomes}"
            )


def test_action6_is_capability_id_but_never_a_scalar_action_token():
    assert Rpc._normalize_actions([1, 6, 7]) == [1, 6, 7]
    for name, accepts in VALIDATORS:
        assert not accepts(6), f"{name} accepted bare ACTION6"
