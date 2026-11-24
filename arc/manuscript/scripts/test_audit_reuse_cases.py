from audit_reuse_cases import inspect_call_paths


def test_wrapper_reaches_unchanged_leg_indirectly():
    previous = {"legs.py": b"def walk(env):\n    env.step(1)\n", "players.py": b"from legs import *\n"}
    current = {**previous, "legs.py": previous["legs.py"] + b"def route(env):\n    walk(env)\n",
               "players.py": b"from legs import *\ndef play_level_2(env):\n    route(env)\n"}
    result = inspect_call_paths(previous, current, 2)
    assert result["unchanged_leg_paths"] == [["players.py:play_level_2", "legs.py:route", "legs.py:walk"]]


def test_effective_player_definition_replaces_earlier_stub():
    previous = {"legs.py": b"def walk(env):\n    env.step(1)\n", "players.py": b"from legs import *\n"}
    current = {**previous, "players.py": b"from legs import *\ndef play_level_2(env):\n    walk(env)\ndef play_level_2(env):\n    for action in [1, 2]:\n        env.step(action)\n"}
    result = inspect_call_paths(previous, current, 2)
    assert result["unchanged_leg_paths"] == []
    assert result["literal_action_loop_only"]
    assert result["literal_action_count"] == 2


def test_changed_helper_is_not_an_unchanged_target():
    previous = {"legs.py": b"def walk(env):\n    env.step(1)\n", "players.py": b"from legs import *\n"}
    current = {"legs.py": b"def walk(env):\n    env.step(2)\n", "players.py": b"from legs import *\ndef play_level_2(env):\n    walk(env)\n"}
    assert inspect_call_paths(previous, current, 2)["unchanged_leg_paths"] == []


def test_uncalled_nested_body_is_not_a_call_path():
    previous = {"legs.py": b"def walk(env):\n    env.step(1)\n", "players.py": b"from legs import *\n"}
    current = {**previous, "players.py": b"from legs import *\ndef play_level_2(env):\n    def unused():\n        walk(env)\n    env.step(2)\n"}
    assert inspect_call_paths(previous, current, 2)["unchanged_leg_paths"] == []


def test_parameter_shadow_does_not_resolve_to_a_retained_leg():
    previous = {"legs.py": b"def walk(env):\n    env.step(1)\n", "players.py": b"from legs import *\n"}
    current = {**previous, "players.py": b"from legs import *\ndef play_level_2(env, walk):\n    walk(env)\n"}
    assert inspect_call_paths(previous, current, 2)["unchanged_leg_paths"] == []
