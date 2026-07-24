import json

import audit_submission_taint as A


def _event(item):
    return json.dumps({"type": "item.completed", "item": item})


def test_codex_traceback_output_does_not_become_private_runtime_taint(tmp_path):
    path = tmp_path / "proposer_last.log"
    path.write_text(
        _event({
            "type": "command_execution",
            "command": "python probe_public_clone.py",
            "aggregated_output": (
                "Traceback: Arena.clone -> self._game = "
                "copy.deepcopy(_clone._game)"
            ),
        }) + "\n"
    )
    assert A.scan_file(path) == []


def test_agent_authored_private_runtime_command_remains_taint(tmp_path):
    path = tmp_path / "turn.jsonl"
    path.write_text(
        _event({
            "type": "command_execution",
            "command": "python -c 'print(env._game)'",
            "aggregated_output": "",
        }) + "\n"
    )
    assert "direct_private_runtime" in A.scan_file(path)


def test_agent_authored_web_search_item_remains_taint(tmp_path):
    path = tmp_path / "proposer_last.log"
    path.write_text(
        _event({"type": "web_search", "query": "ARC game solution"}) + "\n"
    )
    assert "external_web_or_network" in A.scan_file(path)


def test_frontier_scaffold_is_audited_before_future_use(tmp_path):
    artifact = tmp_path / "x_legs"
    level = artifact / "wip_context" / "level_01"
    level.mkdir(parents=True)
    scaffold = level / "frontier_scaffold.json"
    scaffold.write_text('{"strategy":"use public observations"}\n')
    report = A.audit(tmp_path)
    assert report["frontier_scaffolds"] == {
        "files": 1, "hits": [], "verdict": "clean",
    }

    scaffold.write_text('{"strategy":"inspect env._game"}\n')
    report = A.audit(tmp_path)
    assert report["frontier_scaffolds"]["verdict"] == "tainted"
    assert report["frontier_scaffolds"]["hits"][0]["kinds"] == [
        "direct_private_runtime"
    ]
