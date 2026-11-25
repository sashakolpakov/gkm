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


def test_game_source_name_requires_a_filename_boundary():
    clean = (
        "cat > probe_ft09.py <<'PY'\n"
        "python probe_ft09.py\n"
        "cp probe_ft09.py candidate_ft09.py\n"
    )
    assert "hidden_source_or_prior_solution" not in A.scan_text(clean)

    for source_reference in (
        "cat ft09.py",
        "python ./ft09.py",
        "Path('/tmp/ft09.py').read_text()",
        "cat environment_files/ft09/version/renamed.py",
    ):
        assert "hidden_source_or_prior_solution" in A.scan_text(
            source_reference
        )


def test_frontier_scaffold_is_audited_before_future_use(tmp_path):
    artifact = tmp_path / "x_legs"
    level = artifact / "wip_context" / "level_01"
    level.mkdir(parents=True)
    scaffold = level / "frontier_scaffold.json"
    scaffold.write_text('{"strategy":"use public observations"}\n')
    report = A.audit(tmp_path)
    assert report["frontier_scaffolds"]["files"] == 1
    assert report["frontier_scaffolds"]["verdict"] == "tainted"
    assert report["frontier_scaffolds"]["hits"][0]["kinds"] == [
        "supervisor_guidance_artifact"
    ]
    assert A.automated_failed(report) is True

    scaffold.write_text('{"strategy":"inspect env._game"}\n')
    report = A.audit(tmp_path)
    assert report["frontier_scaffolds"]["verdict"] == "tainted"
    assert report["frontier_scaffolds"]["hits"][0]["kinds"] == [
        "direct_private_runtime",
        "supervisor_guidance_artifact",
    ]


def test_named_supervisor_handoff_is_forbidden_even_when_content_is_clean(
    tmp_path,
):
    handoff = (
        tmp_path / "x_legs" / "wip_context" / "level_01" /
        "attempt" / "reviewed_handoff.json"
    )
    handoff.parent.mkdir(parents=True)
    handoff.write_text('{"note":"try another public action"}\n')

    report = A.audit(tmp_path)

    assert report["frontier_scaffolds"]["verdict"] == "tainted"
    assert report["frontier_scaffolds"]["hits"][0]["kinds"] == [
        "supervisor_guidance_artifact"
    ]
    assert A.automated_failed(report) is True
