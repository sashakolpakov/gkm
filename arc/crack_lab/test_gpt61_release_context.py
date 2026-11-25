from pathlib import Path
import hashlib
import arc_agi3_release_gate as gate


def test_gpt61_integrity_dependency_is_bound_and_changes_are_detected(tmp_path):
    controls = gate._default_control_files()
    key = "arc/crack_lab/proposer_integrity.py"
    assert key in controls
    for name in ["codex_campaign_policy.py", "codex_campaign_runner.py"]:
        assert "arc/crack_lab/" + name in controls
    original = gate._control_contract_snapshot(controls)
    changed = tmp_path / "proposer_integrity.py"
    changed.write_bytes(controls[key].read_bytes() + b"\n# changed verification fixture\n")
    other = gate._control_contract_snapshot({**controls, key: changed})
    assert other["sha256"] != original["sha256"]
    assert other["files_sha256"][key] == hashlib.sha256(changed.read_bytes()).hexdigest()
    assert set(other["files_sha256"]) == set(original["files_sha256"])
