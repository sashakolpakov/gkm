from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard import cli
from bongard.action_program_audit import (
    ActionProgramAuditError,
    ActionProgramValidationError,
    PRIVILEGE_CLASS,
    audit_action_program_metadata,
)
from bongard.artifacts import canonical_json
from bongard.corpus import FAMILIES, ShapeBongardCorpus
from bongard.release import load_official_release


REPOSITORY = Path(__file__).resolve().parents[2]
DATA = REPOSITORY / "bongard" / "data"
FULL_RELEASE = (
    REPOSITORY
    / "downloads"
    / "ShapeBongard_V2_full"
    / "ShapeBongard_V2"
)


def _panel_program(action: object = "line_normal_1.000-0.500") -> list[object]:
    return [[action]]


def _task_program(action: object = "line_normal_1.000-0.500") -> list[object]:
    return [
        [_panel_program(action) for _ in range(7)],
        [_panel_program(action) for _ in range(7)],
    ]


def _small_corpus(tmp_path: Path) -> tuple[ShapeBongardCorpus, dict[str, str]]:
    root = tmp_path / "ShapeBongard_V2"
    task_ids: dict[str, str] = {}
    for family in FAMILIES:
        task_id = f"{family}_audit_0000"
        task_ids[family] = task_id
        for label in ("1", "0"):
            directory = root / family / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(7):
                (directory / f"{index}.png").write_bytes(b"inventory-only")
        # Deliberately reproduce upstream's non-canonical json.dump spacing.
        (root / family / f"{family}_action_programs.json").write_text(
            json.dumps({task_id: _task_program()}), encoding="utf-8"
        )
    return ShapeBongardCorpus.from_root(root), task_ids


def _program_path(corpus: ShapeBongardCorpus, family: str) -> Path:
    return corpus.root / family / f"{family}_action_programs.json"


def _rewrite_family(
    corpus: ShapeBongardCorpus, family: str, value: object, *, canonical: bool = False
) -> None:
    separators = (",", ":") if canonical else None
    _program_path(corpus, family).write_text(
        json.dumps(value, sort_keys=canonical, separators=separators),
        encoding="utf-8",
    )


def _checked_report() -> dict[str, object]:
    payload = (DATA / "shape_bongard_v2_action_program_audit_v1.json").read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert payload == canonical_json(value) + b"\n"
    return value


def test_read_only_aggregate_audit_separates_raw_and_canonical_identity(
    tmp_path: Path,
) -> None:
    corpus, _ = _small_corpus(tmp_path)
    before = {
        family: hashlib.sha256(_program_path(corpus, family).read_bytes()).hexdigest()
        for family in FAMILIES
    }

    report = audit_action_program_metadata(corpus)

    assert report.anomaly_count == 0
    assert (report.task_count, report.side_count, report.panel_count) == (3, 6, 42)
    assert (report.shape_program_count, report.action_count) == (42, 42)
    assert report.unique_action_count == 1
    assert all(family.task_keys_exact for family in report.families)
    assert all(family.structure_valid for family in report.families)
    assert all(family.canonical_bytes_equal is False for family in report.families)
    assert report.to_dict()["usage_policy"] == {
        "classification": PRIVILEGE_CLASS,
        "allowed_uses": [
            "post_hoc_release_diagnostics",
            "explicitly_labelled_oracle_upper_bound",
        ],
        "forbidden_inputs": [
            "proposer",
            "support_model",
            "query_model",
            "predicate_synthesis",
            "prediction",
        ],
        "must_never_enter_proposer_or_query_inputs": True,
        "audit_output": "aggregate_counts_and_content_addresses_only",
        "reason": (
            "render recipes expose latent geometric construction information "
            "that is not present in the benchmark PNG interface"
        ),
    }
    assert "line_normal_1.000-0.500" not in json.dumps(report.to_dict())
    assert report.digest == "sha256:" + hashlib.sha256(
        canonical_json(report.content_dict())
    ).hexdigest()
    assert before == {
        family: hashlib.sha256(_program_path(corpus, family).read_bytes()).hexdigest()
        for family in FAMILIES
    }


def test_metadata_keys_must_equal_family_inventory_exactly(tmp_path: Path) -> None:
    corpus, task_ids = _small_corpus(tmp_path)
    _rewrite_family(corpus, "ff", {"ff_uninventoried_9999": _task_program()})

    diagnostic = audit_action_program_metadata(corpus, require_valid=False)
    ff = next(family for family in diagnostic.families if family.family == "ff")
    assert ff.inventory_task_ids_sha256 != ff.metadata_task_ids_sha256
    assert ff.task_keys_exact is False
    assert diagnostic.anomaly_count == 2
    assert {item.code for item in diagnostic.anomalies} == {
        "missing_task_key",
        "extra_task_key",
    }
    assert task_ids["ff"] in diagnostic.anomalies[0].detail

    with pytest.raises(ActionProgramValidationError) as caught:
        audit_action_program_metadata(corpus)
    assert caught.value.report.to_dict() == diagnostic.to_dict()


@pytest.mark.parametrize(
    ("damage", "expected_code"),
    [
        ("side_count", "side_count"),
        ("panel_count", "panel_count"),
        ("shape_count", "shape_program_count"),
        ("action_count", "action_count"),
        ("action_grammar", "action_grammar"),
        ("action_type", "action_type"),
    ],
)
def test_recursive_program_and_action_invariants_fail_closed(
    tmp_path: Path, damage: str, expected_code: str
) -> None:
    corpus, task_ids = _small_corpus(tmp_path)
    program = _task_program()
    if damage == "side_count":
        program.pop()
    elif damage == "panel_count":
        program[0].pop()  # type: ignore[union-attr]
    elif damage == "shape_count":
        program[0][0].extend(  # type: ignore[index,union-attr]
            [["line_normal_1.000-0.500"], ["line_normal_1.000-0.500"]]
        )
    elif damage == "action_count":
        program[0][0][0].extend(  # type: ignore[index,union-attr]
            ["line_normal_1.000-0.500"] * 9
        )
    elif damage == "action_grammar":
        program[0][0][0][0] = "line_normal_1.001-0.500"  # type: ignore[index]
    else:
        program[0][0][0][0] = 7  # type: ignore[index]
    _rewrite_family(corpus, "ff", {task_ids["ff"]: program})

    with pytest.raises(ActionProgramValidationError) as caught:
        audit_action_program_metadata(corpus)
    assert expected_code in {item.code for item in caught.value.report.anomalies}


def test_duplicate_keys_depth_byte_bound_and_symlink_are_rejected(
    tmp_path: Path,
) -> None:
    corpus, task_ids = _small_corpus(tmp_path)
    encoded = json.dumps(_task_program(), separators=(",", ":"))
    _program_path(corpus, "ff").write_text(
        "{" + json.dumps(task_ids["ff"]) + ":" + encoded + ","
        + json.dumps(task_ids["ff"]) + ":" + encoded + "}",
        encoding="utf-8",
    )
    with pytest.raises(ActionProgramAuditError, match="duplicate JSON object key"):
        audit_action_program_metadata(corpus)

    _rewrite_family(corpus, "ff", {task_ids["ff"]: _task_program()})
    with pytest.raises(ActionProgramAuditError, match="byte safety limit"):
        audit_action_program_metadata(corpus, max_file_bytes=1)

    source = _program_path(corpus, "ff")
    target = tmp_path / "outside.json"
    target.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ActionProgramAuditError, match="symlink"):
        audit_action_program_metadata(corpus)


def test_checked_complete_release_report_has_exact_identity_and_counts() -> None:
    report = _checked_report()
    assert report["schema"] == "gkm.shape-bongard-action-program-audit.v1"
    assert report["release_descriptor_digest"] == (
        "sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b"
    )
    assert report["inventory_task_ids_sha256"] == (
        "sha256:4503ae6b40dc7b34520eb5b8a4cca6ff8153635df0f42db5f6715cc349602dd0"
    )
    families = report["families"]
    assert isinstance(families, dict)
    assert {
        family: (
            values["inventory_task_count"],
            values["size_bytes"],
            values["raw_sha256"],
            values["canonical_json_sha256"],
        )
        for family, values in families.items()
    } == {
        "ff": (
            3_600,
            10_708_250,
            "sha256:eedaf901dd698877afe8041635a1553e78089cc615647192010d51cfc4391269",
            "sha256:7883da060da366815e3b32ab4a85e0ee9c706062ec230226dd18497b77767245",
        ),
        "bd": (
            4_000,
            16_415_082,
            "sha256:768b060cc84636036bb85622cb574b52d00476e7fc08afc596ec38edb3cd4b2b",
            "sha256:ebdcfe829f6baa360f97ff5e1b9c9085ceaf2d279db85de54be9227fa87bc863",
        ),
        "hd": (
            4_400,
            10_311_475,
            "sha256:190f3f850d98fa9df0f85cbbafa05fbbaf6d8845586c186ce062af8812ba7e7c",
            "sha256:3dd6b6ecad65635aac3a5fa17ef6bc2074491207ff9d83a5447b4e4c76bb45f3",
        ),
    }
    assert all(values["task_keys_exact"] for values in families.values())
    assert all(values["structure_valid"] for values in families.values())
    assert all(values["canonical_bytes_equal"] is False for values in families.values())
    assert report["totals"] == {
        "tasks": 12_000,
        "sides": 24_000,
        "panels": 168_000,
        "shape_programs": 240_422,
        "actions": 1_270_887,
        "unique_actions": 3_101,
        "raw_bytes": 37_434_807,
        "canonical_bytes": 36_151_923,
        "action_kind_counts": {"arc": 300_247, "line": 970_640},
        "stroke_style_counts": {
            "circle": 152_009,
            "normal": 678_382,
            "square": 143_851,
            "triangle": 144_071,
            "zigzag": 152_574,
        },
    }
    assert report["anomaly_count"] == 0
    assert report["anomalies"] == []
    assert report["digest"] == (
        "sha256:6c51b6218a86ca5dae4b34ca2b829b805d72f1f79ddb9fc32d212356f402667c"
    )
    assert report["usage_policy"][
        "must_never_enter_proposer_or_query_inputs"
    ] is True


@pytest.mark.skipif(
    not (FULL_RELEASE / "ff" / "ff_action_programs.json").is_file(),
    reason="complete local ShapeBongard release is unavailable",
)
def test_local_complete_release_reproduces_checked_report_exactly() -> None:
    corpus = ShapeBongardCorpus.from_root(FULL_RELEASE)
    observed = audit_action_program_metadata(
        corpus,
        official_release=load_official_release(),
    )
    assert observed.to_dict() == _checked_report()


def _install_cli_corpus(
    monkeypatch: pytest.MonkeyPatch,
    corpus: ShapeBongardCorpus,
    discoveries: list[tuple[object, object, object]],
) -> None:
    class DiscoveryOnlyCorpus:
        @staticmethod
        def discover(path: object, *, split_file: object, require_complete: object):
            discoveries.append((path, split_file, require_complete))
            return corpus

    monkeypatch.setattr(cli, "ShapeBongardCorpus", DiscoveryOnlyCorpus)
    # The production command always loads the checked official descriptor.
    # This small-corpus fixture substitutes None so the real aggregate auditor
    # can be exercised without pretending three fixture tasks are official.
    monkeypatch.setattr(cli, "load_official_release", lambda path: None)


def _cli_args(
    corpus: ShapeBongardCorpus,
    *,
    expected_report_digest: str | None = None,
    expected_report: Path | None = None,
    out: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        corpus=str(corpus.root),
        split_file=None,
        ff_action_programs=str(_program_path(corpus, "ff")),
        bd_action_programs=str(_program_path(corpus, "bd")),
        hd_action_programs=str(_program_path(corpus, "hd")),
        release_descriptor="checked-release.json",
        expected_report_digest=expected_report_digest,
        expected_report=str(expected_report) if expected_report is not None else None,
        out=str(out) if out is not None else None,
    )


def test_cli_action_program_audit_requires_explicit_sources_and_has_no_run_inputs(
    tmp_path: Path,
) -> None:
    corpus, _ = _small_corpus(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "audit-action-programs",
            "--corpus",
            str(corpus.root),
            "--ff-action-programs",
            str(_program_path(corpus, "ff")),
            "--bd-action-programs",
            str(_program_path(corpus, "bd")),
            "--hd-action-programs",
            str(_program_path(corpus, "hd")),
        ]
    )
    assert args.handler is cli._audit_action_programs
    assert not hasattr(args, "task_id")
    assert not hasattr(args, "model")
    assert not hasattr(args, "seed")
    assert not hasattr(args, "observer_minutes")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "audit-action-programs",
                "--corpus",
                str(corpus.root),
                "--ff-action-programs",
                str(_program_path(corpus, "ff")),
                "--bd-action-programs",
                str(_program_path(corpus, "bd")),
            ]
        )


def test_cli_regenerates_verifies_and_writes_exact_canonical_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, _ = _small_corpus(tmp_path)
    expected_report = audit_action_program_metadata(corpus)
    expected_bytes = canonical_json(expected_report.to_dict()) + b"\n"
    expected_path = tmp_path / "expected.json"
    expected_path.write_bytes(expected_bytes)
    output_path = tmp_path / "regenerated.json"
    discoveries: list[tuple[object, object, object]] = []
    _install_cli_corpus(monkeypatch, corpus, discoveries)

    # These execution-path functions must remain unreachable from the isolated
    # post-hoc metadata command.
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("benchmark/proposer path was entered")

    monkeypatch.setattr(cli, "prepare_episode", forbidden)
    monkeypatch.setattr(cli, "run_episode", forbidden)
    monkeypatch.setattr(cli, "HeadlessCodexEpisode", forbidden)

    args = _cli_args(
        corpus,
        expected_report_digest=expected_report.digest,
        expected_report=expected_path,
        out=output_path,
    )
    source_hashes = {
        family: hashlib.sha256(_program_path(corpus, family).read_bytes()).hexdigest()
        for family in FAMILIES
    }
    assert cli._audit_action_programs(args) == 0
    assert capsys.readouterr().out.encode("utf-8") == expected_bytes
    assert output_path.read_bytes() == expected_bytes
    assert discoveries == [(str(corpus.root), None, True)]
    assert source_hashes == {
        family: hashlib.sha256(_program_path(corpus, family).read_bytes()).hexdigest()
        for family in FAMILIES
    }


def test_cli_action_program_audit_rejects_unbound_source_digest_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _small_corpus(tmp_path)
    discoveries: list[tuple[object, object, object]] = []
    _install_cli_corpus(monkeypatch, corpus, discoveries)

    malformed = _cli_args(corpus, expected_report_digest="not-a-digest")
    with pytest.raises(cli.CliError, match="expected-report-digest"):
        cli._audit_action_programs(malformed)
    assert discoveries == []

    swapped = _cli_args(corpus)
    swapped.ff_action_programs = swapped.bd_action_programs
    with pytest.raises(cli.CliError, match="explicit ff action-program source"):
        cli._audit_action_programs(swapped)

    wrong_digest = _cli_args(
        corpus,
        expected_report_digest="sha256:" + "0" * 64,
        out=tmp_path / "must-not-exist.json",
    )
    with pytest.raises(cli.CliError, match="report digest is"):
        cli._audit_action_programs(wrong_digest)
    assert not Path(wrong_digest.out).exists()

    wrong_expected = tmp_path / "wrong-expected.json"
    wrong_expected.write_bytes(b"{}\n")
    output = tmp_path / "also-must-not-exist.json"
    mismatch = _cli_args(corpus, expected_report=wrong_expected, out=output)
    with pytest.raises(cli.CliError, match="differs from --expected-report"):
        cli._audit_action_programs(mismatch)
    assert not output.exists()
