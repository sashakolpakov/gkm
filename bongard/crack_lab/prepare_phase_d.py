"""Prepare a Phase D corpus and arm table without constructing a proposer.

Example::

    PYTHONHASHSEED=0 python bongard/crack_lab/prepare_phase_d.py \
      --source both --limit-per-source 25 --scales 1,5,25 \
      --out-dir bongard/crack_lab/phase_d_runs/preregistered

The command requires a hash seed fixed before interpreter startup, samples
once at the declared maximum, writes the canonical base
manifest, freezes every shuffled-side replicate, and writes a track-separated
preregistration.  It performs no model/API work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase_d_protocol as protocol
import collect_phase_d as artifact_io
import semantic_artifacts
import semantic_replay


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


def _require_fixed_python_hash_seed() -> int:
    raw = os.environ.get("PYTHONHASHSEED", "")
    if not raw.isdigit() or int(raw) > 4_294_967_295:
        raise SystemExit(
            "Phase D preparation requires a fixed PYTHONHASHSEED "
            "(use PYTHONHASHSEED=0)")
    return int(raw)


def prepare(args: argparse.Namespace) -> dict:
    # Fail before sampling, directory creation, or any write.  A random hash
    # secret would make the preregistered hash probes unverifiable in the next
    # runner/collector process.
    _require_fixed_python_hash_seed()
    out_dir = os.path.abspath(args.out_dir)
    root = os.path.realpath(str(semantic_replay.BONGARD_ROOT))
    if os.path.commonpath((root, os.path.realpath(out_dir))) != root:
        raise SystemExit("--out-dir must stay inside the bongard working tree")
    problems = protocol.sample_corpus(
        args.dataset_dir,
        limit_per_source=args.limit_per_source,
        seed=args.seed,
        source=args.source,
    )
    corpus_manifest = protocol.build_corpus_manifest(
        problems,
        source=args.source,
        seed=args.seed,
        limit_per_source=args.limit_per_source,
        dataset_revision=protocol.dataset_revision(args.dataset_dir),
        dataset_inputs_digest=protocol.dataset_content_digest(args.dataset_dir),
    )
    corpus_bundle = protocol.build_corpus_bundle(problems, corpus_manifest)
    controls = [
        protocol.build_shuffled_sides_control(
            problems,
            corpus_manifest,
            seed=args.shuffled_seed,
            replicate=replicate,
        ).manifest
        for replicate in range(args.shuffled_replicates)
    ]
    preregistration = protocol.build_preregistration(
        corpus_manifest,
        tracks=args.tracks,
        scales=args.scales,
        shuffled_seed=args.shuffled_seed,
        shuffled_replicates=args.shuffled_replicates,
        no_share_tracks=args.no_share_tracks,
    )
    declared_controls = preregistration["shuffled_sides"]["controls"]
    observed_controls = [
        {
            "replicate": control["replicate"],
            "control_digest": control["control_digest"],
            "panel_set_digests": [
                entry["controlled_panel_set_digest"]
                for entry in control["problems"]
            ],
        }
        for control in controls
    ]
    if declared_controls != observed_controls:
        raise SystemExit(
            "prepared shuffled controls differ from the preregistered plan")

    # Validate the entire campaign in memory before creating any artifact.
    protocol.validate_corpus_manifest(corpus_manifest)
    protocol.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    protocol.validate_preregistration(
        preregistration, corpus_manifest=corpus_manifest)
    for control in controls:
        protocol.validate_shuffled_control_manifest(control, corpus_manifest)

    artifacts: list[tuple[str, dict, Callable[[dict], None]]] = [
        (
            os.path.join(out_dir, "corpus_manifest.json"),
            corpus_manifest,
            protocol.validate_corpus_manifest,
        ),
        (
            os.path.join(out_dir, "corpus_panels.json"),
            corpus_bundle,
            lambda value: protocol.validate_corpus_bundle(
                value, corpus_manifest),
        ),
        (
            os.path.join(out_dir, "phase_d_preregistration.json"),
            preregistration,
            lambda value: protocol.validate_preregistration(
                value, corpus_manifest=corpus_manifest),
        ),
    ]
    for control in controls:
        replicate = control["replicate"]
        artifacts.append((
            os.path.join(
                out_dir, f"shuffled_sides_r{replicate:02d}.json"),
            control,
            lambda value: protocol.validate_shuffled_control_manifest(
                value, corpus_manifest),
        ))

    # A later conflict must not leave earlier files from this invocation
    # behind.  Validate every destination before creating the directory or
    # publishing any missing member of the prepared set.
    for path, payload, validator in artifacts:
        _preflight_write_once(path, payload, validator)
    os.makedirs(out_dir, exist_ok=True)
    for path, payload, validator in artifacts:
        _write_once(path, payload, validator)
    summary = {
        "corpus_digest": corpus_manifest["corpus_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "bundle_digest": corpus_bundle["bundle_digest"],
        "problem_count": corpus_manifest["problem_count"],
        "scales": preregistration["scales"],
        "tracks": preregistration["tracks"],
        "arm_count": len(preregistration["arms"]),
        "execution_tags": sorted({
            arm["execution_tag"] for arm in preregistration["arms"]}),
        "control_digests": [control["control_digest"] for control in controls],
        "out_dir": out_dir,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _preflight_write_once(
        path: str, payload: dict,
        validator: Callable[[dict], None]) -> bool:
    """Validate one destination without mutating it; return whether it exists."""
    try:
        validator(payload)
    except (TypeError, ValueError, protocol.PhaseDProtocolError) as exc:
        raise SystemExit(
            f"new preregistration artifact is invalid: {path}: {exc}") from exc
    expected_bytes = semantic_replay.canonical_json_bytes(payload)
    try:
        existing = artifact_io._load_json(
            path, "existing preregistration artifact")
    except artifact_io.CampaignCollectionError as exc:
        # Absence is the only state in which the subsequent O_EXCL-style
        # creator may proceed.  Every extant special, linked, oversized, or
        # unstable path fails closed in the shared bounded reader.
        if isinstance(exc.__cause__, FileNotFoundError):
            return False
        raise SystemExit(
            f"existing preregistration artifact is invalid: {path}: {exc}") \
            from exc
    try:
        if not isinstance(existing, dict):
            raise TypeError("artifact must be a JSON object")
        validator(existing)
    except (TypeError, ValueError, protocol.PhaseDProtocolError) as exc:
        raise SystemExit(
            f"existing preregistration artifact is invalid: {path}") from exc
    if semantic_replay.canonical_json_bytes(existing) != expected_bytes:
        raise SystemExit(
            f"refusing to redefine existing preregistration artifact: {path}")
    return True


def _write_once(
        path: str, payload: dict,
        validator: Callable[[dict], None]) -> None:
    """Write one canonical artifact, validating content rather than labels."""
    if _preflight_write_once(path, payload, validator):
        return
    if semantic_artifacts.create_json_once(path, payload):
        return
    # A concurrent creator won after preflight.  Validate that winner instead
    # of replacing it.
    _preflight_write_once(path, payload, validator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(repo_root, "downloads", "Bongard-LOGO"),
    )
    parser.add_argument("--source", choices=protocol.SOURCES, default="both")
    parser.add_argument("--limit-per-source", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--scales", type=_csv_ints, default=(1, 5, 25))
    parser.add_argument(
        "--tracks",
        type=_csv_strings,
        default=("UNRESTRICTED", "SEMANTIC-PURE"),
    )
    parser.add_argument("--shuffled-seed", type=int, default=20260805)
    parser.add_argument("--shuffled-replicates", type=int, default=3)
    parser.add_argument(
        "--no-share-tracks",
        type=_csv_strings,
        default=("UNRESTRICTED",),
        help=("tracks with learned cross-problem libraries; semantic-pure is "
              "excluded until it has a learned/base registry split"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(__file__), "phase_d_runs", "preregistered"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
