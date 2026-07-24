#!/usr/bin/env python3
"""Audit promoted ARC artifacts without conflating them with forensic WIP.

The default report has three disjoint ledgers:

* ``canonical``: only files at each ``*_legs`` artifact root;
* ``successful_candidate_wip``: snapshots whose metadata says the active level
  was reached (these remain WIP, not proof of promotion);
* ``discarded_wip``: failed, interrupted, credit-out, or otherwise unverified
  snapshots.
* ``frontier_scaffolds``: reviewed level-scoped context that will be copied into a
  future clean room.

No WIP finding is propagated into the canonical verdict.  Historical ancestry
requires an explicit promotion manifest; file adjacency is not ancestry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROMOTED_FILES = {
    "legs.py", "players.py", "solve.py", "legs_log.md", "checkpoint.json",
    "auto_solve_attempts.json",
}
MAX_TAINT_SCAN_BYTES = 50_000_000
GAME_SOURCE_NAMES = {
    f"{game}.py" for game in (
        "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 "
        "r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30"
    ).split()
}
HIDDEN_SOURCE_RE = re.compile(
    r"environment_files/|/environment_files/|agent_solutions/|/agent_solutions/|"
    + "|".join(re.escape(name) for name in sorted(GAME_SOURCE_NAMES)),
    re.IGNORECASE,
)
NETWORK_RE = re.compile(
    r"(?:^|[\n;&|])\s*(?:sudo\s+)?(?:curl|wget|lynx|links|nc|ncat|netcat|"
    r"telnet|ssh|scp|rsync)(?!\s*=)\s+"
    r"|\b(?:web[_ -]?search|browser\.open|search_query|open_url)\b"
    r"|\b(?:requests|httpx|aiohttp|urllib\.request|http\.client)\."
    r"|\bsocket\.(?:create_connection|socket|getaddrinfo|gethostbyname)\b"
    r"|https?://(?!localhost(?::\d+)?(?:/|\b)|127\.0\.0\.1(?::\d+)?(?:/|\b)|"
    r"\[?::1\]?(?::\d+)?(?:/|\b))",
    re.IGNORECASE,
)
PRIVATE_RUNTIME_RE = re.compile(
    r"\.\s*_(?:game|env|fd)\b|\bvars\s*\(\s*env\b|"
    r"object\.__getattribute__\s*\(\s*env\b",
    re.IGNORECASE,
)
HARNESS_INTROSPECTION_RE = re.compile(
    r"inspect\.getsource\s*\(\s*(?:A\.|gkm_arena)|"
    r"\bdir\s*\(\s*(?:A|gkm_arena)(?:\.|\s*\))|"
    r"\bdir\s*\(\s*env\b|\.\s*_budget\b",
    re.IGNORECASE,
)


def scan_text(text: str, *, strip_inline_code: bool = True) -> list[str]:
    if strip_inline_code:
        text = re.sub(r"`[^`\n]*`", "", text)
    hits = []
    for label, pattern in (
        ("hidden_source_or_prior_solution", HIDDEN_SOURCE_RE),
        ("external_web_or_network", NETWORK_RE),
        ("direct_private_runtime", PRIVATE_RUNTIME_RE),
        ("harness_introspection", HARNESS_INTROSPECTION_RE),
    ):
        if pattern.search(text):
            hits.append(label)
    return hits


def codex_execution_surface(text: str) -> str | None:
    """Extract agent-authored actions from an immutable Codex JSONL transcript.

    Tool output is not an action.  In particular, a traceback emitted by the
    public ``env.clone()`` operation can contain the harness implementation's
    private field names.  The requested command, web-search item, and changed
    file paths remain separately available in the same transcript, while file
    contents are scanned from the corresponding evidence snapshot.
    """
    values: list[str] = []
    parsed = 0
    nonempty = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        nonempty += 1
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed += 1
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "command_execution" and isinstance(item.get("command"), str):
            values.append(item["command"])
        elif item_type in {"web_search", "file_change"}:
            values.append(json.dumps(item, sort_keys=True))
    if parsed and parsed == nonempty:
        return "\n".join(values)
    return None


def scan_file(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_TAINT_SCAN_BYTES:
            return ["oversized_unscanned_evidence"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        surface = (
            codex_execution_surface(text)
            if path.name == "proposer_last.log" or path.suffix == ".jsonl"
            else None
        )
        if surface is not None:
            return scan_text(surface, strip_inline_code=False)
        return scan_text(text)
    except OSError:
        return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_promotion_chain(artifact: Path) -> dict:
    manifests = sorted((artifact / "promotion_evidence").glob("level_*/manifest.json"))
    result = {"manifests": len(manifests), "integrity_errors": [], "taint_hits": [],
              "informational_harness_introspection": []}
    previous = None
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result["integrity_errors"].append(f"{manifest_path}: {exc}")
            continue
        evidence_dir = manifest_path.parent
        transcript = evidence_dir / str(manifest.get("transcript", "proposer_last.log"))
        if not transcript.is_file():
            result["integrity_errors"].append(f"missing transcript: {transcript}")
        elif sha256_file(transcript) != manifest.get("transcript_sha256"):
            result["integrity_errors"].append(f"transcript hash mismatch: {transcript}")
        else:
            hits = scan_file(transcript)
            forbidden = [hit for hit in hits if hit != "harness_introspection"]
            if forbidden:
                result["taint_hits"].append({"path": str(transcript), "kinds": forbidden})
            if "harness_introspection" in hits:
                result["informational_harness_introspection"].append(str(transcript))

        for item in manifest.get("codex_transcripts", []):
            if not isinstance(item, dict):
                result["integrity_errors"].append(
                    f"invalid Codex transcript entry: {manifest_path}"
                )
                continue
            codex_transcript = evidence_dir / str(item.get("path", ""))
            if not codex_transcript.is_file():
                result["integrity_errors"].append(
                    f"missing Codex transcript: {codex_transcript}"
                )
                continue
            if sha256_file(codex_transcript) != item.get("sha256"):
                result["integrity_errors"].append(
                    f"Codex transcript hash mismatch: {codex_transcript}"
                )
                continue
            hits = scan_file(codex_transcript)
            forbidden = [hit for hit in hits if hit != "harness_introspection"]
            if forbidden:
                result["taint_hits"].append({
                    "path": str(codex_transcript), "kinds": forbidden,
                })
            if "harness_introspection" in hits:
                result["informational_harness_introspection"].append(
                    str(codex_transcript)
                )

        for name, expected in manifest.get("promoted_files_sha256", {}).items():
            evidence_file = evidence_dir / "files" / name
            if not evidence_file.is_file():
                result["integrity_errors"].append(f"missing promoted evidence: {evidence_file}")
            elif sha256_file(evidence_file) != expected:
                result["integrity_errors"].append(f"promoted-file hash mismatch: {evidence_file}")
            else:
                hits = scan_file(evidence_file)
                forbidden = [hit for hit in hits if hit != "harness_introspection"]
                if forbidden:
                    result["taint_hits"].append({
                        "path": str(evidence_file), "kinds": forbidden,
                    })
                if "harness_introspection" in hits:
                    result["informational_harness_introspection"].append(
                        str(evidence_file)
                    )

        parent_rel = manifest.get("parent_manifest")
        parent_hash = manifest.get("parent_manifest_sha256")
        if previous is None:
            if parent_rel is not None or parent_hash is not None:
                result["integrity_errors"].append(f"unexpected parent on first manifest: {manifest_path}")
        else:
            expected_rel = str(previous.relative_to(artifact))
            if parent_rel != expected_rel or parent_hash != sha256_file(previous):
                result["integrity_errors"].append(f"broken parent chain: {manifest_path}")
        previous = manifest_path
    result["verdict"] = (
        "clean" if not result["integrity_errors"] and not result["taint_hits"]
        else "tainted_or_invalid"
    )
    return result


def wip_category(metadata: dict) -> str:
    try:
        reached = int(metadata.get("reached", -1))
        level = int(metadata.get("level", 10**9))
    except (TypeError, ValueError):
        return "discarded_wip"
    phase = str(metadata.get("phase", ""))
    success_phase = phase in {
        "reached_before_debrief", "after_debrief", "after_auto_solve_debrief",
        "recovered_existing_path_artifact", "recovered_after_credit_out",
        "recovered_path_artifact", "debrief_credit_out",
    }
    return "successful_candidate_wip" if success_phase and reached >= level else "discarded_wip"


def audit(root: Path) -> dict:
    report = {
        "canonical": {"files": 0, "hits": []},
        "successful_candidate_wip": {"snapshots": 0, "files": 0, "hits": []},
        "discarded_wip": {"snapshots": 0, "files": 0, "hits": []},
        "frontier_scaffolds": {"files": 0, "hits": [], "verdict": "clean"},
        "promotion_chains": {},
        "ancestry_note": (
            "WIP adjacency is not promotion ancestry. Historical ancestry needs an "
            "explicit manifest or an independently reconstructed file-hash chain."
        ),
    }
    for artifact in sorted(root.glob("*_legs")):
        if (artifact / "checkpoint.json").is_file():
            report["promotion_chains"][artifact.name] = audit_promotion_chain(artifact)
            for name in sorted(PROMOTED_FILES):
                path = artifact / name
                if not path.is_file():
                    continue
                report["canonical"]["files"] += 1
                hits = scan_file(path)
                if hits:
                    report["canonical"]["hits"].append({
                        "path": str(path), "kinds": hits,
                    })

        # Cold-start artifacts have no canonical checkpoint yet, but their WIP
        # and reviewed scaffold will enter a future clean room. Audit them too.
        for path in sorted(
            (artifact / "wip_context").glob("level_*/frontier_scaffold.json")
        ):
            report["frontier_scaffolds"]["files"] += 1
            hits = scan_file(path)
            if hits:
                report["frontier_scaffolds"]["hits"].append({
                    "path": str(path), "kinds": hits,
                })

        for metadata_path in sorted((artifact / "wip_context").glob("level_*/*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            category = wip_category(metadata)
            report[category]["snapshots"] += 1
            # A forensic snapshot contains copies of templates and probes from
            # older attempts.  Those are context, not proof of execution.  The
            # proposer transcript is the action ledger, so WIP attempt verdicts
            # are based on that file alone.  Canonical source is scanned above.
            path = metadata_path.parent / "files" / "proposer_last.log"
            snapshot_hits = []
            if path.is_file():
                report[category]["files"] += 1
                hits = scan_file(path)
                if hits:
                    snapshot_hits.append({"path": str(path), "kinds": hits})
            if snapshot_hits:
                report[category]["hits"].append({
                    "attempt": metadata.get("attempt"),
                    "game": metadata.get("game"),
                    "level": metadata.get("level"),
                    "phase": metadata.get("phase"),
                    "files": snapshot_hits,
                })
    report["canonical"]["verdict"] = "clean" if not report["canonical"]["hits"] else "tainted"
    report["frontier_scaffolds"]["verdict"] = (
        "clean" if not report["frontier_scaffolds"]["hits"] else "tainted"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?",
        default=str(Path(__file__).parent / "crack_lab" / "agent_solutions"),
    )
    parser.add_argument(
        "--require-complete-lineage", action="store_true",
        help="fail when a canonical artifact has no promotion manifests",
    )
    args = parser.parse_args()
    report = audit(Path(args.root))
    failed = report["canonical"]["verdict"] != "clean"
    failed = failed or report["frontier_scaffolds"]["verdict"] != "clean"
    for chain in report["promotion_chains"].values():
        failed = failed or chain["verdict"] != "clean"
        if args.require_complete_lineage:
            failed = failed or chain["manifests"] == 0
    report["automated_verdict"] = "FAIL" if failed else "PASS"
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
