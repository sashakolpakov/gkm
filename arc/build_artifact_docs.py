#!/usr/bin/env python3
"""Generate ARC status from the frozen 25-game checkpoint set."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARC = ROOT / "arc"
FROZEN_SOLUTIONS = (
    ARC / "crack_lab" / "releases" / "arc_agi3_gkm_v2_181" / "artifacts"
)
PUBLICATION_READMES = ARC / "crack_lab" / "agent_solutions"
sys.path.insert(0, str(ARC / "crack_lab"))
sys.path.insert(0, str(ARC / "manuscript"))

from gkm_legs import MARGINAL_COMPLEXITY_CONTRACT  # noqa: E402
from build_artifact_history import check as check_artifact_history  # noqa: E402
from history_manifest import get_history  # noqa: E402


AUTHORITATIVE_LEVELS = {
    "ar25": 8, "bp35": 9, "cd82": 6, "cn04": 6, "dc22": 6,
    "ft09": 6, "g50t": 7, "ka59": 7, "lf52": 10, "lp85": 8,
    "ls20": 7, "m0r0": 6, "r11l": 6, "re86": 8, "s5i5": 8,
    "sb26": 8, "sc25": 6, "sk48": 8, "sp80": 6, "su15": 9,
    "tn36": 7, "tr87": 6, "tu93": 9, "vc33": 7, "wa30": 9,
}


@dataclass(frozen=True)
class Artifact:
    game: str
    reached: int
    target: int
    validated: bool
    actions: int
    checkpoint_total: int
    published_total: int
    records: tuple[tuple[int, int], ...]
    checkpoint_records: tuple[tuple[int, int], ...]

    @property
    def recorded_levels(self) -> tuple[int, ...]:
        return tuple(level for level, _ in self.records)

    @property
    def complete_ledger(self) -> bool:
        return self.recorded_levels == tuple(range(1, self.reached + 1))


def load_artifact(game: str) -> Artifact:
    path = FROZEN_SOLUTIONS / f"{game}_legs" / "checkpoint.json"
    data = json.loads(path.read_text())
    checkpoint_records = tuple(
        (int(row["level"]), int(row["marginal_C"])) for row in data["records"]
    )
    history = get_history(game)
    records = tuple((entry.level, entry.marginal_C) for entry in history.ledger)
    artifact = Artifact(
        game=str(data["game"]),
        reached=int(data["reached"]),
        target=AUTHORITATIVE_LEVELS[game],
        validated=bool(data["validated"]),
        actions=len(data["final_path"]),
        checkpoint_total=int(data["total_marginal_C"]),
        published_total=history.total_marginal_C,
        records=records,
        checkpoint_records=checkpoint_records,
    )
    if artifact.game != game:
        raise ValueError(f"{path}: game is {artifact.game!r}, expected {game!r}")
    if artifact.reached != history.max_level or artifact.actions != history.replay_actions:
        raise ValueError(f"{path}: endpoint disagrees with manuscript history")
    if not artifact.complete_ledger:
        raise ValueError(f"{game}: manuscript ledger is not complete")
    checkpoint_levels = tuple(level for level, _ in checkpoint_records)
    if len(set(checkpoint_levels)) != len(checkpoint_records):
        raise ValueError(f"{path}: duplicate checkpoint record")
    if any(level < 1 or level > artifact.reached for level in checkpoint_levels):
        raise ValueError(f"{path}: checkpoint record outside verified depth")
    if sum(cost for _, cost in checkpoint_records) != artifact.checkpoint_total:
        raise ValueError(f"{path}: records do not sum to total_marginal_C")
    if sum(cost for _, cost in artifact.records) != artifact.published_total:
        raise ValueError(f"{game}: manuscript ledger does not sum to its total")
    if not artifact.validated:
        raise ValueError(f"{path}: artifact is not replay validated")
    return artifact


def load_checkpoint_artifact(game: str) -> Artifact:
    """Load one complete checkpoint ledger without requiring a history sidecar."""
    path = FROZEN_SOLUTIONS / f"{game}_legs" / "checkpoint.json"
    data = json.loads(path.read_text())
    records = tuple(
        (int(row["level"]), int(row["marginal_C"])) for row in data["records"]
    )
    reached = int(data["reached"])
    target = AUTHORITATIVE_LEVELS[game]
    artifact = Artifact(
        game=str(data["game"]),
        reached=reached,
        target=target,
        validated=bool(data["validated"]),
        actions=len(data["final_path"]),
        checkpoint_total=int(data["total_marginal_C"]),
        published_total=int(data["total_marginal_C"]),
        records=records,
        checkpoint_records=records,
    )
    if artifact.game != game or not artifact.validated:
        raise ValueError(f"{path}: invalid game or validation field")
    if reached > target or artifact.recorded_levels != tuple(range(1, reached + 1)):
        raise ValueError(f"{path}: invalid depth or incomplete checkpoint ledger")
    if sum(value for _, value in records) != artifact.checkpoint_total:
        raise ValueError(f"{path}: records do not sum to total_marginal_C")
    return artifact


def record_text(records: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"L{level}={cost}" for level, cost in records)


def summary_markdown(artifacts: tuple[Artifact, ...]) -> str:
    by_game = {artifact.game: artifact for artifact in artifacts}
    reached = sum(artifact.reached for artifact in artifacts)
    target = sum(artifact.target for artifact in artifacts)
    actions = sum(artifact.actions for artifact in artifacts)
    rows = [
        "| Scope | Verified levels | Stored replay actions | Ledger charge |",
        "|---|---:|---:|---:|",
        f"| Frozen v2 release (25 games) | {reached}/{target} | {actions} | — |",
    ]
    rows.extend(
        f"| `{a.game}` uniform history | {a.reached}/{a.target} | {a.actions} | "
        f"{a.published_total} |"
        for a in (by_game["wa30"], by_game["ls20"])
    )
    rows.extend(
        [
            "",
            "The frozen release contains one checkpoint record for every promoted "
            "level. The generated all-game marginal table is "
            "`arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` "
            "and `ls20` rows additionally have one uniform, audited history sidecar "
            "per level. "
            f"`{MARGINAL_COMPLEXITY_CONTRACT['field']}` means "
            f"{MARGINAL_COMPLEXITY_CONTRACT['label']}. "
            f"{MARGINAL_COMPLEXITY_CONTRACT['limitation'].capitalize()}.",
        ]
    )
    return "\n".join(rows)


def detail_markdown(artifact: Artifact) -> str:
    return "\n".join(
        [
            f"- Game: `{artifact.game}`",
            f"- Verified through level: {artifact.reached}",
            f"- Replay validated: {artifact.validated}",
            f"- Final replay path length: {artifact.actions}",
            f"- Complete published ledger charge: {artifact.published_total}",
            f"- Complete published ledger: {record_text(artifact.records)}",
            f"- Current operational checkpoint charge: {artifact.checkpoint_total}",
            f"- Current operational checkpoint records: {record_text(artifact.checkpoint_records)}",
            "",
            "The complete ledger and clean-source hashes are in "
            f"`arc/manuscript/artifact_history/{artifact.game}/manifest.json`. "
            "During active acquisition, the artifact root and `wip_context` retain "
            "the clean replay state and restart evidence. Publication authority comes "
            "from the compact schema-v2 release and its frozen receipt; terminal "
            "cleanup may prune mutable WIP after all release consumers are bound.",
            f"The `{MARGINAL_COMPLEXITY_CONTRACT['field']}` field is "
            f"{MARGINAL_COMPLEXITY_CONTRACT['label']}; "
            f"{MARGINAL_COMPLEXITY_CONTRACT['limitation']}.",
        ]
    )


def render_rst(artifacts: tuple[Artifact, ...]) -> str:
    lines = [
        ".. This file is generated by arc/build_artifact_docs.py.",
        "",
        ".. list-table:: Frozen v2 replay endpoints",
        "   :header-rows: 1",
        "",
        "   * - Game",
        "     - Verified levels",
        "     - Replay actions",
        "     - Published ledger charge",
    ]
    for artifact in artifacts:
        lines.extend(
            [
                f"   * - ``{artifact.game}``",
                f"     - {artifact.reached}/{artifact.target}",
                f"     - {artifact.actions}",
                f"     - {artifact.published_total}",
            ]
        )
    lines.extend(
        [
            "",
            "The frozen release contains one checkpoint record for every promoted level. "
            "The all-game per-level matrix is included below. ``wa30`` and ``ls20`` "
            "additionally retain uniform per-promotion manuscript sidecars. "
            f"``{MARGINAL_COMPLEXITY_CONTRACT['field']}`` means "
            f"{MARGINAL_COMPLEXITY_CONTRACT['label']}. "
            f"{MARGINAL_COMPLEXITY_CONTRACT['limitation'].capitalize()}.",
            "",
        ]
    )
    return "\n".join(lines)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def render_tex(artifacts: tuple[Artifact, ...]) -> str:
    by_game = {artifact.game: artifact for artifact in artifacts}
    selected = (by_game["wa30"], by_game["ls20"])
    rows = "\n".join(
        f"\\texttt{{{a.game}}} & {a.reached}/{a.target} & {a.actions} & "
        f"{a.published_total} \\\\" for a in selected
    )
    reached = sum(artifact.reached for artifact in artifacts)
    target = sum(artifact.target for artifact in artifacts)
    actions = sum(artifact.actions for artifact in artifacts)
    return "\n".join(
        [
            "% Generated by arc/build_artifact_docs.py. Do not edit.",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"Game & Levels & Replay actions & Ledger charge \\",
            r"\midrule",
            f"Frozen v2 (25 games) & {reached}/{target} & {actions} & -- " + r"\\",
            rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            "The frozen release contains one checkpoint record per promoted level. The "
            "two displayed game histories additionally contain one uniform manuscript "
            "sidecar per level. "
            f"\\texttt{{{tex_escape(MARGINAL_COMPLEXITY_CONTRACT['field'])}}} denotes "
            f"{tex_escape(MARGINAL_COMPLEXITY_CONTRACT['label'])}; "
            f"{tex_escape(MARGINAL_COMPLEXITY_CONTRACT['limitation'])}.",
            "",
        ]
    )


def replace_block(path: Path, tag: str, content: str) -> None:
    start = f"<!-- BEGIN GENERATED: {tag} -->"
    end = f"<!-- END GENERATED: {tag} -->"
    source = path.read_text()
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"{path}: expected one generated block {tag}")
    before, rest = source.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(f"{before}{start}\n{content}\n{end}{after}")


def main() -> None:
    check_artifact_history()
    artifacts = tuple(
        load_checkpoint_artifact(game) for game in AUTHORITATIVE_LEVELS
    )
    if sum(artifact.reached for artifact in artifacts) != 181:
        raise ValueError("frozen artifact set is not 181/183")
    if sum(artifact.actions for artifact in artifacts) != 7001:
        raise ValueError("frozen artifact set does not contain 7,001 actions")
    summary = summary_markdown(artifacts)
    for path in (
        ROOT / "README.md",
        ARC / "README.md",
        ARC / "ARC.md",
        ROOT / "REPRODUCE_ARC.md",
        ARC / "manuscript" / "gkm_one_page_summary.md",
    ):
        replace_block(path, "ARC_ARTIFACT_STATUS", summary)
    for artifact in (load_artifact("wa30"), load_artifact("ls20")):
        replace_block(
            PUBLICATION_READMES / f"{artifact.game}_legs" / "README.md",
            "ARTIFACT_DETAILS",
            detail_markdown(artifact),
        )
    docs_out = ROOT / "docs" / "generated" / "arc_artifacts.rst"
    tex_out = ARC / "manuscript" / "generated" / "arc_artifacts.tex"
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.write_text(render_rst(artifacts))
    tex_out.write_text(render_tex(artifacts))


if __name__ == "__main__":
    main()
