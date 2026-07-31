#!/usr/bin/env python3
"""Generate the final ARC-AGI-3 empirical tables from sealed checkpoints.

The table records the harness-native ``marginal_C`` value stored when each
level was promoted.  It does not turn that scalar into a semantic-reuse claim:
the separate solved-source audit is responsible for exact adjacent-source and
literal-call witnesses.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterable


AUTHORITATIVE_LEVELS: Final[dict[str, int]] = {
    "ar25": 8,
    "bp35": 9,
    "cd82": 6,
    "cn04": 6,
    "dc22": 6,
    "ft09": 6,
    "g50t": 7,
    "ka59": 7,
    "lf52": 10,
    "lp85": 8,
    "ls20": 7,
    "m0r0": 6,
    "r11l": 6,
    "re86": 8,
    "s5i5": 8,
    "sb26": 8,
    "sc25": 6,
    "sk48": 8,
    "sp80": 6,
    "su15": 9,
    "tn36": 7,
    "tr87": 6,
    "tu93": 9,
    "vc33": 7,
    "wa30": 9,
}

RELEASE_COMMIT: Final[str] = "9235ed26627140460efa1f6ca5e4041470cddc14"
RELEASE_RECEIPT: Final[str] = (
    "140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681"
)
ONLINE_SCORECARD: Final[str] = (
    "https://arcprize.org/scorecards/e293eeae-c0de-4263-a916-0a40ad282cbc"
)
COMPETITION_SCORECARD: Final[str] = (
    "https://arcprize.org/scorecards/cf75e14b-2c25-41cb-bc70-53bd57411edb"
)
OFFICIAL_SCORE: Final[float] = 98.11664037825032
OFFICIAL_API_ACTIONS: Final[int] = 7069


@dataclass(frozen=True)
class GameRow:
    game: str
    reached: int
    total_levels: int
    actions: int
    retained_description: int
    total_marginal_C: int
    marginals: tuple[int, ...]
    direction_reversals: int
    normalized_total_variation: float


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("../crack_lab/agent_solutions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--rst-output", type=Path)
    return parser.parse_args()


def _literal_cost(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            total += len(node.elts)
        elif isinstance(node, ast.Dict):
            total += len(node.keys)
    return total


def description_complexity(code: str) -> int:
    lines = sum(
        bool(line.strip()) and not line.strip().startswith("#")
        for line in code.splitlines()
    )
    return lines + _literal_cost(code)


def _sign_reversals(values: Iterable[int]) -> int:
    values = tuple(values)
    signs = [
        1 if right > left else -1
        for left, right in zip(values, values[1:])
        if right != left
    ]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _normalized_total_variation(values: tuple[int, ...]) -> float:
    if len(values) < 2 or max(values) == 0:
        return 0.0
    return sum(abs(right - left) for left, right in zip(values, values[1:])) / max(
        values
    )


def load_game(solutions_dir: Path, game: str) -> GameRow:
    game_dir = solutions_dir / f"{game}_legs"
    checkpoint_path = game_dir / "checkpoint.json"
    payload = json.loads(checkpoint_path.read_text())
    if payload.get("game") != game:
        raise ValueError(f"{checkpoint_path}: wrong game field")
    if payload.get("validated") is not True:
        raise ValueError(f"{checkpoint_path}: checkpoint is not validated")
    reached = int(payload["reached"])
    total_levels = AUTHORITATIVE_LEVELS[game]
    if reached < 0 or reached > total_levels:
        raise ValueError(f"{checkpoint_path}: invalid reached depth {reached}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != reached:
        raise ValueError(f"{checkpoint_path}: expected one record per reached level")
    levels = [int(record["level"]) for record in records]
    if levels != list(range(1, reached + 1)):
        raise ValueError(f"{checkpoint_path}: nonconsecutive records")
    if any(record.get("reached") is not True for record in records):
        raise ValueError(f"{checkpoint_path}: contains an unpromoted record")
    marginals = tuple(int(record["marginal_C"]) for record in records)
    if any(value < 0 for value in marginals):
        raise ValueError(f"{checkpoint_path}: negative marginal_C")
    if sum(marginals) != int(payload["total_marginal_C"]):
        raise ValueError(f"{checkpoint_path}: marginal total mismatch")
    final_path = payload.get("final_path")
    if not isinstance(final_path, list):
        raise ValueError(f"{checkpoint_path}: final_path is not a list")
    retained = sum(
        description_complexity((game_dir / filename).read_text())
        for filename in ("legs.py", "players.py")
    )
    return GameRow(
        game=game,
        reached=reached,
        total_levels=total_levels,
        actions=len(final_path),
        retained_description=retained,
        total_marginal_C=sum(marginals),
        marginals=marginals,
        direction_reversals=_sign_reversals(marginals),
        normalized_total_variation=_normalized_total_variation(marginals),
    )


def load_all(solutions_dir: Path) -> tuple[GameRow, ...]:
    rows = tuple(load_game(solutions_dir, game) for game in AUTHORITATIVE_LEVELS)
    if len(rows) != 25 or sum(row.total_levels for row in rows) != 183:
        raise ValueError("authoritative inventory is not exactly 25 games / 183 levels")
    if sum(row.reached for row in rows) != 181:
        raise ValueError("frozen release is not exactly 181 promoted levels")
    if sum(row.actions for row in rows) != 7001:
        raise ValueError("frozen release does not contain exactly 7,001 stored actions")
    return rows


def sawtooth_ranking(rows: Iterable[GameRow]) -> list[GameRow]:
    """Rank visible ledger oscillation without interpreting it as reuse."""
    return sorted(
        rows,
        key=lambda row: (
            row.direction_reversals,
            row.normalized_total_variation,
            len(row.marginals),
            row.game,
        ),
        reverse=True,
    )


def _cell(row: GameRow, level: int, *, kind: str) -> str:
    if level <= row.reached:
        return str(row.marginals[level - 1])
    if level <= row.total_levels:
        return "pending"
    return "--" if kind == "md" else (r"$\mathrm{n/a}$" if kind == "tex" else "n/a")


def render_markdown(rows: tuple[GameRow, ...]) -> str:
    headers = [
        "Game",
        "Depth",
        "Path",
        "D(s)",
        *[f"L{level}" for level in range(1, 11)],
        "Total",
    ]
    lines = [
        "<!-- Generated by scripts/generate_empirical_tables.py; do not edit. -->",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:"] + ["---:"] * 13) + " |",
    ]
    for row in rows:
        cells = [
            f"`{row.game}`",
            f"{row.reached}/{row.total_levels}",
            str(row.actions),
            str(row.retained_description),
            *[_cell(row, level, kind="md") for level in range(1, 11)],
            str(row.total_marginal_C),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Values are the checkpoint-recorded harness-native `marginal_C` charges. "
            "`pending` marks an authoritative level with no promoted boundary; `--` "
            "means that the game has no such level. The scalar alone is not a semantic "
            "reuse certificate.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_tex(rows: tuple[GameRow, ...]) -> str:
    body = []
    for row in rows:
        cells = [
            rf"\texttt{{{row.game}}}",
            f"{row.reached}/{row.total_levels}",
            str(row.actions),
            str(row.retained_description),
            *[_cell(row, level, kind="tex") for level in range(1, 11)],
            str(row.total_marginal_C),
        ]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            "% Generated by scripts/generate_empirical_tables.py; do not edit.",
            r"\begin{table}[p]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.7pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrrrrrrrrr@{}}",
            r"\toprule",
            r"Game & Depth & Path & $D(s)$ & L1 & L2 & L3 & L4 & L5 & L6 & L7 & L8 & L9 & L10 & $C_{\leq k}$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Harness-native marginal-complexity ledger for all 25 games in the frozen 181/183 release. Values are the checkpoint-recorded $C_k$ charges. \textnormal{pending} marks the two authoritative but unsolved \texttt{lf52} boundaries; $\mathrm{n/a}$ means that the game has no such level. The scalar profile is descriptive and is not, by itself, a source-reuse witness.}",
            r"\label{tab:all-marginal-complexities}",
            r"\end{table}",
            "",
        ]
    )


def render_rst(rows: tuple[GameRow, ...]) -> str:
    lines = [
        ".. This file is generated by arc/manuscript/scripts/generate_empirical_tables.py.",
        "",
        ".. csv-table:: Harness-native marginal complexity by level",
        "   :header: \"Game\", \"Depth\", \"Path\", \"D(s)\", \"L1\", \"L2\", \"L3\", \"L4\", \"L5\", \"L6\", \"L7\", \"L8\", \"L9\", \"L10\", \"Total\"",
        "   :widths: 8, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7",
        "",
    ]
    for row in rows:
        cells = [
            row.game,
            f"{row.reached}/{row.total_levels}",
            str(row.actions),
            str(row.retained_description),
            *[_cell(row, level, kind="rst") for level in range(1, 11)],
            str(row.total_marginal_C),
        ]
        lines.append("   " + ", ".join(f'\"{cell}\"' for cell in cells))
    lines.extend(
        [
            "",
            "``pending`` marks an authoritative level with no promoted boundary; "
            "``n/a`` means that the game has no such level. Values are stored "
            "harness-native charges, not semantic-reuse labels.",
            "",
        ]
    )
    return "\n".join(lines)


def render_json(rows: tuple[GameRow, ...]) -> str:
    ranking = sawtooth_ranking(rows)
    payload = {
        "schema": 1,
        "release": {
            "commit": RELEASE_COMMIT,
            "receipt_sha256": RELEASE_RECEIPT,
            "online_scorecard": ONLINE_SCORECARD,
            "competition_scorecard": COMPETITION_SCORECARD,
            "official_score_percent": OFFICIAL_SCORE,
            "raw_levels": 181,
            "authoritative_levels": 183,
            "stored_actions": 7001,
            "official_api_actions": OFFICIAL_API_ACTIONS,
        },
        "measurement": (
            "checkpoint-recorded positive net retained-description growth per "
            "legs.py and players.py, including the container-literal surcharge"
        ),
        "games": [asdict(row) for row in rows],
        "sawtooth_ranking": [
            {
                "game": row.game,
                "direction_reversals": row.direction_reversals,
                "normalized_total_variation": row.normalized_total_variation,
            }
            for row in ranking
        ],
        "figure_selection": {
            "strongest_raw_shape": ranking[0].game,
            "manuscript_primary": "wa30",
            "reason": (
                "su15 has the strongest raw oscillation, but wa30 is the strongest "
                "complete uniform promotion-history example with one audited sidecar "
                "per level; ls20 supplies the coupled unchanged-leg reuse example"
            ),
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def main() -> int:
    args = _args()
    rows = load_all(args.solutions_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.output_dir / "marginal_complexity_by_level.md": render_markdown(rows),
        args.output_dir / "marginal_complexity_by_level.tex": render_tex(rows),
        args.output_dir / "marginal_complexity_by_level.json": render_json(rows),
    }
    if args.rst_output:
        args.rst_output.parent.mkdir(parents=True, exist_ok=True)
        outputs[args.rst_output] = render_rst(rows)
    for path, text in outputs.items():
        path.write_text(text)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
