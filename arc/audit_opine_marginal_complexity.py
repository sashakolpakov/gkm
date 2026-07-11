#!/usr/bin/env python3
"""Audit OPINE-World artifacts using an explicit marginal-complexity ledger.

The tool deliberately does not call a carried-forward ``game_engine.py`` free.
For each synthesis snapshot it measures normalized Python-token growth, then
charges every level-specific source reference and cache byte to the level named
by its filename or a nearby ``level_N`` path component.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import re
import tokenize
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath


LEVEL_RE = re.compile(r"(?:^|[_/])l(?:evel)?[_-]?(\d+)(?:[_./-]|$)", re.I)


@dataclass(frozen=True)
class Artifact:
    path: str
    data: bytes


def py_tokens(data: bytes) -> list[str]:
    """Return code tokens, excluding comments, whitespace, and encoding marks."""
    try:
        stream = io.BytesIO(data).readline
        return [
            token.string
            for token in tokenize.tokenize(stream)
            if token.type
            not in {
                tokenize.COMMENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
            }
        ]
    except (SyntaxError, tokenize.TokenError, UnicodeDecodeError):
        return re.findall(r"\S+", data.decode("utf-8", "replace"))


def level_for(path: str) -> str:
    match = LEVEL_RE.search(path)
    return f"L{int(match.group(1))}" if match else "unattributed"


def game_for(path: str) -> str:
    parts = PurePosixPath(path).parts
    # Published archives normally put each game beneath a top-level directory.
    return parts[0] if len(parts) > 1 else "archive-root"


def source_additions(before: list[str], after: list[str]) -> tuple[int, int]:
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "insert"}:
            added += j2 - j1
        if tag in {"replace", "delete"}:
            removed += i2 - i1
    return added, removed


def read_archive(path: str) -> list[Artifact]:
    with zipfile.ZipFile(path) as archive:
        return [
            Artifact(info.filename, archive.read(info))
            for info in archive.infolist()
            if not info.is_dir()
        ]


def audit(artifacts: list[Artifact]) -> list[dict[str, object]]:
    engines = sorted(
        (a for a in artifacts if a.path.endswith("game_engine.py")), key=lambda a: a.path
    )
    rows: list[dict[str, object]] = []
    previous: dict[str, list[str]] = {}

    for engine in engines:
        game = game_for(engine.path)
        tokens = py_tokens(engine.data)
        added, removed = source_additions(previous.get(game, []), tokens)
        previous[game] = tokens
        rows.append(
            {
                "game": game,
                "level": level_for(engine.path),
                "kind": "game_engine",
                "artifact": engine.path,
                "full_tokens": len(tokens),
                "added_tokens": added,
                "removed_tokens": removed,
                "bytes": len(engine.data),
            }
        )

    # Level-entry caches and literal data are not learned dynamics. Charge their
    # full byte cost because the model needs the data verbatim to reproduce it.
    for artifact in artifacts:
        name = PurePosixPath(artifact.path).name.lower()
        if not re.search(r"(?:initial|cache|lookup|replay|checkpoint).*(?:pkl|json|npy|npz)$", name):
            continue
        if level_for(artifact.path) == "unattributed":
            continue
        rows.append(
            {
                "game": game_for(artifact.path),
                "level": level_for(artifact.path),
                "kind": "level_data",
                "artifact": artifact.path,
                "full_tokens": 0,
                "added_tokens": 0,
                "removed_tokens": 0,
                "bytes": len(artifact.data),
            }
        )
    return rows


def write_report(rows: list[dict[str, object]], output: str) -> None:
    csv_path = f"{output}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["game"])
        writer.writeheader()
        writer.writerows(rows)

    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        total = totals[(str(row["game"]), str(row["level"]))]
        total["added_tokens"] += int(row["added_tokens"])
        total["cache_bytes"] += int(row["bytes"]) if row["kind"] == "level_data" else 0
        total["engine_bytes"] = max(total["engine_bytes"], int(row["bytes"]))

    with open(f"{output}.md", "w") as f:
        f.write("# OPINE Marginal-Complexity Ledger\n\n")
        f.write("Charges are source-token additions plus level-specific data bytes. "
                "They are an audit proxy, not machine-independent Kolmogorov complexity.\n\n")
        f.write("| Game | Level | Added code tokens | Level data bytes | Latest engine bytes |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for (game, level), total in sorted(totals.items()):
            f.write(f"| {game} | {level} | {total['added_tokens']} | "
                    f"{total['cache_bytes']} | {total['engine_bytes']} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="OPINE published zip archive")
    parser.add_argument("--output", default="opine_marginal_complexity")
    args = parser.parse_args()
    rows = audit(read_archive(args.archive))
    write_report(rows, args.output)
    print(f"wrote {args.output}.csv and {args.output}.md from {len(rows)} chargeable artifacts")


if __name__ == "__main__":
    main()
