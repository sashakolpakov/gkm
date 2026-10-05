"""Capture and replay the original register benchmark without changing its search.

The historical runner and substrate remain byte-for-byte unchanged. Scoped
wrappers save their inputs and return values; they never rank or repair a solver.
Completed lambda runs are immutable checkpoints, so an interrupted reproduction
can resume without drawing a new seed or discarding a failed condition.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import sys
import tarfile
import time
from unittest.mock import patch

import pattern_fsa as substrate
import run_register_transducer_benchmark as historical

SOURCES = ("pattern_fsa.py", "run_register_transducer_benchmark.py", "register_artifacts.py")
# Transcribed before the new run; used only for reporting after selection.
HISTORICAL = ((1, 0, 3), (0, .5, 3), (1, 0, 7), (1, 0, 5),
              (0, 1 / 3, 3), (1, 0, 8), (1, 0, 10), (1, 0, 7),
              (.5, .25, 2), (1, 0, 4))


def normal(value):
    return json.loads(json.dumps(value, sort_keys=True))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    value = normal(value)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"refusing to overwrite changed evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sources():
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES}


def prepare(output, configs=None):
    configs = historical.CONFIGS if configs is None else configs
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(version=1, kind="new reproduction, not a recovered historical run",
                created_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, platform=platform.platform(),
                configs=[asdict(c) for c in configs], source_hashes=sources(),
                selection="unchanged historical run_config and lambda_sweep_solver",
                metric="exact output at 32-instruction cap; halting not required",
                checkpoints="all four final lambda candidates and their generation histories",
                historical_comparison=normal(HISTORICAL) if list(configs) == historical.CONFIGS else None)
    immutable(output / "plan.json", dict(plan=plan, hash=digest(plan)))
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(Path(__file__).with_name(name), arcname=name)
    return plan


def load_plan(output):
    manifest = read(output / "plan.json")
    plan = manifest["plan"]
    if digest(plan) != manifest["hash"] or plan["source_hashes"] != sources():
        raise ValueError("plan or source changed; replay with the archived sources")
    return plan


def arguments(values):
    return normal({k: asdict(v) if k in ("task", "primitives") else v for k, v in values.items()})


def genome(value):
    return substrate.PatternGenome(value["state_count"], value["alphabet_size"],
                                   [substrate.PatternRule(**r) for r in value["rules"]])


def candidate_return(checkpoint, expected):
    body = checkpoint["result"]
    if digest(body) != checkpoint["hash"] or checkpoint["arguments"] != arguments(expected):
        raise ValueError("candidate checkpoint or original arguments changed")
    program = genome(body["genome"])
    train = substrate.evaluate_genome(program, expected["task"].train_pairs,
        expected["primitives"], expected["lambda_value"], max_steps=expected["max_steps"])
    validation = substrate.evaluate_genome(program, expected["task"].val_pairs,
        expected["primitives"], expected["lambda_value"], max_steps=expected["max_steps"])
    if normal(asdict(train)) != body["train"] or normal(asdict(validation)) != body["validation"]:
        raise ValueError("candidate execution differs")
    history = [substrate.PatternGenerationRecord(**r) for r in body["history"]]
    if ([h.generation for h in history] != list(range(expected["generations"] + 1))
            or any(h.lambda_value != expected["lambda_value"] for h in history)):
        raise ValueError("incomplete generation history")
    return program, history, train, validation


def run_condition(output, index, *, replay=False):
    plan = load_plan(output)
    config = historical.BenchmarkConfig(**plan["configs"][index])
    folder = output / f"condition-{index:02d}"
    original_evolve, original_sweep = substrate.evolve_solver, historical.lambda_sweep_solver
    stream = sys.stdout
    captured = {}

    def evolve(**kwargs):
        ordinal, residue = divmod(kwargs["seed"] - 7, 997)
        if residue or not 1 <= ordinal <= 4:
            raise ValueError("unexpected historical search seed")
        path = folder / f"lambda-{ordinal}.json"
        if path.exists():
            return candidate_return(read(path), kwargs)
        if replay:
            raise ValueError(f"missing candidate: {path}")
        print(json.dumps(dict(event="lambda_start", condition=index, task=config.task,
                              primitive=config.primitive, ordinal=ordinal)), file=stream, flush=True)
        start = time.monotonic()
        result = original_evolve(**kwargs)
        program, history, train, validation = result
        value = dict(genome=asdict(program), history=[asdict(h) for h in history],
                     train=asdict(train), validation=asdict(validation))
        immutable(path, dict(arguments=arguments(kwargs), result=value, hash=digest(value),
                             elapsed_seconds=time.monotonic() - start,
                             peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print(json.dumps(dict(event="lambda_complete", condition=index, ordinal=ordinal,
                              validation_exact=validation.exact_match_rate)), file=stream, flush=True)
        return result

    def sweep(**kwargs):
        captured["arguments"] = arguments(kwargs)
        transcript = io.StringIO()
        with patch.object(substrate, "evolve_solver", evolve), contextlib.redirect_stdout(transcript):
            result = original_sweep(**kwargs)
        program, records, history, train, validation, test = result
        executions = {}
        for split, pairs in (("train", kwargs["task"].train_pairs),
                             ("validation", kwargs["task"].val_pairs),
                             ("hidden", kwargs["task"].test_pairs)):
            executions[split] = [dict(source=source, expected=target,
                run=asdict(substrate.run_transducer(program, source, kwargs["primitives"],
                                                   max_steps=kwargs["max_steps"]))) for source, target in pairs]
        captured.update(genome=asdict(program), solver=substrate.export_solver(program, kwargs["primitives"]),
                        lambda_sweep=[asdict(r) for r in records],
                        history=[asdict(h) for h in history],
                        evaluations=dict(train=asdict(train), validation=asdict(validation), hidden=asdict(test)),
                        executions=executions, transcript=transcript.getvalue())
        return result

    with patch.object(historical, "lambda_sweep_solver", sweep):
        row, rules = historical.run_config(config)
    value = normal(dict(config=asdict(config), row=row, exported_rules=rules, **captured))
    path = folder / "selection.json"
    receipt = dict(result=value, hash=digest(value))
    if replay:
        if read(path) != receipt:
            raise ValueError("fresh selection, transcript or held-out execution differs")
    else:
        immutable(path, receipt)
        print(json.dumps(dict(event="condition_complete", condition=index, row=row)), file=stream, flush=True)
    return value


def replay_all(output):
    plan = load_plan(output)
    rows = [run_condition(output, i, replay=True)["row"] for i in range(len(plan["configs"]))]
    comparisons = []
    if plan["historical_comparison"] is not None:
        for row, old in zip(rows, plan["historical_comparison"]):
            observed = [row["test_exact"], row["test_loss"], row["complexity"]]
            comparisons.append(dict(task=row["task"], primitive=row["primitive"], historical=old,
                reproduced=observed, match=all(abs(a - b) < 1e-9 for a, b in zip(old, observed))))
    return dict(kind=plan["kind"], rows=rows, historical_comparison=comparisons,
                candidate_programs_verified=4 * len(rows), selected_programs_verified=len(rows),
                exact_original_selection_replayed=True)


def finalize(output):
    result = replay_all(output)
    immutable(output / "summary.json", result)
    csvpath = output / "matrix.csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    if csvpath.exists():
        if csvpath.read_bytes() != buffer.getvalue().encode():
            raise ValueError("matrix changed")
    else:
        with csvpath.open("xb") as stream:
            stream.write(buffer.getvalue().encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--conditions", nargs="+", type=int)
    modes.add_argument("--finalize", action="store_true")
    modes.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
    elif args.conditions is not None:
        for index in args.conditions:
            run_condition(args.output, index)
    elif args.finalize:
        print(json.dumps(finalize(args.output), indent=2))
    else:
        result = replay_all(args.output)
        if read(args.output / "summary.json") != result:
            raise ValueError("summary differs from replay")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
