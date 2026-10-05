"""Post-run accounting for the direct proposer comparison; never a selector."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from interface_benchmark import save
from interface_machine import parse


def supported(row):
    return row["admitted"] and all(row.get(k, {}).get("exact", False)
                                   for k in ("hidden", "stress", "audit"))


def recursive(body, name):
    def visit(term):
        return isinstance(term, tuple) and (
            term[:2] in (("call", "self"), ("call", name)) or any(visit(t) for t in term))
    return visit(parse(body))


def analyze(output):
    summary = json.loads((output / "summary.json").read_text())
    plan = json.loads((output / "plan.json").read_text())["plan"]
    if not all(summary[k] for k in ("all_planned_outcomes_replayed",
                                    "no_host_glue_in_model_arm", "independent_libraries_verified")):
        raise ValueError("complete independent replay required")
    rows, costs, failures, rejected_replies = [], {a: Counter() for a in plan["arms"]}, [], []
    certificates = Counter()
    for row in summary["rows"]:
        record = json.loads((output / row["directory"] / "selection.json").read_text())
        effort, cost = record["effort"], costs[row["arm"]]
        certificates["promoted" if row["admitted"] else "rejected"] += len(record.get("certificates", []))
        cost["seconds"] += effort.get("seconds", 0)
        cost["attempted_tasks"] += bool(effort)
        if row["arm"] == "mechanical":
            for part in ("reuse", "proposer"):
                item = effort.get(part, {})
                cost[part + "_candidates"] += item.get("candidates", 0)
                cost["peak_rss_kib"] = max(cost["peak_rss_kib"], item.get("peak_process_group_rss_kib", 0))
        else:
            cost["reply_attempts"] += len(effort.get("attempts", []))
            for number, attempt in enumerate(effort.get("attempts", []), 1):
                if not attempt.get("training", {}).get("exact", False):
                    rejected_replies.append(dict(seed=row["seed"], task=row["task"], reply=number,
                        training=attempt.get("training"), error=attempt.get("error")))
        reused = {name: info for name, info in row.get("reuse", {}).items()
                  if info["hash_unchanged"] and info["used_cases"] and info["elided_correct"] < info["total"]}
        control = {n: v for n, v in reused.items()
                   if recursive(record["input_library"]["cells"][n]["body"], n)}
        rows.append(dict(**row, fully_verified=supported(row), necessary_recursive_reuse=control,
                         seconds=effort.get("seconds")))
        if effort and not row["admitted"]:
            failures.append(dict(seed=row["seed"], arm=row["arm"], task=row["task"],
                status=row["status"], stopped=effort.get("stopped", effort.get("proposer", {}).get("stopped")),
                checks=record.get("checks"),
                public_reply_scores=[a.get("training", a.get("error")) for a in effort.get("attempts", [])],
                mechanical_best_prefix=(effort.get("proposer", {}).get("best") or {}).get("correct")))
    receipts = [json.loads(p.read_text()) for p in output.glob("seed-*/codex/task-*/proposer/proposal-*/receipt.json")]
    requests = list(output.glob("seed-*/codex/task-*/proposer/proposal-*/request.json"))
    usage, audit = Counter(), Counter()
    for receipt in receipts:
        usage.update({k: v for k, v in (receipt.get("usage") or {}).items() if isinstance(v, int)})
    for row in rows:
        audit.update({k: v for k, v in row.get("audit", {}).items() if type(v) is int})
        for split in ("hidden", "stress"):
            audit.update({split + "_" + k: row.get(split, {}).get(k, 0) for k in ("correct", "total")})
    arms = {arm: dict(planned=sum(r["arm"] == arm for r in rows),
        admitted=sum(r["arm"] == arm and r["admitted"] for r in rows),
        fully_verified=sum(r["arm"] == arm and r["fully_verified"] for r in rows),
        blocked=sum(r["arm"] == arm and r["status"].startswith("blocked") for r in rows),
        **costs[arm]) for arm in plan["arms"]}
    curves = [dict(seed=seed, arm=arm,
        nodes=[r["marginal_nodes"] if r["fully_verified"] else None
               for r in rows if r["seed"] == seed and r["arm"] == arm])
        for seed in plan["seeds"] for arm in plan["arms"]]
    result = dict(analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arms=arms, curves=curves, rows=rows, failures=failures, rejected_public_replies=rejected_replies,
        usage=dict(usage), audit=dict(audit),
        verified_pushouts=summary["verified_pushouts"], completed_model_calls=len(receipts),
        certificates=dict(certificates),
        incomplete_model_calls=len(requests) - len(receipts),
        model_receipt_seconds=sum(r["seconds"] for r in receipts),
        peak_model_rss_kib=max((r["peak_process_group_rss_kib"] for r in receipts), default=0))
    save(output / "analysis.json", result)
    return result


def plot(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = sorted({c["seed"] for c in result["curves"]})
    fig, axes = plt.subplots(1, len(seeds), figsize=(10, 4), sharey=True, squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        for curve in result["curves"]:
            if curve["seed"] != seed:
                continue
            values = [v if v is not None else float("nan") for v in curve["nodes"]]
            label = "LLM proposals" if curve["arm"] == "codex" else "Mechanical search"
            ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
        ax.set(title=f"Data seed {seed}", xlabel="Task", xticks=range(1, 7), xlim=(.7, 6.3), ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    axes[0][0].set_ylabel("New expression nodes per verified attachment")
    fig.suptitle("Independent proposal processes; failed/blocked tasks are missing, not zero")
    fig.tight_layout()
    fig.savefig(output / "complexity.png", dpi=160)
    fig.savefig(output / "complexity.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    measured = analyze(args.output)
    if args.plot:
        plot(args.output, measured)
    print(json.dumps({k: v for k, v in measured.items() if k != "rows"}, indent=2))
