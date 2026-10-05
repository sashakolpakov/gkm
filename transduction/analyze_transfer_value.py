"""Post-selection accounting only; never called by a proposer or selector."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from interface_benchmark import save


def supported(row):
    return (row["admitted"] and row.get("hidden", {}).get("exact", False)
            and row.get("stress", {}).get("exact", False)
            and row.get("audit", {}).get("exact", False))


def payback(cold, retained, initial_cost):
    """No failed task is zero-cost, and no unmeasured saving is imputed."""
    if len(cold) != len(retained):
        raise ValueError("matched task sequences required")
    saving, reached, observed = 0, None, []
    complete_prefix = True
    for index, (a, b) in enumerate(zip(cold, retained), 3):
        if a is None or b is None:
            complete_prefix = False
        if complete_prefix:
            saving += a - b
            if reached is None and saving >= initial_cost:
                reached = index
            observed.append(saving)
        else:
            observed.append(None)
    return dict(cumulative_future_savings=observed, pays_back_at_task=reached)


def analyze(output):
    result = json.loads((output / "summary.json").read_text())
    plan = json.loads((output / "plan.json").read_text())["plan"]
    inputs = json.loads((output / "inputs.json").read_text())
    rows = result["rows"]
    if not result["all_planned_outcomes_replayed"] or len(rows) != len(plan["rows"]):
        raise ValueError("only a complete, replayed experiment can be summarized")
    cold = {(r["seed"], r["stage"]): r for r in rows if r["arm"] == "cold"}
    lineages = []
    for item in inputs:
        name = item["trial"]["directory"]
        past = [s["row"]["new_expression_nodes"] for s in item["stages"]]
        before = [r for r in rows if r["arm"] == "before" and r["lineage"] == name]
        after = [r for r in rows if r["arm"] == "after" and r["lineage"] == name]
        sequences = []
        for seed in plan["seeds"]:
            future = [next(r for r in after if r["seed"] == seed and r["stage"] == stage)
                      for stage, _ in plan["tasks"]]
            independent = [cold[seed, stage] for stage, _ in plan["tasks"]]
            a = [r["marginal_nodes"] if supported(r) else None for r in independent]
            b = [r["marginal_nodes"] if supported(r) else None for r in future]
            sequences.append(dict(seed=seed, retained_curve=past + b, cold_future=a,
                **payback(a, b, past[1])))
        extension_names = [p["new"] for p in item["extensions"]]
        elisions = [{name: r.get("reuse", {}).get(name) for name in extension_names} for r in after]
        lineages.append(dict(lineage=name, interface_extension=bool(item["extensions"]),
            before_successes=sum(map(supported, before)), before_total=len(before),
            after_successes=sum(map(supported, after)), after_total=len(after),
            sequences=sequences, extension_elisions=elisions))
    receipts = [json.loads(p.read_text()) for p in output.glob("seed-*/cold/task-*/proposer/proposal-*/receipt.json")]
    attempts = [p for p in output.glob("seed-*/cold/task-*/proposer/proposal-*/request.json")]
    usage = Counter()
    for receipt in receipts:
        usage.update({k: v for k, v in (receipt.get("usage") or {}).items() if isinstance(v, int)})
    reuse_memory = []
    search_seconds = Counter()
    for task in plan["rows"]:
        effort = json.loads((output / task["directory"] / "selection.json").read_text())["effort"]
        reuse_memory.append(effort["reuse"].get("peak_process_group_rss_kib", 0))
        search_seconds[task["arm"]] += effort.get("seconds", effort["reuse"].get("outer_seconds", 0))
    audit = Counter()
    for row in rows:
        for key in ("shapes_correct", "shapes_total", "continuations_correct", "continuations_total"):
            audit[key] += row.get("audit", {}).get(key, 0)
        for split in ("hidden", "stress"):
            for key in ("correct", "total"):
                audit[split + "_" + key] += row.get(split, {}).get(key, 0)
    measured = dict(analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arms={arm: dict(planned=sum(r["arm"] == arm for r in rows),
            admitted=sum(r["arm"] == arm and r["admitted"] for r in rows),
            fully_verified=sum(r["arm"] == arm and supported(r) for r in rows))
            for arm in ("cold", "before", "after")},
        lineages=lineages, audit=dict(audit), verified_pushouts=result["verified_pushouts"],
        failure_modes=dict(Counter(r["reuse_stopped"] for r in rows if not r["admitted"])),
        usage=dict(usage), completed_model_calls=len(receipts), incomplete_model_calls=len(attempts) - len(receipts),
        model_seconds=sum(r["seconds"] for r in receipts), search_seconds=dict(search_seconds),
        peak_model_rss_kib=max((r["peak_process_group_rss_kib"] for r in receipts), default=0),
        peak_reuse_rss_kib=max(reuse_memory, default=0),
        cold_nodes={str(seed): [cold[seed, stage]["marginal_nodes"] if supported(cold[seed, stage]) else None
            for stage, _ in plan["tasks"]] for seed in plan["seeds"]})
    save(output / "analysis.json", measured)
    return measured


def plot(output, measured):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, extension in zip(axes, (True, False)):
        for item in measured["lineages"]:
            if item["interface_extension"] != extension:
                continue
            curves = [s["retained_curve"] for s in item["sequences"]]
            values = [statistics.mean(v) if all(n is not None for n in v) else float("nan") for v in zip(*curves)]
            ax.plot(range(1, 6), values, marker="o", label=item["lineage"].replace("seed-", "S").replace("-trial-", "/"))
        columns = [[v for v in c if v is not None] for c in zip(*measured["cold_nodes"].values())]
        if columns:
            counts = "/".join(str(len(c)) for c in columns)
            ax.plot((3, 4, 5), [statistics.mean(c) if c else float("nan") for c in columns],
                    color="black", linestyle="--", label=f"Cold verified mean (n={counts})")
            ax.fill_between((3, 4, 5), [min(c) if c else float("nan") for c in columns],
                            [max(c) if c else float("nan") for c in columns], color="black", alpha=.10)
        ax.axvline(2.5, color="grey", linewidth=.8)
        ax.set_title("Interface extensions" if extension else "Earlier direct reuse")
        ax.set_xlabel("Task (1–2 frozen; 3–5 prospective)")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("New expression nodes per admitted task")
    fig.suptitle("Cost of acquisition/generalization, followed by retained-call transfer")
    fig.tight_layout()
    fig.savefig(output / "complexity.png", dpi=160)
    fig.savefig(output / "complexity.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = analyze(args.output)
    if args.plot:
        plot(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "lineages"}, indent=2))
