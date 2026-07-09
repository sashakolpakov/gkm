"""Enforced predicate-library orchestration for the Bongard crack.

Sibling of `arc/crack_lab/gkm_legs.py` -- the idioms are reused (LOC+literal
description-length proxy, marginal-C accounting, validated-checkpoint
promotion gating, WIP snapshots, workspace taint markers), the code is fresh
because the ARC types (levels, paths, replay) do not apply: here VERIFY is a
pure function of (predicates source, panels), so re-running the verifier IS
the replay validation.

The discipline, enforced structurally (not requested in a prompt):

  * Logic can only accumulate in the SHARED library `predicates.py`
    (module-level `p_*(panel) -> float|bool` callables). The harness does the
    rule composition itself (exhaustive MDL conjunction search, bongard_arena).
  * Per problem k: PROPOSE (extend predicates.py, minimal new structure) ->
    VERIFY (rotated-LOO on the real panels) -> DEBRIEF (refactor repeats,
    log in predicates_log.md).
  * Admission is structural: predicates.py growth is kept ONLY when the
    problem verifies as solved; a failed attempt's library edits are reverted
    (saved as WIP context, never admitted). F = R + lambda*C_marginal with
    C_marginal = admitted growth of the library; a reused predicate is free.
  * Proposer ladder: Sonnet-first, Opus escalation after N failed rounds;
    every escalation is logged (which problems NEED a strong proposer is a
    novelty signal alongside marginal C).

The proposer and verifier are injectable (`propose_fn`) so the control loop
and accounting are unit-testable offline; the default proposer invokes the
real headless Claude Code agent with tools. Concept names (ground truth) are
never written into the workspace; they live only in the harness-side
results.json in the artifact directory.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_arena as A

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_FILE = "predicates.py"
LOG_FILE = "predicates_log.md"
CHECKPOINT_FILE = "checkpoint.json"
PROMOTED_FILES = (LIBRARY_FILE, LOG_FILE, CHECKPOINT_FILE)

DEFAULT_LADDER = ("sonnet", "sonnet", "opus")
"""Proposer escalation ladder: model per attempt round."""

SOURCE_TAINT_MARKERS = (
    "downloads/bongard-logo",
    "get_action_string_list",
    "human_designed_shapes",
    "basic_sampler",
    "abstract_sampler",
    "action_program",
    "results.json",
)
"""Markers whose presence in a proposer workspace file makes the attempt
inadmissible: they evidence reading the dataset/sampler/ground-truth side."""


class WorkspaceTainted(RuntimeError):
    """The proposer workspace evidences forbidden dataset/ground-truth use."""


PRECONCEPTIONS = """\
You are solving a Bongard problem. You see 12 small images: six in `pos_*`
and six in `neg_*`. All six positive images satisfy a single hidden rule;
all six negative images violate it. The two sides are deliberate near-misses
of each other, and the hidden rule is SIMPLE -- it is the shortest natural
description that separates the sides.

General preconceptions you may carry (nothing problem-specific is given):
each image is a line drawing containing one or more drawn objects on an
empty background. What tends to matter in such problems are properties of
the objects and relations between them -- how many there are, how large,
how they are shaped, where they sit, how they are oriented, and how they
relate to one another. Which of these matters here, and how to measure it
from raw pixels, is yours to discover by experiment.
"""

TESTER = '''import sys
sys.path.insert(0, {labdir!r})
import glob, os
import numpy as np
import bongard_arena as A

ws = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.join(ws, open(os.path.join(ws, "current_problem.txt")).read().strip())
pos = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "pos_*.npy")))]
neg = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "neg_*.npy")))]
problem = A.Problem("current", "?", "?", pos, neg)
preds = A.load_predicates(os.path.join(ws, "predicates.py"))
print(A.verify(preds, problem).result_line())
'''


# ---------------------------------------------------------------------------
# Description-length proxy (idiom from gkm_legs)
# ---------------------------------------------------------------------------

def _loc(code: str) -> int:
    return sum(1 for ln in (code or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def _literal_cost(code: str) -> int:
    """Large literals (lookup tables of panel answers) must carry MDL cost
    even when formatted on one line."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return 0
    cost = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            cost += len(node.elts)
        elif isinstance(node, ast.Dict):
            cost += len(node.keys)
    return cost


def description_complexity(code: str) -> int:
    return _loc(code) + _literal_cost(code)


def marginal_complexity(before: str, after: str) -> int:
    """Admitted growth of the shared library. Reuse is free; only novelty
    is paid for."""
    return max(0, description_complexity(after) - description_complexity(before))


# ---------------------------------------------------------------------------
# Taint check
# ---------------------------------------------------------------------------

def _workspace_taint_reason(ws: str) -> Optional[str]:
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache"}]
        for name in files:
            if name.endswith((".npy", ".png")):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().lower()
            except OSError:
                continue
            for marker in SOURCE_TAINT_MARKERS:
                if marker in text:
                    return f"{marker} in {os.path.relpath(path, ws)}"
    return None


def assert_workspace_not_tainted(ws: str) -> None:
    reason = _workspace_taint_reason(ws)
    if reason:
        raise WorkspaceTainted(
            f"forbidden dataset/ground-truth access tainted workspace: {reason}")


# ---------------------------------------------------------------------------
# Records, checkpoint, artifact
# ---------------------------------------------------------------------------

@dataclass
class ProblemRecord:
    opaque_id: str
    solved: bool
    heldout_accuracy: float
    rule: str
    rule_cost: float
    marginal_C: int
    model: str
    attempts: int
    escalated: bool


@dataclass
class Report:
    tag: str
    records: List[ProblemRecord] = field(default_factory=list)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.records if r.solved)

    @property
    def total_marginal_C(self) -> int:
        return sum(r.marginal_C for r in self.records)

    @property
    def free_energy(self) -> float:
        return A.free_energy(self.solved, self.total_marginal_C)

    def to_json(self) -> dict:
        return {"tag": self.tag, "solved": self.solved,
                "total_marginal_C": self.total_marginal_C,
                "free_energy": self.free_energy,
                "records": [asdict(r) for r in self.records]}


def artifact_dir(tag: str) -> str:
    return os.path.join(LAB_DIR, "agent_solutions", f"{tag}_predicates")


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _save_checkpoint(directory: str, rep: Report) -> None:
    with open(os.path.join(directory, CHECKPOINT_FILE), "w") as f:
        json.dump(rep.to_json(), f, indent=2)


def _load_checkpoint(directory: str) -> Optional[Report]:
    path = os.path.join(directory, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return Report(tag=data.get("tag", ""),
                  records=[ProblemRecord(**r) for r in data.get("records", [])])


def seed_workspace_from_artifact(tag: str, ws: str, verbose: bool = True) -> Optional[Report]:
    """Scratch is disposable; the promoted artifact is the source of truth."""
    art = artifact_dir(tag)
    rep = _load_checkpoint(art)
    if rep is None:
        return None
    for name in PROMOTED_FILES:
        src = os.path.join(art, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ws, name))
    if verbose:
        print(f"seeded workspace from artifact: {art} "
              f"(solved={rep.solved}, C={rep.total_marginal_C})")
    return rep


def promote_verified_artifact(tag: str, ws: str, rep: Report,
                              results: Dict[str, dict],
                              verbose: bool = True) -> bool:
    """Publish the current verified library state. Gated on the taint check;
    verification itself is re-run by the caller (pure function = replay)."""
    assert_workspace_not_tainted(ws)
    art = artifact_dir(tag)
    os.makedirs(art, exist_ok=True)
    _save_checkpoint(ws, rep)
    for name in PROMOTED_FILES:
        src = os.path.join(ws, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(art, name))
    # Ground truth stays harness-side: results.json exists ONLY in the
    # artifact dir, never in the workspace.
    with open(os.path.join(art, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(art, "README.md"), "w") as f:
        f.write(
            f"# {tag} predicate-library artifact\n\n"
            "Latest verified predicate-library state promoted by "
            "`bongard_legs.py`. Re-running `bongard_arena.verify` on the "
            "recorded problems with this `predicates.py` reproduces every "
            "solved verdict (deterministic substrate = replay).\n\n"
            f"- Problems solved: {rep.solved}/{len(rep.records)}\n"
            f"- Total marginal C: {rep.total_marginal_C}\n"
            f"- F = {rep.free_energy:.3f}\n\n"
            "Per-problem novelty:\n\n"
            + "\n".join(f"- {r.opaque_id}: solved={r.solved} "
                        f"marginal_C={r.marginal_C} model={r.model}"
                        for r in rep.records) + "\n")
    if verbose:
        print(f"promoted artifact: {art}")
    return True


def snapshot_wip(tag: str, ws: str, opaque_id: str, verbose: bool = True) -> str:
    """Preserve a failed attempt's workspace files (including the reverted
    library candidate) without admitting them."""
    dst = os.path.join(artifact_dir(tag), "wip_context", opaque_id,
                       time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(ws)):
        path = os.path.join(ws, name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(dst, name))
    if verbose:
        print(f"saved WIP context: {dst}")
    return dst


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------

def build_task(opaque_id: str, tester_cmd: str) -> str:
    return (
        PRECONCEPTIONS
        + f"\nThe current problem's panels are in `{opaque_id}/` in this directory:"
        " `pos_0..5.npy` and `neg_0..5.npy` are 128x128 uint8 arrays"
        " (ink=1, background=0); the `.png` files show the same content"
        " (ink drawn dark on white).\n\n"
        "WORKFLOW (you have Bash/Read/Write/Edit tools in this directory):\n"
        "1. Look at the panels (load the .npy arrays with numpy; view the pngs).\n"
        "2. Write or EXTEND `predicates.py`: module-level pure functions\n"
        "   `p_<name>(panel) -> float | bool` measuring properties of a single\n"
        "   panel. Allowed imports: numpy, math, scipy. Predicates must be\n"
        "   deterministic and must not read files or globals.\n"
        f"3. Run `{tester_cmd}` and read its RESULT line. The harness composes\n"
        "   your predicates into a minimal separating rule itself (an exhaustive\n"
        "   MDL search with leave-one-out rotation); you never write the rule.\n"
        "   `predicate_errors` counts crashes of your predicates -- fix them.\n"
        "4. ITERATE until RESULT shows solved=True.\n\n"
        "GROW A LIBRARY (this is the point -- minimise novelty):\n"
        "- `predicates.py` is a persistent SHARED library that carries over to\n"
        "  later problems. REUSE existing predicates where possible; add as FEW\n"
        "  new ones as possible. A reused predicate is already paid for; only\n"
        "  NEW code costs (free energy F = R + lambda*C_marginal).\n"
        "- After solving, DEBRIEF: refactor any near-duplicate predicates into\n"
        "  one generic parameterized measurement, note the recurring pattern in\n"
        "  `predicates_log.md`, and re-run the tester to confirm solved=True\n"
        "  still holds.\n\n"
        "CONSTRAINTS: work ONLY inside this directory. Do not search the parent\n"
        "repository, the internet, or any dataset; everything you need is in\n"
        "the panels. Hard-coding per-panel answers (lookup tables keyed on\n"
        "panel identity) is charged prohibitively by the complexity accounting\n"
        "and will not generalize across the leave-one-out rotation.\n")


def claude_propose(task: str, ws: str, model: str, minutes: int = 15,
                   verbose: bool = True) -> None:
    """Default proposer: the real headless Claude Code agent with tools."""
    cmd = ["claude", "-p", task,
           "--allowedTools", "Bash", "Read", "Write", "Edit",
           "--dangerously-skip-permissions",
           "--model", model,
           "--output-format", "text"]
    if verbose:
        print(f"invoking {model} proposer in {ws} (up to {minutes} min)")
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                              timeout=minutes * 60)
        if verbose:
            print("=== proposer transcript (tail) ===")
            print((proc.stdout or "")[-1500:])
    except subprocess.TimeoutExpired:
        print(f"proposer timed out after {minutes} min")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _verify_workspace(ws: str, problem: A.Problem) -> A.VerifyResult:
    lib = os.path.join(ws, LIBRARY_FILE)
    if not os.path.exists(lib):
        return A.VerifyResult(False, 0.5, 0.5, "CONST_True", 0.0, 0, 36)
    try:
        preds = A.load_predicates(lib)
    except Exception:
        return A.VerifyResult(False, 0.5, 0.5, "LOAD_ERROR", 0.0, 0, 36)
    return A.verify(preds, problem)


def run(problems: Sequence[A.Problem], tag: str = "logo",
        ws: Optional[str] = None,
        propose_fn: Callable[[str, str, str, int], None] = None,
        ladder: Sequence[str] = DEFAULT_LADDER,
        minutes: int = 15, verbose: bool = True) -> Report:
    """PROPOSE -> VERIFY -> DEBRIEF over a problem sequence, with structural
    admission and Sonnet-first escalation. Resumable: solved problems in the
    promoted artifact are not re-run."""
    propose = propose_fn or (lambda task, w, model, mins:
                             claude_propose(task, w, model, mins, verbose))
    ws = ws or os.path.join("/tmp", f"bongard_ws_{tag}")
    os.makedirs(ws, exist_ok=True)
    prior = seed_workspace_from_artifact(tag, ws, verbose=verbose)
    rep = prior if prior is not None and prior.tag == tag else Report(tag=tag)
    done = {r.opaque_id for r in rep.records if r.solved}

    art = artifact_dir(tag)
    results: Dict[str, dict] = {}
    if os.path.exists(os.path.join(art, "results.json")):
        results = json.loads(_read(os.path.join(art, "results.json")) or "{}")

    with open(os.path.join(ws, "bongard_try.py"), "w") as f:
        f.write(TESTER.format(labdir=LAB_DIR))
    lib_path = os.path.join(ws, LIBRARY_FILE)
    if not os.path.exists(lib_path):
        with open(lib_path, "w") as f:
            f.write("# Shared predicate library. p_<name>(panel) -> float | bool\n")

    tester_cmd = f"{sys.executable} bongard_try.py"
    for k, problem in enumerate(problems):
        oid = f"problem_{k:02d}"
        if oid in done:
            continue
        A.write_panels(ws, problem, oid)
        with open(os.path.join(ws, "current_problem.txt"), "w") as f:
            f.write(oid)
        lib_before = _read(lib_path)

        result = None
        model_used = ladder[0]
        attempts = 0
        for model in ladder:
            attempts += 1
            model_used = model
            propose(build_task(oid, tester_cmd), ws, model, minutes)
            assert_workspace_not_tainted(ws)
            result = _verify_workspace(ws, problem)
            if verbose:
                print(f"{oid} attempt {attempts} ({model}): {result.result_line()}")
            if result.solved:
                break

        lib_after = _read(lib_path)
        if result is not None and result.solved:
            marginal = marginal_complexity(lib_before, lib_after)
        else:
            # Structural admission: failed attempts do not grow the library.
            snapshot_wip(tag, ws, oid, verbose=verbose)
            with open(lib_path, "w") as f:
                f.write(lib_before)
            marginal = 0

        record = ProblemRecord(
            opaque_id=oid,
            solved=bool(result and result.solved),
            heldout_accuracy=result.heldout_accuracy if result else 0.0,
            rule=result.rule if result else "",
            rule_cost=result.rule_cost if result else 0.0,
            marginal_C=marginal,
            model=model_used,
            attempts=attempts,
            escalated=attempts > 1,
        )
        rep.records = [r for r in rep.records if r.opaque_id != oid] + [record]
        rep.records.sort(key=lambda r: r.opaque_id)
        results[oid] = {"problem_id": problem.problem_id,
                        "category": problem.category,
                        "concept": problem.concept,
                        "solved": record.solved,
                        "rule": record.rule}
        _save_checkpoint(ws, rep)
        promote_verified_artifact(tag, ws, rep, results, verbose=verbose)

    if verbose:
        print(f"=== {tag}: solved {rep.solved}/{len(rep.records)} | "
              f"total_marginal_C={rep.total_marginal_C} | F={rep.free_energy:.3f} ===")
        print("marginal-C trace: "
              + ", ".join(f"{r.opaque_id.split('_')[1]}:{r.marginal_C}"
                          for r in rep.records))
    return rep


if __name__ == "__main__":
    dataset = os.path.join(LAB_DIR, "..", "..", "downloads", "Bongard-LOGO")
    limit, seed, source, tag, minutes = 3, 20260709, "basic", "logo", 15
    ladder: Sequence[str] = DEFAULT_LADDER
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
        elif a.startswith("--source="):
            source = a.split("=", 1)[1]
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        elif a.startswith("--minutes="):
            minutes = int(a.split("=", 1)[1])
        elif a.startswith("--ladder="):
            ladder = a.split("=", 1)[1].split(",")
    problems = A.sample_problems(dataset, limit=limit, seed=seed, source=source)
    run(problems, tag=tag, ladder=ladder, minutes=minutes)
