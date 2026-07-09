"""The RAWEST Bongard substrate: the proposer sees ONLY 12 rendered panels.

Sibling of `arc/crack_lab/gkm_arena.py` (house convention: siblings, not
modifications). The engine hands the agent 12 raw bitmaps -- 6 positive, 6
negative -- and nothing else: no action programs, no concept names, no shape
metadata. Everything else -- segmenting objects, measuring them, inventing the
predicates that separate the sides -- must be WRITTEN BY THE AGENT as
`predicates.py`. The harness then does the rule composition itself: an
exhaustive MDL conjunction search over the agent's predicates, verified by
rotated leave-one-out, priced by free energy. The human contribution is exactly
three things: (1) this thin raw harness, (2) a neutral static-vision
preconception prompt, (3) the verify-by-panels admission loop.

Rendering: our own deterministic pure-numpy rasterizer of Bongard-LOGO action
strings (turn/arc denormalization conventions match
`bongard/run_bongard_logo_adapter.py`). Stroke styles (normal/zigzag/...) are
collapsed to plain ink -- panels are a faithful visual realization of the
action programs, not pixel-identical to the published dataset. Determinism =>
bit-exact replays.

Verification protocol (pinned in bongard_crack_plan.md Section 8): the
proposer sees all 12 panels (as a human does) and writes only predicates. For
each of the 36 (pos_i, neg_j) holdouts the selector picks the min-F
conjunction using ONLY the other 10 panels and classifies the held-out pair;
R = error over all 72 held-out predictions. Solved = all 72 correct AND the
full-12-panel selection separates all panels. The articulated rule is the
full-panel winner; the rotation is the overfit guard.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PANEL_SIZE = 128
"""Panels are PANEL_SIZE x PANEL_SIZE uint8 grids, ink=1, background=0."""

CALL_COST = 1.0
"""Description-length fee for using a library predicate as a rule atom."""

BINDING_COST = 0.5
"""Fee for the atom's binding (comparison op + threshold), per the v3
priced-binding discipline: which measurement fills the slot is not free."""

LAMBDA_RULE = 0.1
"""Free-energy weight for rule complexity inside per-problem selection."""

MAX_RULE_ATOMS = 2
"""Conjunction size cap (matches the LOGO adapter's default)."""

MAX_CANDIDATE_ATOMS = 24
"""Search-budget cap: candidate atoms ranked by train separation (not a
post-hoc simplification; same status as the adapter's cap)."""


# ---------------------------------------------------------------------------
# Action-string geometry (conventions copied from run_bongard_logo_adapter.py)
# ---------------------------------------------------------------------------

def trace_shape(actions: Sequence[str]) -> List[Tuple[float, float]]:
    """Trace one shape's action strings into a polyline in shape coordinates.

    `line_<style>_<len>-<turn>` and `arc_<style>_<radius>_<arcangle>-<turn>`,
    all parameters normalized; turn = n*360-180 degrees, arc angle = n*720-360.
    Arcs are stepped at ~5 degrees so the rendered curve is smooth.
    """
    x = y = 0.0
    heading = 0.0
    points: List[Tuple[float, float]] = [(x, y)]
    for action in actions:
        movement, turn_s = action.split("-")
        parts = movement.split("_")
        heading += float(turn_s) * 360.0 - 180.0
        if parts[0] == "line":
            length = float(parts[2])
            rad = math.radians(heading)
            x += length * math.cos(rad)
            y += length * math.sin(rad)
            points.append((x, y))
        elif parts[0] == "arc":
            radius = float(parts[2])
            arc_angle = float(parts[3]) * 720.0 - 360.0
            steps = max(8, int(abs(arc_angle) // 5) + 1)
            step = arc_angle / steps
            chord = 2.0 * radius * math.sin(abs(math.radians(step)) / 2.0)
            for _ in range(steps):
                heading += step / 2.0
                rad = math.radians(heading)
                x += chord * math.cos(rad)
                y += chord * math.sin(rad)
                heading += step / 2.0
                points.append((x, y))
        else:
            raise ValueError(f"unsupported action string {action}")
    return points


def _transform(points: List[Tuple[float, float]], angle_deg: float,
               scale: float, tx: float, ty: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    rad = math.radians(angle_deg)
    rot = np.array([[math.cos(rad), -math.sin(rad)],
                    [math.sin(rad), math.cos(rad)]])
    return pts @ rot.T * scale + np.array([tx, ty])


def _draw_polyline(grid: np.ndarray, pts: np.ndarray) -> None:
    """Paint a polyline with ~1px strokes by dense sub-pixel sampling."""
    size = grid.shape[0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(seg_len / 0.3) + 1)
        ts = np.linspace(0.0, 1.0, n)
        xs = np.clip(np.rint(x0 + (x1 - x0) * ts).astype(int), 0, size - 1)
        ys = np.clip(np.rint(y0 + (y1 - y0) * ts).astype(int), 0, size - 1)
        grid[ys, xs] = 1


def render_panel(image_program: Sequence[Sequence[str]],
                 rng: np.random.RandomState,
                 size: int = PANEL_SIZE) -> np.ndarray:
    """Render one image (a list of shapes) to a size x size uint8 panel.

    Placement is seeded: each shape gets a random rotation, a scale that fits
    it to a fraction of the canvas, and a translation keeping it inside a
    margin; with two shapes, a few rejection attempts reduce bbox overlap.
    """
    grid = np.zeros((size, size), dtype=np.uint8)
    margin = size * 0.08
    n_shapes = max(1, len(image_program))
    target_frac = 0.55 if n_shapes == 1 else 0.38
    placed_boxes: List[Tuple[float, float, float, float]] = []
    for shape_actions in image_program:
        raw = trace_shape(shape_actions)
        angle = rng.uniform(0.0, 360.0)
        base = _transform(raw, angle, 1.0, 0.0, 0.0)
        extent = max(float(np.ptp(base[:, 0])), float(np.ptp(base[:, 1])), 1e-6)
        scale = size * target_frac * rng.uniform(0.85, 1.0) / extent
        pts0 = base * scale
        w, h = float(np.ptp(pts0[:, 0])), float(np.ptp(pts0[:, 1]))
        best = None
        for _attempt in range(8):
            tx = rng.uniform(margin - pts0[:, 0].min(),
                             size - margin - w - pts0[:, 0].min())
            ty = rng.uniform(margin - pts0[:, 1].min(),
                             size - margin - h - pts0[:, 1].min())
            box = (pts0[:, 0].min() + tx, pts0[:, 1].min() + ty,
                   pts0[:, 0].max() + tx, pts0[:, 1].max() + ty)
            overlap = any(not (box[2] < b[0] or b[2] < box[0]
                               or box[3] < b[1] or b[3] < box[1])
                          for b in placed_boxes)
            best = (tx, ty, box)
            if not overlap:
                break
        tx, ty, box = best
        placed_boxes.append(box)
        _draw_polyline(grid, pts0 + np.array([tx, ty]))
    return grid


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """One rendered Bongard problem. `concept` is ground truth and stays
    HARNESS-SIDE ONLY: it is never written into a proposer workspace."""
    problem_id: str
    category: str
    concept: str
    pos: List[np.ndarray]
    neg: List[np.ndarray]

    def panels(self) -> List[Tuple[np.ndarray, bool]]:
        return [(p, True) for p in self.pos] + [(p, False) for p in self.neg]


def _panel_rng(seed: int, problem_id: str, side: str, index: int) -> np.random.RandomState:
    key = f"{seed}:{problem_id}:{side}:{index}".encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:4], "big"))


def sample_problems(dataset_dir: str, limit: int = 10, seed: int = 0,
                    source: str = "basic",
                    panel_size: int = PANEL_SIZE) -> List[Problem]:
    """Fresh-seed Bongard-LOGO problems rendered to raw panels.

    Fresh seeds are the leakage defense: generated instances cannot be
    memorized. Only the harness sees concept names.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)
    from bongard.sampler.basic_sampler import BasicSampler  # type: ignore
    from bongard.sampler.abstract_sampler import AbstractSampler  # type: ignore
    from bongard.util_funcs import (  # type: ignore
        get_attribute_sampling_candidates, get_shape_super_classes)

    shapes_tsv = os.path.join(dataset_dir, "data", "human_designed_shapes.tsv")
    attrs_tsv = os.path.join(dataset_dir, "data",
                             "human_designed_shapes_attributes.tsv")
    rng = np.random.RandomState(seed)
    problems: List[Problem] = []

    def render_problem(pid: str, category: str, concept: str,
                       program) -> Problem:
        pos = [render_panel(img, _panel_rng(seed, pid, "pos", i), panel_size)
               for i, img in enumerate(program[0][:6])]
        neg = [render_panel(img, _panel_rng(seed, pid, "neg", i), panel_size)
               for i, img in enumerate(program[1][:6])]
        return Problem(pid, category, concept, pos, neg)

    if source in ("basic", "both"):
        shape_list = list(get_shape_super_classes(shapes_tsv).keys())
        order = rng.permutation(len(shape_list))
        sampler = BasicSampler(shapes_tsv, attrs_tsv,
                               num_positive_examples=6,
                               num_negative_examples=6, random_state=rng)
        for idx in order[:limit]:
            shape = shape_list[int(idx)]
            sampled = sampler.sample([shape], int(idx))
            problems.append(render_problem(
                sampled.get_problem_name(), "basic", shape,
                sampled.get_action_string_list()))

    if source in ("abstract", "both"):
        candidates = get_attribute_sampling_candidates(attrs_tsv)
        attr_list = [a for a in candidates
                     if len(candidates[a][0]) >= 6 and len(candidates[a][1]) >= 6]
        order = rng.permutation(len(attr_list))
        sampler = AbstractSampler(shapes_tsv, attrs_tsv,
                                  num_positive_examples=6,
                                  num_negative_examples=6, random_state=rng)
        count = 0
        for idx in order:
            if count >= limit:
                break
            attr = attr_list[int(idx)]
            sampled = sampler.sample([attr], int(idx))
            if sampled is None:
                continue
            problems.append(render_problem(
                sampled.get_problem_name(), "abstract", attr,
                sampled.get_action_string_list()))
            count += 1

    return problems


# ---------------------------------------------------------------------------
# Predicate loading (the proposer's contribution)
# ---------------------------------------------------------------------------

def load_predicates(path: str) -> Dict[str, Callable]:
    """Load proposer-authored predicates: module-level callables named `p_*`
    taking one panel (2D uint8 array) and returning a number or bool."""
    spec = importlib.util.spec_from_file_location("agent_predicates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    preds = {name: fn for name, fn in vars(module).items()
             if name.startswith("p_") and callable(fn)}
    return dict(sorted(preds.items()))


def predicate_values(preds: Dict[str, Callable],
                     panels: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[str], int]:
    """Evaluate every predicate on every panel -> (matrix, names, n_errors).

    A predicate that raises or returns a non-finite value scores 0.0 for that
    panel; errors are counted and surface in the RESULT line so the proposer
    can fix its own crashes."""
    names = list(preds)
    values = np.zeros((len(panels), len(names)), dtype=float)
    errors = 0
    for j, name in enumerate(names):
        fn = preds[name]
        for i, panel in enumerate(panels):
            try:
                v = float(fn(panel.copy()))
                if not math.isfinite(v):
                    raise ValueError("non-finite")
                values[i, j] = v
            except Exception:
                errors += 1
    return values, names, errors


# ---------------------------------------------------------------------------
# MDL rule selection (the harness's contribution -- exhaustive, not sampled)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    name: str
    op: str        # '>=' or '<='
    threshold: float

    def holds(self, value: float) -> bool:
        return value >= self.threshold if self.op == ">=" else value <= self.threshold

    def describe(self) -> str:
        return f"{self.name}{self.op}{self.threshold:.4g}"


@dataclass(frozen=True)
class Rule:
    atoms: Tuple[Atom, ...] = ()
    constant: Optional[bool] = None

    def predict(self, row: Dict[str, float]) -> bool:
        if self.constant is not None:
            return self.constant
        return all(a.holds(row[a.name]) for a in self.atoms)

    def cost(self) -> float:
        return len(self.atoms) * (CALL_COST + BINDING_COST)

    def describe(self) -> str:
        if self.constant is not None:
            return f"CONST_{self.constant}"
        return " AND ".join(a.describe() for a in self.atoms) or "CONST_True"


def _candidate_atoms(values: np.ndarray, names: List[str],
                     labels: np.ndarray,
                     max_candidates: int = MAX_CANDIDATE_ATOMS) -> List[Atom]:
    atoms: List[Tuple[float, Atom]] = []
    for j, name in enumerate(names):
        col = values[:, j]
        uniq = np.unique(col)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for t in thresholds:
            for op in (">=", "<="):
                atom = Atom(name, op, float(t))
                pred = col >= t if op == ">=" else col <= t
                acc = float((pred == labels).mean())
                if acc > 0.5:
                    atoms.append((acc, atom))
    atoms.sort(key=lambda p: (-p[0], p[1].name, p[1].op, p[1].threshold))
    return [a for _, a in atoms[:max_candidates]]


def select_rule(values: np.ndarray, names: List[str], labels: np.ndarray,
                lam: float = LAMBDA_RULE,
                max_atoms: int = MAX_RULE_ATOMS,
                max_candidates: int = MAX_CANDIDATE_ATOMS) -> Rule:
    """Exhaustive min-F conjunction over candidate atoms.

    F = train_error + lam * rule_cost. Ties break toward lower cost, then
    lexical description, so the selected rule is deterministic."""
    name_idx = {n: i for i, n in enumerate(names)}

    def rule_error(rule: Rule) -> float:
        if rule.constant is not None:
            pred = np.full(len(labels), rule.constant)
        else:
            pred = np.ones(len(labels), dtype=bool)
            for a in rule.atoms:
                col = values[:, name_idx[a.name]]
                pred &= (col >= a.threshold) if a.op == ">=" else (col <= a.threshold)
        return float((pred != labels).mean())

    candidates = _candidate_atoms(values, names, labels, max_candidates)
    best: Tuple[float, float, str, Rule] = None  # type: ignore[assignment]
    for rule in itertools.chain(
            (Rule(constant=True), Rule(constant=False)),
            (Rule(atoms=c) for size in range(1, max_atoms + 1)
             for c in itertools.combinations(candidates, size))):
        f = rule_error(rule) + lam * rule.cost()
        key = (f, rule.cost(), rule.describe())
        if best is None or key < (best[0], best[1], best[2]):
            best = (key[0], key[1], key[2], rule)
    return best[3]


# ---------------------------------------------------------------------------
# Verification: rotated leave-one-out
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    solved: bool
    heldout_accuracy: float
    train_accuracy: float
    rule: str
    rule_cost: float
    predicate_errors: int
    n_rotations: int

    def result_line(self) -> str:
        return (f"RESULT solved={self.solved} heldout={self.heldout_accuracy:.3f} "
                f"train={self.train_accuracy:.3f} rule=\"{self.rule}\" "
                f"rule_cost={self.rule_cost:.1f} predicate_errors={self.predicate_errors}")


def verify(preds: Dict[str, Callable], problem: Problem,
           lam: float = LAMBDA_RULE,
           max_atoms: int = MAX_RULE_ATOMS) -> VerifyResult:
    """The pure verifier: predicates + panels -> solved or not.

    Deterministic; re-running IS the replay validation."""
    panels = [p for p, _ in problem.panels()]
    labels = np.array([lab for _, lab in problem.panels()])
    if not preds:
        return VerifyResult(False, 0.5, 0.5, "CONST_True", 0.0, 0, 36)
    values, names, errors = predicate_values(preds, panels)

    full_rule = select_rule(values, names, labels, lam, max_atoms)
    name_idx = {n: i for i, n in enumerate(names)}

    def predict(rule: Rule, i: int) -> bool:
        row = {n: values[i, name_idx[n]] for n in names}
        return rule.predict(row)

    train_ok = [predict(full_rule, i) == labels[i] for i in range(12)]
    train_accuracy = float(np.mean(train_ok))

    correct = 0
    total = 0
    n_pos = len(problem.pos)
    for i in range(n_pos):
        for j in range(len(problem.neg)):
            held = {i, n_pos + j}
            mask = np.array([k not in held for k in range(12)])
            rule_ij = select_rule(values[mask], names, labels[mask], lam, max_atoms)
            correct += int(predict(rule_ij, i) == labels[i])
            correct += int(predict(rule_ij, n_pos + j) == labels[n_pos + j])
            total += 2
    heldout_accuracy = correct / total if total else 0.0
    solved = heldout_accuracy == 1.0 and train_accuracy == 1.0
    return VerifyResult(solved, heldout_accuracy, train_accuracy,
                        full_rule.describe(), full_rule.cost(), errors,
                        n_pos * len(problem.neg))


def free_energy(solved: int, total_marginal_C: float, lam: float = 0.02) -> float:
    """Corpus-level F = R + lambda*C with R = -problems_solved, C = total
    marginal novelty in the predicate library (same shape as gkm_legs)."""
    return -float(solved) + lam * float(total_marginal_C)


# ---------------------------------------------------------------------------
# Workspace panel IO (what the proposer is allowed to see)
# ---------------------------------------------------------------------------

def write_panels(ws: str, problem: Problem, opaque_id: str) -> str:
    """Write the 12 panels into the workspace under an OPAQUE id.

    Sampler problem names contain concept names (e.g. `bd_..._triangle_...`)
    and must never reach the proposer; the caller supplies `opaque_id` like
    `problem_03`. Panels are saved as .npy (exact) and .png (viewable)."""
    pdir = os.path.join(ws, opaque_id)
    os.makedirs(pdir, exist_ok=True)
    for side, arrs in (("pos", problem.pos), ("neg", problem.neg)):
        for i, arr in enumerate(arrs):
            np.save(os.path.join(pdir, f"{side}_{i}.npy"), arr)
            try:
                from PIL import Image
                Image.fromarray((255 - arr * 255).astype(np.uint8)).save(
                    os.path.join(pdir, f"{side}_{i}.png"))
            except Exception:
                pass
    return pdir
