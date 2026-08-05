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
import itertools
import math
import os
import stat
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import predicate_pricing as predicate_price

PANEL_SIZE = 128
"""Panels are PANEL_SIZE x PANEL_SIZE uint8 grids, ink=1, background=0."""

PANEL_FILE_SUFFIXES = (".npy", ".png")
"""The complete, harness-owned presentation set for one opaque problem."""

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

FLAT_PRICING = "flat"
SHARED_PRICING = "shared"
NO_SHARE_PRICING = "no-share"
PRICED_SELECTION_POLICY = "risk-then-cost/v2"

AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS = 90.0
"""Parent wall-clock deadline for one unrestricted verification child."""

VERIFIER_CHILD_CPU_LIMIT_SECONDS = 80
"""Child CPU ceiling, deliberately below the parent wall-clock deadline."""

VERIFIER_PREDICATE_LINE_EVENT_LIMIT = 1_000_000
"""Deterministic Python-line budget across one predicate verification."""

VERIFIER_MEMORY_HEADROOM_BYTES = 1 << 30
"""Additional AS/DATA allowance above the child's measured virtual size."""

VERIFIER_RESOURCE_LIMIT_POLICY_ID = \
    "predicate-line-budget-plus-child-rlimit-cpu-as-data/v3"


def verifier_resource_limit_policy() -> dict:
    """Return the deterministic contract for child-process containment."""
    return {
        "policy_id": VERIFIER_RESOURCE_LIMIT_POLICY_ID,
        "parent_wall_timeout_seconds": AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS,
        "child_cpu_limit_seconds": VERIFIER_CHILD_CPU_LIMIT_SECONDS,
        "predicate_python_line_event_limit": (
            VERIFIER_PREDICATE_LINE_EVENT_LIMIT),
        "predicate_line_scope": "compiled-predicate-source-filename",
        "memory_resources": ["RLIMIT_AS", "RLIMIT_DATA"],
        "memory_baseline": "current-process-virtual-memory-bytes",
        "memory_headroom_bytes": VERIFIER_MEMORY_HEADROOM_BYTES,
        "platform_probes": {
            "Linux": "/proc/self/statm:first-field-times-page-size",
            "Darwin": "mach-task-basic-info:virtual-size",
        },
        "limit_application": (
            "exact-soft-and-hard-target;lower-inherited-limit-fails-closed"),
        "unsupported_platform": "fail-closed",
    }


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

def load_predicates_source(
        source: str, *, filename: str = "<agent_predicates>") -> Dict[str, Callable]:
    """Execute predicates from the exact source bytes supplied for pricing.

    Compiling the source directly avoids filesystem/bytecode-cache races in
    which pricing sees one revision while import machinery executes another.
    Runtime/AST parity is enforced by :func:`verify_priced_source`.
    """
    # The loader is public, so enforce the same positive capability gate even
    # when it is used outside verify_priced_source.  The hidden import hook is
    # present only so AST-approved import statements can execute; predicate
    # source cannot name or recover it.
    predicate_price.build_pricing_model(source, filename=filename)
    namespace = {
        "__builtins__": predicate_price.predicate_execution_builtins(),
        "__file__": filename,
        "__name__": "agent_predicates",
        "__package__": None,
    }
    exec(compile(source, filename, "exec"), namespace)
    preds = {name: fn for name, fn in namespace.items()
             if name.startswith("p_") and callable(fn)}
    return dict(sorted(preds.items()))


def load_predicates(path: str) -> Dict[str, Callable]:
    """Load proposer-authored ``p_*`` callables from exact file bytes."""
    source = predicate_price.read_predicate_source(path)
    return load_predicates_source(source, filename=path)


class PredicateEvaluationError(ValueError):
    """A predicate cannot be used as deterministic scientific evidence."""


def _strict_predicate_value(
        name: str, fn: Callable, panel: np.ndarray, panel_index: int) -> float:
    """Evaluate one isolated panel call and require a finite scalar result."""
    try:
        value = float(fn(panel.copy()))
    except BaseException as exc:
        raise PredicateEvaluationError(
            f"predicate {name!r} failed on panel {panel_index}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise PredicateEvaluationError(
            f"predicate {name!r} returned a non-finite value on panel "
            f"{panel_index}")
    return value


def predicate_values(preds: Dict[str, Callable],
                     panels: Sequence[np.ndarray], *,
                     strict: bool = False) -> Tuple[np.ndarray, List[str], int]:
    """Evaluate every predicate on every panel -> (matrix, names, n_errors).

    In the legacy diagnostic mode, a predicate that raises or returns a
    non-finite value scores 0.0 for that panel and the error is counted.  The
    authoritative priced path uses ``strict=True`` and rejects such a
    predicate instead: an exception is not a legitimate negative signal."""
    names = list(preds)
    values = np.zeros((len(panels), len(names)), dtype=float)
    errors = 0
    for j, name in enumerate(names):
        fn = preds[name]
        for i, panel in enumerate(panels):
            if strict:
                values[i, j] = _strict_predicate_value(name, fn, panel, i)
                continue
            try:
                v = float(fn(panel.copy()))
                if not math.isfinite(v):
                    raise ValueError("non-finite")
                values[i, j] = v
            except Exception:
                errors += 1
    return values, names, errors


def audit_predicate_purity(
        source: str,
        panels: Sequence[np.ndarray],
        expected_names: Sequence[str],
        *,
        filename: str = "<agent_predicates>") \
        -> Tuple[Dict[str, Callable], np.ndarray, List[str]]:
    """Require panel-stable outputs across fresh modules and call orders.

    Static definition pricing deliberately permits useful module constants and
    helper objects.  That also means a syntactically valid predicate could
    mutate module state and classify the first six calls as positive without
    inspecting a panel.  We close that evaluation-order channel by executing
    the exact source afresh under canonical, reversed, and content-keyed call
    schedules.  Every panel is therefore a repeated observation from three
    isolated executions.  One deterministic sentinel panel per predicate is
    also repeated inside the canonical module to expose mutable caches/counters.
    The returned canonical matrix is the audited evidence consumed by rule
    selection, avoiding an unaudited fourth execution.
    """
    names = tuple(sorted(expected_names))
    if not names:
        predicates = load_predicates_source(source, filename=filename)
        if predicates:
            raise ValueError(
                "runtime p_* callables differ from the priced module-level functions")
        return predicates, np.zeros((len(panels), 0), dtype=float), []

    canonical = tuple(
        (name, panel_index)
        for name in names
        for panel_index in range(len(panels))
    )
    reversed_schedule = tuple(
        (name, panel_index)
        for name in reversed(names)
        for panel_index in reversed(range(len(panels)))
    )
    keyed_schedule = tuple(sorted(
        canonical,
        key=lambda pair: hashlib.sha256(
            f"predicate-purity/v1\0{pair[0]}\0{pair[1]}".encode("utf-8")
        ).digest(),
    ))

    baseline: Dict[Tuple[str, int], float] = {}
    canonical_predicates: Dict[str, Callable] = {}
    for schedule_name, schedule in (
            ("canonical", canonical),
            ("reversed", reversed_schedule),
            ("keyed", keyed_schedule)):
        predicates = load_predicates_source(source, filename=filename)
        if tuple(sorted(predicates)) != names:
            raise ValueError(
                "runtime p_* callables differ from the priced module-level functions")
        if schedule_name == "canonical":
            canonical_predicates = predicates
        for name, panel_index in schedule:
            value = _strict_predicate_value(
                name, predicates[name], panels[panel_index], panel_index)
            key = (name, panel_index)
            expected = baseline.setdefault(key, value)
            if value != expected:
                raise PredicateEvaluationError(
                    f"predicate {name!r} is not panel-stable on panel "
                    f"{panel_index}: canonical value {expected!r}, "
                    f"{schedule_name} execution returned {value!r}")
        if schedule_name == "canonical" and panels:
            for name in names:
                sentinel = int.from_bytes(
                    hashlib.sha256(
                        f"predicate-sentinel/v1\0{name}".encode("utf-8")
                    ).digest()[:4], "big") % len(panels)
                repeated = _strict_predicate_value(
                    name, predicates[name], panels[sentinel], sentinel)
                expected = baseline[(name, sentinel)]
                if repeated != expected:
                    raise PredicateEvaluationError(
                        f"predicate {name!r} is not panel-stable on repeated "
                        f"panel {sentinel}: canonical value {expected!r}, "
                        f"repeat returned {repeated!r}")

    values = np.empty((len(panels), len(names)), dtype=float)
    for column, name in enumerate(names):
        for panel_index in range(len(panels)):
            values[panel_index, column] = baseline[(name, panel_index)]
    return canonical_predicates, values, list(names)


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

    @property
    def predicate_names(self) -> Tuple[str, ...]:
        """Structured definition uses; formatted rule text is never parsed."""
        return tuple(sorted({atom.name for atom in self.atoms}))

    def structure_cost(self) -> float:
        """Calls and threshold bindings are paid per atom, including repeats."""
        return len(self.atoms) * (CALL_COST + BINDING_COST)

    def cost(self, pricing: Optional["RulePricing"] = None) -> float:
        definition_cost = 0 if pricing is None else pricing.definition_cost(self)
        return self.structure_cost() + definition_cost

    def describe(self) -> str:
        if self.constant is not None:
            return f"CONST_{self.constant}"
        return " AND ".join(a.describe() for a in self.atoms) or "CONST_True"


@dataclass(frozen=True)
class RulePricing:
    """One immutable definition-pricing context for a complete verification.

    ``paid_node_identities`` is the prior-use ledger.  It is frozen while the
    full-data selector and every rotated leave-one-out selector run.
    """

    model: predicate_price.PredicatePricingModel
    sharing_policy: str = SHARED_PRICING
    paid_node_identities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.sharing_policy not in {SHARED_PRICING, NO_SHARE_PRICING}:
            raise ValueError(
                "sharing_policy must be 'shared' or 'no-share' for priced rules")
        identities = frozenset(self.paid_node_identities)
        if any(not isinstance(identity, str) or not identity
               for identity in identities):
            raise ValueError("paid definition identities must be nonempty strings")
        if self.sharing_policy == NO_SHARE_PRICING and identities:
            raise ValueError("no-share pricing cannot carry a paid-definition ledger")
        object.__setattr__(self, "paid_node_identities", identities)

    def receipt(
            self, rule: Rule) -> Optional[predicate_price.DefinitionPrice]:
        if not rule.predicate_names:
            return None
        if self.sharing_policy == NO_SHARE_PRICING:
            return self.model.price_no_share(rule.predicate_names)
        return self.model.price(
            rule.predicate_names,
            promoted_node_identities=self.paid_node_identities,
        )

    def definition_cost(self, rule: Rule) -> int:
        receipt = self.receipt(rule)
        return 0 if receipt is None else receipt.charged_cost


def _candidate_atoms(values: np.ndarray, names: List[str],
                     labels: np.ndarray,
                     max_candidates: int = MAX_CANDIDATE_ATOMS,
                     lam: float = LAMBDA_RULE,
                     pricing: Optional[RulePricing] = None) -> List[Atom]:
    atoms: List[Tuple[Tuple[float, float, str, str, float], Atom]] = []
    for j, name in enumerate(names):
        col = values[:, j]
        uniq = np.unique(col)
        if len(uniq) < 2:
            continue
        # Prefer an interior midpoint because it generalizes better to held-out
        # values than an observed endpoint.  Compute it without overflowing;
        # when IEEE-754 has no representable interior value, use orientation-
        # specific endpoints so neither boundary dichotomy is silently lost.
        for lower, upper in zip(uniq[:-1], uniq[1:]):
            lower_f, upper_f = float(lower), float(upper)
            summed = lower_f + upper_f
            midpoint = (
                summed / 2.0 if math.isfinite(summed)
                else lower_f / 2.0 + upper_f / 2.0
            )
            if lower_f < midpoint < upper_f:
                boundaries = ((">=", midpoint), ("<=", midpoint))
            else:
                boundaries = ((">=", upper_f), ("<=", lower_f))
            for op, threshold in boundaries:
                t = float(threshold)
                atom = Atom(name, op, t)
                pred = col >= t if op == ">=" else col <= t
                acc = float((pred == labels).mean())
                if acc > 0.5:
                    cost = Rule(atoms=(atom,)).cost(pricing)
                    if pricing is None:
                        primary = 1.0 - acc + lam * cost
                    else:
                        # Conditional MDL: empirical risk is the gate and exact
                        # definition/structure cost ranks equally risky atoms.
                        primary = 1.0 - acc
                    key = (primary, cost, atom.name, atom.op, atom.threshold)
                    atoms.append((key, atom))
    atoms.sort(key=lambda pair: pair[0])
    return [atom for _, atom in atoms[:max_candidates]]


def select_rule(values: np.ndarray, names: List[str], labels: np.ndarray,
                lam: float = LAMBDA_RULE,
                max_atoms: int = MAX_RULE_ATOMS,
                max_candidates: int = MAX_CANDIDATE_ATOMS,
                pricing: Optional[RulePricing] = None) -> Rule:
    """Exhaustive conjunction selection over candidate atoms.

    Legacy flat-price calls minimize ``error + lam * rule_cost``.  Priced
    calls use conditional MDL: minimize empirical error first, then exact
    transitive definition plus structure cost.  This avoids an arbitrary LOC
    scale making the constant rule unbeatable while still closing the
    composite-predicate loophole.  Final ties use lexical description."""
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

    candidates = _candidate_atoms(
        values, names, labels, max_candidates, lam, pricing)
    cost_cache: Dict[Rule, float] = {}

    def priced_cost(rule: Rule) -> float:
        if rule not in cost_cache:
            cost_cache[rule] = rule.cost(pricing)
        return cost_cache[rule]

    best: Tuple[float, float, str, Rule] = None  # type: ignore[assignment]
    for rule in itertools.chain(
            (Rule(constant=True), Rule(constant=False)),
            (Rule(atoms=c) for size in range(1, max_atoms + 1)
             for c in itertools.combinations(candidates, size))):
        cost = priced_cost(rule)
        error = rule_error(rule)
        primary = error + lam * cost if pricing is None else error
        key = (primary, cost, rule.describe())
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
    structure_cost: float = 0.0
    definition_cost: int = 0
    full_definition_cost: int = 0
    predicate_names: Tuple[str, ...] = ()
    used_definition_node_identities: Tuple[str, ...] = ()
    charged_definition_node_identities: Tuple[str, ...] = ()
    reused_definition_node_identities: Tuple[str, ...] = ()
    pricing_source_digest: str = ""
    paid_node_identities_digest: str = ""
    sharing_policy: str = FLAT_PRICING
    selection_policy: str = "free-energy/v1"
    selected_rule: Optional[Rule] = None
    fold_rules: Tuple[Rule, ...] = ()
    definition_receipt: Optional[predicate_price.DefinitionPrice] = None

    def result_line(self) -> str:
        return (f"RESULT solved={self.solved} heldout={self.heldout_accuracy:.3f} "
                f"train={self.train_accuracy:.3f} rule=\"{self.rule}\" "
                f"rule_cost={self.rule_cost:.1f} "
                f"definition_cost={self.definition_cost} "
                f"pricing={self.sharing_policy} "
                f"selection={self.selection_policy} "
                f"predicate_errors={self.predicate_errors}")


def verify(preds: Dict[str, Callable], problem: Problem,
           lam: float = LAMBDA_RULE,
           max_atoms: int = MAX_RULE_ATOMS,
           pricing: Optional[RulePricing] = None,
           *,
           _audited_values: Optional[Tuple[np.ndarray, List[str]]] = None
           ) -> VerifyResult:
    """The pure verifier: predicates + panels -> solved or not.

    Deterministic; re-running IS the replay validation."""
    panels = [p for p, _ in problem.panels()]
    labels = np.array([lab for _, lab in problem.panels()])
    if pricing is not None and tuple(sorted(preds)) != \
            tuple(sorted(pricing.model.predicate_names)):
        raise ValueError(
            "runtime p_* callables differ from the priced module-level functions")
    if not preds:
        return VerifyResult(
            False, 0.5, 0.5, "CONST_True", 0.0, 0, 36,
            sharing_policy=(
                pricing.sharing_policy if pricing is not None else FLAT_PRICING),
            selection_policy=(
                PRICED_SELECTION_POLICY if pricing is not None
                else "free-energy/v1"))
    if _audited_values is None:
        values, names, errors = predicate_values(
            preds, panels, strict=pricing is not None)
    else:
        values, names = _audited_values
        if values.shape != (len(panels), len(names)) \
                or names != sorted(preds):
            raise ValueError("audited predicate matrix does not match callables")
        errors = 0

    full_rule = select_rule(
        values, names, labels, lam, max_atoms, pricing=pricing)
    name_idx = {n: i for i, n in enumerate(names)}

    def predict(rule: Rule, i: int) -> bool:
        row = {n: values[i, name_idx[n]] for n in names}
        return rule.predict(row)

    train_ok = [predict(full_rule, i) == labels[i] for i in range(12)]
    train_accuracy = float(np.mean(train_ok))

    correct = 0
    total = 0
    n_pos = len(problem.pos)
    fold_rules: List[Rule] = []
    for i in range(n_pos):
        for j in range(len(problem.neg)):
            held = {i, n_pos + j}
            mask = np.array([k not in held for k in range(12)])
            rule_ij = select_rule(
                values[mask], names, labels[mask], lam, max_atoms,
                pricing=pricing)
            fold_rules.append(rule_ij)
            correct += int(predict(rule_ij, i) == labels[i])
            correct += int(predict(rule_ij, n_pos + j) == labels[n_pos + j])
            total += 2
    heldout_accuracy = correct / total if total else 0.0
    solved = heldout_accuracy == 1.0 and train_accuracy == 1.0
    receipt = pricing.receipt(full_rule) if pricing is not None else None
    paid_digest = ""
    if pricing is not None:
        paid_digest = hashlib.sha256(
            "\0".join(sorted(pricing.paid_node_identities)).encode("utf-8")
        ).hexdigest()
    return VerifyResult(solved, heldout_accuracy, train_accuracy,
                        full_rule.describe(), full_rule.cost(pricing), errors,
                        n_pos * len(problem.neg),
                        structure_cost=full_rule.structure_cost(),
                        definition_cost=(receipt.charged_cost if receipt else 0),
                        full_definition_cost=(receipt.full_cost if receipt else 0),
                        predicate_names=full_rule.predicate_names,
                        used_definition_node_identities=tuple(
                            node.identity for node in receipt.used_nodes)
                        if receipt else (),
                        charged_definition_node_identities=tuple(
                            node.identity for node in receipt.charged_nodes)
                        if receipt else (),
                        reused_definition_node_identities=tuple(
                            node.identity for node in receipt.reused_nodes)
                        if receipt else (),
                        pricing_source_digest=(
                            pricing.model.source_digest if pricing else ""),
                        paid_node_identities_digest=paid_digest,
                        sharing_policy=(
                            pricing.sharing_policy if pricing else FLAT_PRICING),
                        selection_policy=(
                            PRICED_SELECTION_POLICY if pricing
                            else "free-energy/v1"),
                        selected_rule=full_rule,
                        fold_rules=tuple(fold_rules),
                        definition_receipt=receipt)


def verify_priced_source(
        source: str,
        problem: Problem,
        *,
        sharing_policy: str = SHARED_PRICING,
        paid_node_identities: Iterable[str] = (),
        filename: str = "<agent_predicates>",
        lam: float = LAMBDA_RULE,
        max_atoms: int = MAX_RULE_ATOMS) -> VerifyResult:
    """Price and execute the same immutable predicate source snapshot."""
    model = predicate_price.build_pricing_model(source, filename=filename)
    panels = [panel for panel, _ in problem.panels()]
    predicates, values, names = audit_predicate_purity(
        source, panels, model.predicate_names, filename=filename)
    pricing = RulePricing(
        model=model,
        sharing_policy=sharing_policy,
        paid_node_identities=frozenset(paid_node_identities),
    )
    return verify(
        predicates, problem, lam=lam, max_atoms=max_atoms, pricing=pricing,
        _audited_values=(values, names))


def free_energy(solved: int, total_marginal_C: float, lam: float = 0.02) -> float:
    """Corpus-level F = R + lambda*C with R = -problems_solved, C = total
    marginal novelty in the predicate library (same shape as gkm_legs)."""
    return -float(solved) + lam * float(total_marginal_C)


# ---------------------------------------------------------------------------
# Workspace panel IO (what the proposer is allowed to see)
# ---------------------------------------------------------------------------


def canonical_panel_filenames() -> Tuple[str, ...]:
    """Return the only filenames owned by a proposer panel directory."""
    return tuple(
        f"{side}_{index}{suffix}"
        for side in ("pos", "neg")
        for index in range(6)
        for suffix in PANEL_FILE_SUFFIXES
    )


def _require_canonical_opaque_id(opaque_id: str) -> None:
    suffix = opaque_id.removeprefix("problem_")
    if not opaque_id.startswith("problem_") \
            or len(suffix) < 2 or not suffix.isdecimal():
        raise ValueError(
            "opaque panel id must have canonical form problem_<digits>")


def _open_real_directory(
        path: str, description: str, *, dir_fd: Optional[int] = None) -> int:
    """Open an actual directory without following a terminal symlink."""
    try:
        before = (os.lstat(path) if dir_fd is None else os.stat(
            path, dir_fd=dir_fd, follow_symlinks=False))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{description} does not exist") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(
            f"{description} must be a non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise RuntimeError(
            f"{description} must be a non-symlink directory") from exc
    after = os.fstat(descriptor)
    if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != \
            (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise RuntimeError(f"{description} changed during validation")
    return descriptor


def _inspect_panel_directory(
        descriptor: int, opaque_id: str, *, require_complete: bool) -> set:
    expected = set(canonical_panel_filenames())
    observed = set(os.listdir(descriptor))
    extra = sorted(observed - expected)
    if extra:
        raise RuntimeError(
            f"{opaque_id} panel directory contains unexpected files: {extra}")
    missing = sorted(expected - observed)
    if require_complete and missing:
        raise RuntimeError(
            f"{opaque_id} panel directory is incomplete: {missing}")
    for name in sorted(observed):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(
                f"{opaque_id}/{name} must be a non-symlink, singly-linked "
                "regular file")
    return observed


def validate_panel_directory(ws: str, opaque_id: str) -> str:
    """Fail closed unless ``opaque_id`` is the exact canonical panel set."""
    _require_canonical_opaque_id(opaque_id)
    workspace_fd = _open_real_directory(ws, "proposer workspace")
    try:
        panel_fd = _open_real_directory(
            opaque_id, f"{opaque_id} panel directory", dir_fd=workspace_fd)
        try:
            _inspect_panel_directory(
                panel_fd, opaque_id, require_complete=True)
        finally:
            os.close(panel_fd)
    finally:
        os.close(workspace_fd)
    return os.path.join(ws, opaque_id)


def remove_panel_directory(ws: str, opaque_id: str) -> None:
    """Remove only a fully inspected, harness-owned panel directory.

    Partial directories left by a failed renderer are allowed, but every
    entry still has to be one of the canonical regular panel files. Unknown
    content is never silently deleted.
    """
    _require_canonical_opaque_id(opaque_id)
    workspace_fd = _open_real_directory(ws, "proposer workspace")
    panel_fd: Optional[int] = None
    try:
        try:
            panel_fd = _open_real_directory(
                opaque_id, f"{opaque_id} panel directory", dir_fd=workspace_fd)
        except RuntimeError as exc:
            try:
                os.stat(opaque_id, dir_fd=workspace_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise exc
        observed = _inspect_panel_directory(
            panel_fd, opaque_id, require_complete=False)
        opened = os.fstat(panel_fd)
        current = os.stat(
            opaque_id, dir_fd=workspace_fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError(f"{opaque_id} panel directory changed")
        for name in sorted(observed):
            os.unlink(name, dir_fd=panel_fd)
        os.close(panel_fd)
        panel_fd = None
        os.rmdir(opaque_id, dir_fd=workspace_fd)
    finally:
        if panel_fd is not None:
            os.close(panel_fd)
        os.close(workspace_fd)


def write_panels(ws: str, problem: Problem, opaque_id: str) -> str:
    """Write the 12 panels into the workspace under an OPAQUE id.

    Sampler problem names contain concept names (e.g. `bd_..._triangle_...`)
    and must never reach the proposer; the caller supplies `opaque_id` like
    `problem_03`. Panels are saved as .npy (exact) and .png (viewable)."""
    slots: List[Tuple[str, int, np.ndarray]] = []
    for side, arrs in (("pos", problem.pos), ("neg", problem.neg)):
        if len(arrs) != 6:
            raise ValueError(f"{opaque_id} must contain exactly six {side} panels")
        for i, arr in enumerate(arrs):
            panel = np.asarray(arr)
            if panel.dtype != np.uint8 or panel.ndim != 2 \
                    or panel.shape != (PANEL_SIZE, PANEL_SIZE) \
                    or not np.isin(panel, (0, 1)).all():
                raise ValueError(
                    f"{opaque_id} {side}_{i} must be a {PANEL_SIZE}x{PANEL_SIZE} "
                    "binary uint8 panel")
            slots.append((side, i, np.ascontiguousarray(panel)))

    _require_canonical_opaque_id(opaque_id)
    pdir = os.path.join(ws, opaque_id)
    workspace_fd = _open_real_directory(ws, "proposer workspace")
    panel_fd: Optional[int] = None
    created: Dict[str, Tuple[int, int]] = {}
    try:
        try:
            os.stat(opaque_id, dir_fd=workspace_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(opaque_id, mode=0o700, dir_fd=workspace_fd)
        panel_fd = _open_real_directory(
            opaque_id, f"{opaque_id} panel directory", dir_fd=workspace_fd)
        existing = _inspect_panel_directory(
            panel_fd, opaque_id, require_complete=False)
        for name in sorted(existing):
            os.unlink(name, dir_fd=panel_fd)

        def create_file(name: str):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
                | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, 0o600, dir_fd=panel_fd)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(descriptor)
                raise RuntimeError(
                    f"failed to create singly-linked regular panel file {name}")
            created[name] = (info.st_dev, info.st_ino)
            return os.fdopen(descriptor, "wb")

        def open_file(name: str):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=panel_fd)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(descriptor)
                raise RuntimeError(
                    f"panel file {name} is not a singly-linked regular file")
            return os.fdopen(descriptor, "rb")

        from PIL import Image
        for side, i, panel in slots:
            npy_name = f"{side}_{i}.npy"
            with create_file(npy_name) as handle:
                np.save(handle, panel, allow_pickle=False)
            with open_file(npy_name) as handle:
                round_trip_npy = np.load(handle, allow_pickle=False)
            if round_trip_npy.dtype != np.uint8 \
                    or not np.array_equal(round_trip_npy, panel):
                raise RuntimeError(
                    f"failed to reproduce proposer array {side}_{i}.npy")
            presentation = np.where(panel == 1, 0, 255).astype(np.uint8)
            png_name = f"{side}_{i}.png"
            with create_file(png_name) as handle:
                Image.fromarray(presentation, mode="L").save(
                    handle, format="PNG")
            with open_file(png_name) as handle:
                with Image.open(handle) as encoded:
                    round_trip = np.asarray(encoded.convert("L"))
            if not np.isin(round_trip, (0, 255)).all() \
                    or not np.array_equal(
                        (round_trip == 0).astype(np.uint8), panel):
                raise RuntimeError("PNG bytes do not match the panel array")
        _inspect_panel_directory(panel_fd, opaque_id, require_complete=True)
    except Exception as exc:
        if panel_fd is not None:
            for name, identity in created.items():
                try:
                    current = os.stat(
                        name, dir_fd=panel_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == identity:
                        os.unlink(name, dir_fd=panel_fd)
                except FileNotFoundError:
                    pass
        raise RuntimeError(
            f"failed to materialize proposer panels for {opaque_id}") from exc
    finally:
        if panel_fd is not None:
            os.close(panel_fd)
        os.close(workspace_fd)
    return pdir
