#!/usr/bin/env python3
"""Symbolic Bongard-LOGO adapter and first-pass rule-selection experiment.

This script does not vendor Bongard-LOGO. Point ``--dataset-dir`` at a local
Bongard-LOGO checkout/download. The first implemented path uses the public LOGO
samplers to generate action programs without rendering images, then evaluates a
small deterministic feature-rule selector under the same free-energy shape used
elsewhere in this repository:

    F_lambda(rule) = classification_loss_train(rule) + lambda * C(rule)

The goal is not to claim Bongard-LOGO is solved. This is a substrate test: can we
load external LOGO action programs, expose symbolic structure, and run a clean
train/validation/hidden selection loop before adding visual perception.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Atom = str
Scene = Tuple["LogoSceneObject", ...]
LabeledScene = Tuple[Scene, bool]
ActionProgram = List[List[List[List[str]]]]
MeasurementItems = Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class LogoSceneObject:
    """Minimal symbolic object extracted from a LOGO action program."""

    shape_token: str
    action_tokens: Tuple[str, ...]
    position_bin: Optional[str] = None
    size_bin: Optional[str] = None
    orientation_bin: Optional[str] = None
    metadata_shape_names: Tuple[str, ...] = ()
    metadata_super_classes: Tuple[str, ...] = ()
    metadata_attributes: Tuple[str, ...] = ()
    measurements: MeasurementItems = ()


@dataclass(frozen=True)
class LogoProblem:
    """One symbolic Bongard-LOGO problem with generated support examples."""

    problem_id: str
    positives: Tuple[Scene, ...]
    negatives: Tuple[Scene, ...]
    category: str
    concept: str


@dataclass(frozen=True)
class LogoSplit:
    positives: Tuple[Scene, ...]
    negatives: Tuple[Scene, ...]


@dataclass(frozen=True)
class LogoFewShotProblem:
    problem: LogoProblem
    train: LogoSplit
    validation: LogoSplit
    hidden: LogoSplit


@dataclass(frozen=True)
class LogoRule:
    atoms: Tuple[Atom, ...] = ()
    constant: Optional[bool] = None

    def predict(self, scene: Scene, feature_set: str) -> bool:
        if self.constant is not None:
            return self.constant
        features = scene_features(scene, feature_set)
        return all(atom in features for atom in self.atoms)

    def describe(self) -> str:
        if self.constant is True:
            return "CONST_TRUE"
        if self.constant is False:
            return "CONST_FALSE"
        return " AND ".join(self.atoms)


@dataclass(frozen=True)
class RuleEvaluation:
    loss: float
    accuracy: float
    positive_accuracy: float
    negative_accuracy: float
    free_energy: float
    complexity: float


@dataclass(frozen=True)
class LogoRecord:
    problem_id: str
    category: str
    concept: str
    feature_set: str
    lambda_value: float
    train_accuracy: float
    validation_accuracy: float
    hidden_accuracy: float
    train_loss: float
    validation_loss: float
    hidden_loss: float
    complexity: float
    atom_count: int
    rule: str


@dataclass(frozen=True)
class ShapeMetadata:
    function_name: str
    shape_name: str
    super_class: str
    signature: Tuple[str, ...]
    positive_attributes: Tuple[str, ...]


@dataclass(frozen=True)
class ShapeIndex:
    by_signature: Dict[Tuple[str, ...], Tuple[ShapeMetadata, ...]]
    signature_complexity: Dict[Tuple[str, ...], float]
    shape_complexity: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="local Bongard-LOGO checkout or downloaded dataset directory")
    parser.add_argument("--source", choices=("basic", "abstract", "both"), default="both")
    parser.add_argument("--feature-set", choices=("action", "macro", "metadata", "both", "all"), default="both")
    parser.add_argument("--limit", type=int, default=12, help="maximum generated problems per selected source")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--support-count", type=int, default=4, help="positive and negative examples used for training")
    parser.add_argument("--validation-count", type=int, default=1, help="positive and negative examples used for validation elbow selection")
    parser.add_argument("--hidden-count", type=int, default=2, help="positive and negative examples held out until after selection")
    parser.add_argument("--max-rule-atoms", type=int, default=2, help="maximum conjunction size in the sparse feature rule")
    parser.add_argument("--max-candidate-atoms", type=int, default=40, help="cap shared positive atoms before conjunction search, ranked by training-set separation; use 0 for no cap")
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.02)
    parser.add_argument("--lambda-points", type=int, default=5)
    parser.add_argument("--basic-two-shape", action="store_true", help="include two-shape Basic concepts after one-shape concepts")
    parser.add_argument("--abstract-pairs", action="store_true", help="include two-attribute Abstract concepts after one-attribute concepts")
    parser.add_argument("--summary-only", action="store_true", help="print corpus and aggregate summary without per-problem rows")
    parser.add_argument("--show-rules", action="store_true")
    return parser.parse_args()


def ensure_logo_imports(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Bongard-LOGO path does not exist: {dataset_dir}")
    if not (dataset_dir / "bongard").is_dir():
        raise FileNotFoundError(f"Expected a Bongard-LOGO checkout with a bongard/ package: {dataset_dir}")
    sys.path.insert(0, str(dataset_dir.resolve()))
    from bongard.sampler.abstract_sampler import AbstractSampler  # type: ignore
    from bongard.sampler.basic_sampler import BasicSampler  # type: ignore
    from bongard.util_funcs import get_attribute_sampling_candidates, get_shape_super_classes  # type: ignore

    return BasicSampler, AbstractSampler, get_shape_super_classes, get_attribute_sampling_candidates


def load_shape_index(dataset_dir: Path) -> ShapeIndex:
    shape_rows_path = dataset_dir / "data" / "human_designed_shapes.tsv"
    attr_rows_path = dataset_dir / "data" / "human_designed_shapes_attributes.tsv"
    if not shape_rows_path.exists() or not attr_rows_path.exists():
        raise FileNotFoundError("Bongard-LOGO data TSV files are missing from the checkout")

    attr_by_function: Dict[str, Tuple[str, ...]] = {}
    with attr_rows_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty attribute table: {attr_rows_path}")
        attr_names = tuple(name for name in reader.fieldnames[3:] if name != "if stamp")
        for row in reader:
            positives = tuple(name for name in attr_names if row.get(name) == "1")
            attr_by_function[row["shape function name"]] = positives

    by_signature: Dict[Tuple[str, ...], List[ShapeMetadata]] = OrderedDict()
    signature_complexity: Dict[Tuple[str, ...], float] = {}
    shape_complexity: Dict[str, float] = {}
    with shape_rows_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            signature = signature_from_shape_row(row)
            metadata = ShapeMetadata(
                function_name=row["shape function name"],
                shape_name=row["shape name"],
                super_class=row["super class"],
                signature=signature,
                positive_attributes=attr_by_function.get(row["shape function name"], ()),
            )
            by_signature.setdefault(signature, []).append(metadata)
            complexity = float(len(signature) + 1)
            signature_complexity[signature] = complexity
            shape_complexity[metadata.function_name] = complexity

    return ShapeIndex(
        by_signature={signature: tuple(rows) for signature, rows in by_signature.items()},
        signature_complexity=signature_complexity,
        shape_complexity=shape_complexity,
    )


def signature_from_shape_row(row: Dict[str, str]) -> Tuple[str, ...]:
    base_actions = [item.strip() for item in row["set of base actions"].split(",")]
    turn_angles = [item.strip() for item in row["turn angles"].split("--")]
    if len(base_actions) != len(turn_angles):
        raise ValueError(f"base action / turn mismatch in {row['shape function name']}")
    tokens = []
    for base_action, turn_angle in zip(base_actions, turn_angles):
        turn = normalize_turn_angle(turn_angle)
        parts = base_action.split("_")
        if parts[0] == "line":
            tokens.append(f"line:{float(parts[1]):.3f}:{turn:.3f}")
        elif parts[0] == "arc":
            radius = float(parts[1])
            arc_angle = (float(parts[2]) + 360.0) / 720.0
            tokens.append(f"arc:{radius:.3f}:{arc_angle:.3f}:{turn:.3f}")
        else:
            raise ValueError(f"unsupported action {base_action}")
    return tuple(tokens)


def normalize_turn_angle(turn_angle: str) -> float:
    direction = turn_angle[0]
    angle = float(turn_angle[1:])
    if direction == "L":
        return (angle + 180.0) / 360.0
    if direction == "R":
        return (180.0 - angle) / 360.0
    raise ValueError(f"unsupported turn angle {turn_angle}")


def action_skeleton(action_string: str) -> str:
    movement, turn_angle = action_string.split("-")
    turn = float(turn_angle)
    parts = movement.split("_")
    if parts[0] == "line":
        return f"line:{float(parts[2]):.3f}:{turn:.3f}"
    if parts[0] == "arc":
        return f"arc:{float(parts[2]):.3f}:{float(parts[3]):.3f}:{turn:.3f}"
    raise ValueError(f"unsupported action string {action_string}")


def denormalize_signed_turn(normalized_turn: float) -> float:
    return normalized_turn * 360.0 - 180.0


def denormalize_arc_angle(normalized_arc_angle: float) -> float:
    return normalized_arc_angle * 720.0 - 360.0


def canonical_heading_error(degrees: float) -> float:
    return abs(((degrees + 180.0) % 360.0) - 180.0)


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def cross(origin: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return list(unique_points)
    lower: List[Tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def shape_measurements(signature: Tuple[str, ...]) -> MeasurementItems:
    x = y = 0.0
    heading = 0.0
    points: List[Tuple[float, float]] = [(x, y)]
    line_count = 0
    arc_count = 0
    total_line_length = 0.0
    total_arc_angle = 0.0
    abs_turn_total = 0.0
    path_length = 0.0

    for token in signature:
        parts = token.split(":")
        turn = denormalize_signed_turn(float(parts[-1]))
        heading += turn
        abs_turn_total += abs(turn)
        if parts[0] == "line":
            line_count += 1
            length = float(parts[1])
            total_line_length += length
            path_length += length
            radians = math.radians(heading)
            x += length * math.cos(radians)
            y += length * math.sin(radians)
            points.append((round(x, 6), round(y, 6)))
        elif parts[0] == "arc":
            arc_count += 1
            radius = float(parts[1])
            arc_angle = denormalize_arc_angle(float(parts[2]))
            total_arc_angle += abs(arc_angle)
            steps = max(4, int(abs(arc_angle) // 30) + 1)
            step_angle = arc_angle / steps
            chord = 2.0 * radius * math.sin(abs(math.radians(step_angle)) / 2.0)
            for _ in range(steps):
                heading += step_angle / 2.0
                radians = math.radians(heading)
                x += chord * math.cos(radians)
                y += chord * math.sin(radians)
                heading += step_angle / 2.0
                points.append((round(x, 6), round(y, 6)))
            path_length += abs(math.radians(arc_angle) * radius)
        else:
            raise ValueError(f"unsupported skeleton token {token}")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs) if xs else 0.0
    height = max(ys) - min(ys) if ys else 0.0
    min_extent = max(min(width, height), 1e-6)
    max_extent = max(width, height)
    endpoint_distance = math.hypot(x, y)
    hull = convex_hull(points)
    hull_area = polygon_area(hull)
    trace_area = polygon_area(points)
    hull_fill = trace_area / hull_area if hull_area > 1e-9 else 0.0
    measurements = {
        "action_count": float(len(signature)),
        "line_count": float(line_count),
        "arc_count": float(arc_count),
        "curve_fraction": arc_count / max(1, len(signature)),
        "line_fraction": line_count / max(1, len(signature)),
        "total_line_length": total_line_length,
        "total_arc_angle": total_arc_angle,
        "abs_turn_total": abs_turn_total,
        "path_length": path_length,
        "closure_error": endpoint_distance / max(path_length, 1e-6),
        "aspect_ratio": max_extent / min_extent if max_extent else 1.0,
        "bbox_area": width * height,
        "hull_area": hull_area,
        "hull_fill": min(1.0, hull_fill),
        "heading_error": canonical_heading_error(heading) / 180.0,
    }
    return tuple(sorted(measurements.items()))


def measurements_dict(obj: LogoSceneObject) -> Dict[str, float]:
    return dict(obj.measurements)


def threshold_atom(name: str, op: str, threshold: float) -> str:
    if threshold == int(threshold):
        threshold_text = str(int(threshold))
    else:
        threshold_text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return f"macro:{name}{op}{threshold_text}"


def macro_features_for_object(obj: LogoSceneObject) -> Set[Atom]:
    measurements = measurements_dict(obj)
    atoms: Set[Atom] = set()
    for name in ("line_count", "arc_count", "action_count"):
        value = measurements.get(name, 0.0)
        for threshold in range(1, 10):
            if value >= threshold:
                atoms.add(threshold_atom(name, ">=", float(threshold)))
            if value <= threshold:
                atoms.add(threshold_atom(name, "<=", float(threshold)))
    for name, thresholds in {
        "curve_fraction": (0.001, 0.25, 0.50, 0.75),
        "line_fraction": (0.25, 0.50, 0.75, 0.999),
        "closure_error": (0.03, 0.08, 0.15, 0.30),
        "aspect_ratio": (1.25, 1.75, 2.50, 4.00),
        "hull_fill": (0.25, 0.50, 0.75, 0.90),
        "heading_error": (0.05, 0.15, 0.30, 0.60),
        "abs_turn_total": (180.0, 360.0, 540.0, 720.0),
        "total_arc_angle": (90.0, 180.0, 270.0, 360.0),
    }.items():
        value = measurements.get(name, 0.0)
        for threshold in thresholds:
            if value >= threshold:
                atoms.add(threshold_atom(name, ">=", threshold))
            if value <= threshold:
                atoms.add(threshold_atom(name, "<=", threshold))
    if measurements.get("closure_error", 1.0) <= 0.08:
        atoms.add("macro:closed")
    if measurements.get("closure_error", 0.0) > 0.15:
        atoms.add("macro:open")
    if measurements.get("aspect_ratio", 1.0) >= 2.5:
        atoms.add("macro:thin_candidate")
    if measurements.get("hull_fill", 0.0) >= 0.9 and measurements.get("closure_error", 1.0) <= 0.15:
        atoms.add("macro:convex_fill_candidate")
    if measurements.get("arc_count", 0.0) > 0 and measurements.get("line_count", 0.0) > 0:
        atoms.add("macro:mixed_line_arc")
    return atoms


def scene_from_image_program(image_program: List[List[str]], shape_index: ShapeIndex) -> Scene:
    objects = []
    for shape_program in image_program:
        signature = tuple(action_skeleton(action) for action in shape_program)
        metadata_rows = shape_index.by_signature.get(signature, ())
        shared_super_classes = sorted({row.super_class for row in metadata_rows})
        if len(shared_super_classes) != 1:
            shared_super_classes = []
        shared_attributes: Set[str] = set()
        if metadata_rows:
            shared_attributes = set(metadata_rows[0].positive_attributes)
            for row in metadata_rows[1:]:
                shared_attributes &= set(row.positive_attributes)
        objects.append(
            LogoSceneObject(
                shape_token="/".join(row.function_name for row in metadata_rows) if metadata_rows else "unknown",
                action_tokens=signature,
                metadata_shape_names=tuple(row.function_name for row in metadata_rows),
                metadata_super_classes=tuple(shared_super_classes),
                metadata_attributes=tuple(sorted(shared_attributes)),
                measurements=shape_measurements(signature),
            )
        )
    return tuple(objects)


def problem_from_action_program(problem_id: str, action_program: ActionProgram, category: str, concept: str, shape_index: ShapeIndex) -> LogoProblem:
    positives = tuple(scene_from_image_program(image, shape_index) for image in action_program[0])
    negatives = tuple(scene_from_image_program(image, shape_index) for image in action_program[1])
    return LogoProblem(problem_id=problem_id, positives=positives, negatives=negatives, category=category, concept=concept)


def abstract_concept_has_capacity(
    attr_candidates: Dict[str, Tuple[Sequence[str], Sequence[str]]],
    attrs: Tuple[str, ...],
    total_examples: int,
    min_pair_bucket: int = 10,
) -> bool:
    if len(attrs) == 1:
        positives, negatives = attr_candidates[attrs[0]]
        return len(positives) >= total_examples and len(negatives) >= total_examples
    if len(attrs) != 2:
        return False

    pos0, neg0 = attr_candidates[attrs[0]]
    pos1, neg1 = attr_candidates[attrs[1]]
    pos0_set = set(pos0)
    neg0_set = set(neg0)
    pos1_set = set(pos1)
    neg1_set = set(neg1)
    pos_pos = pos0_set & pos1_set
    pos_neg = pos0_set & neg1_set
    neg_pos = neg0_set & pos1_set
    negatives_per_bucket = (total_examples + 1) // 2
    required_pair_bucket = max(min_pair_bucket, negatives_per_bucket)
    return (
        len(pos_pos) >= max(total_examples, min_pair_bucket)
        and len(pos_neg) >= required_pair_bucket
        and len(neg_pos) >= required_pair_bucket
    )


def generate_logo_problems(args: argparse.Namespace, shape_index: ShapeIndex) -> List[LogoProblem]:
    BasicSampler, AbstractSampler, get_shape_super_classes, get_attribute_sampling_candidates = ensure_logo_imports(args.dataset_dir)
    shape_actions_path = args.dataset_dir / "data" / "human_designed_shapes.tsv"
    shape_attributes_path = args.dataset_dir / "data" / "human_designed_shapes_attributes.tsv"
    total_examples = args.support_count + args.validation_count + args.hidden_count
    if total_examples < 3:
        raise ValueError("support + validation + hidden examples must leave a meaningful split")

    rng = np.random.RandomState(args.seed)
    problems: List[LogoProblem] = []
    sources = ("basic", "abstract") if args.source == "both" else (args.source,)

    if "basic" in sources:
        shape_sup_class_dict = get_shape_super_classes(str(shape_actions_path))
        shape_list = list(shape_sup_class_dict.keys())
        concepts: List[Tuple[str, ...]] = [(shape,) for shape in shape_list]
        if args.basic_two_shape:
            concepts.extend(itertools.combinations(shape_list, 2))
        sampler = BasicSampler(
            str(shape_actions_path),
            str(shape_attributes_path),
            num_positive_examples=total_examples,
            num_negative_examples=total_examples,
            random_state=rng,
        )
        for idx, shapes in enumerate(concepts[: args.limit]):
            sampled = sampler.sample(list(shapes), idx)
            problems.append(
                problem_from_action_program(
                    sampled.get_problem_name(),
                    sampled.get_action_string_list(),
                    category="basic",
                    concept="&".join(shapes),
                    shape_index=shape_index,
                )
            )

    if "abstract" in sources:
        attr_candidates = get_attribute_sampling_candidates(str(shape_attributes_path))
        attr_list = list(attr_candidates.keys())
        concepts = [(attr,) for attr in attr_list]
        if args.abstract_pairs:
            concepts.extend(itertools.combinations(attr_list, 2))
        sampler = AbstractSampler(
            str(shape_actions_path),
            str(shape_attributes_path),
            num_positive_examples=total_examples,
            num_negative_examples=total_examples,
            random_state=rng,
        )
        count = 0
        task_id = 0
        for attrs in concepts:
            if count >= args.limit:
                break
            if not abstract_concept_has_capacity(attr_candidates, tuple(attrs), total_examples):
                continue
            sampled = sampler.sample(list(attrs), task_id)
            task_id += 1
            if sampled is None:
                continue
            problems.append(
                problem_from_action_program(
                    sampled.get_problem_name(),
                    sampled.get_action_string_list(),
                    category="abstract",
                    concept="&".join(attrs),
                    shape_index=shape_index,
                )
            )
            count += 1

    return problems


def split_problem(problem: LogoProblem, support_count: int, validation_count: int, hidden_count: int) -> LogoFewShotProblem:
    needed = support_count + validation_count + hidden_count
    if len(problem.positives) < needed or len(problem.negatives) < needed:
        raise ValueError(f"{problem.problem_id} has too few examples for requested split")

    def split_side(items: Tuple[Scene, ...]) -> Tuple[Tuple[Scene, ...], Tuple[Scene, ...], Tuple[Scene, ...]]:
        train = items[:support_count]
        validation = items[support_count : support_count + validation_count]
        hidden = items[support_count + validation_count : needed]
        return train, validation, hidden

    train_pos, val_pos, hidden_pos = split_side(problem.positives)
    train_neg, val_neg, hidden_neg = split_side(problem.negatives)
    return LogoFewShotProblem(
        problem=problem,
        train=LogoSplit(train_pos, train_neg),
        validation=LogoSplit(val_pos, val_neg),
        hidden=LogoSplit(hidden_pos, hidden_neg),
    )


def iter_labeled(split: LogoSplit) -> Iterable[LabeledScene]:
    for scene in split.positives:
        yield scene, True
    for scene in split.negatives:
        yield scene, False


def scene_features(scene: Scene, feature_set: str) -> Set[Atom]:
    features: Set[Atom] = {f"object_count={len(scene)}"}
    for idx, obj in enumerate(scene):
        signature = "|".join(obj.action_tokens)
        action_names = tuple(token.split(":", 1)[0] for token in obj.action_tokens)
        features.add(f"sig:{signature}")
        features.add(f"action_count={len(obj.action_tokens)}")
        features.add("has_line" if "line" in action_names else "no_line")
        features.add("has_arc" if "arc" in action_names else "no_arc")
        features.add("type_sequence:" + ",".join(action_names))
        features.add(f"slot{idx}:sig:{signature}")
        if feature_set == "macro":
            for atom in macro_features_for_object(obj):
                features.add(atom)
                features.add(f"slot{idx}:{atom}")
        if feature_set == "metadata":
            for shape_name in obj.metadata_shape_names:
                features.add(f"shape:{shape_name}")
                features.add(f"slot{idx}:shape:{shape_name}")
            for super_class in obj.metadata_super_classes:
                features.add(f"super:{super_class}")
                features.add(f"slot{idx}:super:{super_class}")
            for attr in obj.metadata_attributes:
                features.add(f"attr:{attr}")
                features.add(f"slot{idx}:attr:{attr}")
    return features


def atom_complexity(atom: Atom, shape_index: ShapeIndex) -> float:
    if atom.startswith("slot") and ":sig:" in atom:
        signature_text = atom.split(":sig:", 1)[1]
        return signature_text.count("|") + 2.0
    if atom.startswith("sig:"):
        signature_text = atom.split(":", 1)[1]
        return signature_text.count("|") + 2.0
    if ":shape:" in atom:
        shape_name = atom.split(":shape:", 1)[1]
        return shape_index.shape_complexity.get(shape_name, 4.0)
    if atom.startswith("shape:"):
        shape_name = atom.split(":", 1)[1]
        return shape_index.shape_complexity.get(shape_name, 4.0)
    if ":attr:" in atom or atom.startswith("attr:"):
        return 1.5
    if ":super:" in atom or atom.startswith("super:"):
        return 1.5
    if atom.startswith("macro:") or ":macro:" in atom:
        return macro_atom_complexity(atom)
    if atom.startswith("object_count="):
        return 1.0
    if atom.startswith("action_count="):
        return 1.0
    if atom in {"has_line", "no_line", "has_arc", "no_arc"}:
        return 1.0
    if atom.startswith("type_sequence:"):
        return float(atom.count(",") + 2)
    return 2.0


def macro_atom_complexity(atom: Atom) -> float:
    macro = atom.split("macro:", 1)[1]
    if macro in {"closed", "open", "thin_candidate", "convex_fill_candidate", "mixed_line_arc"}:
        return 2.5
    if macro.startswith(("line_count", "arc_count", "action_count")):
        return 2.0
    if macro.startswith(("curve_fraction", "line_fraction")):
        return 2.5
    if macro.startswith(("closure_error", "heading_error")):
        return 3.0
    if macro.startswith(("aspect_ratio", "hull_fill")):
        return 3.5
    if macro.startswith(("abs_turn_total", "total_arc_angle")):
        return 2.5
    return 3.0


def rule_complexity(rule: LogoRule, shape_index: ShapeIndex) -> float:
    if rule.constant is not None:
        return 0.0
    return 1.0 + sum(atom_complexity(atom, shape_index) for atom in rule.atoms)


def evaluate_rule(rule: LogoRule, split: LogoSplit, feature_set: str, lambda_value: float, shape_index: ShapeIndex) -> RuleEvaluation:
    labeled = list(iter_labeled(split))
    correct = 0
    pos_total = pos_correct = neg_total = neg_correct = 0
    for scene, label in labeled:
        prediction = rule.predict(scene, feature_set)
        if label:
            pos_total += 1
            pos_correct += int(prediction == label)
        else:
            neg_total += 1
            neg_correct += int(prediction == label)
        correct += int(prediction == label)
    accuracy = correct / len(labeled) if labeled else 0.0
    positive_accuracy = pos_correct / pos_total if pos_total else 0.0
    negative_accuracy = neg_correct / neg_total if neg_total else 0.0
    loss = 1.0 - accuracy
    complexity = rule_complexity(rule, shape_index)
    return RuleEvaluation(
        loss=loss,
        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        free_energy=loss + lambda_value * complexity,
        complexity=complexity,
    )


def make_lambda_values(lambda_min: float, lambda_max: float, points: int) -> List[float]:
    if points <= 1:
        return [lambda_min]
    step = (lambda_max - lambda_min) / (points - 1)
    return [lambda_min + idx * step for idx in range(points)]


def candidate_rules(split: LogoSplit, feature_set: str, max_rule_atoms: int, max_candidate_atoms: int) -> List[LogoRule]:
    positive_features = [scene_features(scene, feature_set) for scene in split.positives]
    negative_features = [scene_features(scene, feature_set) for scene in split.negatives]
    if not positive_features:
        return [LogoRule(constant=False), LogoRule(constant=True)]
    shared_positive = set.intersection(*positive_features)

    def score_atom(atom: Atom) -> Tuple[float, int, str]:
        negative_hits = sum(atom in features for features in negative_features)
        negative_rate = negative_hits / max(1, len(negative_features))
        # Every candidate is true on all positives. Rank by how many negatives it excludes,
        # then prefer compact, non-slot predicates before exact signatures.
        coarse_bonus = 0.05 if atom.startswith("macro:") or atom.startswith("attr:") or atom.startswith("super:") else 0.0
        slot_penalty = 1 if atom.startswith("slot") else 0
        return (1.0 - negative_rate + coarse_bonus, -slot_penalty, atom)

    ranked_atoms = sorted(shared_positive, key=score_atom, reverse=True)
    if max_candidate_atoms > 0:
        ranked_atoms = ranked_atoms[:max_candidate_atoms]
    atoms = sorted(ranked_atoms)
    candidates = [LogoRule(constant=False), LogoRule(constant=True)]
    for size in range(1, max_rule_atoms + 1):
        for combo in itertools.combinations(atoms, size):
            candidates.append(LogoRule(atoms=tuple(combo)))
    return candidates


def validation_elbow(records: Sequence[Tuple[LogoRecord, LogoRule]]) -> Tuple[LogoRecord, LogoRule]:
    best_validation_loss = min(record.validation_loss for record, _rule in records)
    allowed = [(record, rule) for record, rule in records if record.validation_loss <= best_validation_loss]
    return min(allowed, key=lambda item: (item[0].complexity, item[0].train_loss, item[0].lambda_value))


def run_rule_selection(fewshot: LogoFewShotProblem, feature_set: str, lambda_values: Sequence[float], max_rule_atoms: int, max_candidate_atoms: int, shape_index: ShapeIndex) -> Tuple[LogoRecord, LogoRule]:
    candidates = candidate_rules(fewshot.train, feature_set, max_rule_atoms, max_candidate_atoms)
    selected_by_lambda: List[Tuple[LogoRecord, LogoRule]] = []
    for lambda_value in lambda_values:
        scored = []
        for rule in candidates:
            train_eval = evaluate_rule(rule, fewshot.train, feature_set, lambda_value, shape_index)
            scored.append((train_eval.free_energy, train_eval.loss, train_eval.complexity, rule, train_eval))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3].describe()))
        _free_energy, _train_loss, _complexity, selected_rule, train_eval = scored[0]
        validation_eval = evaluate_rule(selected_rule, fewshot.validation, feature_set, lambda_value, shape_index)
        hidden_eval = evaluate_rule(selected_rule, fewshot.hidden, feature_set, lambda_value, shape_index)
        selected_by_lambda.append(
            (
                LogoRecord(
                    problem_id=fewshot.problem.problem_id,
                    category=fewshot.problem.category,
                    concept=fewshot.problem.concept,
                    feature_set=feature_set,
                    lambda_value=lambda_value,
                    train_accuracy=train_eval.accuracy,
                    validation_accuracy=validation_eval.accuracy,
                    hidden_accuracy=hidden_eval.accuracy,
                    train_loss=train_eval.loss,
                    validation_loss=validation_eval.loss,
                    hidden_loss=hidden_eval.loss,
                    complexity=train_eval.complexity,
                    atom_count=len(selected_rule.atoms),
                    rule=selected_rule.describe(),
                ),
                selected_rule,
            )
        )
    return validation_elbow(selected_by_lambda)


def summarize_problem_corpus(problems: Sequence[LogoProblem]) -> str:
    category_counts = Counter(problem.category for problem in problems)
    object_counts = Counter(len(scene) for problem in problems for scene in problem.positives + problem.negatives)
    action_counts = Counter(len(obj.action_tokens) for problem in problems for scene in problem.positives + problem.negatives for obj in scene)
    concepts = Counter(problem.concept for problem in problems)
    return (
        f"loaded_problems={len(problems)} "
        f"categories={dict(category_counts)} "
        f"unique_concepts={len(concepts)} "
        f"object_counts={dict(object_counts)} "
        f"object_action_lengths={dict(action_counts)}"
    )


def print_records(records: Sequence[LogoRecord], show_rules: bool, summary_only: bool = False) -> None:
    if not summary_only:
        print("per_problem")
        print("problem_id,category,concept,feature_set,lambda,train_acc,val_acc,hidden_acc,train_loss,val_loss,hidden_loss,complexity,atoms,rule")
        for record in records:
            rule = record.rule if show_rules else compact_rule(record.rule)
            print(
                f"{record.problem_id},{record.category},{record.concept},{record.feature_set},{record.lambda_value:.6f},"
                f"{record.train_accuracy:.2f},{record.validation_accuracy:.2f},{record.hidden_accuracy:.2f},"
                f"{record.train_loss:.4f},{record.validation_loss:.4f},{record.hidden_loss:.4f},"
                f"{record.complexity:.1f},{record.atom_count},{csv_escape(rule)}"
            )
        print()

    print("summary")
    print("category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity")
    keys = sorted({(record.category, record.feature_set) for record in records})
    for category, feature_set in keys:
        subset = [record for record in records if record.category == category and record.feature_set == feature_set]
        print(
            f"{category},{feature_set},{len(subset)},"
            f"{mean(record.train_accuracy for record in subset):.3f},"
            f"{mean(record.validation_accuracy for record in subset):.3f},"
            f"{mean(record.hidden_accuracy for record in subset):.3f},"
            f"{sum(record.hidden_accuracy == 1.0 for record in subset)},"
            f"{mean(record.complexity for record in subset):.1f}"
        )


def compact_rule(rule: str, max_len: int = 96) -> str:
    return rule if len(rule) <= max_len else rule[: max_len - 3] + "..."


def csv_escape(value: str) -> str:
    if any(ch in value for ch in [",", "\n", '"']):
        return '"' + value.replace('"', '""') + '"'
    return value


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return math.nan
    return sum(items) / len(items)


def main() -> None:
    args = parse_args()
    shape_index = load_shape_index(args.dataset_dir)
    problems = generate_logo_problems(args, shape_index)
    print(summarize_problem_corpus(problems))
    lambda_values = make_lambda_values(args.lambda_min, args.lambda_max, args.lambda_points)
    if args.feature_set == "both":
        feature_sets = ("action", "metadata")
    elif args.feature_set == "all":
        feature_sets = ("action", "macro", "metadata")
    else:
        feature_sets = (args.feature_set,)
    records: List[LogoRecord] = []
    for problem in problems:
        fewshot = split_problem(problem, args.support_count, args.validation_count, args.hidden_count)
        for feature_set in feature_sets:
            record, _rule = run_rule_selection(fewshot, feature_set, lambda_values, args.max_rule_atoms, args.max_candidate_atoms, shape_index)
            records.append(record)
    print_records(records, show_rules=args.show_rules, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
