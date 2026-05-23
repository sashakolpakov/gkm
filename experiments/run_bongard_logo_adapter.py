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
    parser.add_argument("--feature-set", choices=("action", "metadata", "both"), default="both")
    parser.add_argument("--limit", type=int, default=12, help="maximum generated problems per selected source")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--support-count", type=int, default=4, help="positive and negative examples used for training")
    parser.add_argument("--validation-count", type=int, default=1, help="positive and negative examples used for validation elbow selection")
    parser.add_argument("--hidden-count", type=int, default=2, help="positive and negative examples held out until after selection")
    parser.add_argument("--max-rule-atoms", type=int, default=2, help="maximum conjunction size in the sparse feature rule")
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
    if atom.startswith("object_count="):
        return 1.0
    if atom.startswith("action_count="):
        return 1.0
    if atom in {"has_line", "no_line", "has_arc", "no_arc"}:
        return 1.0
    if atom.startswith("type_sequence:"):
        return float(atom.count(",") + 2)
    return 2.0


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


def candidate_rules(split: LogoSplit, feature_set: str, max_rule_atoms: int) -> List[LogoRule]:
    positive_features = [scene_features(scene, feature_set) for scene in split.positives]
    if not positive_features:
        return [LogoRule(constant=False), LogoRule(constant=True)]
    shared_positive = set.intersection(*positive_features)
    candidates = [LogoRule(constant=False), LogoRule(constant=True)]
    atoms = sorted(shared_positive)
    for size in range(1, max_rule_atoms + 1):
        for combo in itertools.combinations(atoms, size):
            candidates.append(LogoRule(atoms=tuple(combo)))
    return candidates


def validation_elbow(records: Sequence[Tuple[LogoRecord, LogoRule]]) -> Tuple[LogoRecord, LogoRule]:
    best_validation_loss = min(record.validation_loss for record, _rule in records)
    allowed = [(record, rule) for record, rule in records if record.validation_loss <= best_validation_loss]
    return min(allowed, key=lambda item: (item[0].complexity, item[0].train_loss, item[0].lambda_value))


def run_rule_selection(fewshot: LogoFewShotProblem, feature_set: str, lambda_values: Sequence[float], max_rule_atoms: int, shape_index: ShapeIndex) -> Tuple[LogoRecord, LogoRule]:
    candidates = candidate_rules(fewshot.train, feature_set, max_rule_atoms)
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
    feature_sets = ("action", "metadata") if args.feature_set == "both" else (args.feature_set,)
    records: List[LogoRecord] = []
    for problem in problems:
        fewshot = split_problem(problem, args.support_count, args.validation_count, args.hidden_count)
        for feature_set in feature_sets:
            record, _rule = run_rule_selection(fewshot, feature_set, lambda_values, args.max_rule_atoms, shape_index)
            records.append(record)
    print_records(records, show_rules=args.show_rules, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
