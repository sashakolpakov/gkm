#!/usr/bin/env python3
"""Internal abstraction-emergence experiment for predicate libraries.

This is a controlled, non-visual Bongard-style scaffold. Each example is an
opaque object described only by primitive symbolic observations. The latent
regularity is not supplied as metadata. A solver can either repeat primitive
conditions inside every task rule or create a predicate macro and call it from
several task rules.

The free-energy objective is:

    F = total task loss + lambda * (library complexity + task rule complexity)

The intended diagnostic is not that predicate invention is new. It is whether a
shared predicate becomes cheaper than duplicated inline logic only when multiple
tasks reuse the same latent substructure.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Atom = str
ObjectAtoms = frozenset[Atom]
LabeledObject = Tuple[ObjectAtoms, bool]

PRIMITIVE_ATOMS: Tuple[Atom, ...] = (
    "low_closure_error",
    "high_hull_fill",
    "turn_balanced",
    "has_curve",
    "thin",
    "symmetric_hint",
    "many_segments",
    "acute_turns",
    "line_dominant",
    "large_scale",
)

CORE_ABSTRACTION: Tuple[Atom, ...] = (
    "low_closure_error",
    "high_hull_fill",
    "turn_balanced",
)

TASK_DEFINITIONS: Dict[str, Tuple[Atom, ...]] = {
    "solid_loop": CORE_ABSTRACTION,
    "solid_loop_curve": CORE_ABSTRACTION + ("has_curve",),
    "solid_loop_thin": CORE_ABSTRACTION + ("thin",),
    "solid_loop_symmetric": CORE_ABSTRACTION + ("symmetric_hint",),
    "solid_loop_many": CORE_ABSTRACTION + ("many_segments",),
}

SCENARIOS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "single": (("solid_loop_curve",), ("solid_loop_thin",)),
    "multi": (("solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"), ("solid_loop_many",)),
    "with_direct": (("solid_loop", "solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"), ("solid_loop_many",)),
}

CONDITIONS = ("inline", "shared", "no_share", "oracle")


@dataclass(frozen=True)
class Split:
    positives: Tuple[ObjectAtoms, ...]
    negatives: Tuple[ObjectAtoms, ...]


@dataclass(frozen=True)
class Task:
    name: str
    required_atoms: Tuple[Atom, ...]
    train: Split
    validation: Split
    hidden: Split


@dataclass(frozen=True)
class Macro:
    name: str
    atoms: Tuple[Atom, ...]

    def holds(self, obj: ObjectAtoms) -> bool:
        return all(atom in obj for atom in self.atoms)


@dataclass(frozen=True)
class Rule:
    atoms: Tuple[Atom, ...] = ()
    constant: Optional[bool] = None

    def describe(self) -> str:
        if self.constant is True:
            return "CONST_TRUE"
        if self.constant is False:
            return "CONST_FALSE"
        return " AND ".join(self.atoms)


@dataclass(frozen=True)
class RuleScore:
    rule: Rule
    train_loss: float
    validation_loss: float
    hidden_loss: float
    complexity: float
    train_accuracy: float
    validation_accuracy: float
    hidden_accuracy: float


@dataclass(frozen=True)
class LibraryResult:
    condition: str
    scenario: str
    lambda_value: float
    macro: Optional[Macro]
    train_loss: float
    validation_loss: float
    hidden_loss: float
    complexity: float
    free_energy: float
    exact_hidden_tasks: int
    task_count: int
    task_scores: Tuple[RuleScore, ...]

    @property
    def hidden_accuracy(self) -> float:
        total = 0.0
        for score in self.task_scores:
            total += score.hidden_accuracy
        return total / max(1, len(self.task_scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS) + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--train-count", type=int, default=18)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--hidden-count", type=int, default=32)
    parser.add_argument("--lambda-values", default="0.005,0.01,0.02,0.04")
    parser.add_argument("--max-inline-atoms", type=int, default=4)
    parser.add_argument("--max-solver-atoms", type=int, default=2)
    parser.add_argument("--max-macro-atoms", type=int, default=3)
    parser.add_argument("--call-cost", type=float, default=0.35)
    parser.add_argument("--macro-overhead", type=float, default=1.0)
    parser.add_argument("--rule-overhead", type=float, default=1.0)
    parser.add_argument("--show-rules", action="store_true")
    return parser.parse_args()


@functools.lru_cache(maxsize=1)
def make_universe() -> Tuple[ObjectAtoms, ...]:
    objects: List[ObjectAtoms] = []
    for mask in range(1 << len(PRIMITIVE_ATOMS)):
        atoms = frozenset(atom for idx, atom in enumerate(PRIMITIVE_ATOMS) if mask & (1 << idx))
        objects.append(atoms)
    return tuple(objects)


def predicate(obj: ObjectAtoms, required_atoms: Sequence[Atom]) -> bool:
    return all(atom in obj for atom in required_atoms)


def split_candidates(universe: Sequence[ObjectAtoms], required_atoms: Sequence[Atom]) -> Tuple[List[ObjectAtoms], List[ObjectAtoms]]:
    positives = [obj for obj in universe if predicate(obj, required_atoms)]
    negatives = [obj for obj in universe if not predicate(obj, required_atoms)]
    return positives, negatives


def overlap_score(obj: ObjectAtoms, required_atoms: Sequence[Atom]) -> int:
    return sum(atom in obj for atom in required_atoms)


def sample_side(
    rng: random.Random,
    candidates: Sequence[ObjectAtoms],
    required_atoms: Sequence[Atom],
    count: int,
    label: bool,
    used: set[ObjectAtoms],
) -> Tuple[ObjectAtoms, ...]:
    available = [obj for obj in candidates if obj not in used]
    if label:
        rng.shuffle(available)
    else:
        available.sort(key=lambda obj: (overlap_score(obj, required_atoms), rng.random()), reverse=True)
    if len(available) < count:
        raise RuntimeError("not enough unique examples for split")
    chosen = tuple(available[:count])
    used.update(chosen)
    return chosen


def make_task(name: str, seed: int, train_count: int, validation_count: int, hidden_count: int) -> Task:
    required_atoms = TASK_DEFINITIONS[name]
    universe = make_universe()
    positives, negatives = split_candidates(universe, required_atoms)
    rng = random.Random(seed)
    used_pos: set[ObjectAtoms] = set()
    used_neg: set[ObjectAtoms] = set()

    def make_split(count: int) -> Split:
        return Split(
            positives=sample_side(rng, positives, required_atoms, count, True, used_pos),
            negatives=sample_side(rng, negatives, required_atoms, count, False, used_neg),
        )

    return Task(
        name=name,
        required_atoms=required_atoms,
        train=make_split(train_count),
        validation=make_split(validation_count),
        hidden=make_split(hidden_count),
    )


def iter_labeled(split: Split) -> Iterable[LabeledObject]:
    for obj in split.positives:
        yield obj, True
    for obj in split.negatives:
        yield obj, False


def macro_feature_name(macro: Macro) -> Atom:
    return f"macro:{macro.name}"


def oracle_feature_name(task: Task) -> Atom:
    return f"oracle:{task.name}"


def features_for_object(obj: ObjectAtoms, macros: Sequence[Macro], oracle_task: Optional[Task] = None) -> frozenset[Atom]:
    features = set(obj)
    for macro in macros:
        if macro.holds(obj):
            features.add(macro_feature_name(macro))
    if oracle_task is not None and predicate(obj, oracle_task.required_atoms):
        features.add(oracle_feature_name(oracle_task))
    return frozenset(features)


def predict_rule(rule: Rule, obj: ObjectAtoms, macros: Sequence[Macro], oracle_task: Optional[Task] = None) -> bool:
    if rule.constant is not None:
        return rule.constant
    features = features_for_object(obj, macros, oracle_task=oracle_task)
    return all(atom in features for atom in rule.atoms)


def evaluate_accuracy(rule: Rule, split: Split, macros: Sequence[Macro], oracle_task: Optional[Task] = None) -> float:
    labeled = list(iter_labeled(split))
    correct = sum(1 for obj, label in labeled if predict_rule(rule, obj, macros, oracle_task=oracle_task) == label)
    return correct / len(labeled) if labeled else 0.0


def macro_definition_complexity(macro: Macro, macro_overhead: float) -> float:
    return macro_overhead + float(len(macro.atoms))


def rule_complexity(
    rule: Rule,
    macros: Sequence[Macro],
    call_cost: float,
    rule_overhead: float,
    no_share: bool = False,
    macro_overhead: float = 1.0,
) -> float:
    if rule.constant is not None:
        return 0.0
    by_name = {macro_feature_name(macro): macro for macro in macros}
    total = rule_overhead
    for atom in rule.atoms:
        macro = by_name.get(atom)
        if macro is None:
            total += 1.0
        elif no_share:
            total += call_cost + macro_definition_complexity(macro, macro_overhead)
        else:
            total += call_cost
    return total


@functools.lru_cache(maxsize=None)
def candidate_macros(max_macro_atoms: int) -> Tuple[Macro, ...]:
    macros: List[Macro] = []
    for size in range(2, max_macro_atoms + 1):
        for atoms in itertools.combinations(PRIMITIVE_ATOMS, size):
            name = "+".join(atoms)
            macros.append(Macro(name=name, atoms=tuple(atoms)))
    return tuple(macros)


@functools.lru_cache(maxsize=None)
def candidate_rules(atom_pool: Sequence[Atom], max_atoms: int) -> Tuple[Rule, ...]:
    rules: List[Rule] = [Rule(constant=False), Rule(constant=True)]
    for size in range(1, max_atoms + 1):
        for atoms in itertools.combinations(sorted(atom_pool), size):
            rules.append(Rule(atoms=tuple(atoms)))
    return tuple(rules)


@functools.lru_cache(maxsize=None)
def score_rule(
    task: Task,
    rule: Rule,
    macros: Sequence[Macro],
    call_cost: float,
    rule_overhead: float,
    no_share: bool,
    macro_overhead: float,
    oracle: bool = False,
) -> RuleScore:
    oracle_task = task if oracle else None
    train_accuracy = evaluate_accuracy(rule, task.train, macros, oracle_task=oracle_task)
    validation_accuracy = evaluate_accuracy(rule, task.validation, macros, oracle_task=oracle_task)
    hidden_accuracy = evaluate_accuracy(rule, task.hidden, macros, oracle_task=oracle_task)
    return RuleScore(
        rule=rule,
        train_loss=1.0 - train_accuracy,
        validation_loss=1.0 - validation_accuracy,
        hidden_loss=1.0 - hidden_accuracy,
        complexity=rule_complexity(
            rule,
            macros,
            call_cost=call_cost,
            rule_overhead=rule_overhead,
            no_share=no_share,
            macro_overhead=macro_overhead,
        ),
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        hidden_accuracy=hidden_accuracy,
    )


def select_best_task_rule(
    task: Task,
    macros: Sequence[Macro],
    condition: str,
    lambda_value: float,
    max_inline_atoms: int,
    max_solver_atoms: int,
    call_cost: float,
    rule_overhead: float,
    macro_overhead: float,
) -> RuleScore:
    if condition == "inline":
        atom_pool = PRIMITIVE_ATOMS
        max_atoms = max_inline_atoms
        no_share = False
        oracle = False
    elif condition == "oracle":
        atom_pool = tuple(PRIMITIVE_ATOMS) + (oracle_feature_name(task),)
        max_atoms = 1
        no_share = False
        oracle = True
    else:
        atom_pool = tuple(PRIMITIVE_ATOMS) + tuple(macro_feature_name(macro) for macro in macros)
        # If no library predicate is present, the shared-library condition must
        # remain allowed to solve the task inline. Otherwise macro use would be
        # forced by an artificial rule-width bottleneck rather than selected by
        # the free-energy complexity tradeoff.
        max_atoms = max_solver_atoms if macros else max_inline_atoms
        no_share = condition == "no_share"
        oracle = False
    rules = candidate_rules(atom_pool, max_atoms)
    scores = [
        score_rule(
            task,
            rule,
            macros,
            call_cost=call_cost,
            rule_overhead=rule_overhead,
            no_share=no_share,
            macro_overhead=macro_overhead,
            oracle=oracle,
        )
        for rule in rules
    ]
    return min(scores, key=lambda score: (score.train_loss + lambda_value * score.complexity, score.train_loss, score.complexity, score.rule.describe()))


def evaluate_library(
    scenario: str,
    tasks: Sequence[Task],
    condition: str,
    macro: Optional[Macro],
    lambda_value: float,
    max_inline_atoms: int,
    max_solver_atoms: int,
    call_cost: float,
    macro_overhead: float,
    rule_overhead: float,
    charge_library: bool = True,
) -> LibraryResult:
    macros: Tuple[Macro, ...]
    if condition in {"shared", "no_share"} and macro is not None:
        macros = (macro,)
    else:
        macros = ()
    task_scores = tuple(
        select_best_task_rule(
            task,
            macros,
            condition=condition,
            lambda_value=lambda_value,
            max_inline_atoms=max_inline_atoms,
            max_solver_atoms=max_solver_atoms,
            call_cost=call_cost,
            rule_overhead=rule_overhead,
            macro_overhead=macro_overhead,
        )
        for task in tasks
    )
    library_complexity = 0.0
    if charge_library and condition == "shared" and macro is not None:
        library_complexity = macro_definition_complexity(macro, macro_overhead)
    train_loss = sum(score.train_loss for score in task_scores)
    validation_loss = sum(score.validation_loss for score in task_scores)
    hidden_loss = sum(score.hidden_loss for score in task_scores)
    complexity = library_complexity + sum(score.complexity for score in task_scores)
    return LibraryResult(
        condition=condition,
        scenario=scenario,
        lambda_value=lambda_value,
        macro=macro if condition in {"shared", "no_share"} else None,
        train_loss=train_loss,
        validation_loss=validation_loss,
        hidden_loss=hidden_loss,
        complexity=complexity,
        free_energy=train_loss + lambda_value * complexity,
        exact_hidden_tasks=sum(score.hidden_accuracy == 1.0 for score in task_scores),
        task_count=len(tasks),
        task_scores=task_scores,
    )


def select_condition_result(
    scenario: str,
    tasks: Sequence[Task],
    condition: str,
    lambda_values: Sequence[float],
    max_inline_atoms: int,
    max_solver_atoms: int,
    max_macro_atoms: int,
    call_cost: float,
    macro_overhead: float,
    rule_overhead: float,
) -> LibraryResult:
    macros: Tuple[Optional[Macro], ...]
    if condition in {"shared", "no_share"}:
        macros = (None,) + candidate_macros(max_macro_atoms)
    else:
        macros = (None,)

    train_selected: List[LibraryResult] = []
    for lambda_value in lambda_values:
        candidates = [
            evaluate_library(
                scenario,
                tasks,
                condition,
                macro,
                lambda_value=lambda_value,
                max_inline_atoms=max_inline_atoms,
                max_solver_atoms=max_solver_atoms,
                call_cost=call_cost,
                macro_overhead=macro_overhead,
                rule_overhead=rule_overhead,
            )
            for macro in macros
        ]
        train_selected.append(
            min(candidates, key=lambda result: (result.free_energy, result.train_loss, result.complexity, describe_macro(result.macro)))
        )
    best_validation = min(result.validation_loss for result in train_selected)
    allowed = [result for result in train_selected if result.validation_loss <= best_validation]
    return min(allowed, key=lambda result: (result.complexity, result.hidden_loss, result.lambda_value, describe_macro(result.macro)))


def select_with_fixed_macro(
    scenario: str,
    tasks: Sequence[Task],
    condition: str,
    macro: Optional[Macro],
    lambda_values: Sequence[float],
    max_inline_atoms: int,
    max_solver_atoms: int,
    call_cost: float,
    macro_overhead: float,
    rule_overhead: float,
    charge_library: bool = True,
) -> LibraryResult:
    candidates = [
        evaluate_library(
            scenario,
            tasks,
            condition,
            macro,
            lambda_value=lambda_value,
            max_inline_atoms=max_inline_atoms,
            max_solver_atoms=max_solver_atoms,
            call_cost=call_cost,
            macro_overhead=macro_overhead,
            rule_overhead=rule_overhead,
            charge_library=charge_library,
        )
        for lambda_value in lambda_values
    ]
    best_validation = min(result.validation_loss for result in candidates)
    allowed = [result for result in candidates if result.validation_loss <= best_validation]
    return min(allowed, key=lambda result: (result.complexity, result.hidden_loss, result.lambda_value))


def make_tasks(task_names: Sequence[str], seed: int, train_count: int, validation_count: int, hidden_count: int) -> Tuple[Task, ...]:
    return tuple(
        make_task(
            name,
            seed=seed + 997 * idx,
            train_count=train_count,
            validation_count=validation_count,
            hidden_count=hidden_count,
        )
        for idx, name in enumerate(task_names)
    )


def parse_lambda_values(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one lambda value is required")
    return values


def describe_macro(macro: Optional[Macro]) -> str:
    if macro is None:
        return "none"
    return " AND ".join(macro.atoms)


def print_result(result: LibraryResult) -> None:
    print(
        f"{result.scenario},{result.condition},{result.lambda_value:.4f},"
        f"{result.train_loss:.4f},{result.validation_loss:.4f},{result.hidden_loss:.4f},"
        f"{result.hidden_accuracy:.3f},{result.exact_hidden_tasks}/{result.task_count},"
        f"{result.complexity:.2f},{result.free_energy:.4f},{describe_macro(result.macro)}"
    )


def print_rules(result: LibraryResult) -> None:
    print(f"\n{result.scenario} / {result.condition} / macro={describe_macro(result.macro)}")
    for score in result.task_scores:
        print(
            f"  rule={score.rule.describe()} "
            f"train={score.train_accuracy:.2f} val={score.validation_accuracy:.2f} "
            f"hidden={score.hidden_accuracy:.2f} complexity={score.complexity:.2f}"
        )


def run_scenario(args: argparse.Namespace, scenario: str, lambda_values: Sequence[float]) -> Tuple[LibraryResult, ...]:
    support_names, transfer_names = SCENARIOS[scenario]
    support_tasks = make_tasks(support_names, args.seed, args.train_count, args.validation_count, args.hidden_count)
    results: List[LibraryResult] = []
    for condition in CONDITIONS:
        results.append(
            select_condition_result(
                scenario,
                support_tasks,
                condition,
                lambda_values=lambda_values,
                max_inline_atoms=args.max_inline_atoms,
                max_solver_atoms=args.max_solver_atoms,
                max_macro_atoms=args.max_macro_atoms,
                call_cost=args.call_cost,
                macro_overhead=args.macro_overhead,
                rule_overhead=args.rule_overhead,
            )
        )

    shared = next(result for result in results if result.condition == "shared")
    transfer_tasks = make_tasks(transfer_names, args.seed + 10_000, args.train_count, args.validation_count, args.hidden_count)
    for condition in ("inline", "shared", "no_share", "oracle"):
        macro = shared.macro if condition in {"shared", "no_share"} else None
        results.append(
            select_with_fixed_macro(
                f"{scenario}_transfer",
                transfer_tasks,
                condition,
                macro,
                lambda_values=lambda_values,
                max_inline_atoms=args.max_inline_atoms,
                max_solver_atoms=args.max_solver_atoms,
                call_cost=args.call_cost,
                macro_overhead=args.macro_overhead,
                rule_overhead=args.rule_overhead,
                charge_library=condition != "shared",
            )
        )
    return tuple(results)


def main() -> None:
    args = parse_args()
    lambda_values = parse_lambda_values(args.lambda_values)
    scenarios = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    print("scenario,condition,lambda,train_loss,val_loss,hidden_loss,mean_hidden_acc,exact_hidden,complexity,free_energy,macro")
    all_results: List[LibraryResult] = []
    for scenario in scenarios:
        results = run_scenario(args, scenario, lambda_values)
        all_results.extend(results)
        for result in results:
            print_result(result)
    if args.show_rules:
        for result in all_results:
            print_rules(result)


if __name__ == "__main__":
    main()
