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

TASK_DEFINITIONS: Dict[str, Tuple[Tuple[Atom, ...], ...]] = {
    "solid_loop": (CORE_ABSTRACTION,),
    "solid_loop_curve": (CORE_ABSTRACTION + ("has_curve",),),
    "solid_loop_thin": (CORE_ABSTRACTION + ("thin",),),
    "solid_loop_symmetric": (CORE_ABSTRACTION + ("symmetric_hint",),),
    "solid_loop_many": (CORE_ABSTRACTION + ("many_segments",),),
    "curve_or_thin": (("has_curve",), ("thin",)),
    "solid_loop_curve_or_thin": (
        CORE_ABSTRACTION + ("has_curve",),
        CORE_ABSTRACTION + ("thin",),
    ),
}

SCENARIOS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "single": (("solid_loop_curve",), ("solid_loop_thin",)),
    "multi": (("solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"), ("solid_loop_many",)),
    "with_direct": (("solid_loop", "solid_loop_curve", "solid_loop_thin", "solid_loop_symmetric"), ("solid_loop_many",)),
    "or_control": (("curve_or_thin",), ("solid_loop_many",)),
    "or_factor": (("solid_loop_curve_or_thin",), ("solid_loop_symmetric",)),
}

CONDITIONS = ("inline", "shared", "no_share", "oracle")
MAX_DNF_CANDIDATE_CLAUSES = 80


@dataclass(frozen=True)
class Split:
    positives: Tuple[ObjectAtoms, ...]
    negatives: Tuple[ObjectAtoms, ...]


@dataclass(frozen=True)
class Task:
    name: str
    positive_clauses: Tuple[Tuple[Atom, ...], ...]
    train: Split
    validation: Split
    hidden: Split

    @property
    def required_atoms(self) -> Tuple[Atom, ...]:
        if len(self.positive_clauses) != 1:
            raise ValueError(f"{self.name} is disjunctive and has no single required-atom clause")
        return self.positive_clauses[0]


@dataclass(frozen=True)
class Macro:
    name: str
    atoms: Tuple[Atom, ...]

    def holds(self, obj: ObjectAtoms) -> bool:
        return all(atom in obj for atom in self.atoms)


@dataclass(frozen=True)
class Rule:
    atoms: Tuple[Atom, ...] = ()
    clauses: Tuple[Tuple[Atom, ...], ...] = ()
    constant: Optional[bool] = None

    def active_clauses(self) -> Tuple[Tuple[Atom, ...], ...]:
        if self.clauses:
            return self.clauses
        if self.atoms:
            return (self.atoms,)
        return ()

    def describe(self) -> str:
        if self.constant is True:
            return "CONST_TRUE"
        if self.constant is False:
            return "CONST_FALSE"
        clauses = self.active_clauses()
        if len(clauses) == 1:
            return " AND ".join(clauses[0])
        return " OR ".join("(" + " AND ".join(clause) + ")" for clause in clauses)


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


def predicate(obj: ObjectAtoms, positive_clauses: Sequence[Sequence[Atom]]) -> bool:
    return any(all(atom in obj for atom in clause) for clause in positive_clauses)


def split_candidates(
    universe: Sequence[ObjectAtoms], positive_clauses: Sequence[Sequence[Atom]]
) -> Tuple[List[ObjectAtoms], List[ObjectAtoms]]:
    positives = [obj for obj in universe if predicate(obj, positive_clauses)]
    negatives = [obj for obj in universe if not predicate(obj, positive_clauses)]
    return positives, negatives


def overlap_score(obj: ObjectAtoms, positive_clauses: Sequence[Sequence[Atom]]) -> int:
    return max((sum(atom in obj for atom in clause) for clause in positive_clauses), default=0)


def sample_side(
    rng: random.Random,
    candidates: Sequence[ObjectAtoms],
    positive_clauses: Sequence[Sequence[Atom]],
    count: int,
    label: bool,
    used: set[ObjectAtoms],
) -> Tuple[ObjectAtoms, ...]:
    available = [obj for obj in candidates if obj not in used]
    if label:
        rng.shuffle(available)
    else:
        available.sort(key=lambda obj: (overlap_score(obj, positive_clauses), rng.random()), reverse=True)
    if len(available) < count:
        raise RuntimeError("not enough unique examples for split")
    chosen = tuple(available[:count])
    used.update(chosen)
    return chosen


def make_task(name: str, seed: int, train_count: int, validation_count: int, hidden_count: int) -> Task:
    positive_clauses = TASK_DEFINITIONS[name]
    universe = make_universe()
    positives, negatives = split_candidates(universe, positive_clauses)
    rng = random.Random(seed)
    used_pos: set[ObjectAtoms] = set()
    used_neg: set[ObjectAtoms] = set()

    def make_split(count: int) -> Split:
        return Split(
            positives=sample_side(rng, positives, positive_clauses, count, True, used_pos),
            negatives=sample_side(rng, negatives, positive_clauses, count, False, used_neg),
        )

    return Task(
        name=name,
        positive_clauses=positive_clauses,
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
    if oracle_task is not None and predicate(obj, oracle_task.positive_clauses):
        features.add(oracle_feature_name(oracle_task))
    return frozenset(features)


def predict_rule(rule: Rule, obj: ObjectAtoms, macros: Sequence[Macro], oracle_task: Optional[Task] = None) -> bool:
    if rule.constant is not None:
        return rule.constant
    features = features_for_object(obj, macros, oracle_task=oracle_task)
    return any(all(atom in features for atom in clause) for clause in rule.active_clauses())


def evaluate_accuracy(rule: Rule, split: Split, macros: Sequence[Macro], oracle_task: Optional[Task] = None) -> float:
    labeled = list(iter_labeled(split))
    correct = sum(1 for obj, label in labeled if predict_rule(rule, obj, macros, oracle_task=oracle_task) == label)
    return correct / len(labeled) if labeled else 0.0


def macro_definition_complexity(macro: Macro, macro_overhead: float) -> float:
    return macro_overhead + float(len(macro.atoms))


def atom_reference_complexity(
    atom: Atom,
    macros: Sequence[Macro],
    call_cost: float,
    no_share: bool,
    macro_overhead: float,
) -> float:
    by_name = {macro_feature_name(macro): macro for macro in macros}
    macro = by_name.get(atom)
    if macro is None:
        return 1.0
    if no_share:
        return call_cost + macro_definition_complexity(macro, macro_overhead)
    return call_cost


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
    return rule_overhead + sum(
        atom_reference_complexity(
            atom,
            macros,
            call_cost=call_cost,
            no_share=no_share,
            macro_overhead=macro_overhead,
        )
        for clause in rule.active_clauses()
        for atom in clause
    )


@functools.lru_cache(maxsize=None)
def candidate_macros(max_macro_atoms: int) -> Tuple[Macro, ...]:
    macros: List[Macro] = []
    for size in range(2, max_macro_atoms + 1):
        for atoms in itertools.combinations(PRIMITIVE_ATOMS, size):
            name = "+".join(atoms)
            macros.append(Macro(name=name, atoms=tuple(atoms)))
    return tuple(macros)


@functools.lru_cache(maxsize=None)
def candidate_clauses(atom_pool: Sequence[Atom], max_atoms: int) -> Tuple[Tuple[Atom, ...], ...]:
    clauses: List[Tuple[Atom, ...]] = []
    for size in range(1, max_atoms + 1):
        clauses.extend(tuple(atoms) for atoms in itertools.combinations(sorted(atom_pool), size))
    return tuple(clauses)


@functools.lru_cache(maxsize=None)
def candidate_rules(atom_pool: Sequence[Atom], max_atoms: int, max_clauses: int) -> Tuple[Rule, ...]:
    clauses = candidate_clauses(tuple(atom_pool), max_atoms)
    rules: List[Rule] = [Rule(constant=False), Rule(constant=True)]
    rules.extend(Rule(atoms=clause) for clause in clauses)
    if max_clauses >= 2:
        for left_idx, left in enumerate(clauses):
            left_set = set(left)
            for right in clauses[left_idx + 1 :]:
                right_set = set(right)
                if left_set <= right_set or right_set <= left_set:
                    continue
                rules.append(Rule(clauses=(left, right)))
    return tuple(rules)


def clause_holds(clause: Sequence[Atom], features: Sequence[Atom]) -> bool:
    feature_set = set(features)
    return all(atom in feature_set for atom in clause)


def ranked_candidate_clauses(
    task: Task,
    atom_pool: Sequence[Atom],
    max_atoms: int,
    macros: Sequence[Macro],
    oracle: bool,
) -> Tuple[Tuple[Atom, ...], ...]:
    oracle_task = task if oracle else None
    positive_features = [features_for_object(obj, macros, oracle_task=oracle_task) for obj in task.train.positives]
    negative_features = [features_for_object(obj, macros, oracle_task=oracle_task) for obj in task.train.negatives]
    scored: List[Tuple[int, int, int, Tuple[Atom, ...]]] = []
    for clause in candidate_clauses(tuple(atom_pool), max_atoms):
        positive_hits = sum(clause_holds(clause, features) for features in positive_features)
        if positive_hits == 0:
            continue
        negative_hits = sum(clause_holds(clause, features) for features in negative_features)
        scored.append((negative_hits, -positive_hits, len(clause), clause))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return tuple(item[3] for item in scored[:MAX_DNF_CANDIDATE_CLAUSES])


def candidate_dnf_rules(
    task: Task,
    atom_pool: Sequence[Atom],
    max_atoms: int,
    macros: Sequence[Macro],
    oracle: bool,
) -> Tuple[Rule, ...]:
    clauses = ranked_candidate_clauses(task, tuple(atom_pool), max_atoms, tuple(macros), oracle)
    rules: List[Rule] = [Rule(constant=False), Rule(constant=True)]
    rules.extend(Rule(atoms=clause) for clause in clauses)
    for left_idx, left in enumerate(clauses):
        left_set = set(left)
        for right in clauses[left_idx + 1 :]:
            right_set = set(right)
            if left_set <= right_set or right_set <= left_set:
                continue
            rules.append(Rule(clauses=(left, right)))
    return tuple(rules)


def macros_from_repeated_inline_branches(
    tasks: Sequence[Task],
    max_inline_atoms: int,
    max_macro_atoms: int,
) -> Tuple[Macro, ...]:
    macros: List[Macro] = []
    seen: set[Tuple[Atom, ...]] = set()
    for task in tasks:
        if len(task.positive_clauses) < 2:
            continue
        rules = candidate_dnf_rules(task, PRIMITIVE_ATOMS, max_inline_atoms, (), False)
        scores = [
            score_rule(
                task,
                rule,
                (),
                call_cost=0.0,
                rule_overhead=1.0,
                no_share=False,
                macro_overhead=1.0,
                oracle=False,
            )
            for rule in rules
        ]
        best = min(scores, key=lambda score: (score.train_loss, score.complexity, score.rule.describe()))
        clauses = best.rule.active_clauses()
        if len(clauses) < 2:
            continue
        repeated_set = set.intersection(*(set(clause) for clause in clauses))
        repeated_atoms = [atom for atom in PRIMITIVE_ATOMS if atom in repeated_set]
        max_size = min(max_macro_atoms, len(repeated_atoms))
        for size in range(2, max_size + 1):
            for atoms in itertools.combinations(repeated_atoms, size):
                if atoms in seen:
                    continue
                seen.add(atoms)
                macros.append(Macro(name="+".join(atoms), atoms=atoms))
    return tuple(macros)


def candidate_macro_pool_for_tasks(
    tasks: Sequence[Task],
    max_inline_atoms: int,
    max_macro_atoms: int,
) -> Tuple[Macro, ...]:
    repeated_branch_macros = macros_from_repeated_inline_branches(tasks, max_inline_atoms, max_macro_atoms)
    if any(len(task.positive_clauses) > 1 for task in tasks):
        return repeated_branch_macros
    return candidate_macros(max_macro_atoms)


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
    max_clauses = 1 if oracle else min(2, max(1, len(task.positive_clauses)))
    if max_clauses == 1:
        rules = candidate_rules(tuple(atom_pool), max_atoms, max_clauses)
    else:
        rules = candidate_dnf_rules(task, tuple(atom_pool), max_atoms, tuple(macros), oracle)
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


def rule_uses_macro(rule: Rule, macro: Macro) -> bool:
    name = macro_feature_name(macro)
    return any(atom == name for clause in rule.active_clauses() for atom in clause)


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
    macro_used = macro is not None and any(rule_uses_macro(score.rule, macro) for score in task_scores)
    library_complexity = 0.0
    if charge_library and condition == "shared" and macro_used:
        library_complexity = macro_definition_complexity(macro, macro_overhead)
    train_loss = sum(score.train_loss for score in task_scores)
    validation_loss = sum(score.validation_loss for score in task_scores)
    hidden_loss = sum(score.hidden_loss for score in task_scores)
    complexity = library_complexity + sum(score.complexity for score in task_scores)
    return LibraryResult(
        condition=condition,
        scenario=scenario,
        lambda_value=lambda_value,
        macro=macro if condition in {"shared", "no_share"} and macro_used else None,
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
        macros = (None,) + candidate_macro_pool_for_tasks(tasks, max_inline_atoms, max_macro_atoms)
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
