#!/usr/bin/env python3
"""Procedural Bongard-style concept-induction baseline.

This is a local harness for the Bongard-LOGO direction. It does not load the
external dataset. It generates positive/negative examples over opaque object
sequences, uses disjoint train/validation/hidden object pools, sweeps lambda,
and selects a compact deterministic hypothesis by validation loss/complexity.

Run from the repository root:

    python3 experiments/run_bongard_symbolic_baseline.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Symbol = int
Example = Tuple[Symbol, ...]


@dataclass(frozen=True)
class Concept:
    name: str
    predicate: Callable[[Example], bool]
    complexity: float
    primitive_tier: str


@dataclass(frozen=True)
class Split:
    positives: Tuple[Example, ...]
    negatives: Tuple[Example, ...]


@dataclass(frozen=True)
class Problem:
    target: Concept
    train: Split
    validation: Split
    hidden_test: Split


@dataclass(frozen=True)
class LambdaRecord:
    lambda_value: float
    selected_rule: str
    train_loss: float
    validation_loss: float
    hidden_loss: float
    train_accuracy: float
    validation_accuracy: float
    hidden_accuracy: float
    complexity: float
    selected: bool = False


def length_even(example: Example) -> bool:
    return len(example) % 2 == 0


def has_adjacent_duplicate(example: Example) -> bool:
    return any(left == right for left, right in zip(example, example[1:]))


def first_equals_last(example: Example) -> bool:
    return bool(example) and example[0] == example[-1]


def first_equals_penultimate(example: Example) -> bool:
    return len(example) >= 3 and example[0] == example[-2]


def second_equals_last(example: Example) -> bool:
    return len(example) >= 3 and example[1] == example[-1]


def first_equals_second(example: Example) -> bool:
    return len(example) >= 2 and example[0] == example[1]


def last_two_equal(example: Example) -> bool:
    return len(example) >= 2 and example[-2] == example[-1]


def second_equals_penultimate(example: Example) -> bool:
    return len(example) >= 4 and example[1] == example[-2]


def length_multiple_of_three(example: Example) -> bool:
    return len(example) % 3 == 0


def is_palindrome(example: Example) -> bool:
    return len(example) >= 2 and example == tuple(reversed(example))


def contains_duplicate(example: Example) -> bool:
    return len(set(example)) < len(example)


def all_unique(example: Example) -> bool:
    return len(set(example)) == len(example)


CONCEPTS = (
    Concept("length_even", length_even, complexity=2.0, primitive_tier="stream"),
    Concept("length_multiple_of_three", length_multiple_of_three, complexity=3.0, primitive_tier="stream"),
    Concept("first_equals_second", first_equals_second, complexity=4.0, primitive_tier="compare"),
    Concept("last_two_equal", last_two_equal, complexity=4.0, primitive_tier="bidirectional_compare"),
    Concept("has_adjacent_duplicate", has_adjacent_duplicate, complexity=5.0, primitive_tier="compare"),
    Concept("first_equals_last", first_equals_last, complexity=6.0, primitive_tier="bidirectional_compare"),
    Concept("first_equals_penultimate", first_equals_penultimate, complexity=7.0, primitive_tier="bidirectional_compare"),
    Concept("second_equals_last", second_equals_last, complexity=7.0, primitive_tier="bidirectional_compare"),
    Concept("second_equals_penultimate", second_equals_penultimate, complexity=7.0, primitive_tier="bidirectional_compare"),
    Concept("palindrome", is_palindrome, complexity=8.0, primitive_tier="bidirectional_compare"),
    Concept("contains_duplicate", contains_duplicate, complexity=9.0, primitive_tier="compare_memory"),
    Concept("all_unique", all_unique, complexity=9.0, primitive_tier="compare_memory"),
)


def random_example(rng: random.Random, object_pool: Sequence[Symbol], min_length: int, max_length: int) -> Example:
    length = rng.randint(min_length, max_length)
    return tuple(rng.choice(object_pool) for _ in range(length))


def nonmatching_symbol(rng: random.Random, object_pool: Sequence[Symbol], symbol: Symbol) -> Symbol:
    alternatives = [item for item in object_pool if item != symbol]
    return rng.choice(alternatives or list(object_pool))


def hard_first_equals_last(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    first = rng.choice(object_pool)
    if label:
        middle = [nonmatching_symbol(rng, object_pool, first) for _ in range(length - 2)]
        return tuple([first] + middle + [first])
    last = nonmatching_symbol(rng, object_pool, first)
    middle = [rng.choice(object_pool) for _ in range(length - 2)]
    middle[rng.randrange(len(middle))] = first
    return tuple([first] + middle + [last])


def hard_first_equals_penultimate(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    first = rng.choice(object_pool)
    if label:
        middle = [nonmatching_symbol(rng, object_pool, first) for _ in range(length - 3)]
        last = nonmatching_symbol(rng, object_pool, first)
        return tuple([first] + middle + [first, last])
    penultimate = nonmatching_symbol(rng, object_pool, first)
    middle = [rng.choice(object_pool) for _ in range(length - 3)]
    if middle:
        middle[rng.randrange(len(middle))] = first
    last = first
    return tuple([first] + middle + [penultimate, last])


def hard_second_equals_last(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    second = rng.choice(object_pool)
    first = nonmatching_symbol(rng, object_pool, second)
    if label:
        middle = [nonmatching_symbol(rng, object_pool, second) for _ in range(length - 3)]
        return tuple([first, second] + middle + [second])
    last = nonmatching_symbol(rng, object_pool, second)
    middle = [rng.choice(object_pool) for _ in range(length - 3)]
    if middle:
        middle[rng.randrange(len(middle))] = second
    return tuple([second, second] + middle + [last])


def hard_first_equals_second(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    first = rng.choice(object_pool)
    if label:
        rest = [nonmatching_symbol(rng, object_pool, first) for _ in range(length - 2)]
        return tuple([first, first] + rest)
    second = nonmatching_symbol(rng, object_pool, first)
    rest = [rng.choice(object_pool) for _ in range(length - 2)]
    if rest:
        rest[rng.randrange(len(rest))] = first
    return tuple([first, second] + rest)


def hard_last_two_equal(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    last = rng.choice(object_pool)
    prefix = [nonmatching_symbol(rng, object_pool, last) for _ in range(length - 2)]
    if label:
        return tuple(prefix + [last, last])
    penultimate = nonmatching_symbol(rng, object_pool, last)
    if prefix:
        prefix[rng.randrange(len(prefix))] = last
    return tuple(prefix + [penultimate, last])


def hard_second_equals_penultimate(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(4, min_length), max(4, max_length))
    second = rng.choice(object_pool)
    first = nonmatching_symbol(rng, object_pool, second)
    last = nonmatching_symbol(rng, object_pool, second)
    if label:
        middle = [nonmatching_symbol(rng, object_pool, second) for _ in range(length - 4)]
        return tuple([first, second] + middle + [second, last])
    penultimate = nonmatching_symbol(rng, object_pool, second)
    middle = [rng.choice(object_pool) for _ in range(length - 4)]
    if middle:
        middle[rng.randrange(len(middle))] = second
    return tuple([first, second] + middle + [penultimate, last])


def hard_length_multiple_of_three(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    valid_lengths = [length for length in range(min_length, max_length + 1) if (length % 3 == 0) == label]
    length = rng.choice(valid_lengths or list(range(min_length, max_length + 1)))
    return tuple(rng.choice(object_pool) for _ in range(length))


def hard_has_adjacent_duplicate(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    if label:
        example = [rng.choice(object_pool) for _ in range(length)]
        pos = rng.randrange(length - 1)
        example[pos + 1] = example[pos]
        return tuple(example)
    while True:
        first = rng.choice(object_pool)
        second = nonmatching_symbol(rng, object_pool, first)
        example = [first, second, first]
        while len(example) < length:
            candidates = [item for item in object_pool if item != example[-1]]
            example.append(rng.choice(candidates or list(object_pool)))
        result = tuple(example)
        if not has_adjacent_duplicate(result) and contains_duplicate(result):
            return result


def hard_palindrome(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(4, min_length), max(4, max_length))
    if label:
        half = [rng.choice(object_pool) for _ in range((length + 1) // 2)]
        return tuple(half + half[: length // 2][::-1])
    while True:
        first = rng.choice(object_pool)
        middle = [rng.choice(object_pool) for _ in range(length - 2)]
        result = tuple([first] + middle + [first])
        if not is_palindrome(result):
            return result


def hard_duplicate_memory(
    rng: random.Random, object_pool: Sequence[Symbol], label: bool, min_length: int, max_length: int
) -> Example:
    length = rng.randint(max(3, min_length), max(3, max_length))
    if label:
        return hard_has_adjacent_duplicate(rng, object_pool, False, length, length)
    example: List[Symbol] = []
    for _idx in range(length):
        candidates = [item for item in object_pool if item not in example]
        example.append(rng.choice(candidates or list(object_pool)))
    return tuple(example)


def hard_example(
    rng: random.Random,
    concept: Concept,
    object_pool: Sequence[Symbol],
    label: bool,
    min_length: int,
    max_length: int,
) -> Example:
    if concept.name == "first_equals_last":
        return hard_first_equals_last(rng, object_pool, label, min_length, max_length)
    if concept.name == "first_equals_penultimate":
        return hard_first_equals_penultimate(rng, object_pool, label, min_length, max_length)
    if concept.name == "second_equals_last":
        return hard_second_equals_last(rng, object_pool, label, min_length, max_length)
    if concept.name == "first_equals_second":
        return hard_first_equals_second(rng, object_pool, label, min_length, max_length)
    if concept.name == "last_two_equal":
        return hard_last_two_equal(rng, object_pool, label, min_length, max_length)
    if concept.name == "second_equals_penultimate":
        return hard_second_equals_penultimate(rng, object_pool, label, min_length, max_length)
    if concept.name == "length_multiple_of_three":
        return hard_length_multiple_of_three(rng, object_pool, label, min_length, max_length)
    if concept.name == "has_adjacent_duplicate":
        return hard_has_adjacent_duplicate(rng, object_pool, label, min_length, max_length)
    if concept.name == "palindrome":
        return hard_palindrome(rng, object_pool, label, min_length, max_length)
    if concept.name == "contains_duplicate":
        return hard_duplicate_memory(rng, object_pool, label, min_length, max_length)
    if concept.name == "all_unique":
        return hard_duplicate_memory(rng, object_pool, not label, min_length, max_length)
    return random_example(rng, object_pool, min_length, max_length)


def sample_examples(
    rng: random.Random,
    concept: Concept,
    object_pool: Sequence[Symbol],
    count: int,
    label: bool,
    min_length: int,
    max_length: int,
    counterexample_rich: bool = False,
) -> Tuple[Example, ...]:
    examples: List[Example] = []
    seen = set()
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError(f"could not sample {count} examples for {concept.name}={label}")
        if counterexample_rich:
            example = hard_example(rng, concept, object_pool, label, min_length, max_length)
        else:
            example = random_example(rng, object_pool, min_length, max_length)
        if example in seen or concept.predicate(example) != label:
            continue
        seen.add(example)
        examples.append(example)
    return tuple(examples)


def make_split(
    rng: random.Random,
    concept: Concept,
    object_pool: Sequence[Symbol],
    positives: int,
    negatives: int,
    min_length: int,
    max_length: int,
    counterexample_rich: bool = False,
) -> Split:
    return Split(
        positives=sample_examples(
            rng, concept, object_pool, positives, True, min_length, max_length, counterexample_rich
        ),
        negatives=sample_examples(
            rng, concept, object_pool, negatives, False, min_length, max_length, counterexample_rich
        ),
    )


def make_problem(
    concept: Concept,
    seed: int,
    train_count: int = 6,
    validation_count: int = 4,
    hidden_count: int = 12,
    objects_per_split: int = 8,
    min_length: int = 2,
    max_length: int = 5,
    counterexample_train: bool = False,
    counterexample_validation: bool = True,
    train_positive_count: Optional[int] = None,
    train_negative_count: Optional[int] = None,
    validation_positive_count: Optional[int] = None,
    validation_negative_count: Optional[int] = None,
    hidden_positive_count: Optional[int] = None,
    hidden_negative_count: Optional[int] = None,
) -> Problem:
    rng = random.Random(seed)
    train_pool = tuple(range(objects_per_split))
    validation_pool = tuple(range(objects_per_split, 2 * objects_per_split))
    hidden_pool = tuple(range(2 * objects_per_split, 3 * objects_per_split))
    train_positive_count = train_count if train_positive_count is None else train_positive_count
    train_negative_count = train_count if train_negative_count is None else train_negative_count
    validation_positive_count = validation_count if validation_positive_count is None else validation_positive_count
    validation_negative_count = validation_count if validation_negative_count is None else validation_negative_count
    hidden_positive_count = hidden_count if hidden_positive_count is None else hidden_positive_count
    hidden_negative_count = hidden_count if hidden_negative_count is None else hidden_negative_count
    return Problem(
        target=concept,
        train=make_split(
            rng,
            concept,
            train_pool,
            train_positive_count,
            train_negative_count,
            min_length,
            max_length,
            counterexample_train,
        ),
        validation=make_split(
            rng,
            concept,
            validation_pool,
            validation_positive_count,
            validation_negative_count,
            min_length,
            max_length,
            counterexample_validation,
        ),
        hidden_test=make_split(
            rng,
            concept,
            hidden_pool,
            hidden_positive_count,
            hidden_negative_count,
            min_length,
            max_length,
            counterexample_validation,
        ),
    )


def iter_labeled(split: Split) -> Iterable[Tuple[Example, bool]]:
    for example in split.positives:
        yield example, True
    for example in split.negatives:
        yield example, False


def accuracy(concept: Concept, split: Split) -> float:
    labeled = list(iter_labeled(split))
    correct = sum(1 for example, label in labeled if concept.predicate(example) == label)
    return correct / len(labeled)


def loss(concept: Concept, split: Split) -> float:
    return 1.0 - accuracy(concept, split)


def make_lambda_values(lambda_min: float, lambda_max: float, points: int) -> List[float]:
    if points <= 1:
        return [lambda_min]
    step = (lambda_max - lambda_min) / (points - 1)
    return [lambda_min + idx * step for idx in range(points)]


def select_for_lambda(problem: Problem, lambda_value: float) -> Concept:
    return min(
        CONCEPTS,
        key=lambda concept: (
            loss(concept, problem.train) + lambda_value * concept.complexity,
            concept.complexity,
            concept.name,
        ),
    )


def validation_elbow(records: Sequence[LambdaRecord], tolerance: float = 0.0) -> LambdaRecord:
    best_validation_loss = min(record.validation_loss for record in records)
    allowed = [record for record in records if record.validation_loss <= best_validation_loss + tolerance]
    return min(allowed, key=lambda record: (record.complexity, record.validation_loss, record.train_loss))


def run_problem(problem: Problem, lambda_values: Sequence[float]) -> List[LambdaRecord]:
    records: List[LambdaRecord] = []
    for lambda_value in lambda_values:
        selected = select_for_lambda(problem, lambda_value)
        records.append(
            LambdaRecord(
                lambda_value=lambda_value,
                selected_rule=selected.name,
                train_loss=loss(selected, problem.train),
                validation_loss=loss(selected, problem.validation),
                hidden_loss=loss(selected, problem.hidden_test),
                train_accuracy=accuracy(selected, problem.train),
                validation_accuracy=accuracy(selected, problem.validation),
                hidden_accuracy=accuracy(selected, problem.hidden_test),
                complexity=selected.complexity,
            )
        )
    chosen = validation_elbow(records)
    return [record if record is not chosen else LambdaRecord(**{**record.__dict__, "selected": True}) for record in records]


def main() -> None:
    lambda_values = make_lambda_values(0.0, 0.08, 5)
    print("target,selected_rule,lambda,train_acc,val_acc,hidden_acc,hidden_loss,complexity,primitive_tier,selected")
    for idx, concept in enumerate(CONCEPTS, 1):
        problem = make_problem(concept, seed=100 + idx)
        records = run_problem(problem, lambda_values)
        chosen = next(record for record in records if record.selected)
        chosen_concept = next(concept for concept in CONCEPTS if concept.name == chosen.selected_rule)
        print(
            f"{problem.target.name},{chosen.selected_rule},{chosen.lambda_value:.4f},"
            f"{chosen.train_accuracy:.2f},{chosen.validation_accuracy:.2f},{chosen.hidden_accuracy:.2f},"
            f"{chosen.hidden_loss:.4f},{chosen.complexity:.1f},{chosen_concept.primitive_tier},{chosen.selected}"
        )


if __name__ == "__main__":
    main()
