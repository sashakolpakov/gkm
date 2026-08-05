"""Content-distinct latent-program/style-pose HYBRID support/query holdout.

The PURE nuisance benchmark deliberately renders the same action strings
twice.  This module instead asks Bongard-LOGO for a deterministic oversized
pool, keeps twenty-four content-distinct action-string programs, and freezes a
6+6 support / 6+6 query split before either presentation is rendered.  For a
basic-category problem these remain style/pose variants of one fixed template;
this does not test semantic-instance or cross-template generalization.

It is a separate module so existing content-bound PURE campaigns do not have
their source bindings changed by this additional benchmark protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset import ImageProgram, PANEL_SIZE, Problem, render_panel


PROGRAM_SPLIT_SCHEMA = "bongard.hybrid-program-split/v1"
DEFAULT_POOL_SIZE = 64
CASES_PER_SIDE_PER_SPLIT = 6


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _freeze_programs(
    programs: Sequence[Sequence[Sequence[str]]],
) -> tuple[ImageProgram, ...]:
    return tuple(
        tuple(tuple(str(action) for action in shape) for shape in image)
        for image in programs
    )


def program_digest(program: ImageProgram) -> str:
    return canonical_digest({"program": program})


def _program_entries(programs: Sequence[ImageProgram]) -> list[dict[str, Any]]:
    return [
        {"program": [list(shape) for shape in program],
         "program_digest": program_digest(program)}
        for program in programs
    ]


def _parse_entries(value: Any, name: str) -> tuple[ImageProgram, ...]:
    if not isinstance(value, list) or len(value) != CASES_PER_SIDE_PER_SPLIT:
        raise ValueError(f"{name} must contain exactly six programs")
    result: list[ImageProgram] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
                "program", "program_digest"}:
            raise ValueError(f"{name} program entry fields differ")
        program_raw = raw["program"]
        if not isinstance(program_raw, list) or not program_raw:
            raise ValueError(f"{name} program must contain shapes")
        program: ImageProgram = tuple(
            tuple(str(action) for action in shape)
            for shape in program_raw
        )
        if any(not isinstance(shape, list) or not shape for shape in program_raw) \
                or any(not action for shape in program for action in shape) \
                or raw["program_digest"] != program_digest(program):
            raise ValueError(f"{name} program digest does not reproduce")
        result.append(program)
    return tuple(result)


def _render_rng(
    seed: int, problem_id: str, split: str, side: str, index: int,
    program: ImageProgram,
) -> np.random.RandomState:
    envelope = canonical_json({
        "schema": "bongard.hybrid-render-rng/v1",
        "seed": seed,
        "problem_id": problem_id,
        "split": split,
        "side": side,
        "index": index,
        "program_digest": program_digest(program),
    }).encode("utf-8")
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(envelope).digest()[:4], "big"))


@dataclass(frozen=True)
class HybridProgramSplit:
    problem_id: str
    category: str
    concept: str
    sampling_seed: int
    pool_size: int
    dataset_inputs: tuple[tuple[str, str], ...]
    support_pos: tuple[ImageProgram, ...]
    support_neg: tuple[ImageProgram, ...]
    query_pos: tuple[ImageProgram, ...]
    query_neg: tuple[ImageProgram, ...]

    def __post_init__(self) -> None:
        groups = (
            self.support_pos, self.support_neg,
            self.query_pos, self.query_neg,
        )
        if not self.problem_id or self.category != "basic" or not self.concept:
            raise ValueError("hybrid split identity is malformed")
        if isinstance(self.sampling_seed, bool) \
                or not isinstance(self.sampling_seed, int) \
                or isinstance(self.pool_size, bool) \
                or not isinstance(self.pool_size, int) \
                or self.pool_size < 2 * CASES_PER_SIDE_PER_SPLIT:
            raise ValueError("hybrid sampling parameters are malformed")
        if any(len(group) != CASES_PER_SIDE_PER_SPLIT for group in groups):
            raise ValueError("hybrid split requires 6+6 support and query")
        digests = [program_digest(program) for group in groups for program in group]
        if len(digests) != len(set(digests)):
            raise ValueError(
                "support and query latent programs must be content-disjoint")
        if not self.dataset_inputs \
                or any(not name or not digest.startswith("sha256:")
                       for name, digest in self.dataset_inputs):
            raise ValueError("dataset input bindings are malformed")

    def render(
        self, split: str, seed: int, panel_size: int = PANEL_SIZE,
    ) -> Problem:
        if split not in {"support", "query"}:
            raise ValueError("split must be support or query")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("render seed must be an integer")
        pos_programs = self.support_pos if split == "support" else self.query_pos
        neg_programs = self.support_neg if split == "support" else self.query_neg

        def side(programs: Sequence[ImageProgram], side_name: str) \
                -> tuple[np.ndarray, ...]:
            return tuple(
                render_panel(
                    program,
                    _render_rng(
                        seed, self.problem_id, split, side_name, index, program),
                    panel_size,
                )
                for index, program in enumerate(programs)
            )

        return Problem(
            self.problem_id, self.category, self.concept,
            side(pos_programs, "pos"), side(neg_programs, "neg"),
        )

    def to_manifest(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": PROGRAM_SPLIT_SCHEMA,
            "problem_id": self.problem_id,
            "category": self.category,
            "concept": self.concept,
            "sampling_seed": self.sampling_seed,
            "pool_size": self.pool_size,
            "dataset_inputs": [
                {"path": path, "sha256": digest}
                for path, digest in self.dataset_inputs
            ],
            "selection": {
                "policy": "first-unique-content-digest/v1",
                "support_indices": list(range(6)),
                "query_indices": list(range(6, 12)),
                "global_cross-side-disjoint": True,
            },
            "support": {
                "pos": _program_entries(self.support_pos),
                "neg": _program_entries(self.support_neg),
            },
            "query": {
                "pos": _program_entries(self.query_pos),
                "neg": _program_entries(self.query_neg),
            },
        }
        body["program_split_digest"] = canonical_digest(body)
        return body

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "HybridProgramSplit":
        keys = {
            "schema", "problem_id", "category", "concept", "sampling_seed",
            "pool_size", "dataset_inputs", "selection", "support", "query",
            "program_split_digest",
        }
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError("hybrid program split fields differ")
        unsigned = {key: item for key, item in value.items()
                    if key != "program_split_digest"}
        if value["schema"] != PROGRAM_SPLIT_SCHEMA \
                or value["program_split_digest"] != canonical_digest(unsigned):
            raise ValueError("hybrid program split digest does not reproduce")
        selection = value["selection"]
        if selection != {
            "policy": "first-unique-content-digest/v1",
            "support_indices": list(range(6)),
            "query_indices": list(range(6, 12)),
            "global_cross-side-disjoint": True,
        }:
            raise ValueError("hybrid split selection contract differs")
        dataset_inputs_raw = value["dataset_inputs"]
        if not isinstance(dataset_inputs_raw, list) \
                or any(not isinstance(item, Mapping)
                       or set(item) != {"path", "sha256"}
                       for item in dataset_inputs_raw):
            raise ValueError("hybrid dataset input fields differ")
        support = value["support"]
        query = value["query"]
        if not isinstance(support, Mapping) or set(support) != {"pos", "neg"} \
                or not isinstance(query, Mapping) \
                or set(query) != {"pos", "neg"}:
            raise ValueError("hybrid support/query program groups differ")
        return cls(
            problem_id=str(value["problem_id"]),
            category=str(value["category"]),
            concept=str(value["concept"]),
            sampling_seed=value["sampling_seed"],
            pool_size=value["pool_size"],
            dataset_inputs=tuple(
                (str(item["path"]), str(item["sha256"]))
                for item in dataset_inputs_raw
            ),
            support_pos=_parse_entries(support["pos"], "support.pos"),
            support_neg=_parse_entries(support["neg"], "support.neg"),
            query_pos=_parse_entries(query["pos"], "query.pos"),
            query_neg=_parse_entries(query["neg"], "query.neg"),
        )


def _take_unique(
    programs: Sequence[ImageProgram], count: int, used: set[str], name: str,
) -> tuple[ImageProgram, ...]:
    selected: list[ImageProgram] = []
    for program in programs:
        digest = program_digest(program)
        if digest in used:
            continue
        used.add(digest)
        selected.append(program)
        if len(selected) == count:
            return tuple(selected)
    raise RuntimeError(
        f"{name} has only {len(selected)} content-distinct programs in the "
        "deterministic pool; content-distinct latent-program/style-pose "
        "holdout unavailable")


def sample_basic_program_splits(
    dataset_dir: str,
    *,
    limit: int = 1,
    seed: int = 0,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> list[HybridProgramSplit]:
    """Sample deterministic, content-disjoint basic-category holdouts.

    Bongard-LOGO's ``BasicSampler`` internally duplicates its positive shape
    function list.  Its public ``sample`` method nevertheless materializes
    exactly the requested prefix.  We request an oversized prefix and greedily
    retain the first globally unique action programs.  Any category with fewer
    than 24 unique programs fails closed rather than being reported as a
    content-distinct latent-program/style-pose holdout.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be positive")
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) \
            or pool_size < 2 * CASES_PER_SIDE_PER_SPLIT:
        raise ValueError("pool_size must be at least twelve")
    dataset_dir = os.path.abspath(dataset_dir)
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)
    from bongard.sampler.basic_sampler import BasicSampler  # type: ignore
    from bongard.util_funcs import get_shape_super_classes  # type: ignore

    shapes_tsv = os.path.join(
        dataset_dir, "data", "human_designed_shapes.tsv")
    attrs_tsv = os.path.join(
        dataset_dir, "data", "human_designed_shapes_attributes.tsv")
    inputs = (
        ("data/human_designed_shapes.tsv", file_digest(shapes_tsv)),
        ("data/human_designed_shapes_attributes.tsv", file_digest(attrs_tsv)),
        ("bongard/__init__.py", file_digest(os.path.join(
            dataset_dir, "bongard", "__init__.py"))),
        ("bongard/bongard.py", file_digest(os.path.join(
            dataset_dir, "bongard", "bongard.py"))),
        ("bongard/bongard_painter.py", file_digest(os.path.join(
            dataset_dir, "bongard", "bongard_painter.py"))),
        ("bongard/util_funcs.py", file_digest(os.path.join(
            dataset_dir, "bongard", "util_funcs.py"))),
        ("bongard/sampler/bongard_sampler.py", file_digest(os.path.join(
            dataset_dir, "bongard", "sampler", "bongard_sampler.py"))),
        ("bongard/sampler/basic_sampler.py", file_digest(os.path.join(
            dataset_dir, "bongard", "sampler", "basic_sampler.py"))),
    )
    rng = np.random.RandomState(seed)
    shape_list = list(get_shape_super_classes(shapes_tsv).keys())
    order = rng.permutation(len(shape_list))
    sampler = BasicSampler(
        shapes_tsv, attrs_tsv,
        num_positive_examples=pool_size,
        num_negative_examples=pool_size,
        random_state=rng,
    )
    result: list[HybridProgramSplit] = []
    for raw_index in order:
        shape_index = int(raw_index)
        shape = shape_list[shape_index]
        sampled = sampler.sample([shape], shape_index)
        raw = sampled.get_action_string_list()
        if not isinstance(raw, (tuple, list)) or len(raw) != 2 \
                or len(raw[0]) != pool_size or len(raw[1]) != pool_size:
            raise RuntimeError(
                "BasicSampler did not materialize the requested program pool")
        pos_pool = _freeze_programs(raw[0])
        neg_pool = _freeze_programs(raw[1])
        used: set[str] = set()
        try:
            selected_pos = _take_unique(pos_pool, 12, used, f"{shape}/pos")
            selected_neg = _take_unique(neg_pool, 12, used, f"{shape}/neg")
        except RuntimeError:
            # A low-entropy category cannot support this content-distinct
            # latent-program/style-pose protocol. Continue deterministically
            # until ``limit`` admissible categories have been found.
            continue
        result.append(HybridProgramSplit(
            problem_id=sampled.get_problem_name(),
            category="basic",
            concept=shape,
            sampling_seed=seed,
            pool_size=pool_size,
            dataset_inputs=inputs,
            support_pos=selected_pos[:6],
            support_neg=selected_neg[:6],
            query_pos=selected_pos[6:],
            query_neg=selected_neg[6:],
        ))
        if len(result) == limit:
            break
    if len(result) != limit:
        raise RuntimeError(
            f"only {len(result)} basic categories admit the requested "
            f"{limit} content-disjoint HYBRID splits")
    return result


__all__ = [
    "CASES_PER_SIDE_PER_SPLIT",
    "DEFAULT_POOL_SIZE",
    "HybridProgramSplit",
    "PROGRAM_SPLIT_SCHEMA",
    "canonical_digest",
    "canonical_json",
    "file_digest",
    "program_digest",
    "sample_basic_program_splits",
]
