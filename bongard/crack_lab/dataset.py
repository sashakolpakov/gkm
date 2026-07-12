"""Deterministic Bongard-LOGO panel substrate.

This module is intentionally small. It keeps only the basic data machinery
needed by the semantic-cone harness: render action strings, sample public
Bongard-LOGO problems, and write opaque panel files. It contains no predicate
selector and no legacy verifier.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

PANEL_SIZE = 128


@dataclass(frozen=True)
class Problem:
    problem_id: str
    category: str
    concept: str
    pos: tuple[np.ndarray, ...]
    neg: tuple[np.ndarray, ...]

    def panels(self) -> list[tuple[np.ndarray, bool]]:
        return [(p, True) for p in self.pos] + [(p, False) for p in self.neg]


def trace_shape(actions: Sequence[str]) -> list[tuple[float, float]]:
    x = y = 0.0
    heading = 0.0
    points: list[tuple[float, float]] = [(x, y)]
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
            raise ValueError(f"unsupported action string {action!r}")
    return points


def _transform(points: Sequence[tuple[float, float]], angle_deg: float,
               scale: float, tx: float, ty: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    rad = math.radians(angle_deg)
    rot = np.array([[math.cos(rad), -math.sin(rad)],
                    [math.sin(rad), math.cos(rad)]])
    return pts @ rot.T * scale + np.array([tx, ty])


def _draw_polyline(grid: np.ndarray, pts: np.ndarray) -> None:
    size = grid.shape[0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg_len = math.hypot(float(x1 - x0), float(y1 - y0))
        n = max(2, int(seg_len / 0.3) + 1)
        ts = np.linspace(0.0, 1.0, n)
        xs = np.clip(np.rint(x0 + (x1 - x0) * ts).astype(int), 0, size - 1)
        ys = np.clip(np.rint(y0 + (y1 - y0) * ts).astype(int), 0, size - 1)
        grid[ys, xs] = 1


def render_panel(image_program: Sequence[Sequence[str]],
                 rng: np.random.RandomState,
                 size: int = PANEL_SIZE) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    margin = size * 0.08
    n_shapes = max(1, len(image_program))
    target_frac = 0.55 if n_shapes == 1 else 0.38
    placed_boxes: list[tuple[float, float, float, float]] = []
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


def _panel_rng(seed: int, problem_id: str, side: str, index: int) -> np.random.RandomState:
    key = f"{seed}:{problem_id}:{side}:{index}".encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:4], "big"))


def sample_problems(dataset_dir: str, limit: int = 10, seed: int = 0,
                    source: str = "basic",
                    panel_size: int = PANEL_SIZE) -> list[Problem]:
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
    problems: list[Problem] = []

    def render_problem(pid: str, category: str, concept: str, program) -> Problem:
        pos = tuple(render_panel(img, _panel_rng(seed, pid, "pos", i), panel_size)
                    for i, img in enumerate(program[0][:6]))
        neg = tuple(render_panel(img, _panel_rng(seed, pid, "neg", i), panel_size)
                    for i, img in enumerate(program[1][:6]))
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


def interleave_corpus(basic: Sequence[Problem], abstract: Sequence[Problem],
                      every: int = 5) -> list[Problem]:
    out: list[Problem] = []
    bi = ai = 0
    while bi < len(basic) or ai < len(abstract):
        for _ in range(every - 1):
            if bi < len(basic):
                out.append(basic[bi])
                bi += 1
        if ai < len(abstract):
            out.append(abstract[ai])
            ai += 1
    return out


def write_panels(ws: str, problem: Problem, opaque_id: str) -> str:
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
