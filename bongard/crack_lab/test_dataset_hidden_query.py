"""Contracts for support/query rerendering from one latent Bongard program."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import ProgrammedProblem


_TRIANGLE = (
    "line_normal_0.3-0.5",
    "line_normal_0.3-0.8333333333",
    "line_normal_0.3-0.8333333333",
)


def test_hidden_query_changes_pixels_without_changing_latent_program() -> None:
    latent = ProgrammedProblem(
        "latent-0",
        "synthetic",
        "triangle",
        pos_programs=((_TRIANGLE,),) * 6,
        neg_programs=(((_TRIANGLE[0],),),) * 6,
    )
    support = latent.render(17)
    replay = latent.render(17)
    query = latent.render(23)

    assert support.problem_id == query.problem_id == latent.problem_id
    assert support.concept == query.concept == latent.concept
    assert all(np.array_equal(a, b) for a, b in zip(
        support.pos + support.neg, replay.pos + replay.neg))
    assert all(not np.array_equal(a, b) for a, b in zip(
        support.pos + support.neg, query.pos + query.neg))

