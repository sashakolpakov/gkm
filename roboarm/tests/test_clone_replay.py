from __future__ import annotations

import hashlib

import numpy as np

from roboarm_game import make_env


def test_clone_replays_exactly_and_is_independent() -> None:
    original = make_env(seed=42)
    original.reset()
    for action in (2, 4, 1, 6):
        original.step(action)

    clone = original.clone()
    assert clone is not original
    assert np.array_equal(clone.frame(), original.frame())
    assert clone.actions == original.actions
    assert clone.levels_completed == original.levels_completed

    for action in (4, 2, 5, 3, 1, 6):
        assert np.array_equal(original.step(action), clone.step(action))

    original_before_divergence = original.frame()
    clone_before_divergence = clone.frame()
    assert np.array_equal(original_before_divergence, clone_before_divergence)

    original_after = original.step(1)
    assert np.array_equal(clone.frame(), clone_before_divergence)
    clone_after = clone.step(2)
    assert not np.array_equal(original_after, clone_after)


def test_fresh_replay_reproduces_every_observation() -> None:
    sequence = (4, 2, 2, 6, 3, 1, 5, 3, 2)

    discovery = make_env(seed=20260730)
    expected = [discovery.reset()]
    expected.extend(discovery.step(action) for action in sequence)

    replay = make_env(seed=20260730)
    observed = [replay.reset()]
    observed.extend(replay.step(action) for action in sequence)

    assert len(observed) == len(expected)
    assert all(
        np.array_equal(expected_frame, observed_frame)
        for expected_frame, observed_frame in zip(expected, observed, strict=True)
    )
    assert replay.levels_completed == discovery.levels_completed == 0
    assert replay.terminal() is discovery.terminal() is False


def test_phase0_calibration_golden_replay() -> None:
    env = make_env(seed=0)
    frames = [env.reset()]
    frames.extend(env.step(action) for action in (2, 1, 4, 3, 6, 5))
    observed = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
    assert observed == [
        "0d20eb1fed93e9cfb5890c87f8b4b49545d670a8cefce92ec6581a0c6e2224a3",
        "5950a7372341ba32b6ce9c8d8cfbe2b616ca0ab9e9b0f86878ba22ed19155bb7",
        "2fb6885db674dc3d29f7b0247c7c475b5f53b10ba39d44f770332e0ab1e2fca4",
        "3e1889601220bdcfc277b7a0aa85a8552c76de63bfcbeca421a156d7db5d8e0c",
        "cd8ea48cc6ed0d34db4a443cf42a4f1d79b914442065cf6dfbb5cc3927429ca4",
        "cb1a13d5f7c5f5387c9519c7bee4236671ee7642ea794afb66f48a3827841745",
        "224a863d5faf14b3954420dc990deb009b1f9ca84bef14974ab5a2102e2b5d51",
    ]
