"""Exact proposer-presentation contracts shared by both Bongard runners."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena as arena
import dataset


WRITERS = (
    (arena, lambda pos, neg: arena.Problem("hidden", "basic", "secret", pos, neg)),
    (dataset, lambda pos, neg: dataset.Problem(
        "hidden", "basic", "secret", tuple(pos), tuple(neg))),
)


def _panels():
    panels = []
    for index in range(12):
        panel = np.zeros((128, 128), dtype=np.uint8)
        panel[index: index + 3, 2 * index: 2 * index + 5] = 1
        panels.append(panel)
    return panels[:6], panels[6:]


@pytest.mark.parametrize("module,factory", WRITERS)
def test_panel_npy_and_png_round_trips_are_exact(tmp_path, module, factory):
    pos, neg = _panels()
    pdir = module.write_panels(
        str(tmp_path), factory(pos, neg), "problem_00")
    assert sorted(os.listdir(pdir)) == sorted(
        f"{side}_{index}{suffix}"
        for side in ("pos", "neg") for index in range(6)
        for suffix in (".npy", ".png"))
    for side, panels in (("pos", pos), ("neg", neg)):
        for index, expected in enumerate(panels):
            npy = np.load(
                os.path.join(pdir, f"{side}_{index}.npy"), allow_pickle=False)
            with Image.open(os.path.join(pdir, f"{side}_{index}.png")) as encoded:
                png = np.asarray(encoded.convert("L"))
            assert npy.dtype == np.uint8
            assert np.array_equal(npy, expected)
            assert set(np.unique(png)) <= {0, 255}
            assert np.array_equal((png == 0).astype(np.uint8), expected)


@pytest.mark.parametrize("module,factory", WRITERS)
@pytest.mark.parametrize("invalid", ("float", "nonbinary", "shape"))
def test_invalid_panel_is_rejected_before_existing_files_are_touched(
        tmp_path, module, factory, invalid):
    pos, neg = _panels()
    pdir = module.write_panels(
        str(tmp_path), factory(pos, neg), "problem_00")
    before = {
        name: open(os.path.join(pdir, name), "rb").read()
        for name in os.listdir(pdir)
    }
    changed = list(pos)
    if invalid == "float":
        changed[4] = changed[4].astype(float)
    elif invalid == "nonbinary":
        changed[4] = changed[4].copy()
        changed[4][0, 0] = 2
    else:
        changed[4] = changed[4][:-1]
    with pytest.raises(ValueError, match="binary uint8 panel"):
        module.write_panels(
            str(tmp_path), factory(changed, neg), "problem_00")
    after = {
        name: open(os.path.join(pdir, name), "rb").read()
        for name in os.listdir(pdir)
    }
    assert after == before


@pytest.mark.parametrize("module,factory", WRITERS)
def test_render_failure_removes_every_owned_presentation(
        tmp_path, monkeypatch, module, factory):
    pos, neg = _panels()
    pdir = module.write_panels(
        str(tmp_path), factory(pos, neg), "problem_00")

    def fail_save(_self, _path, *args, **kwargs):
        raise OSError("simulated PNG failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(RuntimeError, match="failed to materialize"):
        module.write_panels(
            str(tmp_path), factory(pos, neg), "problem_00")
    assert not any(
        name.endswith((".npy", ".png")) for name in os.listdir(pdir))
