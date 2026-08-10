from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bongard import panel_action_count_skeleton_graph_dev_command as subject


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = REPOSITORY_ROOT / "downloads/ShapeBongard_V2_full"
LIVE_FIT_PRECOMMIT = (
    LIVE_ROOT / "panel_action_count_cnn_fit_20260810_v3/fit_pixel_precommit.json"
)
LIVE_DATASET_ROOT = LIVE_ROOT / "ShapeBongard_V2"


def _png(*, faint: bool = False) -> bytes:
    image = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.line((4, 16, 27, 16), fill=247 if faint else 0, width=2)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_feature_bank_is_exact_and_candidate_projection_is_fail_closed() -> None:
    first = subject.extract_feature_vector(_png())
    second = subject.extract_feature_vector(_png())
    assert first.shape == (112,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert len(subject.FEATURE_NAMES) == len(set(subject.FEATURE_NAMES)) == 112
    assert len(subject.VALID_PAIR_CLASS_ORDER) == 54
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="not an integer"):
        subject.decode_pair_class(True)
    assert subject.catalog_candidate_projection([-1, 1]) == {
        "disposition": "indeterminate",
        "reason": "catalog_unresolved_in_candidate_set",
        "candidates": [],
    }
    assert subject.catalog_candidate_projection([0, 1])["disposition"] == "indeterminate"
    assert subject.catalog_candidate_projection([1])["disposition"] == "present"
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="exact integer"):
        subject.catalog_candidate_projection([True])
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="exact integer"):
        subject.catalog_candidate_projection([0.5])
    malformed = np.zeros((1, len(subject.OBSERVED_TRAIN_PAIR_CLASS_ORDER)))
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="probability"):
        subject.pair_marginals(malformed)
    malformed[0, 0] = np.nan
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="probability"):
        subject.pair_marginals(malformed)
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="frozen threshold"):
        subject.extract_feature_vector(_png(faint=True))


def test_verified_model_is_factory_only_and_hash_precedes_unpickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="factory-only"):
        subject.VerifiedDevelopmentModel(
            _model_bytes=b"evil",
            model_file_sha256="sha256:" + "0" * 64,
            precommit_record_digest="sha256:" + "1" * 64,
            promoted_heads=("direct_pair",),
            result_record_digest="sha256:" + "2" * 64,
        )
    value = subject._make_verified_development_model(
        model_bytes=b"evil-pickle-transport",
        model_file_sha256="sha256:" + "0" * 64,
        precommit_record_digest="sha256:" + "1" * 64,
        promoted_heads=("direct_pair",),
        result_record_digest="sha256:" + "2" * 64,
    )
    called = False

    def forbidden_loads(_payload: bytes):
        nonlocal called
        called = True
        raise AssertionError("pickle.loads ran before address authentication")

    monkeypatch.setattr(subject.pickle, "loads", forbidden_loads)
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="bytes changed"):
        value.predict(head="direct_pair", features=np.zeros((1, 112), np.float32))
    assert called is False


def test_fresh_subprocess_fits_and_cold_reloads_exact_probabilities() -> None:
    program = r'''
import hashlib
import pickle
import numpy as np
from bongard import panel_action_count_skeleton_graph_dev_command as s

pairs = s.OBSERVED_TRAIN_PAIR_CLASS_ORDER
rows = []
labels = []
for index, encoded in enumerate(pairs):
    straight, arc = divmod(encoded, 10)
    for catalog in (-1, 0, 1):
        vector = np.zeros(112, dtype=np.float32)
        vector[index % 112] = 1.0
        vector[(index * 7 + catalog + 2) % 112] += np.float32(0.25 * (catalog + 2))
        rows.append(vector)
        labels.append((straight, arc, catalog))
features = np.stack(rows)
targets = np.asarray(labels, dtype=np.int64)
heads = s.fit_authoritative_estimators(features, targets)
pair_a, catalog_a = s.predict_authoritative_probabilities(heads, features)
payload = pickle.dumps(heads, protocol=5)
restored = pickle.loads(payload)
pair_b, catalog_b = s.predict_authoritative_probabilities(restored, features)
assert np.array_equal(pair_a, pair_b)
assert np.array_equal(catalog_a, catalog_b)
straight, arc = s.pair_marginals(pair_b)
assert np.allclose(straight.sum(axis=1), np.ones(len(features)), rtol=0.0, atol=1e-12)
assert np.allclose(arc.sum(axis=1), np.ones(len(features)), rtol=0.0, atol=1e-12)
digest = hashlib.sha256(pair_b.tobytes() + catalog_b.tobytes()).hexdigest()
restored["direct_pair"].set_params(n_jobs=2)
try:
    s.predict_authoritative_probabilities(restored, features)
except s.SkeletonGraphDevelopmentError:
    pass
else:
    raise AssertionError("mutated estimator parameters were accepted")
print(digest)
'''
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.strip()) == 64


def test_module_cli_is_loadable_in_a_fresh_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bongard.panel_action_count_skeleton_graph_dev_command",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "prepare" in completed.stdout
    assert "train" in completed.stdout
    assert "replay" in completed.stdout


@pytest.mark.skipif(not LIVE_FIT_PRECOMMIT.is_file(), reason="pinned dev metadata unavailable")
def test_real_prepare_replays_metadata_without_opening_any_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subject._stable_regular_bytes

    def png_tripwire(path: Path, *, maximum: int) -> bytes:
        if path.suffix.lower() == ".png":
            raise AssertionError(f"prepare tried to open PNG {path}")
        return original(path, maximum=maximum)

    monkeypatch.setattr(subject, "_stable_regular_bytes", png_tripwire)
    value = subject.create_development_precommit(
        repository_root=REPOSITORY_ROOT,
        dataset_root=LIVE_DATASET_ROOT,
        fit_precommit_path=LIVE_FIT_PRECOMMIT,
        model_path=tmp_path / "model.pkl",
        feature_manifest_path=tmp_path / "features.json",
        predictions_path=tmp_path / "predictions.json",
        result_path=tmp_path / "result.json",
        replay_path=tmp_path / "replay.json",
        output_path=tmp_path / "precommit.json",
        maximum_seconds=600,
    )
    assert value["pixels_read_by_precommit"] == 0
    assert value["fit_inventory_audit"]["effective_group_counts"] == {
        "train": 11_143,
        "validation": 1_392,
    }
    assert value["fit_inventory_audit"]["cross_cohort_task_overlap"] == 0


@pytest.mark.skipif(not LIVE_FIT_PRECOMMIT.is_file(), reason="pinned dev metadata unavailable")
def test_precommit_rejects_output_root_symlink_substitution(tmp_path: Path) -> None:
    bound = tmp_path / "bound"
    value = subject.create_development_precommit(
        repository_root=REPOSITORY_ROOT,
        dataset_root=LIVE_DATASET_ROOT,
        fit_precommit_path=LIVE_FIT_PRECOMMIT,
        model_path=bound / "model.pkl",
        feature_manifest_path=bound / "features.json",
        predictions_path=bound / "predictions.json",
        result_path=bound / "result.json",
        replay_path=bound / "replay.json",
        output_path=bound / "precommit.json",
        maximum_seconds=600,
    )
    original = tmp_path / "original"
    attacker = tmp_path / "attacker"
    bound.rename(original)
    attacker.mkdir()
    shutil.copyfile(original / "precommit.json", attacker / "precommit.json")
    bound.symlink_to(attacker, target_is_directory=True)
    with pytest.raises(subject.SkeletonGraphDevelopmentError, match="output root"):
        subject._load_development_precommit(
            bound / "precommit.json",
            expected_record_digest=value["record_digest"],
        )
