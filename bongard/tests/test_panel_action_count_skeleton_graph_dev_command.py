from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import pickle
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
LIVE_FAILED_V1_ROOT = (
    LIVE_ROOT / "panel_action_count_skeleton_graph_dev_20260810_v1"
)
CACHED_FEATURE_MATRIX = Path("/private/tmp/gkm_skeleton_graph_features_v1.npz")


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


def test_capacity_repair_ledger_is_closed_immutable_and_selects_fixed_32() -> None:
    ledger = subject.CAPACITY_SELECTION_LEDGER
    rows = list(ledger["candidate_rows"])
    assert subject.SCHEMA_PRECOMMIT.endswith(".v2")
    assert subject.MODEL_SCHEMA.endswith(".v2")
    assert subject.PROTOCOL["n_estimators"] == 32
    assert subject._expected_estimator_params(260813)["n_estimators"] == 32
    assert ledger["candidate_tree_count_order"] == (
        16, 32, 48, 64, 96, 128, 192, 256,
    )
    assert [row["n_estimators"] for row in rows] == list(
        ledger["candidate_tree_count_order"]
    )
    passing = [
        row for row in rows
        if row["direct_pair_passed"]
        and row["catalog_three_class_passed"]
        and row["within_model_cap"]
    ]
    assert passing[0]["n_estimators"] == ledger["selected_n_estimators"] == 32
    assert rows[0]["direct_pair_passed"] is False
    assert rows[0]["catalog_three_class_passed"] is True
    selected = rows[1]
    assert selected["selected"] is True
    assert selected["estimators_only_serialized_bytes"] == 97_903_781
    assert selected["model_bundle_serialized_bytes"] == 97_911_851
    assert (
        selected["model_bundle_serialized_bytes"]
        - selected["estimators_only_serialized_bytes"]
    ) == 8_070
    assert selected["direct_pair_total_node_count"] == 261_776
    assert selected["catalog_three_class_total_node_count"] == 136_500
    assert ledger["unchanged_model_max_bytes"] == subject.MODEL_MAX_BYTES
    assert dict(ledger["unchanged_engineering_thresholds"]) == dict(
        subject.ENGINEERING_THRESHOLDS
    )
    assert ledger["v2_runtime_capacity_search"] is False
    failed = ledger["failed_v1_attempt"]
    assert failed["precommit_record_digest"] == subject.PRIOR_FAILED_PRECOMMIT_RECORD_DIGEST
    assert failed["precommit_file_sha256"] == subject.PRIOR_FAILED_PRECOMMIT_FILE_SHA256
    assert failed["model_serialized_bytes"] == 780_044_909
    assert failed["durable_output_count_beyond_precommit"] == 0
    with pytest.raises(TypeError):
        ledger["selected_n_estimators"] = 16
    with pytest.raises(TypeError):
        rows[1]["selected"] = False


def test_sklearn_prefix_semantics_match_separate_16_and_32_fits() -> None:
    from sklearn import base as sklearn_base
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import _forest
    from sklearn.tree import _classes, _tree

    expected_sources = subject.CAPACITY_SELECTION_LEDGER["estimator_protocol"][
        "source_addresses"
    ]
    for module in (sklearn_base, _forest, _classes, _tree):
        raw = Path(module.__file__).resolve().read_bytes()
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == expected_sources[
            module.__name__
        ]

    rng = np.random.default_rng(260810)
    matrix = rng.normal(size=(396, 112)).astype(np.float32)
    pair_target = np.tile(
        np.asarray(subject.OBSERVED_TRAIN_PAIR_CLASS_ORDER, dtype=np.int64), 12
    )
    catalog_target = np.tile(np.asarray(subject.CATALOG_CLASS_ORDER), 132)
    common = {
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": 1,
    }

    def state_bytes(tree) -> bytes:
        state = tree.tree_.__getstate__()
        return pickle.dumps(state, protocol=5)

    for target, seed in (
        (pair_target, subject.FIXED_CLASSIFIER_SEEDS["direct_pair"]),
        (catalog_target, subject.FIXED_CLASSIFIER_SEEDS["catalog_three_class"]),
    ):
        full = ExtraTreesClassifier(
            n_estimators=256, random_state=seed, **common
        ).fit(matrix, target)
        for count in (16, 32):
            separate = ExtraTreesClassifier(
                n_estimators=count, random_state=seed, **common
            ).fit(matrix, target)
            assert [tree.random_state for tree in separate.estimators_] == [
                tree.random_state for tree in full.estimators_[:count]
            ]
            assert [state_bytes(tree) for tree in separate.estimators_] == [
                state_bytes(tree) for tree in full.estimators_[:count]
            ]
            prefix_probability = sum(
                (tree.predict_proba(matrix) for tree in full.estimators_[:count]),
                start=np.zeros((len(matrix), len(full.classes_)), dtype=np.float64),
            ) / count
            assert np.array_equal(prefix_probability, separate.predict_proba(matrix))


@pytest.mark.skipif(
    not CACHED_FEATURE_MATRIX.is_file(), reason="raw-addressed cached feature matrix unavailable"
)
def test_cached_selected_model_exact_size_nodes_without_reading_pixels() -> None:
    raw = CACHED_FEATURE_MATRIX.read_bytes()
    cache = subject.CAPACITY_SELECTION_LEDGER["cached_feature_matrix"]
    assert len(raw) == cache["raw_file_bytes"]
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == cache["raw_file_sha256"]
    with np.load(BytesIO(raw), allow_pickle=False) as arrays:
        matrix = np.ascontiguousarray(arrays["X"], dtype=np.float32)
        labels = np.ascontiguousarray(arrays["Y"], dtype=np.int64)
        cohort = arrays["S"]
    estimators = subject.fit_authoritative_estimators(
        matrix[cohort == "train"], labels[cohort == "train"]
    )
    bundle = subject._build_model_bundle(
        estimators, precommit_record_digest="sha256:" + "0" * 64
    )
    model_bytes = pickle.dumps(bundle, protocol=5)
    structure = subject._model_structure(estimators, len(model_bytes))
    selected = subject.CAPACITY_SELECTION_LEDGER["candidate_rows"][1]
    assert len(model_bytes) == selected["model_bundle_serialized_bytes"]
    assert structure["heads"]["direct_pair"]["total_node_count"] == selected[
        "direct_pair_total_node_count"
    ]
    assert structure["heads"]["catalog_three_class"]["total_node_count"] == selected[
        "catalog_three_class_total_node_count"
    ]


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


@pytest.mark.skipif(
    not LIVE_FIT_PRECOMMIT.is_file() or not LIVE_FAILED_V1_ROOT.is_dir(),
    reason="pinned dev metadata or failed-v1 root unavailable",
)
def test_real_prepare_replays_metadata_without_opening_any_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_root_before = LIVE_FAILED_V1_ROOT.stat()
    failed_entries_before = sorted(path.name for path in LIVE_FAILED_V1_ROOT.iterdir())
    assert failed_entries_before == ["precommit.json"]
    failed_precommit_before = (LIVE_FAILED_V1_ROOT / "precommit.json").read_bytes()
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
    assert value["schema"] == subject.SCHEMA_PRECOMMIT
    assert value["protocol"]["n_estimators"] == 32
    assert value["capacity_selection_ledger"] == subject._plain(
        subject.CAPACITY_SELECTION_LEDGER
    )
    assert value["prior_failed_capacity_attempt"] == (
        subject._verify_prior_failed_capacity_attempt()
    )
    assert value["fit_inventory_audit"]["effective_group_counts"] == {
        "train": 11_143,
        "validation": 1_392,
    }
    assert value["fit_inventory_audit"]["cross_cohort_task_overlap"] == 0
    failed_root_after = LIVE_FAILED_V1_ROOT.stat()
    assert (failed_root_after.st_dev, failed_root_after.st_ino) == (
        failed_root_before.st_dev, failed_root_before.st_ino,
    )
    assert sorted(path.name for path in LIVE_FAILED_V1_ROOT.iterdir()) == failed_entries_before
    assert (LIVE_FAILED_V1_ROOT / "precommit.json").read_bytes() == failed_precommit_before


@pytest.mark.skipif(
    not LIVE_FIT_PRECOMMIT.is_file() or not LIVE_FAILED_V1_ROOT.is_dir(),
    reason="pinned dev metadata or failed-v1 root unavailable",
)
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
