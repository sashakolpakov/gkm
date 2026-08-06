from __future__ import annotations

import json
from pathlib import Path

from bongard import load_historical_exposure, load_official_release
from bongard.artifacts import canonical_json


DATA = Path(__file__).resolve().parents[1] / "data"


def _canonical_record(name: str) -> dict[str, object]:
    payload = (DATA / name).read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert payload == canonical_json(value) + b"\n"
    return value


def test_checked_in_complete_corpus_cohort_summary_is_bound_to_frozen_inputs() -> None:
    record = _canonical_record("shape_bongard_v2_cohort_summary_v1.json")
    release = load_official_release()
    historical = load_historical_exposure()

    assert record["schema"] == "gkm.shape-bongard-v2-cohort-summary.v1"
    assert record["release_descriptor_digest"] == release.digest
    assert record["historical_seed_digest"] == historical.seed_digest
    counts = record["counts"]
    assert isinstance(counts, dict)
    assert counts["tasks"] == 12_000
    assert counts["ff"] + counts["bd"] + counts["hd"] == 12_000
    assert counts["historically_clean"] == (
        counts["drill"]
        + counts["dev"]
        + counts["sealed"]
    )
    assert (counts["drill"], counts["dev"], counts["sealed"]) == (
        2_769,
        542,
        557,
    )
    assert record["cohort_report_digest"] == (
        "sha256:55de04a582ffa3a4fbf26466ab88f265ddd7839ae10004210cca4d9ffa4f8e9d"
    )
    assert record["membership_digests"] == {
        "all": "sha256:4503ae6b40dc7b34520eb5b8a4cca6ff8153635df0f42db5f6715cc349602dd0",
        "dev": "sha256:ea0334e538cfe3b6fed58fa4d575f85ac077afdb650fcb8d8043d0337f8d3f74",
        "drill": "sha256:15c95adcffe7e858b8007a3b7f20df4acb5c6fdaa7a994f2a32a5aa595abdbe9",
        "exact_task_recorded": (
            "sha256:4c9b9236c62d2e2b8e6f43dbe7297aeab380f45f473061bc80bcc7fd525b7477"
        ),
        "historically_clean": (
            "sha256:0c1bc85f24bf7491a3882c4994e57aaf50e0b66cf31fd92476d15341399b0336"
        ),
        "sealed": "sha256:e130e2281ce0209ee35ce292e1d7abd7c184f469699f99710f3fd82230ae30d0",
    }
    assert "do not certify" in str(record["qualification"])


def test_checked_in_complete_corpus_image_audit_is_strict_and_bound() -> None:
    record = _canonical_record("shape_bongard_v2_image_audit_v1.json")
    release = load_official_release()

    assert record["schema"] == "gkm.shape-bongard-image-audit.v1"
    assert record["corpus_manifest_digest"] == release.corpus_manifest_sha256
    assert record["task_count"] == 12_000
    assert record["panel_count"] == 168_000
    assert record["family_task_counts"] == {"bd": 4_000, "ff": 3_600, "hd": 4_400}
    assert record["family_panel_counts"] == {
        "bd": 56_000,
        "ff": 50_400,
        "hd": 61_600,
    }
    assert record["format_counts"] == {"PNG": 168_000}
    assert record["mode_counts"] == {"RGB": 168_000}
    assert record["size_counts"] == [
        {"count": 168_000, "height": 512, "width": 512}
    ]
    assert record["info_key_set_counts"] == [
        {"count": 168_000, "info_keys": []}
    ]
    assert record["frame_count_counts"] == [
        {"count": 168_000, "frame_count": 1}
    ]
    assert record["expectations"] == {
        "frame_count": 1,
        "height": 512,
        "info_keys": [],
        "mode": "RGB",
        "width": 512,
    }
    assert record["require_expected_properties"] is True
    assert record["anomaly_count"] == 0
    assert record["anomalies"] == []
    assert record["anomalies_truncated"] is False
    assert record["digest"] == (
        "sha256:d3485ada3605d708db82fbcfe6ecfc73506ce51ed85fcd1ce6ccd798e3bff9f8"
    )
