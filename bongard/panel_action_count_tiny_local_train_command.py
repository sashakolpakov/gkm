"""Runnable development-only trainer for the tiny pose-free action observer.

``prepare`` reads only already-exposed metadata/action programs and writes an
authorization plus source-bound precommits.  ``train`` is the sole PNG-reading
entrypoint and can reread only the exact decontaminated development digest
groups from the predecessor fit precommit.  ``replay`` can reread that same
development cohort.  There is intentionally no CAL, evaluation, family,
support, query, or target argument or subcommand.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_tiny_local_dev_command as core
from bongard import panel_action_local_supervision_authority as authority_module
from bongard import panel_action_local_duplicate_conflict_audit as conflict_module


AUTHORIZATION_SCHEMA = "gkm.bongard-tiny-local-action-development-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-tiny-local-action-development-training-precommit.v1"
RESULT_SCHEMA = "gkm.bongard-tiny-local-action-development-result.v1"
REPLAY_SCHEMA = "gkm.bongard-tiny-local-action-development-replay.v1"

FIT_PRECOMMIT_DIGEST = (
    "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
)
CATALOG_TO_INDEX = {-1: 0, 0: 1, 1: 2}
COMMITTED_CONFLICT_AUDIT_DIGEST = (
    "sha256:ac74773f5dfc05fcd935822eee5567d06d806a71f79b17b011557b648c2e4a25"
)
FINALIZATION_RESERVE_SECONDS = 30.0


class TinyLocalTrainingError(RuntimeError):
    """Development custody, training, or replay differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TinyLocalTrainingError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise TinyLocalTrainingError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise TinyLocalTrainingError(f"{label} digest differs")
    return value


def _paths(output_root: Path) -> dict[str, Path]:
    root = output_root.resolve()
    return {
        "authorization": root / "authorization.json",
        "checkpoint": root / "model.pt",
        "core_precommit": root / "core_precommit.json",
        "precommit": root / "training_precommit.json",
        "replay": root / "replay.json",
        "result": root / "result.json",
    }


def _verify_dataset_root(repository_root: Path, dataset_root: Path) -> Path:
    repository = repository_root.resolve()
    expected = repository / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
    supplied = dataset_root.absolute()
    if supplied != expected or dataset_root.resolve() != expected:
        raise TinyLocalTrainingError("dataset root is not the pinned official release path")
    current = repository
    for part in ("downloads", "ShapeBongard_V2_full", "ShapeBongard_V2", "hd", "images"):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TinyLocalTrainingError(f"cannot stat dataset layout: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise TinyLocalTrainingError("dataset layout contains a symlink/non-directory")
    return expected


def _stable_panel_bytes(path: Path, *, dataset_root: Path) -> bytes:
    root = dataset_root.resolve()
    try:
        relative = path.relative_to(dataset_root)
        current = dataset_root
        for component in relative.parts[:-1]:
            current = current / component
            parent_info = current.lstat()
            if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(
                parent_info.st_mode
            ):
                raise TinyLocalTrainingError(
                    "panel parent contains a symlink/non-directory"
                )
        resolved_parent = path.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
        before = path.lstat()
    except (OSError, ValueError) as exc:
        raise TinyLocalTrainingError(f"panel path escapes pinned dataset: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TinyLocalTrainingError("panel path is not a regular nonsymlink file")
    if before.st_size <= 0 or before.st_size > 16 * 1024 * 1024:
        raise TinyLocalTrainingError("panel byte size leaves frozen limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        remaining = 16 * 1024 * 1024 + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not fingerprint(before) == fingerprint(opened) == fingerprint(after_read) == fingerprint(after):
        raise TinyLocalTrainingError("panel changed during stable read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size or len(raw) > 16 * 1024 * 1024:
        raise TinyLocalTrainingError("panel stable-read size differs")
    return raw


def _compile_all_supervisions(repository_root: Path):
    authority = authority_module.load_development_authority(
        repository_root=repository_root.resolve()
    )
    rows = tuple(
        authority_module.compile_pose_free_panel(authority, panel_id)
        for _cohort, panel_ids in authority.cohort_panel_ids
        for panel_id in panel_ids
    )
    if len(rows) != 12_600:
        raise TinyLocalTrainingError("development supervision count differs")
    return authority, rows


def _committed_conflict_binding(
    artifact: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    if artifact.get("record_digest") != COMMITTED_CONFLICT_AUDIT_DIGEST:
        raise TinyLocalTrainingError("committed descriptor-conflict audit digest differs")
    conflict_module.verify_duplicate_target_conflict_audit(
        dict(artifact), repository_root=repository_root.resolve()
    )
    result = artifact.get("result")
    cohorts = artifact.get("cohorts")
    if (
        not isinstance(result, Mapping)
        or result.get("all_effective_png_groups_descriptor_loss_eligible") is not True
        or result.get("pose_free_target_conflict_group_count") != 0
        or result.get("straight_arc_pair_conflict_group_count") != 0
        or result.get("catalog_convexity_conflict_group_count") != 0
        or not isinstance(cohorts, Mapping)
    ):
        raise TinyLocalTrainingError("committed descriptor-conflict audit failed")
    train = cohorts.get("train", {}).get("descriptor_loss_eligibility", {})
    validation = cohorts.get("validation", {}).get("descriptor_loss_eligibility", {})
    if (
        train.get("eligible_occurrence_count") != 11_200
        or train.get("eligible_group_count") != 11_143
        or validation.get("eligible_occurrence_count") != 1_392
        or validation.get("eligible_group_count") != 1_392
    ):
        raise TinyLocalTrainingError("descriptor eligibility counts differ")
    bindings = artifact.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TinyLocalTrainingError("descriptor-conflict bindings are missing")
    return {
        "authority_gap_occurrences": 0,
        "committed_audit_record_digest": artifact["record_digest"],
        "committed_audit_source_sha256": bindings["audit_source_sha256"],
        "count_and_catalog_supervision_occurrences": 12_592,
        "descriptor_conflict_occurrences": 0,
        "descriptor_eligible_group_counts": {"train": 11_143, "validation": 1_392},
        "descriptor_eligible_occurrences": 12_592,
        "descriptor_gap_is_never_none_or_zero": True,
        "effective_occurrence_count": 12_592,
        "eligibility_index_digests": {
            "train": train["eligible_index_digest"],
            "validation": validation["eligible_index_digest"],
        },
        "all_effective_png_groups_descriptor_loss_eligible": True,
    }


def _authorization_body(
    *,
    repository_root: Path,
    dataset_root: Path,
    fit_precommit_path: Path,
    failed_baseline_path: Path,
    retired_spatial_outcome_path: Path,
    descriptor_conflict_audit_path: Path,
    output_paths: Mapping[str, Path],
    authority_record: Mapping[str, Any],
    coverage: Mapping[str, Any],
    conflict: Mapping[str, Any],
    runtime_probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture_id": core.ARCHITECTURE_ID,
        "authority": "already_exposed_development_only",
        "core_source_sha256": core.source_sha256(),
        "dataset_root": str(dataset_root.resolve()),
        "failed_baseline_path": str(failed_baseline_path.resolve()),
        "descriptor_conflict_audit_path": str(descriptor_conflict_audit_path.resolve()),
        "fit_precommit_path": str(fit_precommit_path.resolve()),
        "forbidden_cohorts": list(core.PROTOCOL["forbidden_cohorts"]),
        "intended_outputs": {
            name: str(output_paths[name])
            for name in ("checkpoint", "core_precommit", "precommit", "result", "replay")
        },
        "pixels_read_by_prepare": 0,
        "repository_root": str(repository_root.resolve()),
        "retired_spatial_outcome_path": str(retired_spatial_outcome_path.resolve()),
        "runtime_probe": dict(runtime_probe),
        "schema": AUTHORIZATION_SCHEMA,
        "source_sha256": source_sha256(),
        "supervision_authority_record_digest": authority_record["record_digest"],
        "supervision_coverage": dict(coverage),
        "target_conflict_audit": dict(conflict),
    }


def _training_precommit_body(
    *,
    authorization: Mapping[str, Any],
    core_precommit: Mapping[str, Any],
    authority_record: Mapping[str, Any],
    conflict_artifact: Mapping[str, Any],
    conflict: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture_id": core.ARCHITECTURE_ID,
        "authorization_record_digest": authorization["record_digest"],
        "config_digest": core.successor_config_digest(),
        "core_precommit_record_digest": core_precommit["record_digest"],
        "core_source_sha256": core.source_sha256(),
        "decontaminated_occurrence_counts": {"train": 11_200, "validation": 1_392},
        "descriptor_conflict_audit_record_digest": conflict_artifact["record_digest"],
        "descriptor_target_conflict_audit": dict(conflict),
        "fit_precommit_record_digest": FIT_PRECOMMIT_DIGEST,
        "forbidden_cohorts": list(core.PROTOCOL["forbidden_cohorts"]),
        "intended_outputs": authorization["intended_outputs"],
        "pixels_read_by_precommit": 0,
        "protocol": json.loads(canonical_json(dict(core.PROTOCOL))),
        "schema": PRECOMMIT_SCHEMA,
        "source_sha256": source_sha256(),
        "supervision_authority_record_digest": authority_record["record_digest"],
    }


def prepare_training(
    *,
    repository_root: Path,
    dataset_root: Path,
    fit_precommit_path: Path,
    failed_baseline_path: Path,
    retired_spatial_outcome_path: Path,
    descriptor_conflict_audit_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write authorization/precommits after metadata-only audits; read zero PNGs."""

    _verify_dataset_root(repository_root, dataset_root)
    paths = _paths(output_root)
    authority, supervisions = _compile_all_supervisions(repository_root)
    authority_record = authority.to_record()
    coverage = core.audit_supervision_coverage(supervisions)
    fit_precommit = _load(fit_precommit_path, label="fit precommit")
    if fit_precommit.get("record_digest") != FIT_PRECOMMIT_DIGEST:
        raise TinyLocalTrainingError("fit precommit binding differs")
    conflict_artifact = _load(
        descriptor_conflict_audit_path, label="committed descriptor-conflict audit"
    )
    conflict = _committed_conflict_binding(
        conflict_artifact, repository_root=repository_root
    )
    if paths["authorization"].exists():
        authorization = _load(paths["authorization"], label="authorization")
        runtime_probe = authorization.get("runtime_probe")
        if not isinstance(runtime_probe, dict):
            raise TinyLocalTrainingError("authorization runtime probe is missing")
    else:
        runtime_probe = core.synthetic_runtime_probe(repetitions=3)
    expected_authorization = _seal(
        _authorization_body(
            repository_root=repository_root,
            dataset_root=dataset_root,
            fit_precommit_path=fit_precommit_path,
            failed_baseline_path=failed_baseline_path,
            retired_spatial_outcome_path=retired_spatial_outcome_path,
            descriptor_conflict_audit_path=descriptor_conflict_audit_path,
            output_paths=paths,
            authority_record=authority_record,
            coverage=coverage,
            conflict=conflict,
            runtime_probe=runtime_probe,
        )
    )
    core._write_once(paths["authorization"], expected_authorization)
    if _load(paths["authorization"], label="authorization") != expected_authorization:
        raise TinyLocalTrainingError("authorization fresh replay differs")
    core_precommit = core.create_successor_precommit(
        failed_baseline_path=failed_baseline_path,
        retired_spatial_outcome_path=retired_spatial_outcome_path,
        fit_precommit_path=fit_precommit_path,
        supervision_authority_record=authority_record,
        supervision_coverage=coverage,
        descriptor_conflict_audit=conflict,
        runtime_probe=runtime_probe,
        trainer_source_sha256=source_sha256(),
        training_entrypoint_status="live_runnable_development_only",
        intended_checkpoint=paths["checkpoint"],
        intended_result=paths["result"],
        output=paths["core_precommit"],
    )
    body = _training_precommit_body(
        authorization=expected_authorization,
        core_precommit=core_precommit,
        authority_record=authority_record,
        conflict_artifact=conflict_artifact,
        conflict=conflict,
    )
    precommit = _seal(body)
    core._write_once(paths["precommit"], precommit)
    if _load(paths["precommit"], label="training precommit") != precommit:
        raise TinyLocalTrainingError("training precommit fresh replay differs")
    return {
        "authorization": expected_authorization,
        "core_precommit": core_precommit,
        "precommit": precommit,
    }


@dataclass(frozen=True)
class TrainingGroup:
    cohort: str
    png_sha256: str
    ink: np.ndarray
    straight: int
    arc: int
    catalog: int
    multiplicity: int
    descriptor_targets: tuple[Mapping[str, Any], ...] | None


def _panel_path(dataset_root: Path, panel_id: str) -> Path:
    parts = panel_id.split("/")
    if len(parts) != 4 or parts[0] != "hd" or parts[2] not in {"0", "1"}:
        raise TinyLocalTrainingError("development panel ID is invalid")
    if parts[1].startswith("hd_convex-has_four_straight_lines_"):
        raise TinyLocalTrainingError("target family entered development trainer")
    return dataset_root / "hd/images" / parts[1] / parts[2] / parts[3]


def materialize_groups(
    *,
    repository_root: Path,
    dataset_root: Path,
    fit_precommit: Mapping[str, Any],
    conflict_audit: Mapping[str, Any],
    deadline: core.WallDeadline,
) -> tuple[TrainingGroup, ...]:
    """Read only precommitted effective development PNG occurrences."""

    dataset_root = _verify_dataset_root(repository_root, dataset_root)
    authority, all_supervisions = _compile_all_supervisions(repository_root)
    del authority
    supervision_by_panel = {}
    for row in all_supervisions:
        data = row.to_data()
        targets = core.authority_panel_targets(data)
        exact_target = conflict_module._pose_free_target(data)
        supervision_by_panel[row.panel_id] = (
            targets,
            "sha256:" + canonical_digest(exact_target),
        )
    observations = fit_precommit.get("exact_png_observations")
    groups = fit_precommit.get("path_independent_digest_groups")
    if not isinstance(observations, list) or not isinstance(groups, list):
        raise TinyLocalTrainingError("fit precommit pixel inventory differs")
    observed = {row["panel_id"]: row for row in observations}
    if (
        conflict_audit.get("committed_audit_record_digest")
        != COMMITTED_CONFLICT_AUDIT_DIGEST
        or conflict_audit.get("all_effective_png_groups_descriptor_loss_eligible")
        is not True
    ):
        raise TinyLocalTrainingError("committed descriptor eligibility is not bound")
    result: list[TrainingGroup] = []
    occurrence_count = 0
    for group_index, group in enumerate(groups):
        if group_index % 64 == 0:
            deadline.check()
        panel_ids = group.get("panel_ids")
        if not isinstance(panel_ids, list) or group.get("multiplicity") != len(panel_ids):
            raise TinyLocalTrainingError("fit digest group structure differs")
        representative: bytes | None = None
        descriptor_digests: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for panel_id in panel_ids:
            raw = _stable_panel_bytes(
                _panel_path(dataset_root, panel_id), dataset_root=dataset_root
            )
            expected = observed.get(panel_id)
            if (
                not isinstance(expected, Mapping)
                or expected.get("png_sha256") != group.get("png_sha256")
                or expected.get("png_size_bytes") != len(raw)
                or core._address(raw) != group.get("png_sha256")
            ):
                raise TinyLocalTrainingError("development PNG changed after precommit")
            if representative is None:
                representative = raw
            target_row = supervision_by_panel.get(panel_id)
            if target_row is None:
                raise TinyLocalTrainingError("development panel lacks supervision")
            target, digest = target_row
            descriptor_digests[digest] = target
        assert representative is not None
        if len(descriptor_digests) != 1:
            raise TinyLocalTrainingError("committed descriptor eligibility replay differs")
        labels = group.get("label_triple")
        if (
            not isinstance(labels, list)
            or len(labels) != 3
            or labels[0] not in range(10)
            or labels[1] not in range(10)
            or labels[0] + labels[1] > 9
            or labels[2] not in CATALOG_TO_INDEX
        ):
            raise TinyLocalTrainingError("development group labels differ")
        result.append(
            TrainingGroup(
                cohort=str(group["fit_cohort"]),
                png_sha256=str(group["png_sha256"]),
                ink=core.preprocess_png_bytes(representative),
                straight=int(labels[0]),
                arc=int(labels[1]),
                catalog=CATALOG_TO_INDEX[int(labels[2])],
                multiplicity=len(panel_ids),
                descriptor_targets=(
                    next(iter(descriptor_digests.values()))
                ),
            )
        )
        occurrence_count += len(panel_ids)
    if occurrence_count != 12_592:
        raise TinyLocalTrainingError("materialized development occurrence count differs")
    deadline.check()
    return tuple(result)


def _d4(array: np.ndarray, *, digest: str, epoch: int) -> np.ndarray:
    key = hashlib.sha256(
        f"{core.PROTOCOL['random_seed']}\0{epoch}\0{digest}".encode("utf-8")
    ).digest()
    index = int.from_bytes(key, "big") % 8
    value = array if index < 4 else np.fliplr(array)
    return np.ascontiguousarray(np.rot90(value, k=index % 4))


def _batch(groups: Sequence[TrainingGroup], indices: Sequence[int], *, epoch: int, augment: bool):
    torch, _, _ = core._torch_runtime()
    arrays = [
        _d4(groups[index].ink, digest=groups[index].png_sha256, epoch=epoch)
        if augment
        else groups[index].ink
        for index in indices
    ]
    ink = torch.from_numpy(np.stack(arrays)[:, None])
    return core.input_channels(ink), [groups[index] for index in indices]


def group_normalized_loss(output: Mapping[str, Any], groups: Sequence[TrainingGroup]):
    """Occurrence-weight counts/catalog; each digest group gets one descriptor vote."""

    torch, _, functional = core._torch_runtime()
    logits = output["slot_logits"]
    geometry = output["geometry"]
    catalog_logits = output["catalog_logits"]
    if logits.shape[0] != len(groups):
        raise TinyLocalTrainingError("loss group cardinality differs")
    slot_probabilities = logits.softmax(dim=-1)
    joint = core.joint_count_probabilities(slot_probabilities)
    multiplicities = logits.new_tensor([group.multiplicity for group in groups])
    count_losses = -torch.stack(
        [
            joint[index, group.straight, group.arc].clamp_min(1e-12).log()
            for index, group in enumerate(groups)
        ]
    )
    catalog_targets = torch.tensor(
        [group.catalog for group in groups], dtype=torch.long, device=logits.device
    )
    catalog_losses = functional.cross_entropy(
        catalog_logits, catalog_targets, reduction="none"
    )
    primary = ((count_losses + catalog_losses) * multiplicities).sum() / multiplicities.sum()
    descriptor_classification = logits.new_zeros(())
    descriptor_geometry = logits.new_zeros(())
    eligible_groups = 0
    for batch_index, group in enumerate(groups):
        if group.descriptor_targets is None:
            continue
        targets = tuple(core.normalize_pose_free_action(item) for item in group.descriptor_targets)
        matches = core._hungarian_matches(
            logits[batch_index], geometry[batch_index], targets
        )
        classes = torch.zeros(core.MAX_ACTION_SLOTS, dtype=torch.long, device=logits.device)
        geometry_group = logits.new_zeros(())
        for slot_index, target_index in matches:
            target = targets[target_index]
            class_index = 1 if target["kind"] == "line" else 2
            classes[slot_index] = class_index
            prediction = geometry[batch_index, slot_index]
            if class_index == 1:
                distances = (
                    core._distance_outside_interval(prediction[0], target["line_length"]),
                )
            else:
                distances = (
                    core._distance_outside_interval(prediction[1], target["arc_radius"]),
                    core._distance_outside_interval(
                        prediction[2], target["arc_sweep_magnitude"]
                    ),
                )
            geometry_group = geometry_group + sum(
                functional.smooth_l1_loss(
                    distance, prediction.new_zeros(()), reduction="sum"
                )
                for distance in distances
            )
        descriptor_classification = descriptor_classification + functional.cross_entropy(
            logits[batch_index], classes
        )
        descriptor_geometry = descriptor_geometry + geometry_group / max(1, len(matches))
        eligible_groups += 1
    if eligible_groups:
        descriptor_classification = descriptor_classification / eligible_groups
        descriptor_geometry = descriptor_geometry / eligible_groups
    total = primary + descriptor_classification + descriptor_geometry
    return {
        "catalog_and_count_occurrence_weighted": primary,
        "descriptor_classification_group_normalized": descriptor_classification,
        "descriptor_geometry_group_normalized": descriptor_geometry,
        "descriptor_eligible_group_count": eligible_groups,
        "total": total,
    }


def _group_order(groups: Sequence[TrainingGroup], *, epoch: int) -> list[int]:
    return sorted(
        range(len(groups)),
        key=lambda index: (
            hashlib.sha256(
                f"{core.PROTOCOL['random_seed']}\0{epoch}\0{groups[index].png_sha256}".encode(
                    "utf-8"
                )
            ).digest(),
            groups[index].png_sha256,
        ),
    )


def validation_metrics(
    model, groups: Sequence[TrainingGroup], *, deadline: core.WallDeadline
) -> dict[str, Any]:
    torch, _, _ = core._torch_runtime()
    model.eval()
    totals = {"panel": 0, "straight": 0, "arc": 0, "catalog": 0}
    catalog_by_true = {1: [0, 0], 2: [0, 0]}
    descriptor = {
        "eligible_groups": 0,
        "interval_fields": 0,
        "interval_hits": 0,
        "matched_actions": 0,
        "matched_primitive_correct": 0,
        "primitive_multiset_exact_groups": 0,
    }
    rows = []
    batch_size = int(core.PROTOCOL["batch_size"])
    with torch.no_grad():
        for start in range(0, len(groups), batch_size):
            deadline.check()
            indices = list(range(start, min(len(groups), start + batch_size)))
            pixels, selected = _batch(groups, indices, epoch=0, augment=False)
            output = model(pixels)
            slot_probabilities = output["slot_logits"].softmax(dim=-1)
            joint = core.joint_count_probabilities(slot_probabilities)
            catalog = output["catalog_logits"].argmax(dim=-1)
            for index, group in enumerate(selected):
                flat_index = int(joint[index].argmax().item())
                predicted_straight, predicted_arc = divmod(flat_index, 10)
                predicted_catalog = int(catalog[index].item())
                multiplicity = group.multiplicity
                totals["panel"] += multiplicity
                totals["straight"] += multiplicity * (predicted_straight == group.straight)
                totals["arc"] += multiplicity * (predicted_arc == group.arc)
                totals["catalog"] += multiplicity * (predicted_catalog == group.catalog)
                if group.catalog in catalog_by_true:
                    catalog_by_true[group.catalog][1] += multiplicity
                    catalog_by_true[group.catalog][0] += multiplicity * (
                        predicted_catalog == group.catalog
                    )
                descriptor_exact = None
                matched_primitive_accuracy = None
                interval_hit_rate = None
                if group.descriptor_targets is not None:
                    targets = tuple(
                        core.normalize_pose_free_action(item)
                        for item in group.descriptor_targets
                    )
                    matches = core._hungarian_matches(
                        output["slot_logits"][index], output["geometry"][index], targets
                    )
                    predicted_classes = output["slot_logits"][index].argmax(dim=-1)
                    predicted_line = int((predicted_classes == 1).sum().item())
                    predicted_arc_slots = int((predicted_classes == 2).sum().item())
                    true_line = sum(item["kind"] == "line" for item in targets)
                    true_arc = sum(item["kind"] == "arc" for item in targets)
                    descriptor_exact = (
                        predicted_line == true_line and predicted_arc_slots == true_arc
                    )
                    descriptor["eligible_groups"] += 1
                    descriptor["primitive_multiset_exact_groups"] += descriptor_exact
                    primitive_correct = 0
                    field_hits = 0
                    field_total = 0
                    for slot_index, target_index in matches:
                        target = targets[target_index]
                        expected_class = 1 if target["kind"] == "line" else 2
                        primitive_correct += int(predicted_classes[slot_index].item()) == expected_class
                        prediction = output["geometry"][index, slot_index]
                        fields = (
                            ((prediction[0], target["line_length"]),)
                            if expected_class == 1
                            else (
                                (prediction[1], target["arc_radius"]),
                                (prediction[2], target["arc_sweep_magnitude"]),
                            )
                        )
                        for value, interval in fields:
                            number = float(value.item())
                            field_hits += interval["lower"] <= number <= interval["upper"]
                            field_total += 1
                    descriptor["matched_actions"] += len(matches)
                    descriptor["matched_primitive_correct"] += primitive_correct
                    descriptor["interval_fields"] += field_total
                    descriptor["interval_hits"] += field_hits
                    matched_primitive_accuracy = primitive_correct / max(1, len(matches))
                    interval_hit_rate = field_hits / max(1, field_total)
                rows.append(
                    {
                        "arc": predicted_arc,
                        "catalog": predicted_catalog,
                        "multiplicity": multiplicity,
                        "png_sha256": group.png_sha256,
                        "straight": predicted_straight,
                        "descriptor_interval_hit_rate": interval_hit_rate,
                        "descriptor_matched_primitive_accuracy": matched_primitive_accuracy,
                        "descriptor_primitive_multiset_exact": descriptor_exact,
                    }
                )
            deadline.check()
    if totals["panel"] <= 0 or any(total == 0 for _correct, total in catalog_by_true.values()):
        raise TinyLocalTrainingError("validation metric population differs")
    if (
        descriptor["eligible_groups"] != len(groups)
        or descriptor["matched_actions"] <= 0
        or descriptor["interval_fields"] <= 0
    ):
        raise TinyLocalTrainingError("validation descriptor population differs")
    return {
        "arc_top1": totals["arc"] / totals["panel"],
        "catalog_all_class_top1": totals["catalog"] / totals["panel"],
        "known_catalog_binary_balanced_accuracy": sum(
            correct / total for correct, total in catalog_by_true.values()
        )
        / 2,
        "panel_occurrences": totals["panel"],
        "descriptor_deployment_authority": False,
        "descriptor_eligible_digest_groups": descriptor["eligible_groups"],
        "descriptor_geometry_interval_hit": (
            descriptor["interval_hits"] / descriptor["interval_fields"]
        ),
        "descriptor_geometry_interval_hit_denominator": descriptor["interval_fields"],
        "descriptor_geometry_interval_hit_numerator": descriptor["interval_hits"],
        "descriptor_matched_primitive_accuracy": (
            descriptor["matched_primitive_correct"] / descriptor["matched_actions"]
        ),
        "descriptor_matched_primitive_denominator": descriptor["matched_actions"],
        "descriptor_matched_primitive_numerator": descriptor[
            "matched_primitive_correct"
        ],
        "descriptor_primitive_multiset_exact": (
            descriptor["primitive_multiset_exact_groups"]
            / descriptor["eligible_groups"]
        ),
        "descriptor_primitive_multiset_exact_denominator": descriptor[
            "eligible_groups"
        ],
        "descriptor_primitive_multiset_exact_numerator": descriptor[
            "primitive_multiset_exact_groups"
        ],
        "prediction_rows_by_digest_group": rows,
        "straight_top1": totals["straight"] / totals["panel"],
    }


def train_core(groups: Sequence[TrainingGroup], *, deadline: core.WallDeadline):
    torch = core._configure_torch(int(core.PROTOCOL["random_seed"]))
    training = tuple(group for group in groups if group.cohort == "train")
    validation = tuple(group for group in groups if group.cohort == "validation")
    if (
        sum(group.multiplicity for group in training) != 11_200
        or sum(group.multiplicity for group in validation) != 1_392
    ):
        raise TinyLocalTrainingError("decontaminated training split differs")
    model = core.build_model(seed=int(core.PROTOCOL["random_seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(core.PROTOCOL["learning_rate"]),
        weight_decay=float(core.PROTOCOL["weight_decay"]),
    )
    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_metrics = None
    batch_size = int(core.PROTOCOL["batch_size"])
    for epoch in range(int(core.PROTOCOL["epochs"])):
        model.train()
        order = _group_order(training, epoch=epoch)
        epoch_loss = 0.0
        group_count = 0
        for start in range(0, len(order), batch_size):
            deadline.check()
            indices = order[start : start + batch_size]
            pixels, selected = _batch(training, indices, epoch=epoch, augment=True)
            optimizer.zero_grad(set_to_none=True)
            losses = group_normalized_loss(model(pixels), selected)
            losses["total"].backward()
            optimizer.step()
            epoch_loss += float(losses["total"].item()) * len(selected)
            group_count += len(selected)
            deadline.check()
        metrics = validation_metrics(model, validation, deadline=deadline)
        metrics["epoch"] = epoch
        metrics["training_group_mean_loss"] = epoch_loss / group_count
        history.append(metrics)
        key = (
            metrics["straight_top1"],
            metrics["known_catalog_binary_balanced_accuracy"],
            metrics["descriptor_primitive_multiset_exact"],
            metrics["descriptor_matched_primitive_accuracy"],
            metrics["descriptor_geometry_interval_hit"],
            metrics["arc_top1"],
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    assert best_epoch is not None and best_state is not None and best_metrics is not None
    model.load_state_dict(best_state, strict=True)
    replay_metrics = validation_metrics(model, validation, deadline=deadline)
    comparable = {key: value for key, value in best_metrics.items() if key not in {"epoch", "training_group_mean_loss"}}
    if replay_metrics != comparable:
        raise TinyLocalTrainingError("selected in-memory validation replay differs")
    return {
        "best_epoch": best_epoch,
        "history": history,
        "metrics": replay_metrics,
        "model": model,
        "state": best_state,
        "validation_groups": validation,
    }


def _verify_prepared(
    *, repository_root: Path, dataset_root: Path, output_root: Path
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_root = _verify_dataset_root(repository_root, dataset_root)
    paths = _paths(output_root)
    authorization = _load(paths["authorization"], label="authorization")
    required_paths = (
        "fit_precommit_path",
        "failed_baseline_path",
        "retired_spatial_outcome_path",
        "descriptor_conflict_audit_path",
    )
    if any(not isinstance(authorization.get(name), str) for name in required_paths):
        raise TinyLocalTrainingError("authorization input paths are missing")
    fit_precommit_path = Path(authorization["fit_precommit_path"])
    failed_baseline_path = Path(authorization["failed_baseline_path"])
    retired_spatial_path = Path(authorization["retired_spatial_outcome_path"])
    conflict_path = Path(authorization["descriptor_conflict_audit_path"])
    fit_precommit = _load(fit_precommit_path, label="fit precommit")
    if fit_precommit.get("record_digest") != FIT_PRECOMMIT_DIGEST:
        raise TinyLocalTrainingError("fit precommit differs")
    authority, supervisions = _compile_all_supervisions(repository_root)
    authority_record = authority.to_record()
    coverage = core.audit_supervision_coverage(supervisions)
    conflict_artifact = _load(conflict_path, label="committed descriptor-conflict audit")
    conflict = _committed_conflict_binding(
        conflict_artifact, repository_root=repository_root
    )
    runtime_probe = authorization.get("runtime_probe")
    if not isinstance(runtime_probe, Mapping):
        raise TinyLocalTrainingError("authorization runtime probe is missing")
    expected_authorization = _seal(
        _authorization_body(
            repository_root=repository_root,
            dataset_root=dataset_root,
            fit_precommit_path=fit_precommit_path,
            failed_baseline_path=failed_baseline_path,
            retired_spatial_outcome_path=retired_spatial_path,
            descriptor_conflict_audit_path=conflict_path,
            output_paths=paths,
            authority_record=authority_record,
            coverage=coverage,
            conflict=conflict,
            runtime_probe=runtime_probe,
        )
    )
    if authorization != expected_authorization:
        raise TinyLocalTrainingError("authorization exact reconstruction differs")
    expected_core_precommit = core.create_successor_precommit(
        failed_baseline_path=failed_baseline_path,
        retired_spatial_outcome_path=retired_spatial_path,
        fit_precommit_path=fit_precommit_path,
        supervision_authority_record=authority_record,
        supervision_coverage=coverage,
        descriptor_conflict_audit=conflict,
        runtime_probe=runtime_probe,
        trainer_source_sha256=source_sha256(),
        training_entrypoint_status="live_runnable_development_only",
        intended_checkpoint=paths["checkpoint"],
        intended_result=paths["result"],
        output=paths["core_precommit"],
    )
    core_precommit = _load(paths["core_precommit"], label="core precommit")
    if core_precommit != expected_core_precommit:
        raise TinyLocalTrainingError("core precommit exact reconstruction differs")
    expected_precommit = _seal(
        _training_precommit_body(
            authorization=authorization,
            core_precommit=core_precommit,
            authority_record=authority_record,
            conflict_artifact=conflict_artifact,
            conflict=conflict,
        )
    )
    precommit = _load(paths["precommit"], label="training precommit")
    if precommit != expected_precommit:
        raise TinyLocalTrainingError("training precommit exact reconstruction differs")
    return paths, authorization, core_precommit, precommit, fit_precommit


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> bytes:
    torch, _, _ = core._torch_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raw = path.read_bytes()
        existing = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        if (
            not isinstance(existing, dict)
            or set(existing) != set(payload)
            or any(existing[key] != payload[key] for key in payload if key != "state_dict")
            or core.state_dict_digest(existing["state_dict"])
            != core.state_dict_digest(payload["state_dict"])
        ):
            raise TinyLocalTrainingError("refusing to overwrite nonidentical checkpoint")
        return raw
    temporary = path.with_name(path.name + ".tmp-tiny-local-training")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return path.read_bytes()
    except OSError as exc:
        raise TinyLocalTrainingError(f"cannot save checkpoint: {exc}") from exc


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "prediction_rows_by_digest_group"
    }


def _prediction_digest(metrics: Mapping[str, Any]) -> str:
    rows = metrics.get("prediction_rows_by_digest_group")
    if not isinstance(rows, list):
        raise TinyLocalTrainingError("validation prediction rows are missing")
    return "sha256:" + canonical_digest(rows)


def _validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = {
        "arc_top1": 0.80,
        "known_catalog_binary_balanced_accuracy": 0.65,
        "straight_top1": 0.65,
    }
    checks = {
        name: float(metrics[name]) >= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "checks": checks,
        "descriptor_diagnostics": {
            "deployment_authority": False,
            "geometry_interval_hit": metrics["descriptor_geometry_interval_hit"],
            "matched_primitive_accuracy": metrics[
                "descriptor_matched_primitive_accuracy"
            ],
            "primitive_multiset_exact": metrics[
                "descriptor_primitive_multiset_exact"
            ],
            "role": "reported_and_checkpoint_selection_tiebreaker_not_deployment_gate",
        },
        "on_failure": (
            "development_GAP;_fresh_CAL_eval_family_target_and_query_remain_sealed"
        ),
        "passed": all(checks.values()),
        "thresholds": thresholds,
    }


def run_training(
    *, repository_root: Path, dataset_root: Path, output_root: Path
) -> dict[str, Any]:
    """Run one bounded development fit; no later-cohort transition exists here."""

    paths, authorization, _core_precommit, precommit, fit_precommit = _verify_prepared(
        repository_root=repository_root,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    deadline = core.WallDeadline()
    started = time.monotonic()
    groups = materialize_groups(
        repository_root=repository_root,
        dataset_root=dataset_root,
        fit_precommit=fit_precommit,
        conflict_audit=precommit["descriptor_target_conflict_audit"],
        deadline=deadline,
    )
    trained = train_core(groups, deadline=deadline)
    state_digest = core.state_dict_digest(trained["state"])
    checkpoint = {
        "architecture_id": core.ARCHITECTURE_ID,
        "config_digest": core.successor_config_digest(),
        "selected_epoch": trained["best_epoch"],
        "source_sha256": core.source_sha256(),
        "state_dict": trained["state"],
        "state_dict_sha256": state_digest,
        "training_precommit_record_digest": precommit["record_digest"],
    }
    deadline.check(reserve_seconds=FINALIZATION_RESERVE_SECONDS)
    checkpoint_raw = _save_checkpoint(paths["checkpoint"], checkpoint)
    deadline.check(reserve_seconds=20.0)
    # Fresh checkpoint load and validation replay happen before result sealing.
    torch, _, _ = core._torch_runtime()
    loaded = torch.load(BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    fresh_model = core.build_model(seed=int(core.PROTOCOL["random_seed"]))
    fresh_model.load_state_dict(loaded["state_dict"], strict=True)
    fresh_metrics = validation_metrics(
        fresh_model, trained["validation_groups"], deadline=deadline
    )
    if fresh_metrics != trained["metrics"]:
        raise TinyLocalTrainingError("fresh checkpoint validation replay differs")
    deadline.check(reserve_seconds=5.0)
    elapsed = time.monotonic() - started
    body = {
        "architecture_id": core.ARCHITECTURE_ID,
        "authorization_record_digest": authorization["record_digest"],
        "checkpoint_raw_sha256": core._address(checkpoint_raw),
        "checkpoint_state_dict_sha256": state_digest,
        "config_digest": core.successor_config_digest(),
        "decontaminated_occurrence_counts": {"train": 11_200, "validation": 1_392},
        "descriptor_target_conflict_audit": precommit[
            "descriptor_target_conflict_audit"
        ],
        "forbidden_cohorts_opened": 0,
        "history": [
            _metric_summary(value) for value in trained["history"]
        ],
        "pixel_occurrences_reread": 12_592,
        "runtime_budget": {
            "cooperative_batch_boundary_deadline": True,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "limit_seconds": float(core.PROTOCOL["maximum_wall_runtime_seconds"]),
            "passed_before_result_seal": elapsed
            < float(core.PROTOCOL["maximum_wall_runtime_seconds"]),
        },
        "runtime_seconds": elapsed,
        "schema": RESULT_SCHEMA,
        "selected_epoch": trained["best_epoch"],
        "source_sha256": source_sha256(),
        "training_precommit_record_digest": precommit["record_digest"],
        "validation_gate": _validation_gate(fresh_metrics),
        "validation_metrics": _metric_summary(fresh_metrics),
        "validation_prediction_rows_digest": _prediction_digest(fresh_metrics),
    }
    result = _seal(body)
    deadline.check(reserve_seconds=5.0)
    core._write_once(paths["result"], result)
    deadline.check()
    if _load(paths["result"], label="training result") != result:
        raise TinyLocalTrainingError("training result fresh replay differs")
    core.load_verified_checkpoint(
        paths["checkpoint"],
        expected_training_precommit_record_digest=precommit["record_digest"],
        training_result=result,
        expected_training_result_record_digest=result["record_digest"],
        require_passed_development_gate=False,
    )
    deadline.check()
    return result


def replay_training(
    *, repository_root: Path, dataset_root: Path, output_root: Path
) -> dict[str, Any]:
    """Replay the selected checkpoint on the same exposed validation cohort."""

    paths, _authorization, _core_precommit, precommit, fit_precommit = _verify_prepared(
        repository_root=repository_root,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    archived = _load(paths["result"], label="training result")
    model, checkpoint, checkpoint_raw_sha256 = core.load_verified_checkpoint(
        paths["checkpoint"],
        expected_training_precommit_record_digest=precommit["record_digest"],
        training_result=archived,
        expected_training_result_record_digest=archived["record_digest"],
        require_passed_development_gate=False,
    )
    deadline = core.WallDeadline()
    groups = materialize_groups(
        repository_root=repository_root,
        dataset_root=dataset_root,
        fit_precommit=fit_precommit,
        conflict_audit=precommit["descriptor_target_conflict_audit"],
        deadline=deadline,
    )
    validation = tuple(group for group in groups if group.cohort == "validation")
    metrics = validation_metrics(model, validation, deadline=deadline)
    if (
        _metric_summary(metrics) != archived.get("validation_metrics")
        or _prediction_digest(metrics)
        != archived.get("validation_prediction_rows_digest")
        or checkpoint_raw_sha256 != archived.get("checkpoint_raw_sha256")
        or checkpoint["state_dict_sha256"]
        != archived.get("checkpoint_state_dict_sha256")
    ):
        raise TinyLocalTrainingError("archived training replay differs")
    body = {
        "archived_result_record_digest": archived["record_digest"],
        "checkpoint_raw_sha256": checkpoint_raw_sha256,
        "forbidden_cohorts_opened": 0,
        "metrics_exact": True,
        "pixel_occurrences_reread": 12_592,
        "predictions_exact": True,
        "schema": REPLAY_SCHEMA,
        "source_sha256": source_sha256(),
        "training_precommit_record_digest": precommit["record_digest"],
    }
    replay = _seal(body)
    deadline.check(reserve_seconds=5.0)
    core._write_once(paths["replay"], replay)
    deadline.check()
    if _load(paths["replay"], label="training replay") != replay:
        raise TinyLocalTrainingError("training replay fresh load differs")
    return replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--fit-precommit", type=Path, required=True)
    prepare.add_argument("--failed-baseline", type=Path, required=True)
    prepare.add_argument("--retired-spatial-outcome", type=Path, required=True)
    prepare.add_argument("--descriptor-conflict-audit", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    for name in ("train", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    exit_code = 0
    if args.command == "prepare":
        result = prepare_training(
            repository_root=args.repository_root,
            dataset_root=args.dataset_root,
            fit_precommit_path=args.fit_precommit,
            failed_baseline_path=args.failed_baseline,
            retired_spatial_outcome_path=args.retired_spatial_outcome,
            descriptor_conflict_audit_path=args.descriptor_conflict_audit,
            output_root=args.output_root,
        )["precommit"]
    elif args.command == "train":
        result = run_training(
            repository_root=args.repository_root,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
        if result.get("validation_gate", {}).get("passed") is not True:
            exit_code = 2
    else:
        result = replay_training(
            repository_root=args.repository_root,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
    summary = {"record_digest": result["record_digest"]}
    if args.command == "train":
        summary["validation_gate_passed"] = (
            result.get("validation_gate", {}).get("passed") is True
        )
    print(json.dumps(summary, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
