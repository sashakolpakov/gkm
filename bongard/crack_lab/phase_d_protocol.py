"""Preregistered corpus and control protocol for Bongard Phase D.

The experiment runners historically accepted a sampler ``limit`` without
pinning whether it meant a corpus size or a per-source pool size.  They also
assigned opaque IDs after applying runner-specific ordering.  Both behaviours
are unsafe for cross-track comparisons: the same ``problem_04`` can otherwise
refer to different panels.

This module provides the runner-independent pieces of the Phase D protocol:

* sample one deterministic maximum corpus, with an explicit per-source count
  unit and one shared ordering policy;
* freeze the ordered, labelled panel bytes into a ground-truth-free manifest;
* derive deterministic balanced shuffled-side controls from that manifest;
* compute shared-library and no-share definition-cost traces;
* preregister track-separated arms and validate one-arm result documents.

The manifest deliberately omits dataset problem IDs and concept names.  Those
remain in harness-only result files.  Scaling is always a prefix operation on
one frozen maximum corpus; changing the sampler limit creates a different
corpus rather than silently redefining an existing scale.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # package import in tests/tools, script import in crack_lab runners
    from . import (
        bongard_arena,
        codex_proposer,
        dataset,
        predicate_pricing,
        semantic_replay,
    )
    from .semantic_selection import RISK_FIELDS, Track
except ImportError:  # pragma: no cover - exercised by script entry points
    import bongard_arena  # type: ignore
    import codex_proposer  # type: ignore
    import dataset  # type: ignore
    import predicate_pricing  # type: ignore
    import semantic_replay  # type: ignore
    from semantic_selection import RISK_FIELDS, Track  # type: ignore


CORPUS_SCHEMA = "bongard.phase-d-corpus/v1"
CORPUS_BUNDLE_SCHEMA = "bongard.phase-d-corpus-bundle/v1"
SHUFFLED_SIDES_SCHEMA = "bongard.phase-d-shuffled-sides/v1"
COMPLEXITY_TRACE_SCHEMA = "bongard.phase-d-complexity-trace/v1"
PREREGISTRATION_SCHEMA = "bongard.phase-d-preregistration/v6"
TRACK_REPORT_SCHEMA = "bongard.phase-d-track-report/v7"
PROPOSER_RECEIPT_SCHEMA = codex_proposer.CODEX_RECEIPT_SCHEMA
SEMANTIC_PROPOSER_RECEIPT_SCHEMA = \
    "bongard.semantic-proposer-model-receipt/v1"
SEMANTIC_MAX_MODEL_ATTEMPTS_PER_ROUND = 3
EXECUTION_BINDING_SCHEMA = "bongard.phase-d-execution-binding/v1"

COUNT_POLICY = "limit-per-selected-source/v1"
SAMPLER_ORDER = "sampler-order/v1"
INTERLEAVED_ORDER = "four-basic-then-one-abstract/v1"
SHUFFLE_POLICY = "balanced-three-from-each-original-side/v1"

OBSERVED = "observed"
SHUFFLED_SIDES = "shuffled-sides"
SHARED = "shared"
NO_SHARE = "no-share"

# HYBRID exists in the general semantic selector, but Phase D has no runner
# that can emit a HYBRID result.  Advertising it here would create arms that
# can never be completed.
TRACK_VALUES = (Track.UNRESTRICTED.value, Track.SEMANTIC_PURE.value)
SOURCES = ("basic", "abstract", "both")

SCALE_POLICY = "ordered-prefixes-of-one-frozen-maximum-corpus/v1"
AGGREGATION_POLICY = "report-each-track-and-arm-separately/v1"
NO_SHARE_DEFINITION_AVAILABILITY = "held-fixed"
NO_SHARE_DEFINITION_CHARGE = "full-cost-per-accepted-rule-use"
NO_SHARE_SELECTION_POLICY = "copy-primary-rules-risks-and-outcomes"
NO_SHARE_CAUSAL_CLAIM = "accounting-only; no independent solve-rate claim"
EXECUTION_POLICY_SCHEMA = "bongard.phase-d-execution-policy/v5"

_HARNESS_SOURCE_FILES = (
    "phase_d_protocol.py",
    "prepare_phase_d.py",
    "collect_phase_d.py",
    "bongard_arena.py",
    "predicate_pricing.py",
    "bongard_legs.py",
    "codex_proposer.py",
    "run_semantic_cone.py",
    "cofibered_proposer.py",
    "semantic_legs.py",
    "semantic_selection.py",
    "semantic_verifier.py",
    "semantic_compiler.py",
    "semantic_ir.py",
    "semantic_requirements.py",
    "soft_semantics.py",
    "semantic_replay.py",
    "semantic_artifacts.py",
    "replay_semantic_runspec.py",
)

_ARM_KEYS = {
    "arm_id", "track", "condition", "label_policy", "sharing_policy",
    "scale", "replicate", "control_digest", "execution_tag",
}

_EXECUTION_TAG_SCHEMA = "bongard.phase-d-execution-tag/v1"

# A no-share report is a held-fixed re-accounting of its primary source, not a
# second scientific run.  These are the only record fields permitted to
# change.  In particular, rules, risks, scores, source bytes/identities,
# attempts, model evidence, panel identities, and solve outcomes must match.
NO_SHARE_ACCOUNTING_DIFFERENCE_FIELDS = frozenset({
    "condition",
    "runner_condition",
    "sharing_policy",
    "report_source_trace_digest",
    "parent_source_trace_digest",
    "rule_cost",
    "marginal_C",
    "charged_definition_node_identities",
    "reused_definition_node_identities",
    "definition_charge",
    "total_charge",
    "pricing_context_digest",
    "verification_digest",
    "phase_execution_binding_digest",
})

_TERMINAL_STATUSES = {
    Track.UNRESTRICTED.value: {
        "SOLVED_UNRESTRICTED": True,
        "UNSOLVED_UNRESTRICTED": False,
        "VERIFIER_FAILURE_UNRESTRICTED": False,
    },
    Track.SEMANTIC_PURE.value: {
        "SOLVED_SEMANTIC_PURE": True,
        "SOLVED_SEMANTIC_PURE_STRESS_FLAGGED": True,
        "APPROXIMATE_SEMANTIC_FIT": False,
        "EXACT_SEMANTIC_FIT_TOLERANT_RUN_POLICY": False,
        "MISSING_LEG": False,
        "COMPILE_FAILED": False,
        "MEASUREMENT_ONLY": False,
        "MORPHISM_UNCHECKED": False,
        "NATURALITY_FAILURE": False,
        "COFIBRATION_FAILURE": False,
        "COUNTEREXAMPLE_FAILURE": False,
        "PROPOSER_PARSE_FAILED": False,
        "NO_PROPOSALS": False,
    },
}


class PhaseDProtocolError(ValueError):
    """A corpus, control, accounting trace, or report violates the protocol."""


@dataclass(frozen=True)
class ShuffledSidesControl:
    """A transformed corpus plus the exact assignment evidence that made it."""

    problems: tuple[dataset.Problem, ...]
    manifest: dict[str, Any]


def dataset_revision(dataset_dir: str) -> str:
    """Return the dataset Git revision, or an explicit unavailable marker."""
    try:
        proc = subprocess.run(
            ["git", "-C", os.path.abspath(dataset_dir), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = proc.stdout.strip().lower()
    if proc.returncode == 0 and len(revision) == 40 \
            and all(char in "0123456789abcdef" for char in revision):
        return revision
    return "unavailable"


def dataset_content_digest(dataset_dir: str) -> str:
    """Hash sampler Python and TSV inputs, including uncommitted changes."""
    root = os.path.abspath(dataset_dir)
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in {".git", "__pycache__", ".pytest_cache"})
        for filename in sorted(filenames):
            if filename.endswith((".py", ".tsv")):
                paths.append(os.path.join(dirpath, filename))
    if not paths:
        return "unavailable"
    digest = hashlib.sha256()
    try:
        for path in sorted(paths):
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with open(path, "rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
            digest.update(b"\0")
    except OSError:
        return "unavailable"
    return "sha256:" + digest.hexdigest()


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PhaseDProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhaseDProtocolError(f"{label} must be a nonempty string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PhaseDProtocolError(
            f"{label} keys differ (missing={missing}, extra={extra})")


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value[7:]
    return len(suffix) == 64 and all(char in "0123456789abcdef" for char in suffix)


def _is_raw_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _without_digest(value: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != digest_key}


def _source_file_digest(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise PhaseDProtocolError(
            f"cannot fingerprint Phase D harness source {filename!r}") from exc
    return "sha256:" + digest.hexdigest()


def _codex_cli_fingerprint() -> dict[str, str]:
    try:
        return codex_proposer.codex_cli_fingerprint()
    except codex_proposer.CodexProposerFailure as exc:
        raise PhaseDProtocolError(
            "cannot fingerprint the installed Codex CLI") from exc


def canonical_execution_policy(
        *, require_unrestricted_cli: bool = True) -> dict[str, Any]:
    """Return the fixed runner policy and exact implementation boundary."""
    body: dict[str, Any] = {
        "schema": EXECUTION_POLICY_SCHEMA,
        "unrestricted": {
            "runner": "bongard_legs.py",
            "checkpoint_schema": "bongard.unrestricted-report/v8",
            "verifier_failure_policy": (
                "canonical-zero-admission-exact-cold-replay/v1"),
            "selection_policy": bongard_arena.PRICED_SELECTION_POLICY,
            "definition_pricing": (
                "ast-transitive-closure-loc-literals-call-cardinality/v3"),
            "predicate_pricing_policy_id": (
                predicate_pricing.PREDICATE_PRICING_POLICY_ID),
            "predicate_purity_policy_id": (
                predicate_pricing.PREDICATE_PURITY_POLICY_ID),
            "predicate_capability_manifest": (
                json.loads(json.dumps(
                    predicate_pricing.predicate_capability_manifest(),
                    sort_keys=True, separators=(",", ":")))),
            "allowed_import_roots": sorted(predicate_pricing.ALLOWED_IMPORT_ROOTS),
            "max_rule_atoms": bongard_arena.MAX_RULE_ATOMS,
            "max_candidate_atoms": bongard_arena.MAX_CANDIDATE_ATOMS,
            "call_cost": bongard_arena.CALL_COST,
            "binding_cost": bongard_arena.BINDING_COST,
            "proposer": "codex-cli",
            "proposer_ladder": [
                codex_proposer.DEFAULT_CODEX_MODEL,
                codex_proposer.DEFAULT_CODEX_MODEL,
                codex_proposer.DEFAULT_CODEX_MODEL,
            ],
            "requested_reasoning_effort": (
                codex_proposer.DEFAULT_REASONING_EFFORT),
            "minutes_per_attempt": 15,
            "infrastructure_retry_wait_seconds": 30,
            "maximum_infrastructure_retries_per_rung": 2,
            "restore_wip_context": True,
            "proposer_tool_surface": codex_proposer.CODEX_ISOLATION_POLICY,
            "default_workspace_policy": (
                "private-harness-workspace-plus-separate-auth-and-image-only-"
                "mode0700-codex-views/v1"),
            "proposer_result_policy": (
                "codex-jsonl-one-turn-schema-positive-usage-no-tool-events-"
                "causal-input-output-chain/v2"),
            "proposer_receipt_schema": PROPOSER_RECEIPT_SCHEMA,
            "proposer_input_digest_schema": (
                codex_proposer.PREDICATE_INPUT_DIGEST_SCHEMA),
            "proposer_output_schema_digest": (
                codex_proposer.PREDICATE_PROPOSAL_SCHEMA_DIGEST),
            "proposer_turn_identity_policy": (
                "unique-thread-and-event-stream-per-adaptive-turn/v1"),
            "authoritative_verifier_resource_limits": (
                bongard_arena.verifier_resource_limit_policy()),
            "rotated_loo_folds": 36,
        },
        "semantic_pure": {
            "runner": "run_semantic_cone.py",
            "proposer": "anthropic",
            "model": "sonnet",
            "concrete_model": "claude-sonnet-5",
            "proposer_receipt_schema": SEMANTIC_PROPOSER_RECEIPT_SCHEMA,
            "max_tokens": 8000,
            "rounds": 4,
            "max_model_attempts_per_round": (
                SEMANTIC_MAX_MODEL_ATTEMPTS_PER_ROUND),
            "max_support_errors": 0,
            "max_loo_errors": 0,
            "max_rotated_loo_errors": 0,
            "selection_method": "conditional_free_energy",
            "lambda": 0.02,
            "selection_risk_fields": [
                "R_support", "R_rotated_LOO", "R_naturality",
                "R_parser_stability",
            ],
            "selection_unmeasured_risks": [
                name for name in RISK_FIELDS
                if name not in {
                    "R_support", "R_rotated_LOO", "R_naturality",
                    "R_parser_stability",
                }
            ],
        },
        "harness_sources": {
            filename: _source_file_digest(filename)
            for filename in _HARNESS_SOURCE_FILES
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_cache_tag": getattr(sys.implementation, "cache_tag", ""),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "byteorder": sys.byteorder,
            "python_hash_seed_env": os.environ.get("PYTHONHASHSEED", "random"),
            "python_hash_probes": [
                hash(f"bongard-phase-d/v6/{index}") for index in range(4)],
            "codex_cli": (
                _codex_cli_fingerprint() if require_unrestricted_cli
                else {
                    "version": "not-required:no-unrestricted-track",
                    "launcher_digest": "",
                }),
            "dependencies": semantic_replay.capture_dependency_versions(
                ("Pillow", "anthropic", "numpy", "scikit-image", "scipy"),
                strict=True,
            ),
        },
        "runner_arguments_must_match": True,
    }
    body["policy_digest"] = semantic_replay.canonical_json_digest(body)
    return body


_EXECUTION_BINDING_KEYS = {
    "schema", "preregistration_digest", "execution_policy_digest",
    "arm_id", "track", "condition", "scale", "execution_tag",
    "binding_digest",
}


def validate_execution_binding(value: Mapping[str, Any]) -> None:
    """Validate one closed Phase arm-to-run provenance receipt."""
    if not isinstance(value, Mapping):
        raise PhaseDProtocolError("Phase execution binding must be a mapping")
    _exact_keys(value, _EXECUTION_BINDING_KEYS, "Phase execution binding")
    if value["schema"] != EXECUTION_BINDING_SCHEMA \
            or not _is_digest(value["preregistration_digest"]) \
            or not _is_digest(value["execution_policy_digest"]) \
            or not _is_digest(value["binding_digest"]):
        raise PhaseDProtocolError("Phase execution binding identity is malformed")
    for name in ("arm_id", "track", "condition", "execution_tag"):
        _strict_nonempty_string(value[name], f"execution_binding.{name}")
    _strict_int(value["scale"], "execution_binding.scale", minimum=1)
    reproduced = semantic_replay.canonical_json_digest(
        _without_digest(value, "binding_digest"))
    if value["binding_digest"] != reproduced:
        raise PhaseDProtocolError(
            "Phase execution binding digest does not reproduce")


def execution_binding(
        preregistration: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    """Bind a runner checkpoint to one exact preregistered execution arm."""
    matching = [
        arm for arm in preregistration.get("arms", [])
        if isinstance(arm, Mapping) and arm.get("arm_id") == arm_id
    ]
    if len(matching) != 1:
        raise PhaseDProtocolError("execution binding arm was not preregistered")
    arm = matching[0]
    policy = preregistration.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise PhaseDProtocolError("execution binding lacks execution policy")
    body: dict[str, Any] = {
        "schema": EXECUTION_BINDING_SCHEMA,
        "preregistration_digest": preregistration.get(
            "preregistration_digest"),
        "execution_policy_digest": policy.get("policy_digest"),
        "arm_id": arm["arm_id"],
        "track": arm["track"],
        "condition": arm["condition"],
        "scale": arm["scale"],
        "execution_tag": arm["execution_tag"],
    }
    body["binding_digest"] = semantic_replay.canonical_json_digest(body)
    validate_execution_binding(body)
    return body


def execution_binding_family(
        preregistration: Mapping[str, Any], arm: Mapping[str, Any]) \
        -> list[dict[str, Any]]:
    """Return current and earlier bindings in the same monotone run family."""
    candidates = sorted(
        (
            candidate for candidate in preregistration.get("arms", [])
            if isinstance(candidate, Mapping)
            and candidate.get("track") == arm.get("track")
            and candidate.get("condition") == arm.get("condition")
            and candidate.get("replicate") == arm.get("replicate")
            and candidate.get("control_digest") == arm.get("control_digest")
            and candidate.get("execution_tag") == arm.get("execution_tag")
            and isinstance(candidate.get("scale"), int)
            and candidate["scale"] <= arm.get("scale", 0)
        ),
        key=lambda item: item["scale"],
    )
    return [
        execution_binding(preregistration, candidate["arm_id"])
        for candidate in candidates
    ]


def sample_corpus(
        dataset_dir: str,
        *,
        limit_per_source: int,
        seed: int,
        source: str,
        panel_size: int = dataset.PANEL_SIZE,
        interleave_every: int = 5) -> list[dataset.Problem]:
    """Sample the maximum corpus once under explicit count/order semantics.

    For ``source='both'``, ``limit_per_source=N`` requests up to N basic and N
    abstract problems and then interleaves them.  It therefore does *not* mean
    a total corpus size.  Experiments at sizes 1, 5, and 25 must take prefixes
    of this returned list, never call this function again with smaller limits.
    """
    limit = _strict_int(limit_per_source, "limit_per_source", minimum=1)
    sampling_seed = _strict_int(seed, "seed")
    size = _strict_int(panel_size, "panel_size", minimum=1)
    every = _strict_int(interleave_every, "interleave_every", minimum=2)
    if source not in SOURCES:
        raise PhaseDProtocolError(
            f"source must be one of {', '.join(SOURCES)}, got {source!r}")
    if source != "both":
        return dataset.sample_problems(
            dataset_dir,
            limit=limit,
            seed=sampling_seed,
            source=source,
            panel_size=size,
        )
    # The legacy sampler consumes one mutable RNG stream for basic work before
    # abstract ordering. Calling each source independently makes the abstract
    # stream identical alone and inside a combined corpus.
    basic = dataset.sample_problems(
        dataset_dir,
        limit=limit,
        seed=sampling_seed,
        source="basic",
        panel_size=size,
    )
    abstract = dataset.sample_problems(
        dataset_dir,
        limit=limit,
        seed=sampling_seed,
        source="abstract",
        panel_size=size,
    )
    return dataset.interleave_corpus(
        basic,
        abstract,
        every=every,
    )


def _problem_manifest(problem: Any, opaque_id: str) -> dict[str, Any]:
    try:
        positives = tuple(problem.pos)
        negatives = tuple(problem.neg)
        category = str(problem.category)
    except (AttributeError, TypeError) as exc:
        raise PhaseDProtocolError(
            f"{opaque_id} must expose category, pos, and neg") from exc
    if len(positives) != 6 or len(negatives) != 6:
        raise PhaseDProtocolError(
            f"{opaque_id} must contain exactly six panels on each side")
    if not category:
        raise PhaseDProtocolError(f"{opaque_id}.category must be nonempty")
    try:
        panel_records = semantic_replay.panel_records_from_problem(problem)
    except semantic_replay.SemanticReplayError as exc:
        raise PhaseDProtocolError(f"{opaque_id}: invalid panel data: {exc}") from exc
    return {
        "opaque_id": opaque_id,
        "category": category,
        "panel_count": len(panel_records),
        "panel_set_digest": semantic_replay.panel_set_digest(panel_records),
        "panels": [record.manifest_entry() for record in panel_records],
    }


def build_corpus_manifest(
        problems: Sequence[Any],
        *,
        source: str,
        seed: int,
        limit_per_source: int,
        panel_size: int = dataset.PANEL_SIZE,
        dataset_revision: str = "unavailable",
        dataset_inputs_digest: str = "unavailable",
        interleave_every: int = 5) -> dict[str, Any]:
    """Freeze ordered labelled panels without exposing IDs or concept names."""
    if source not in SOURCES:
        raise PhaseDProtocolError(f"unsupported source {source!r}")
    sampling_seed = _strict_int(seed, "seed")
    limit = _strict_int(limit_per_source, "limit_per_source", minimum=1)
    size = _strict_int(panel_size, "panel_size", minimum=1)
    every = _strict_int(interleave_every, "interleave_every", minimum=2)
    revision = _strict_nonempty_string(dataset_revision, "dataset_revision")
    inputs_digest = _strict_nonempty_string(
        dataset_inputs_digest, "dataset_inputs_digest")
    if inputs_digest != "unavailable" and not _is_digest(inputs_digest):
        raise PhaseDProtocolError("dataset_inputs_digest must be sha256 or unavailable")
    if not problems:
        raise PhaseDProtocolError("cannot freeze an empty corpus")

    entries = [
        _problem_manifest(problem, f"problem_{index:02d}")
        for index, problem in enumerate(problems)
    ]
    counts: dict[str, int] = {}
    for entry in entries:
        category = entry["category"]
        counts[category] = counts.get(category, 0) + 1
    allowed_categories = {source} if source != "both" else {"basic", "abstract"}
    if not set(counts).issubset(allowed_categories):
        raise PhaseDProtocolError(
            f"corpus categories {sorted(counts)} contradict source {source!r}")

    body: dict[str, Any] = {
        "schema": CORPUS_SCHEMA,
        "sampling": {
            "source": source,
            "seed": sampling_seed,
            "count_policy": COUNT_POLICY,
            "limit_per_source": limit,
            "panel_size": size,
            "order_policy": (
                INTERLEAVED_ORDER if source == "both" else SAMPLER_ORDER),
            "interleave_every": every if source == "both" else None,
            "dataset_revision": revision,
            "dataset_inputs_digest": inputs_digest,
        },
        "generator": {
            "dataset_module": semantic_replay.source_object_fingerprint(
                dataset, require_source=True),
            "protocol_sampler": semantic_replay.source_object_fingerprint(
                sample_corpus, require_source=True),
            "numpy_version": str(dataset.np.__version__),
            "sampler_rng": "independent-source-numpy-RandomState-MT19937/v1",
            "panel_rng": "sha256(seed:source-problem-id:side:index)-uint32/v1",
        },
        "problem_count": len(entries),
        "source_counts": dict(sorted(counts.items())),
        "problems": entries,
        "ground_truth_included": False,
    }
    body["corpus_digest"] = semantic_replay.canonical_json_digest(body)
    validate_corpus_manifest(body)
    return body


def validate_corpus_manifest(manifest: Mapping[str, Any]) -> None:
    """Strictly validate manifest structure and its self-digest."""
    if not isinstance(manifest, Mapping):
        raise PhaseDProtocolError("corpus manifest must be a mapping")
    _exact_keys(
        manifest,
        {
            "schema", "sampling", "problem_count", "source_counts",
            "generator", "problems", "ground_truth_included", "corpus_digest",
        },
        "corpus manifest",
    )
    if manifest["schema"] != CORPUS_SCHEMA:
        raise PhaseDProtocolError(f"unsupported corpus schema {manifest['schema']!r}")
    if manifest["ground_truth_included"] is not False:
        raise PhaseDProtocolError("corpus manifest must not include ground truth")
    if not _is_digest(manifest["corpus_digest"]):
        raise PhaseDProtocolError("corpus_digest is malformed")
    observed_digest = semantic_replay.canonical_json_digest(
        _without_digest(manifest, "corpus_digest"))
    if observed_digest != manifest["corpus_digest"]:
        raise PhaseDProtocolError("corpus manifest digest does not reproduce")

    sampling = manifest["sampling"]
    if not isinstance(sampling, Mapping):
        raise PhaseDProtocolError("sampling must be a mapping")
    _exact_keys(
        sampling,
        {
            "source", "seed", "count_policy", "limit_per_source",
            "panel_size", "order_policy", "interleave_every",
            "dataset_revision", "dataset_inputs_digest",
        },
        "sampling",
    )
    source = sampling["source"]
    if source not in SOURCES or sampling["count_policy"] != COUNT_POLICY:
        raise PhaseDProtocolError("sampling source/count policy is invalid")
    _strict_int(sampling["seed"], "sampling.seed")
    _strict_int(sampling["limit_per_source"], "sampling.limit_per_source", minimum=1)
    _strict_int(sampling["panel_size"], "sampling.panel_size", minimum=1)
    _strict_nonempty_string(sampling["dataset_revision"], "sampling.dataset_revision")
    inputs_digest = _strict_nonempty_string(
        sampling["dataset_inputs_digest"], "sampling.dataset_inputs_digest")
    if inputs_digest != "unavailable" and not _is_digest(inputs_digest):
        raise PhaseDProtocolError("sampling.dataset_inputs_digest is malformed")
    if source == "both":
        if sampling["order_policy"] != INTERLEAVED_ORDER:
            raise PhaseDProtocolError("combined corpus must use the interleaved order")
        _strict_int(sampling["interleave_every"], "sampling.interleave_every", minimum=2)
    elif sampling["order_policy"] != SAMPLER_ORDER \
            or sampling["interleave_every"] is not None:
        raise PhaseDProtocolError("single-source corpus has invalid ordering metadata")

    generator = manifest["generator"]
    if not isinstance(generator, Mapping):
        raise PhaseDProtocolError("generator must be a mapping")
    _exact_keys(
        generator,
        {"dataset_module", "protocol_sampler", "numpy_version", "sampler_rng",
         "panel_rng"},
        "generator",
    )
    for name in ("numpy_version", "sampler_rng", "panel_rng"):
        _strict_nonempty_string(generator[name], f"generator.{name}")
    for name in ("dataset_module", "protocol_sampler"):
        fingerprint = generator[name]
        if not isinstance(fingerprint, Mapping) \
                or fingerprint.get("source_complete") is not True \
                or not _is_digest(fingerprint.get("source_digest")) \
                or not _is_digest(fingerprint.get("module_source_digest")):
            raise PhaseDProtocolError(f"generator.{name} is incomplete")

    problems = manifest["problems"]
    count = _strict_int(manifest["problem_count"], "problem_count", minimum=1)
    if not isinstance(problems, list) or len(problems) != count:
        raise PhaseDProtocolError("problem_count does not match problems")
    source_counts = manifest["source_counts"]
    if not isinstance(source_counts, Mapping):
        raise PhaseDProtocolError("source_counts must be a mapping")
    validated_counts: dict[str, int] = {}
    for category, value in source_counts.items():
        validated_counts[_strict_nonempty_string(category, "source_counts key")] = \
            _strict_int(value, f"source_counts.{category}")
    if sum(validated_counts.values()) != count:
        raise PhaseDProtocolError("source_counts do not sum to problem_count")
    allowed_categories = {source} if source != "both" else {"basic", "abstract"}
    if not set(validated_counts).issubset(allowed_categories):
        raise PhaseDProtocolError("source_counts contradict the selected source")
    limit = sampling["limit_per_source"]
    if any(value > limit for value in validated_counts.values()):
        raise PhaseDProtocolError("a source count exceeds limit_per_source")

    observed_counts: dict[str, int] = {}
    for index, entry in enumerate(problems):
        if not isinstance(entry, Mapping):
            raise PhaseDProtocolError(f"problems[{index}] must be a mapping")
        _exact_keys(
            entry,
            {"opaque_id", "category", "panel_count", "panel_set_digest", "panels"},
            f"problems[{index}]",
        )
        expected_oid = f"problem_{index:02d}"
        if entry["opaque_id"] != expected_oid:
            raise PhaseDProtocolError(
                f"problems[{index}].opaque_id must be {expected_oid!r}")
        category = _strict_nonempty_string(entry["category"], f"{expected_oid}.category")
        observed_counts[category] = observed_counts.get(category, 0) + 1
        if entry["panel_count"] != 12 or not _is_digest(entry["panel_set_digest"]):
            raise PhaseDProtocolError(f"{expected_oid} has invalid panel metadata")
        panels = entry["panels"]
        if not isinstance(panels, list) or len(panels) != 12:
            raise PhaseDProtocolError(f"{expected_oid} must bind twelve panels")
        sides = []
        for panel_index, panel in enumerate(panels):
            if not isinstance(panel, Mapping):
                raise PhaseDProtocolError(
                    f"{expected_oid}.panels[{panel_index}] must be a mapping")
            _exact_keys(
                panel,
                {"side", "index", "shape", "dtype", "encoding", "content_digest"},
                f"{expected_oid}.panels[{panel_index}]",
            )
            expected_side = "pos" if panel_index < 6 else "neg"
            expected_index = panel_index if panel_index < 6 else panel_index - 6
            if panel.get("side") != expected_side or panel.get("index") != expected_index:
                raise PhaseDProtocolError(
                    f"{expected_oid}.panels[{panel_index}] is out of canonical order")
            if panel.get("shape") != [sampling["panel_size"], sampling["panel_size"]]:
                raise PhaseDProtocolError(
                    f"{expected_oid}.panels[{panel_index}] shape contradicts panel_size")
            if not _is_digest(panel.get("content_digest")):
                raise PhaseDProtocolError(
                    f"{expected_oid}.panels[{panel_index}] has malformed digest")
            sides.append(panel.get("side"))
        if sides.count("pos") != 6 or sides.count("neg") != 6:
            raise PhaseDProtocolError(f"{expected_oid} side balance is invalid")
        expected_panel_set_digest = semantic_replay.canonical_json_digest({
            "schema": semantic_replay.PANEL_SCHEMA,
            "panels": panels,
        })
        if entry["panel_set_digest"] != expected_panel_set_digest:
            raise PhaseDProtocolError(
                f"{expected_oid}.panel_set_digest does not reproduce")
    if dict(sorted(observed_counts.items())) != dict(sorted(validated_counts.items())):
        raise PhaseDProtocolError("source_counts do not match problem categories")


def assert_corpus_matches(
        manifest: Mapping[str, Any], problems: Sequence[Any]) -> None:
    """Re-hash live panels and reject any corpus/order/label drift."""
    validate_corpus_manifest(manifest)
    sampling = manifest["sampling"]
    rebuilt = build_corpus_manifest(
        problems,
        source=sampling["source"],
        seed=sampling["seed"],
        limit_per_source=sampling["limit_per_source"],
        panel_size=sampling["panel_size"],
        dataset_revision=sampling["dataset_revision"],
        dataset_inputs_digest=sampling["dataset_inputs_digest"],
        interleave_every=sampling["interleave_every"] or 5,
    )
    if rebuilt["corpus_digest"] != manifest["corpus_digest"]:
        raise PhaseDProtocolError("live corpus differs from frozen corpus manifest")


def assert_corpus_prefix_matches(
        manifest: Mapping[str, Any], problems: Sequence[Any]) -> None:
    """Validate an active scale as an exact prefix of a frozen maximum corpus."""
    validate_corpus_manifest(manifest)
    if not problems:
        raise PhaseDProtocolError("active corpus prefix must be nonempty")
    if len(problems) > manifest["problem_count"]:
        raise PhaseDProtocolError("active corpus exceeds the frozen maximum")
    for index, problem in enumerate(problems):
        observed = _problem_manifest(problem, f"problem_{index:02d}")
        expected = manifest["problems"][index]
        if semantic_replay.canonical_json_digest(observed) != \
                semantic_replay.canonical_json_digest(expected):
            raise PhaseDProtocolError(
                f"problem_{index:02d} differs from the frozen corpus prefix")


def corpus_prefix_ids(manifest: Mapping[str, Any], scale: int) -> tuple[str, ...]:
    validate_corpus_manifest(manifest)
    requested = _strict_int(scale, "scale", minimum=1)
    if requested > manifest["problem_count"]:
        raise PhaseDProtocolError(
            f"scale {requested} exceeds frozen corpus size {manifest['problem_count']}")
    return tuple(entry["opaque_id"] for entry in manifest["problems"][:requested])


def build_corpus_bundle(
        problems: Sequence[Any], corpus_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Embed canonical panel bytes for every outcome, including failures."""
    assert_corpus_matches(corpus_manifest, problems)
    entries: list[dict[str, Any]] = []
    for problem, manifest_entry in zip(problems, corpus_manifest["problems"]):
        records = semantic_replay.panel_records_from_problem(problem)
        entries.append({
            "opaque_id": manifest_entry["opaque_id"],
            "panel_set_digest": semantic_replay.panel_set_digest(records),
            "panels": [record.to_dict() for record in records],
        })
    body: dict[str, Any] = {
        "schema": CORPUS_BUNDLE_SCHEMA,
        "corpus_digest": corpus_manifest["corpus_digest"],
        "problem_count": len(entries),
        "problems": entries,
        "ground_truth_included": False,
    }
    body["bundle_digest"] = semantic_replay.canonical_json_digest(body)
    validate_corpus_bundle(body, corpus_manifest)
    return body


def validate_corpus_bundle(
        bundle: Mapping[str, Any], corpus_manifest: Mapping[str, Any]) -> None:
    validate_corpus_manifest(corpus_manifest)
    if not isinstance(bundle, Mapping):
        raise PhaseDProtocolError("corpus bundle must be a mapping")
    _exact_keys(
        bundle,
        {"schema", "corpus_digest", "problem_count", "problems",
         "ground_truth_included", "bundle_digest"},
        "corpus bundle",
    )
    if bundle["schema"] != CORPUS_BUNDLE_SCHEMA \
            or bundle["ground_truth_included"] is not False:
        raise PhaseDProtocolError("unsupported or ground-truth-bearing corpus bundle")
    if bundle["corpus_digest"] != corpus_manifest["corpus_digest"]:
        raise PhaseDProtocolError("corpus bundle belongs to a different manifest")
    if not _is_digest(bundle["bundle_digest"]) \
            or semantic_replay.canonical_json_digest(
                _without_digest(bundle, "bundle_digest")) != bundle["bundle_digest"]:
        raise PhaseDProtocolError("corpus bundle digest does not reproduce")
    if bundle["problem_count"] != corpus_manifest["problem_count"] \
            or not isinstance(bundle["problems"], list) \
            or len(bundle["problems"]) != bundle["problem_count"]:
        raise PhaseDProtocolError("corpus bundle problem count differs")
    for index, (entry, manifest_entry) in enumerate(
            zip(bundle["problems"], corpus_manifest["problems"])):
        if not isinstance(entry, Mapping):
            raise PhaseDProtocolError(f"bundle problems[{index}] must be a mapping")
        _exact_keys(
            entry,
            {"opaque_id", "panel_set_digest", "panels"},
            f"bundle problems[{index}]",
        )
        if entry["opaque_id"] != manifest_entry["opaque_id"] \
                or entry["panel_set_digest"] != manifest_entry["panel_set_digest"] \
                or not isinstance(entry["panels"], list) \
                or len(entry["panels"]) != 12:
            raise PhaseDProtocolError(
                f"bundle {manifest_entry['opaque_id']} identity differs")
        try:
            records = tuple(
                semantic_replay.PanelRecord.from_dict(panel)
                for panel in entry["panels"])
        except semantic_replay.SemanticReplayError as exc:
            raise PhaseDProtocolError(
                f"bundle {manifest_entry['opaque_id']} panel is invalid: {exc}") \
                from exc
        if semantic_replay.panel_set_digest(records) != entry["panel_set_digest"]:
            raise PhaseDProtocolError(
                f"bundle {manifest_entry['opaque_id']} panel set differs")
        if [record.manifest_entry() for record in records] != manifest_entry["panels"]:
            raise PhaseDProtocolError(
                f"bundle {manifest_entry['opaque_id']} panel manifest differs")


def problems_from_corpus_bundle(
        bundle: Mapping[str, Any], corpus_manifest: Mapping[str, Any]) \
        -> tuple[dataset.Problem, ...]:
    """Reconstruct opaque, ground-truth-free problems from embedded bytes."""
    validate_corpus_bundle(bundle, corpus_manifest)
    problems: list[dataset.Problem] = []
    for bundle_entry, manifest_entry in zip(
            bundle["problems"], corpus_manifest["problems"]):
        records = tuple(
            semantic_replay.PanelRecord.from_dict(panel)
            for panel in bundle_entry["panels"])
        positives = tuple(record.decode() for record in records if record.side == "pos")
        negatives = tuple(record.decode() for record in records if record.side == "neg")
        problems.append(dataset.Problem(
            problem_id=manifest_entry["opaque_id"],
            category=manifest_entry["category"],
            concept="",
            pos=positives,
            neg=negatives,
        ))
    return tuple(problems)


def _rank_key(*parts: Any) -> bytes:
    return hashlib.sha256(semantic_replay.canonical_json_bytes(list(parts))).digest()


def _balanced_assignment(
        entry: Mapping[str, Any], seed: int, replicate: int) -> list[dict[str, Any]]:
    oid = entry["opaque_id"]
    sources = [dict(panel) for panel in entry["panels"]]
    positives = [panel for panel in sources if panel["side"] == "pos"]
    negatives = [panel for panel in sources if panel["side"] == "neg"]
    positives.sort(key=lambda panel: _rank_key(
        "select", seed, replicate, oid, "pos", panel["index"],
        panel["content_digest"]))
    negatives.sort(key=lambda panel: _rank_key(
        "select", seed, replicate, oid, "neg", panel["index"],
        panel["content_digest"]))
    target_pos = positives[:3] + negatives[:3]
    target_neg = positives[3:] + negatives[3:]
    target_pos.sort(key=lambda panel: _rank_key(
        "order-pos", seed, replicate, oid, panel["side"], panel["index"],
        panel["content_digest"]))
    target_neg.sort(key=lambda panel: _rank_key(
        "order-neg", seed, replicate, oid, panel["side"], panel["index"],
        panel["content_digest"]))
    assignment: list[dict[str, Any]] = []
    for target_side, selected in (("pos", target_pos), ("neg", target_neg)):
        for target_index, panel in enumerate(selected):
            assignment.append({
                "target_side": target_side,
                "target_index": target_index,
                "source_side": panel["side"],
                "source_index": panel["index"],
                "content_digest": panel["content_digest"],
            })
    return assignment


def _controlled_panel_set_digest(
        entry: Mapping[str, Any],
        assignment: Sequence[Mapping[str, Any]]) -> str:
    """Derive a controlled panel-set digest without loading panel bytes."""
    base_by_slot = {
        (panel["side"], panel["index"]): panel
        for panel in entry["panels"]
    }
    target_manifests: list[dict[str, Any]] = []
    for item in assignment:
        source = (item["source_side"], item["source_index"])
        panel_manifest = dict(base_by_slot[source])
        panel_manifest["side"] = item["target_side"]
        panel_manifest["index"] = item["target_index"]
        target_manifests.append(panel_manifest)
    target_manifests.sort(key=lambda panel: (
        0 if panel["side"] == "pos" else 1, panel["index"]))
    return semantic_replay.canonical_json_digest({
        "schema": semantic_replay.PANEL_SCHEMA,
        "panels": target_manifests,
    })


def build_shuffled_control_manifest(
        corpus_manifest: Mapping[str, Any], *, seed: int,
        replicate: int = 0) -> dict[str, Any]:
    """Build exact shuffled-side assignment evidence from a frozen manifest.

    Panel bytes are unnecessary because the assignment and transformed panel
    identities are deterministic functions of the frozen slot digests.  This
    lets preregistration bind controls before either runner is invoked.
    """
    validate_corpus_manifest(corpus_manifest)
    control_seed = _strict_int(seed, "control seed")
    control_replicate = _strict_int(replicate, "replicate")
    evidence: list[dict[str, Any]] = []
    for entry in corpus_manifest["problems"]:
        assignment = _balanced_assignment(
            entry, control_seed, control_replicate)
        evidence.append({
            "opaque_id": entry["opaque_id"],
            "assignment": assignment,
            "controlled_panel_set_digest": _controlled_panel_set_digest(
                entry, assignment),
        })
    body: dict[str, Any] = {
        "schema": SHUFFLED_SIDES_SCHEMA,
        "control_kind": SHUFFLED_SIDES,
        "assignment_policy": SHUFFLE_POLICY,
        "base_corpus_digest": corpus_manifest["corpus_digest"],
        "seed": control_seed,
        "replicate": control_replicate,
        "problem_count": corpus_manifest["problem_count"],
        "problems": evidence,
    }
    body["control_digest"] = semantic_replay.canonical_json_digest(body)
    validate_shuffled_control_manifest(body, corpus_manifest)
    return body


def build_shuffled_sides_control(
        problems: Sequence[Any],
        corpus_manifest: Mapping[str, Any],
        *,
        seed: int,
        replicate: int = 0) -> ShuffledSidesControl:
    """Construct a balanced deterministic negative-control corpus.

    Every controlled side receives exactly three original positives and three
    original negatives.  Conditioning on this balance prevents an unchanged
    or almost-unchanged random split from being silently accepted as a useful
    shuffled-side replicate.
    """
    control_seed = _strict_int(seed, "control seed")
    control_replicate = _strict_int(replicate, "replicate")
    assert_corpus_matches(corpus_manifest, problems)
    controlled: list[dataset.Problem] = []
    frozen_manifest = build_shuffled_control_manifest(
        corpus_manifest, seed=control_seed, replicate=control_replicate)
    for problem, entry, control_entry in zip(
            problems, corpus_manifest["problems"], frozen_manifest["problems"]):
        assignment = control_entry["assignment"]
        source_arrays = {
            (side, index): panel
            for side, panels in (("pos", tuple(problem.pos)),
                                 ("neg", tuple(problem.neg)))
            for index, panel in enumerate(panels)
        }
        target_pos = tuple(
            source_arrays[(item["source_side"], item["source_index"])]
            for item in assignment if item["target_side"] == "pos")
        target_neg = tuple(
            source_arrays[(item["source_side"], item["source_index"])]
            for item in assignment if item["target_side"] == "neg")
        transformed = dataset.Problem(
            problem_id=f"{entry['opaque_id']}::shuffled::{control_replicate}",
            category=problem.category,
            concept="shuffled-sides-control",
            pos=target_pos,
            neg=target_neg,
        )
        controlled.append(transformed)
        transformed_entry = _problem_manifest(transformed, entry["opaque_id"])
        if transformed_entry["panel_set_digest"] != \
                control_entry["controlled_panel_set_digest"]:
            raise PhaseDProtocolError(
                f"{entry['opaque_id']} controlled panel identity drifted")
    return ShuffledSidesControl(tuple(controlled), frozen_manifest)


def validate_shuffled_control_manifest(
        control_manifest: Mapping[str, Any],
        corpus_manifest: Mapping[str, Any]) -> None:
    """Validate the exact balanced mapping against base panel-slot digests."""
    validate_corpus_manifest(corpus_manifest)
    if not isinstance(control_manifest, Mapping):
        raise PhaseDProtocolError("control manifest must be a mapping")
    _exact_keys(
        control_manifest,
        {
            "schema", "control_kind", "assignment_policy",
            "base_corpus_digest", "seed", "replicate", "problem_count",
            "problems", "control_digest",
        },
        "control manifest",
    )
    if control_manifest["schema"] != SHUFFLED_SIDES_SCHEMA \
            or control_manifest["control_kind"] != SHUFFLED_SIDES \
            or control_manifest["assignment_policy"] != SHUFFLE_POLICY:
        raise PhaseDProtocolError("unsupported shuffled-side control protocol")
    if not _is_digest(control_manifest["control_digest"]):
        raise PhaseDProtocolError("control_digest is malformed")
    observed_digest = semantic_replay.canonical_json_digest(
        _without_digest(control_manifest, "control_digest"))
    if observed_digest != control_manifest["control_digest"]:
        raise PhaseDProtocolError("control manifest digest does not reproduce")
    if control_manifest["base_corpus_digest"] != corpus_manifest.get("corpus_digest"):
        raise PhaseDProtocolError("control is bound to a different base corpus")
    _strict_int(control_manifest["seed"], "control seed")
    _strict_int(control_manifest["replicate"], "control replicate")
    if control_manifest["problem_count"] != corpus_manifest["problem_count"]:
        raise PhaseDProtocolError("control problem_count differs from base corpus")
    evidence = control_manifest["problems"]
    if not isinstance(evidence, list) or len(evidence) != control_manifest[
            "problem_count"]:
        raise PhaseDProtocolError("control problems do not match problem_count")

    for index, (control_entry, base_entry) in enumerate(
            zip(evidence, corpus_manifest["problems"])):
        if not isinstance(control_entry, Mapping):
            raise PhaseDProtocolError(f"control problems[{index}] must be a mapping")
        _exact_keys(
            control_entry,
            {"opaque_id", "assignment", "controlled_panel_set_digest"},
            f"control problems[{index}]",
        )
        oid = base_entry["opaque_id"]
        if control_entry["opaque_id"] != oid \
                or not _is_digest(control_entry["controlled_panel_set_digest"]):
            raise PhaseDProtocolError(f"control problem identity is invalid at {index}")
        assignments = control_entry["assignment"]
        if not isinstance(assignments, list) or len(assignments) != 12:
            raise PhaseDProtocolError(f"{oid} control must assign twelve panels")
        base_by_slot = {
            (panel["side"], panel["index"]): panel
            for panel in base_entry["panels"]
        }
        sources: list[tuple[str, int]] = []
        targets: list[tuple[str, int]] = []
        target_manifests: list[dict[str, Any]] = []
        for assignment_index, assignment in enumerate(assignments):
            if not isinstance(assignment, Mapping):
                raise PhaseDProtocolError(
                    f"{oid}.assignment[{assignment_index}] must be a mapping")
            _exact_keys(
                assignment,
                {"target_side", "target_index", "source_side", "source_index",
                 "content_digest"},
                f"{oid}.assignment[{assignment_index}]",
            )
            target = (assignment["target_side"], assignment["target_index"])
            source = (assignment["source_side"], assignment["source_index"])
            if target[0] not in {"pos", "neg"} \
                    or source[0] not in {"pos", "neg"} \
                    or isinstance(target[1], bool) or not isinstance(target[1], int) \
                    or isinstance(source[1], bool) or not isinstance(source[1], int) \
                    or not 0 <= target[1] < 6 or not 0 <= source[1] < 6:
                raise PhaseDProtocolError(f"{oid} contains an invalid control slot")
            source_panel = base_by_slot.get(source)
            if source_panel is None \
                    or assignment["content_digest"] != source_panel["content_digest"]:
                raise PhaseDProtocolError(
                    f"{oid} control source digest differs from base panel")
            panel_manifest = dict(source_panel)
            panel_manifest["side"] = target[0]
            panel_manifest["index"] = target[1]
            target_manifests.append(panel_manifest)
            sources.append(source)
            targets.append(target)
        expected_slots = {
            (side, panel_index)
            for side in ("pos", "neg") for panel_index in range(6)
        }
        if set(sources) != expected_slots or set(targets) != expected_slots:
            raise PhaseDProtocolError(
                f"{oid} control must use every source and target slot exactly once")
        for target_side in ("pos", "neg"):
            source_sides = [
                assignment["source_side"] for assignment in assignments
                if assignment["target_side"] == target_side
            ]
            if source_sides.count("pos") != 3 or source_sides.count("neg") != 3:
                raise PhaseDProtocolError(
                    f"{oid} {target_side} side is not balanced three-plus-three")
        target_manifests.sort(key=lambda panel: (
            0 if panel["side"] == "pos" else 1, panel["index"]))
        derived_digest = semantic_replay.canonical_json_digest({
            "schema": semantic_replay.PANEL_SCHEMA,
            "panels": target_manifests,
        })
        if derived_digest != control_entry["controlled_panel_set_digest"]:
            raise PhaseDProtocolError(
                f"{oid} controlled_panel_set_digest does not reproduce")


def assert_shuffled_control_prefix_matches(
        control_manifest: Mapping[str, Any],
        corpus_manifest: Mapping[str, Any],
        controlled_problems: Sequence[Any]) -> None:
    """Reject live controlled panels that are not the frozen control prefix."""
    validate_shuffled_control_manifest(control_manifest, corpus_manifest)
    if not controlled_problems or len(controlled_problems) > control_manifest[
            "problem_count"]:
        raise PhaseDProtocolError("controlled prefix length is invalid")
    for problem, entry in zip(controlled_problems, control_manifest["problems"]):
        observed = _problem_manifest(problem, entry["opaque_id"])["panel_set_digest"]
        if observed != entry["controlled_panel_set_digest"]:
            raise PhaseDProtocolError(
                f"{entry['opaque_id']} controlled panels differ from manifest")


def assert_shuffled_control_matches(
        control_manifest: Mapping[str, Any],
        corpus_manifest: Mapping[str, Any],
        original_problems: Sequence[Any],
        controlled_problems: Sequence[Any]) -> None:
    """Reproduce a shuffled control and reject assignment or panel drift."""
    validate_shuffled_control_manifest(control_manifest, corpus_manifest)
    reproduced = build_shuffled_sides_control(
        original_problems,
        corpus_manifest,
        seed=control_manifest["seed"],
        replicate=control_manifest["replicate"],
    )
    if reproduced.manifest["control_digest"] != control_manifest["control_digest"]:
        raise PhaseDProtocolError("shuffled-side assignments do not reproduce")
    if len(controlled_problems) != len(reproduced.problems):
        raise PhaseDProtocolError("controlled corpus length differs")
    for expected, observed, entry in zip(
            reproduced.problems, controlled_problems, control_manifest["problems"]):
        expected_digest = _problem_manifest(expected, entry["opaque_id"])[
            "panel_set_digest"]
        observed_digest = _problem_manifest(observed, entry["opaque_id"])[
            "panel_set_digest"]
        if expected_digest != observed_digest \
                or observed_digest != entry["controlled_panel_set_digest"]:
            raise PhaseDProtocolError(
                f"{entry['opaque_id']} controlled panels differ from manifest")


def complexity_trace(
        problem_definitions: Sequence[tuple[str, Iterable[str]]],
        definition_costs: Mapping[str, int],
        *,
        sharing_policy: str,
        structure_costs: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Price the same definition uses under shared and no-share policies.

    Definition availability is held fixed.  ``shared`` charges a definition on
    its first corpus use; ``no-share`` charges every used definition in every
    problem.  Per-problem rule/call/binding costs are supplied separately and
    never receive a sharing discount.
    """
    if sharing_policy not in {SHARED, NO_SHARE}:
        raise PhaseDProtocolError(
            f"sharing_policy must be {SHARED!r} or {NO_SHARE!r}")
    costs: dict[str, int] = {}
    for name, value in definition_costs.items():
        key = _strict_nonempty_string(name, "definition name")
        costs[key] = _strict_int(value, f"definition_costs.{key}")
    extras = dict(structure_costs or {})
    seen: set[str] = set()
    seen_oids: set[str] = set()
    records: list[dict[str, Any]] = []
    for opaque_id, definitions in problem_definitions:
        oid = _strict_nonempty_string(opaque_id, "opaque_id")
        if oid in seen_oids:
            raise PhaseDProtocolError(f"duplicate complexity record {oid!r}")
        seen_oids.add(oid)
        used = sorted(set(definitions))
        unknown = sorted(set(used) - set(costs))
        if unknown:
            raise PhaseDProtocolError(
                f"{oid} uses definitions with no registered cost: {unknown}")
        if sharing_policy == SHARED:
            charged = sorted(set(used) - seen)
        else:
            charged = used
        definition_charge = sum(costs[name] for name in charged)
        structure_charge = _strict_int(extras.get(oid, 0), f"structure_costs.{oid}")
        records.append({
            "opaque_id": oid,
            "used_definitions": used,
            "charged_definitions": charged,
            "definition_charge": definition_charge,
            "structure_charge": structure_charge,
            "total_charge": definition_charge + structure_charge,
        })
        seen.update(used)
    body: dict[str, Any] = {
        "schema": COMPLEXITY_TRACE_SCHEMA,
        "sharing_policy": sharing_policy,
        "definition_costs": dict(sorted(costs.items())),
        "records": records,
        "total_definition_charge": sum(
            record["definition_charge"] for record in records),
        "total_structure_charge": sum(
            record["structure_charge"] for record in records),
        "total_charge": sum(record["total_charge"] for record in records),
    }
    body["trace_digest"] = semantic_replay.canonical_json_digest(body)
    return body


def _execution_tag(
        *, track: str, condition: str, scale: int,
        replicate: int | None, control_digest: str, corpus_digest: str,
        execution_policy_digest: str) -> str:
    """Return one bounded artifact tag for an exact execution family.

    Primary and shuffled runs grow one checkpoint through the preregistered
    scale prefixes, so scale is intentionally absent from those identities.
    A no-share artifact is an immutable reprice of one exact prefix, so its
    scale is part of the identity and each scale receives a distinct tag.
    """
    family_scale = scale if condition == NO_SHARE else None
    identity = {
        "schema": _EXECUTION_TAG_SCHEMA,
        "track": track,
        "condition": condition,
        "scale": family_scale,
        "replicate": replicate,
        "control_digest": control_digest,
        "corpus_digest": corpus_digest,
        "execution_policy_digest": execution_policy_digest,
    }
    digest = semantic_replay.canonical_json_digest(identity)[7:39]
    track_label = "u" if track == Track.UNRESTRICTED.value else "sp"
    if condition == "primary":
        family_label = "primary"
    elif condition == SHUFFLED_SIDES:
        family_label = f"shuffle-r{replicate:02d}"
    else:
        family_label = f"no-share-n{scale}"
    tag = f"phase-d-{track_label}-{family_label}-{digest}"
    if len(tag) > 64:
        raise PhaseDProtocolError("derived Phase D execution tag is too long")
    return tag


def _canonical_arm_table(
        tracks: Sequence[str], scales: Sequence[int],
        controls: Sequence[Mapping[str, Any]],
        no_share_tracks: Sequence[str], *, corpus_digest: str,
        execution_policy_digest: str) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for track in tracks:
        for scale in scales:
            arms.append({
                "arm_id": f"{track}:primary:n{scale}",
                "track": track,
                "condition": "primary",
                "label_policy": OBSERVED,
                "sharing_policy": SHARED,
                "scale": scale,
                "replicate": None,
                "control_digest": "",
                "execution_tag": _execution_tag(
                    track=track, condition="primary", scale=scale,
                    replicate=None, control_digest="",
                    corpus_digest=corpus_digest,
                    execution_policy_digest=execution_policy_digest),
            })
            for control in controls:
                replicate = control["replicate"]
                arms.append({
                    "arm_id": f"{track}:shuffled-sides:n{scale}:r{replicate}",
                    "track": track,
                    "condition": SHUFFLED_SIDES,
                    "label_policy": SHUFFLED_SIDES,
                    "sharing_policy": SHARED,
                    "scale": scale,
                    "replicate": replicate,
                    "control_digest": control["control_digest"],
                    "execution_tag": _execution_tag(
                        track=track, condition=SHUFFLED_SIDES, scale=scale,
                        replicate=replicate,
                        control_digest=control["control_digest"],
                        corpus_digest=corpus_digest,
                        execution_policy_digest=execution_policy_digest),
                })
            if track in no_share_tracks:
                arms.append({
                    "arm_id": f"{track}:no-share:n{scale}",
                    "track": track,
                    "condition": NO_SHARE,
                    "label_policy": OBSERVED,
                    "sharing_policy": NO_SHARE,
                    "scale": scale,
                    "replicate": None,
                    "control_digest": "",
                    "execution_tag": _execution_tag(
                        track=track, condition=NO_SHARE, scale=scale,
                        replicate=None, control_digest="",
                        corpus_digest=corpus_digest,
                        execution_policy_digest=execution_policy_digest),
                })
    return arms


def _control_bindings(
        corpus_manifest: Mapping[str, Any], *, seed: int,
        replicate_count: int) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for replicate in range(replicate_count):
        control = build_shuffled_control_manifest(
            corpus_manifest, seed=seed, replicate=replicate)
        bindings.append({
            "replicate": replicate,
            "control_digest": control["control_digest"],
            "panel_set_digests": [
                entry["controlled_panel_set_digest"]
                for entry in control["problems"]
            ],
        })
    return bindings


def build_preregistration(
        corpus_manifest: Mapping[str, Any],
        *,
        tracks: Sequence[str],
        scales: Sequence[int],
        shuffled_seed: int,
        shuffled_replicates: int = 3,
        no_share_tracks: Sequence[str] | None = None) -> dict[str, Any]:
    """Build the complete, track-separated Phase D arm table."""
    validate_corpus_manifest(corpus_manifest)
    selected_tracks = tuple(tracks)
    if not selected_tracks \
            or any(not isinstance(track, str) or not track
                   for track in selected_tracks) \
            or len(set(selected_tracks)) != len(selected_tracks):
        raise PhaseDProtocolError("tracks must be nonempty and unique")
    unknown_tracks = sorted(set(selected_tracks) - set(TRACK_VALUES))
    if unknown_tracks:
        raise PhaseDProtocolError(
            f"tracks have no implemented Phase D runner: {unknown_tracks}")
    selected_scales = tuple(
        _strict_int(scale, "scale", minimum=1) for scale in scales)
    if not selected_scales or tuple(sorted(set(selected_scales))) != selected_scales:
        raise PhaseDProtocolError("scales must be unique and strictly increasing")
    if selected_scales[-1] > corpus_manifest["problem_count"]:
        raise PhaseDProtocolError("largest scale exceeds the frozen corpus")
    control_seed = _strict_int(shuffled_seed, "shuffled_seed")
    replicate_count = _strict_int(
        shuffled_replicates, "shuffled_replicates", minimum=1)
    expected_no_share = tuple(
        track for track in selected_tracks if track == Track.UNRESTRICTED.value)
    no_share = expected_no_share if no_share_tracks is None \
        else tuple(no_share_tracks)
    if no_share != expected_no_share:
        raise PhaseDProtocolError(
            "no-share is defined only for the learned unrestricted library "
            "and is required when that track is selected")

    controls = _control_bindings(
        corpus_manifest, seed=control_seed, replicate_count=replicate_count)
    execution_policy = canonical_execution_policy(
        require_unrestricted_cli=Track.UNRESTRICTED.value in selected_tracks)
    arms = _canonical_arm_table(
        selected_tracks, selected_scales, controls, no_share,
        corpus_digest=corpus_manifest["corpus_digest"],
        execution_policy_digest=execution_policy["policy_digest"])
    body: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "corpus_digest": corpus_manifest["corpus_digest"],
        "corpus_problem_count": corpus_manifest["problem_count"],
        "corpus_panel_set_digests": [
            entry["panel_set_digest"] for entry in corpus_manifest["problems"]],
        "tracks": list(selected_tracks),
        "scales": list(selected_scales),
        "scale_policy": SCALE_POLICY,
        "shuffled_sides": {
            "seed": control_seed,
            "replicates": list(range(replicate_count)),
            "controls": controls,
            "assignment_policy": SHUFFLE_POLICY,
            "discard_failed_controls": False,
        },
        "no_share": {
            "tracks": list(no_share),
            "definition_availability": NO_SHARE_DEFINITION_AVAILABILITY,
            "definition_charge": NO_SHARE_DEFINITION_CHARGE,
            "selection_policy": NO_SHARE_SELECTION_POLICY,
            "causal_claim": NO_SHARE_CAUSAL_CLAIM,
        },
        "execution_policy": execution_policy,
        "aggregation_policy": AGGREGATION_POLICY,
        "arms": arms,
    }
    body["preregistration_digest"] = semantic_replay.canonical_json_digest(body)
    validate_preregistration(body, corpus_manifest=corpus_manifest)
    return body


def validate_preregistration(
        value: Mapping[str, Any], *,
        corpus_manifest: Mapping[str, Any] | None = None) -> None:
    """Validate and reconstruct the closed Phase D execution matrix."""
    if not isinstance(value, Mapping) or value.get("schema") != PREREGISTRATION_SCHEMA:
        raise PhaseDProtocolError("unsupported Phase D preregistration")
    _exact_keys(
        value,
        {
            "schema", "corpus_digest", "corpus_problem_count",
            "corpus_panel_set_digests", "tracks", "scales", "scale_policy",
            "shuffled_sides", "no_share", "execution_policy",
            "aggregation_policy", "arms", "preregistration_digest",
        },
        "preregistration",
    )
    digest = value["preregistration_digest"]
    if not _is_digest(digest):
        raise PhaseDProtocolError("preregistration_digest is malformed")
    observed = semantic_replay.canonical_json_digest(
        _without_digest(value, "preregistration_digest"))
    if observed != digest:
        raise PhaseDProtocolError("preregistration digest does not reproduce")
    if not _is_digest(value["corpus_digest"]):
        raise PhaseDProtocolError("preregistration corpus_digest is malformed")
    corpus_count = _strict_int(
        value["corpus_problem_count"], "corpus_problem_count", minimum=1)
    base_panels = value["corpus_panel_set_digests"]
    if not isinstance(base_panels, list) or len(base_panels) != corpus_count \
            or any(not _is_digest(item) for item in base_panels):
        raise PhaseDProtocolError("corpus panel digest table is invalid")

    tracks = value["tracks"]
    if not isinstance(tracks, list) or not tracks \
            or any(not isinstance(track, str) or not track for track in tracks) \
            or len(set(tracks)) != len(tracks):
        raise PhaseDProtocolError("preregistered tracks are invalid")
    unknown_tracks = sorted(set(tracks) - set(TRACK_VALUES))
    if unknown_tracks:
        raise PhaseDProtocolError(
            f"tracks have no implemented Phase D runner: {unknown_tracks}")
    scales_value = value["scales"]
    if not isinstance(scales_value, list):
        raise PhaseDProtocolError("preregistered scales must be a list")
    scales = tuple(
        _strict_int(scale, f"scales[{index}]", minimum=1)
        for index, scale in enumerate(scales_value))
    if not scales or tuple(sorted(set(scales))) != scales:
        raise PhaseDProtocolError("scales must be unique and strictly increasing")
    if scales[-1] > corpus_count:
        raise PhaseDProtocolError("largest scale exceeds the frozen corpus")
    if value["scale_policy"] != SCALE_POLICY:
        raise PhaseDProtocolError("preregistration scale policy is not fixed")
    if value["aggregation_policy"] != AGGREGATION_POLICY:
        raise PhaseDProtocolError("preregistration aggregation policy is not fixed")
    if value["execution_policy"] != canonical_execution_policy(
            require_unrestricted_cli=Track.UNRESTRICTED.value in tracks):
        raise PhaseDProtocolError(
            "preregistration execution policy or harness fingerprint differs")

    shuffled = value["shuffled_sides"]
    if not isinstance(shuffled, Mapping):
        raise PhaseDProtocolError("shuffled_sides must be a mapping")
    _exact_keys(
        shuffled,
        {"seed", "replicates", "controls", "assignment_policy",
         "discard_failed_controls"},
        "shuffled_sides",
    )
    _strict_int(shuffled["seed"], "shuffled_sides.seed")
    if shuffled["assignment_policy"] != SHUFFLE_POLICY \
            or shuffled["discard_failed_controls"] is not False:
        raise PhaseDProtocolError("shuffled-side policy is not fixed")
    replicates_value = shuffled["replicates"]
    controls_value = shuffled["controls"]
    if not isinstance(replicates_value, list) or not replicates_value \
            or not isinstance(controls_value, list):
        raise PhaseDProtocolError("shuffled-side replicate table is invalid")
    replicates = tuple(
        _strict_int(item, f"shuffled_sides.replicates[{index}]")
        for index, item in enumerate(replicates_value))
    if replicates != tuple(range(len(replicates))):
        raise PhaseDProtocolError(
            "shuffled-side replicates must be contiguous from zero")
    if len(controls_value) != len(replicates):
        raise PhaseDProtocolError("control table does not match replicates")
    controls: list[Mapping[str, Any]] = []
    for index, control in enumerate(controls_value):
        if not isinstance(control, Mapping):
            raise PhaseDProtocolError(f"controls[{index}] must be a mapping")
        _exact_keys(
            control,
            {"replicate", "control_digest", "panel_set_digests"},
            f"controls[{index}]",
        )
        if control["replicate"] != replicates[index] \
                or not _is_digest(control["control_digest"]):
            raise PhaseDProtocolError("control identity does not match replicate")
        panel_digests = control["panel_set_digests"]
        if not isinstance(panel_digests, list) \
                or len(panel_digests) != corpus_count \
                or any(not _is_digest(item) for item in panel_digests):
            raise PhaseDProtocolError("control panel digest table is invalid")
        controls.append(control)

    no_share = value["no_share"]
    if not isinstance(no_share, Mapping):
        raise PhaseDProtocolError("no_share must be a mapping")
    _exact_keys(
        no_share,
        {"tracks", "definition_availability", "definition_charge",
         "selection_policy", "causal_claim"},
        "no_share",
    )
    expected_no_share = [
        track for track in tracks if track == Track.UNRESTRICTED.value]
    if no_share["tracks"] != expected_no_share:
        raise PhaseDProtocolError(
            "no-share must apply exactly to the unrestricted track")
    expected_no_share_policy = {
        "definition_availability": NO_SHARE_DEFINITION_AVAILABILITY,
        "definition_charge": NO_SHARE_DEFINITION_CHARGE,
        "selection_policy": NO_SHARE_SELECTION_POLICY,
        "causal_claim": NO_SHARE_CAUSAL_CLAIM,
    }
    for name, expected in expected_no_share_policy.items():
        if no_share[name] != expected:
            raise PhaseDProtocolError(f"no-share {name} policy is not fixed")

    arms = value["arms"]
    if not isinstance(arms, list) or not arms:
        raise PhaseDProtocolError("preregistration contains no arms")
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise PhaseDProtocolError(f"arms[{index}] must be a mapping")
        _exact_keys(arm, _ARM_KEYS, f"arms[{index}]")
    expected_arms = _canonical_arm_table(
        tracks, scales, controls, expected_no_share,
        corpus_digest=value["corpus_digest"],
        execution_policy_digest=value["execution_policy"]["policy_digest"])
    if arms != expected_arms:
        raise PhaseDProtocolError(
            "preregistration arms are not the exact Cartesian execution table")

    if corpus_manifest is not None:
        validate_corpus_manifest(corpus_manifest)
        expected_base_panels = [
            entry["panel_set_digest"] for entry in corpus_manifest["problems"]]
        if value["corpus_digest"] != corpus_manifest["corpus_digest"] \
                or corpus_count != corpus_manifest["problem_count"] \
                or base_panels != expected_base_panels:
            raise PhaseDProtocolError(
                "preregistration differs from its frozen corpus")
        expected_controls = _control_bindings(
            corpus_manifest,
            seed=shuffled["seed"],
            replicate_count=len(replicates),
        )
        if controls_value != expected_controls:
            raise PhaseDProtocolError(
                "preregistration control table does not reproduce")


def build_track_report(
        preregistration: Mapping[str, Any],
        *,
        arm_id: str,
        records: Sequence[Mapping[str, Any]],
        report_source_trace_digest: str | None = None,
        parent_source_trace_digest: str = "") -> dict[str, Any]:
    """Build one exact preregistered arm report from terminal records."""
    validate_preregistration(preregistration)
    matching = [
        arm for arm in preregistration["arms"]
        if arm["arm_id"] == arm_id
    ]
    if len(matching) != 1:
        raise PhaseDProtocolError("track report arm was not preregistered")
    arm = matching[0]
    if any(not isinstance(record, Mapping) for record in records):
        raise PhaseDProtocolError("track report records must be mappings")
    copied_records: list[dict[str, Any]] = []
    record_identity = {
        "track": arm["track"],
        "condition": arm["condition"],
        "label_policy": arm["label_policy"],
        "sharing_policy": arm["sharing_policy"],
        "corpus_digest": preregistration["corpus_digest"],
        "control_digest": arm["control_digest"],
    }
    panel_digests = _panel_digest_table_for_arm(preregistration, arm)
    record_source_traces: set[str] = set()
    record_parent_traces: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PhaseDProtocolError(f"records[{index}] must be a mapping")
        copied = dict(record)
        existing_source_trace = copied.pop("report_source_trace_digest", "")
        existing_parent_trace = copied.pop("parent_source_trace_digest", "")
        if existing_source_trace:
            record_source_traces.add(existing_source_trace)
        if existing_parent_trace:
            record_parent_traces.add(existing_parent_trace)
        for name, expected in record_identity.items():
            if name in copied and copied[name] != expected:
                raise PhaseDProtocolError(
                    f"records[{index}].{name} differs from its arm")
            copied[name] = expected
        expected_panel_digest = panel_digests[index] \
            if index < len(panel_digests) else None
        if "panel_set_digest" in copied \
                and copied["panel_set_digest"] != expected_panel_digest:
            raise PhaseDProtocolError(
                f"records[{index}].panel_set_digest differs from its arm")
        copied["panel_set_digest"] = expected_panel_digest
        copied_records.append(copied)
    reproduced_trace = _report_source_trace_digest(
        arm["track"], copied_records)
    trace_digest = report_source_trace_digest or reproduced_trace
    if trace_digest != reproduced_trace:
        raise PhaseDProtocolError(
            "report source trace does not reproduce from its records")
    if record_source_traces - {trace_digest}:
        raise PhaseDProtocolError(
            "record source trace differs from the report source trace")
    if not _is_digest(trace_digest):
        raise PhaseDProtocolError("track report source trace digest is malformed")
    if arm["condition"] == NO_SHARE:
        if not _is_digest(parent_source_trace_digest):
            raise PhaseDProtocolError(
                "no-share report requires its primary source trace digest")
        if record_parent_traces - {parent_source_trace_digest}:
            raise PhaseDProtocolError(
                "record parent trace differs from the report parent trace")
    elif parent_source_trace_digest or record_parent_traces:
        raise PhaseDProtocolError(
            "only no-share reports may bind a parent source trace")
    for copied in copied_records:
        copied["report_source_trace_digest"] = trace_digest
        copied["parent_source_trace_digest"] = parent_source_trace_digest
    report = {
        "schema": TRACK_REPORT_SCHEMA,
        "preregistration_digest": preregistration["preregistration_digest"],
        "corpus_digest": preregistration["corpus_digest"],
        "arm_id": arm["arm_id"],
        "execution_tag": arm["execution_tag"],
        "track": arm["track"],
        "condition": arm["condition"],
        "label_policy": arm["label_policy"],
        "sharing_policy": arm["sharing_policy"],
        "scale": arm["scale"],
        "replicate": arm["replicate"],
        "control_digest": arm["control_digest"],
        "report_source_trace_digest": trace_digest,
        "parent_source_trace_digest": parent_source_trace_digest,
        "records": copied_records,
        "solved": sum(
            int(record.get("solved") is True) for record in copied_records),
        "attempted": len(copied_records),
    }
    validate_track_report(
        report, preregistration, _preregistration_validated=True)
    return report


def _panel_digest_table_for_arm(
        preregistration: Mapping[str, Any],
        arm: Mapping[str, Any]) -> Sequence[str]:
    if arm["condition"] != SHUFFLED_SIDES:
        return preregistration["corpus_panel_set_digests"]
    matching = [
        control for control in preregistration["shuffled_sides"]["controls"]
        if control["replicate"] == arm["replicate"]
    ]
    if len(matching) != 1 \
            or matching[0]["control_digest"] != arm["control_digest"]:
        raise PhaseDProtocolError(
            "shuffled arm is not bound to one prepared control")
    return matching[0]["panel_set_digests"]


def _unstamp_track_report_records(
        track: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reverse only fields added by the Phase D report publisher.

    The resulting mappings are the exact runner records that must occur in
    the originating checkpoint.  This is intentionally shared by source-trace
    reproduction and the campaign collector's artifact certification.
    """
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise PhaseDProtocolError("track report record must be a mapping")
        value = dict(record)
        value.pop("report_source_trace_digest", None)
        value.pop("parent_source_trace_digest", None)
        if track == Track.SEMANTIC_PURE.value:
            # The semantic runner's ProblemResult has no label_policy field;
            # this is a report-only arm stamp added by its publisher.
            value.pop("label_policy", None)
        runner_condition = value.pop("runner_condition", None)
        if runner_condition is not None:
            value["condition"] = runner_condition
        elif value.get("condition") == "primary":
            value["condition"] = OBSERVED
        normalized.append(value)
    return normalized


def runner_records_from_track_report(
        report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exact pre-publication records from a validated report shape."""
    if not isinstance(report, Mapping):
        raise PhaseDProtocolError("track report must be a mapping")
    track = report.get("track")
    if track not in TRACK_VALUES:
        raise PhaseDProtocolError("track report has no implemented track")
    records = report.get("records")
    if not isinstance(records, list):
        raise PhaseDProtocolError("track report records must be a list")
    return _unstamp_track_report_records(track, records)


def _report_source_trace_digest(
        track: str, records: Sequence[Mapping[str, Any]]) -> str:
    """Reproduce the runner trace from stamped track-report records."""
    normalized = _unstamp_track_report_records(track, records)
    if track == Track.UNRESTRICTED.value:
        encoded = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
    return semantic_replay.canonical_json_digest(normalized)


def _validate_unrestricted_proposer_receipts(
        record: Mapping[str, Any], opaque_id: str,
        execution_policy: Mapping[str, Any], *,
        seen_threads: set[str] | None = None,
        seen_event_streams: set[str] | None = None) -> None:
    policy = execution_policy["unrestricted"]
    expected_ladder = policy["proposer_ladder"]
    runtime = execution_policy["runtime"]["codex_cli"]
    receipts = record.get("proposer_receipts")
    attempts = record.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) \
            or attempts < 1 or not isinstance(receipts, list) \
            or len(receipts) != attempts:
        raise PhaseDProtocolError(
            f"{opaque_id} proposer receipt count differs from attempts")
    feedback = record.get("proposer_feedback")
    baseline_source = record.get("baseline_source_digest")
    attempted_source = record.get("attempted_source_digest")
    baseline_log = record.get("baseline_log_digest")
    attempted_log = record.get("attempted_log_digest")
    proposer_panels = record.get("proposer_panel_set_digest")
    if not isinstance(feedback, list) or len(feedback) != attempts \
            or not feedback or feedback[0] != "" \
            or any(not isinstance(item, str) for item in feedback) \
            or not all(_is_raw_digest(item) for item in (
                baseline_source, attempted_source, baseline_log, attempted_log)) \
            or proposer_panels != record.get("panel_set_digest"):
        raise PhaseDProtocolError(
            f"{opaque_id} proposer causal record binding is invalid")
    threads = seen_threads if seen_threads is not None else set()
    streams = seen_event_streams if seen_event_streams is not None else set()
    expected_source = baseline_source
    expected_log = baseline_log
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise PhaseDProtocolError(
                f"{opaque_id} has malformed proposer receipt fields")
        try:
            codex_proposer.validate_codex_receipt(dict(receipt))
        except codex_proposer.CodexProposerFailure as exc:
            raise PhaseDProtocolError(
                f"{opaque_id} Codex proposer receipt is invalid: {exc}") \
                from exc
        if receipt["schema"] != PROPOSER_RECEIPT_SCHEMA \
                or receipt["requested_reasoning_effort"] != \
                policy["requested_reasoning_effort"] \
                or receipt["input_digest_schema"] != \
                policy["proposer_input_digest_schema"] \
                or receipt["output_schema_digest"] != \
                policy["proposer_output_schema_digest"] \
                or receipt["isolation_policy"] != \
                policy["proposer_tool_surface"] \
                or receipt["codex_cli_version"] != runtime["version"] \
                or receipt["codex_launcher_digest"] != \
                runtime["launcher_digest"]:
            raise PhaseDProtocolError(
                f"{opaque_id} Codex receipt differs from execution policy")
        if receipt["current_source_digest"] != expected_source \
                or receipt["current_log_digest"] != expected_log \
                or receipt["panel_set_digest"] != proposer_panels:
            raise PhaseDProtocolError(
                f"{opaque_id} Codex receipt input chain differs")
        if receipt["thread_id"] in threads \
                or receipt["event_stream_digest"] in streams:
            raise PhaseDProtocolError(
                f"{opaque_id} reuses Codex turn identity evidence")
        threads.add(receipt["thread_id"])
        streams.add(receipt["event_stream_digest"])
        expected_source = receipt["proposed_source_digest"]
        expected_log = receipt["proposed_log_digest"]
    if expected_source != attempted_source or expected_log != attempted_log:
        raise PhaseDProtocolError(
            f"{opaque_id} Codex final output chain differs")
    if receipts[-1]["requested_model"] != record.get("model"):
        raise PhaseDProtocolError(
            f"{opaque_id} final proposer receipt model differs")
    if attempts > len(expected_ladder) or [
            receipt["requested_model"] for receipt in receipts
            ] != list(expected_ladder[:attempts]):
        raise PhaseDProtocolError(
            f"{opaque_id} proposer receipts differ from the preregistered ladder")


def validate_semantic_proposer_receipts(
        record: Mapping[str, Any], opaque_id: str,
        policy: Mapping[str, Any]) -> None:
    """Require durable exact Anthropic model/usage evidence for every round."""
    terminal = record.get("terminal_evidence")
    rounds = terminal.get("rounds") if isinstance(terminal, Mapping) else None
    if not isinstance(rounds, list) or not rounds \
            or len(rounds) > policy.get("rounds", 0):
        raise PhaseDProtocolError(
            f"{opaque_id} lacks semantic proposer round evidence")
    expected_keys = {
        "schema", "source", "requested_model", "actual_model",
        "input_tokens", "output_tokens", "stop_reason", "receipt_digest",
    }
    for round_index, round_record in enumerate(rounds):
        if not isinstance(round_record, Mapping) \
                or round_record.get("round") != round_index \
                or round_record.get("proposer_kind") != "anthropic":
            raise PhaseDProtocolError(
                f"{opaque_id} semantic proposer kind/round differs")
        receipts = round_record.get("model_receipts")
        if not isinstance(receipts, list) or not receipts \
                or len(receipts) > policy.get(
                    "max_model_attempts_per_round", 0):
            raise PhaseDProtocolError(
                f"{opaque_id} semantic round has an impossible receipt count")
        for receipt in receipts:
            if not isinstance(receipt, Mapping) \
                    or set(receipt) != expected_keys \
                    or receipt["schema"] != SEMANTIC_PROPOSER_RECEIPT_SCHEMA \
                    or receipt["source"] != "anthropic-messages-api" \
                    or receipt["requested_model"] != policy["concrete_model"] \
                    or receipt["actual_model"] != policy["concrete_model"]:
                raise PhaseDProtocolError(
                    f"{opaque_id} semantic model identity differs")
            input_tokens = receipt["input_tokens"]
            output_tokens = receipt["output_tokens"]
            if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) \
                    or isinstance(output_tokens, bool) \
                    or not isinstance(output_tokens, int) \
                    or input_tokens < 0 or output_tokens < 0 \
                    or input_tokens + output_tokens <= 0 \
                    or not isinstance(receipt["stop_reason"], str) \
                    or not receipt["stop_reason"]:
                raise PhaseDProtocolError(
                    f"{opaque_id} semantic token usage is malformed")
            reproduced = semantic_replay.canonical_json_digest(
                _without_digest(receipt, "receipt_digest"))
            if receipt["receipt_digest"] != reproduced:
                raise PhaseDProtocolError(
                    f"{opaque_id} semantic receipt digest does not reproduce")


def validate_track_report(
        report: Mapping[str, Any], preregistration: Mapping[str, Any], *,
        _preregistration_validated: bool = False) -> None:
    """Validate a one-track, one-condition, one-scale result document."""
    if not _preregistration_validated:
        validate_preregistration(preregistration)
    if not isinstance(report, Mapping):
        raise PhaseDProtocolError("track report must be a mapping")
    required = {
        "schema", "preregistration_digest", "corpus_digest", "arm_id",
        "execution_tag",
        "track", "condition", "label_policy", "sharing_policy", "scale",
        "replicate", "control_digest", "report_source_trace_digest",
        "parent_source_trace_digest", "records", "solved", "attempted",
    }
    _exact_keys(report, required, "track report")
    if report["schema"] != TRACK_REPORT_SCHEMA:
        raise PhaseDProtocolError("unsupported track report schema")
    if report["preregistration_digest"] != preregistration["preregistration_digest"] \
            or report["corpus_digest"] != preregistration["corpus_digest"]:
        raise PhaseDProtocolError("track report is bound to a different protocol/corpus")
    matching = [
        arm for arm in preregistration["arms"]
        if arm["arm_id"] == report["arm_id"]
    ]
    if len(matching) != 1:
        raise PhaseDProtocolError("track report arm was not preregistered")
    arm = matching[0]
    for name in (
            "execution_tag", "track", "condition", "label_policy",
            "sharing_policy", "scale",
            "replicate", "control_digest"):
        if report[name] != arm[name]:
            raise PhaseDProtocolError(f"track report {name} differs from its arm")
    records = report["records"]
    scale = _strict_int(report["scale"], "report.scale", minimum=1)
    if not isinstance(records, list) or len(records) != scale:
        raise PhaseDProtocolError("track report must contain exactly scale records")
    expected_ids = tuple(f"problem_{index:02d}" for index in range(scale))
    observed_ids: list[str] = []
    solved = 0
    source_traces: set[str] = set()
    parent_traces: set[str] = set()
    panel_digests = _panel_digest_table_for_arm(preregistration, arm)
    execution_bindings = execution_binding_family(preregistration, arm)
    seen_codex_threads: set[str] = set()
    seen_codex_event_streams: set[str] = set()
    current_unrestricted_log_digest = hashlib.sha256(b"").hexdigest()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PhaseDProtocolError(f"records[{index}] must be a mapping")
        oid = record.get("opaque_id")
        if oid != expected_ids[index]:
            raise PhaseDProtocolError("track report records are not the frozen prefix")
        if not isinstance(record.get("solved"), bool):
            raise PhaseDProtocolError(f"{oid}.solved must be boolean")
        status = _strict_nonempty_string(record.get("status"), f"{oid}.status")
        terminal_statuses = _TERMINAL_STATUSES[report["track"]]
        if status not in terminal_statuses:
            raise PhaseDProtocolError(
                f"{oid}.status is not a terminal {report['track']} status")
        if record["solved"] != terminal_statuses[status]:
            raise PhaseDProtocolError(
                f"{oid}.solved contradicts its terminal status")
        if report["track"] == Track.UNRESTRICTED.value:
            failure_status = "VERIFIER_FAILURE_UNRESTRICTED"
            failure_rule = "PRICING_OR_LOAD_ERROR"
            if status == failure_status:
                canonical_failure = {
                    "solved": False,
                    "heldout_accuracy": 0.0,
                    "train_accuracy": 0.0,
                    "rule": failure_rule,
                    "rule_cost": 0.0,
                    "marginal_C": 0,
                    "accepted_source_digest": "",
                    "accepted_source": "",
                    "predicate_names": [],
                    "rule_atoms": [],
                    "fold_rule_atoms": [],
                    "used_definition_nodes": [],
                    "charged_definition_node_identities": [],
                    "reused_definition_node_identities": [],
                    "full_definition_cost": 0,
                    "definition_charge": 0,
                    "structure_charge": 0.0,
                    "total_charge": 0.0,
                    "predicate_errors": 12,
                    "n_rotations": 36,
                }
                for name, expected in canonical_failure.items():
                    if record.get(name) != expected:
                        raise PhaseDProtocolError(
                            f"{oid}.{name} is not canonical verifier-failure "
                            "evidence")
            elif record.get("rule") == failure_rule:
                raise PhaseDProtocolError(
                    f"{oid}.status does not identify its verifier failure")
        for name in (
                "track", "condition", "label_policy", "sharing_policy"):
            if record.get(name) != report[name]:
                raise PhaseDProtocolError(
                    f"{oid} mixes a different {name} into the report")
        if record.get("corpus_digest") != report["corpus_digest"]:
            raise PhaseDProtocolError(
                f"{oid} is bound to a different report corpus")
        if record.get("control_digest") != report["control_digest"]:
            raise PhaseDProtocolError(
                f"{oid} is bound to a different report control")
        if record.get("panel_set_digest") != panel_digests[index]:
            raise PhaseDProtocolError(
                f"{oid} is bound to a different panel set")
        expected_execution_binding = next(
            (binding for binding in execution_bindings
             if index < binding["scale"]),
            None,
        )
        if expected_execution_binding is None \
                or record.get("phase_execution_binding_digest") != \
                expected_execution_binding["binding_digest"]:
            raise PhaseDProtocolError(
                f"{oid} has incorrect preregistered execution tranche")
        if report["track"] == Track.UNRESTRICTED.value:
            if record.get("baseline_log_digest") != \
                    current_unrestricted_log_digest:
                raise PhaseDProtocolError(
                    f"{oid} proposer log trace is not sequential")
            _validate_unrestricted_proposer_receipts(
                record, oid, preregistration["execution_policy"],
                seen_threads=seen_codex_threads,
                seen_event_streams=seen_codex_event_streams)
            current_unrestricted_log_digest = record["attempted_log_digest"]
        else:
            validate_semantic_proposer_receipts(
                record, oid,
                preregistration["execution_policy"]["semantic_pure"])
        source_trace = record.get("report_source_trace_digest")
        parent_trace = record.get("parent_source_trace_digest")
        if not _is_digest(source_trace) or not isinstance(parent_trace, str):
            raise PhaseDProtocolError(f"{oid} has malformed source-trace evidence")
        source_traces.add(source_trace)
        parent_traces.add(parent_trace)
        solved += int(record["solved"])
        observed_ids.append(oid)
    if tuple(observed_ids) != expected_ids:
        raise PhaseDProtocolError("track report problem IDs are incomplete or reordered")
    if report["attempted"] != scale or report["solved"] != solved:
        raise PhaseDProtocolError("track report aggregate counts do not reproduce")
    if len(source_traces) != 1 or len(parent_traces) != 1:
        raise PhaseDProtocolError("track report source-trace evidence is inconsistent")
    if not _is_digest(report["report_source_trace_digest"]) \
            or not isinstance(report["parent_source_trace_digest"], str):
        raise PhaseDProtocolError("track report source trace is malformed")
    if source_traces != {report["report_source_trace_digest"]} \
            or parent_traces != {report["parent_source_trace_digest"]}:
        raise PhaseDProtocolError(
            "track report top-level source trace does not match its records")
    reproduced_trace = _report_source_trace_digest(
        report["track"], records)
    if report["report_source_trace_digest"] != reproduced_trace:
        raise PhaseDProtocolError(
            "track report source trace does not reproduce from its records")
    parent_trace = next(iter(parent_traces))
    if report["condition"] == NO_SHARE:
        if not _is_digest(parent_trace):
            raise PhaseDProtocolError(
                "no-share report lacks a primary source-trace binding")
    elif parent_trace:
        raise PhaseDProtocolError(
            "non-no-share report unexpectedly carries a parent source trace")


def validate_report_collection(
        reports: Sequence[Mapping[str, Any]],
        preregistration: Mapping[str, Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Validate reports while preserving a hard track boundary in the result."""
    validate_preregistration(preregistration)
    by_track: dict[str, list[Mapping[str, Any]]] = {
        track: [] for track in preregistration["tracks"]}
    seen_arms: set[str] = set()
    for report in reports:
        validate_track_report(
            report, preregistration, _preregistration_validated=True)
        arm_id = report["arm_id"]
        if arm_id in seen_arms:
            raise PhaseDProtocolError(f"duplicate track report for arm {arm_id!r}")
        seen_arms.add(arm_id)
        by_track[report["track"]].append(report)
    return {track: tuple(items) for track, items in by_track.items()}


def validate_complete_report_collection(
        reports: Sequence[Mapping[str, Any]],
        preregistration: Mapping[str, Any]) \
        -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Finalize all arms and prove scale prefixes and paired controls agree."""
    by_track = validate_report_collection(reports, preregistration)
    by_arm = {report["arm_id"]: report for report in reports}
    expected = {arm["arm_id"] for arm in preregistration["arms"]}
    missing = sorted(expected - set(by_arm))
    extra = sorted(set(by_arm) - expected)
    if missing or extra:
        raise PhaseDProtocolError(
            f"report collection is incomplete (missing={missing}, extra={extra})")

    families: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for report in reports:
        key = (
            report["track"], report["condition"], report["label_policy"],
            report["sharing_policy"], report["replicate"],
        )
        families.setdefault(key, []).append(report)
    for family, members in families.items():
        ordered = sorted(members, key=lambda report: report["scale"])
        for smaller, larger in zip(ordered, ordered[1:]):
            def prefix_identity(record: Mapping[str, Any]) -> dict[str, Any]:
                value = dict(record)
                value.pop("report_source_trace_digest", None)
                value.pop("parent_source_trace_digest", None)
                # Every report already validates its exact execution tranche.
                # No-share scales are distinct derived artifacts, so their
                # record provenance digest legitimately differs by scale.
                value.pop("phase_execution_binding_digest", None)
                return value

            prefix = [
                prefix_identity(record)
                for record in larger["records"][:smaller["scale"]]
            ]
            smaller_records = [
                prefix_identity(record) for record in smaller["records"]]
            if semantic_replay.canonical_json_digest(prefix) != \
                    semantic_replay.canonical_json_digest(smaller_records):
                raise PhaseDProtocolError(
                    f"Phase D scale reports are not nested prefixes for {family}")

    # The no-share accounting control must inherit the exact primary outcomes,
    # rules, and source-verification evidence at the same scale.
    for report in reports:
        if report["condition"] != NO_SHARE:
            continue
        primary_id = f"{report['track']}:primary:n{report['scale']}"
        primary = by_arm.get(primary_id)
        if primary is None:
            raise PhaseDProtocolError("no-share report lacks its primary source arm")
        primary_traces = {
            record["report_source_trace_digest"]
            for record in primary["records"]
        }
        parent_traces = {
            record["parent_source_trace_digest"]
            for record in report["records"]
        }
        if len(primary_traces) != 1 or parent_traces != primary_traces:
            raise PhaseDProtocolError(
                "no-share report is not bound to its primary source trace")
        for no_share_record, primary_record in zip(
                report["records"], primary["records"]):
            def held_fixed_fields(record: Mapping[str, Any]) -> dict[str, Any]:
                scientific = {
                    name: value for name, value in record.items()
                    if name not in NO_SHARE_ACCOUNTING_DIFFERENCE_FIELDS
                }
                nodes = scientific.get("used_definition_nodes")
                if isinstance(nodes, list):
                    scientific["used_definition_nodes"] = [
                        {name: value for name, value in node.items()
                         if name != "charged"}
                        if isinstance(node, Mapping) else node
                        for node in nodes
                    ]
                return scientific

            no_share_scientific = held_fixed_fields(no_share_record)
            primary_scientific = held_fixed_fields(primary_record)
            if no_share_scientific != primary_scientific:
                names = sorted(
                    set(no_share_scientific) | set(primary_scientific))
                changed = next(
                    name for name in names
                    if no_share_scientific.get(name) !=
                    primary_scientific.get(name))
                raise PhaseDProtocolError(
                    f"no-share record changes primary {changed}")
    return by_track
