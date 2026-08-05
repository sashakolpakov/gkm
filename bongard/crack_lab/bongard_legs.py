"""Enforced predicate-library orchestration for the Bongard crack.

Sibling of `arc/crack_lab/gkm_legs.py` -- the idioms are reused (LOC+literal
description-length proxy, marginal-C accounting, validated-checkpoint
promotion gating, WIP snapshots, workspace taint markers), the code is fresh
because the ARC types (levels, paths, replay) do not apply: here VERIFY is a
pure function of (predicates source, panels), so re-running the verifier IS
the replay validation.

The discipline, enforced structurally (not requested in a prompt):

  * Logic can only accumulate in the SHARED library `predicates.py`
    (module-level `p_*(panel) -> float|bool` callables). The harness does the
    rule composition itself (exhaustive MDL conjunction search, bongard_arena).
  * Per problem k: PROPOSE (extend predicates.py, minimal new structure) ->
    VERIFY (rotated-LOO on the real panels) -> DEBRIEF (refactor repeats,
    log in predicates_log.md).
  * Admission is structural: predicates.py growth is kept ONLY when the
    problem verifies as solved; a failed attempt's library edits are reverted
    (saved as WIP context, never admitted). F = R + lambda*C_marginal with
    C_marginal = admitted growth of the library; a reused predicate is free.
  * Proposer ladder: repeated, independently receipted Codex attempts; every
    retry is logged alongside marginal C.

The proposer and verifier are injectable (`propose_fn`) so the control loop
and accounting are unit-testable offline; the default proposer invokes
headless Codex in an ephemeral, tool-free, read-only image view and applies
only its schema-validated structured response. Concept names (ground truth) are
never written into the workspace; they live only in the harness-side
results.json in the artifact directory.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_arena as A
import codex_proposer as codex_headless
import phase_d_protocol as P
import predicate_pricing as predicate_price
import semantic_artifacts as artifact_io
import semantic_replay

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_FILE = "predicates.py"
LOG_FILE = "predicates_log.md"
CHECKPOINT_FILE = "checkpoint.json"
PENDING_CHECKPOINT_FILE = "pending_checkpoint.json"
PENDING_PROMOTION_SCHEMA = "bongard.unrestricted-pending-promotion/v1"
PRICING_CONTRACT_FILE = "pricing_contract.json"
PROMOTED_FILES = (LIBRARY_FILE, LOG_FILE, CHECKPOINT_FILE)
WORKSPACE_CONTROL_FILES = (
    LIBRARY_FILE, LOG_FILE, CHECKPOINT_FILE, "bongard_try.py",
    "current_problem.txt", PRICING_CONTRACT_FILE,
)
REPORT_SCHEMA = "bongard.unrestricted-report/v8"
PRICING_CONTRACT_SCHEMA = "bongard.predicate-pricing-context/v3"
VERIFIER_FINGERPRINT_SCHEMA = "bongard.unrestricted-verifier/v3"
AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS = \
    A.AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS
MAX_PROPOSER_LOG_UTF8_BYTES = 1_000_000
MAX_OTHER_WORKSPACE_TEXT_BYTES = 2_000_000
MAX_PHASE_JSON_BYTES = 64 * 1024 * 1024
INITIAL_LIBRARY_SOURCE = \
    "# Shared predicate library. p_<name>(panel) -> float | bool\n"

DEFAULT_LADDER = (codex_headless.DEFAULT_CODEX_MODEL,) * 3
"""Proposer escalation ladder: model per attempt round."""
DEFAULT_INFRA_WAIT_SECONDS = 30
DEFAULT_MAX_INFRA_WAITS = 2
VERIFIER_FAILURE_STATUS = "VERIFIER_FAILURE_UNRESTRICTED"
LEGACY_CLAUDE_RECEIPT_SCHEMA = "bongard.unrestricted-proposer-receipt/v1"

SOURCE_TAINT_MARKERS = (
    "downloads/bongard-logo",
    "get_action_string_list",
    "human_designed_shapes",
    "basic_sampler",
    "abstract_sampler",
    "action_program",
    "results.json",
)
"""Markers whose presence in a proposer workspace file makes the attempt
inadmissible: they evidence reading the dataset/sampler/ground-truth side."""


class WorkspaceTainted(RuntimeError):
    """The proposer workspace evidences forbidden dataset/ground-truth use."""


class ProposerInfrastructureFailure(RuntimeError):
    """The proposer process failed before producing a scientific attempt."""


@dataclass(frozen=True)
class ProposerOutcome:
    """A consuming proposer result with its exact model-usage receipt."""

    transcript: str
    receipt: Dict[str, object]


PRECONCEPTIONS = """\
You are solving a Bongard problem. You see 12 small images: six in `pos_*`
and six in `neg_*`. All six positive images satisfy a single hidden rule;
all six negative images violate it. The two sides are deliberate near-misses
of each other, and the hidden rule is SIMPLE -- it is the shortest natural
description that separates the sides.

General preconceptions you may carry (nothing problem-specific is given):
each image is a line drawing containing one or more drawn objects on an
empty background. What tends to matter in such problems are properties of
the objects and relations between them -- how many there are, how large,
how they are shaped, where they sit, how they are oriented, and how they
relate to one another. Which of these matters here, and how to measure it
from raw pixels, is yours to discover by experiment.
"""

TESTER = '''import sys
sys.path.insert(0, {labdir!r})
import glob, json, os
import numpy as np
import bongard_arena as A

ws = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.join(ws, open(os.path.join(ws, "current_problem.txt")).read().strip())
pos = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "pos_*.npy")))]
neg = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "neg_*.npy")))]
problem = A.Problem("current", "?", "?", pos, neg)
source_path = os.path.join(ws, "predicates.py")
source = open(source_path, encoding="utf-8").read()
contract = json.load(open(os.path.join(ws, "pricing_contract.json")))
try:
    result = A.verify_priced_source(
        source,
        problem,
        sharing_policy=contract["sharing_policy"],
        paid_node_identities=contract["paid_node_identities"],
        filename=source_path,
    )
except Exception as exc:
    print("RESULT solved=False heldout=0.000 train=0.000 "
          "rule=\\\"PRICING_OR_LOAD_ERROR\\\" rule_cost=0.0 "
          "definition_cost=0 pricing=%s predicate_errors=12 error=%s:%s"
          % (contract.get("sharing_policy", "unknown"),
             type(exc).__name__, str(exc).replace("\\n", " ")))
else:
    print(result.result_line())
'''


# ---------------------------------------------------------------------------
# Description-length proxy (idiom from gkm_legs)
# ---------------------------------------------------------------------------

def _loc(code: str) -> int:
    return sum(1 for ln in (code or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def _literal_cost(code: str) -> int:
    """Large literals (lookup tables of panel answers) must carry MDL cost
    even when formatted on one line."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return 0
    cost = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            cost += len(node.elts)
        elif isinstance(node, ast.Dict):
            cost += len(node.keys)
        elif isinstance(node, ast.Call):
            # Constructor-style tables such as dict(k0=..., k999=...) have no
            # container AST node; their positional/keyword cardinality is
            # nevertheless literal payload and must not collapse to one LOC.
            cost += len(node.args) + len(node.keywords)
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                size = len(value.encode("utf-8"))
            elif isinstance(value, bytes):
                size = len(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                size = max(1, (abs(value).bit_length() + 7) // 8)
            elif isinstance(value, (float, complex)):
                size = len(repr(value).encode("ascii"))
            else:
                size = 0
            cost += max(0, (size + 15) // 16 - 1)
    return cost


def description_complexity(code: str) -> int:
    return _loc(code) + _literal_cost(code)


def marginal_complexity(before: str, after: str) -> int:
    """Admitted growth of the shared library. Reuse is free; only novelty
    is paid for."""
    return max(0, description_complexity(after) - description_complexity(before))


# ---------------------------------------------------------------------------
# Taint check
# ---------------------------------------------------------------------------

def _workspace_taint_reason(ws: str) -> Optional[str]:
    for root, dirs, files in os.walk(ws):
        retained_dirs = []
        for name in dirs:
            path = os.path.join(root, name)
            try:
                info = os.lstat(path)
            except OSError as exc:
                return f"unreadable workspace directory {path!r}: {exc}"
            if not stat.S_ISDIR(info.st_mode):
                return f"non-directory workspace entry {os.path.relpath(path, ws)}"
            if name not in {"__pycache__", ".pytest_cache"}:
                retained_dirs.append(name)
        dirs[:] = retained_dirs
        for name in files:
            path = os.path.join(root, name)
            try:
                info = os.lstat(path)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    return ("non-regular workspace entry "
                            f"{os.path.relpath(path, ws)}")
                if name.endswith((".npy", ".png")):
                    continue
                if name in WORKSPACE_CONTROL_FILES \
                        and name not in {LIBRARY_FILE, LOG_FILE}:
                    continue
                if info.st_size > MAX_OTHER_WORKSPACE_TEXT_BYTES:
                    return ("oversized workspace entry "
                            f"{os.path.relpath(path, ws)}")
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().lower()
            except OSError:
                continue
            for marker in SOURCE_TAINT_MARKERS:
                if marker in text:
                    return f"{marker} in {os.path.relpath(path, ws)}"
    return None


def assert_workspace_not_tainted(ws: str) -> None:
    reason = _workspace_taint_reason(ws)
    if reason:
        raise WorkspaceTainted(
            f"forbidden dataset/ground-truth access tainted workspace: {reason}")


# ---------------------------------------------------------------------------
# Records, checkpoint, artifact
# ---------------------------------------------------------------------------

@dataclass
class ProblemRecord:
    opaque_id: str
    solved: bool
    heldout_accuracy: float
    rule: str
    rule_cost: float
    marginal_C: int
    model: str
    attempts: int
    escalated: bool
    proposer_receipts: List[dict] = field(default_factory=list)
    proposer_feedback: List[str] = field(default_factory=list)
    proposer_panel_set_digest: str = ""
    baseline_log_digest: str = ""
    attempted_log_digest: str = ""
    phase_execution_binding_digest: str = ""
    status: str = "UNRECORDED"
    track: str = "UNRESTRICTED"
    condition: str = P.OBSERVED
    sharing_policy: str = P.SHARED
    corpus_digest: str = ""
    panel_set_digest: str = ""
    control_digest: str = ""
    label_policy: str = P.OBSERVED
    selection_policy: str = A.PRICED_SELECTION_POLICY
    baseline_source_digest: str = ""
    attempted_source_digest: str = ""
    attempted_source: str = ""
    accepted_source_digest: str = ""
    accepted_source: str = ""
    predicate_names: List[str] = field(default_factory=list)
    rule_atoms: List[dict] = field(default_factory=list)
    used_definition_nodes: List[dict] = field(default_factory=list)
    charged_definition_node_identities: List[str] = field(default_factory=list)
    reused_definition_node_identities: List[str] = field(default_factory=list)
    full_definition_cost: int = 0
    definition_charge: int = 0
    structure_charge: float = 0.0
    total_charge: float = 0.0
    pricing_context_digest: str = ""
    verification_digest: str = ""
    source_verification_digest: str = ""
    train_accuracy: float = 0.0
    predicate_errors: int = 0
    n_rotations: int = 0
    fold_rule_atoms: List[List[dict]] = field(default_factory=list)
    verifier_fingerprint_digest: str = ""


@dataclass
class Report:
    tag: str
    records: List[ProblemRecord] = field(default_factory=list)
    track: str = "UNRESTRICTED"
    condition: str = P.OBSERVED
    sharing_policy: str = P.SHARED
    corpus_digest: str = ""
    corpus_bundle_digest: str = ""
    control_digest: str = ""
    schema: str = REPORT_SCHEMA
    label_policy: str = P.OBSERVED
    source_trace_digest: str = ""
    parent_source_trace_digest: str = ""
    verifier_fingerprint: Dict[str, object] = field(default_factory=dict)
    phase_execution_binding: Dict[str, object] = field(default_factory=dict)
    phase_execution_binding_history: List[dict] = field(default_factory=list)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.records if r.solved)

    @property
    def total_marginal_C(self) -> int:
        return sum(r.marginal_C for r in self.records)

    @property
    def total_definition_charge(self) -> int:
        return sum(r.definition_charge for r in self.records)

    @property
    def total_structure_charge(self) -> float:
        return sum(r.structure_charge for r in self.records)

    @property
    def total_charge(self) -> float:
        return self.total_definition_charge + self.total_structure_charge

    @property
    def paid_node_identities(self) -> List[str]:
        if self.sharing_policy != P.SHARED:
            return []
        return sorted({
            node["identity"]
            for record in self.records if record.solved
            for node in record.used_definition_nodes
        })

    @property
    def free_energy(self) -> float:
        return A.free_energy(self.solved, self.total_charge)

    def to_json(self) -> dict:
        body = {"schema": self.schema, "tag": self.tag, "solved": self.solved,
                "track": self.track, "condition": self.condition,
                "label_policy": self.label_policy,
                "sharing_policy": self.sharing_policy,
                "corpus_digest": self.corpus_digest,
                "corpus_bundle_digest": self.corpus_bundle_digest,
                "control_digest": self.control_digest,
                "total_marginal_C": self.total_marginal_C,
                "total_definition_charge": self.total_definition_charge,
                "total_structure_charge": self.total_structure_charge,
                "total_charge": self.total_charge,
                "paid_node_identities": self.paid_node_identities,
                "free_energy": self.free_energy,
                "source_trace_digest": self.source_trace_digest,
                "parent_source_trace_digest": self.parent_source_trace_digest,
                "verifier_fingerprint": self.verifier_fingerprint,
                "phase_execution_binding": self.phase_execution_binding,
                "phase_execution_binding_history": (
                    self.phase_execution_binding_history),
                "records": [asdict(r) for r in self.records]}
        if not body["source_trace_digest"]:
            body["source_trace_digest"] = _source_trace_digest(self.records)
        return body


_REPORT_KEYS = frozenset({
    "schema", "tag", "solved", "track", "condition", "label_policy",
    "sharing_policy", "corpus_digest", "corpus_bundle_digest",
    "control_digest", "total_marginal_C", "total_definition_charge",
    "total_structure_charge", "total_charge", "paid_node_identities",
    "free_energy", "source_trace_digest", "parent_source_trace_digest",
    "verifier_fingerprint", "phase_execution_binding",
    "phase_execution_binding_history", "records",
})
_RECORD_KEYS = frozenset(item.name for item in fields(ProblemRecord))
_PENDING_PROMOTION_KEYS = frozenset({
    "schema", "report", "predicates_log", "predicates_log_digest",
    "pending_digest",
})


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_LEGACY_PROPOSER_RECEIPT_KEYS = frozenset({
    "schema", "source", "requested_model", "actual_model", "input_tokens",
    "output_tokens", "model_usage", "model_usage_digest", "outcome",
    "permission_denials", "receipt_digest",
})


def _build_proposer_receipt(
        *, source: str, requested_model: str, actual_model: str,
        input_tokens: int, output_tokens: int, model_usage: dict,
        outcome: str, permission_denials: Sequence[object]) -> dict:
    body = {
        "schema": LEGACY_CLAUDE_RECEIPT_SCHEMA,
        "source": source,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_usage": model_usage,
        "model_usage_digest": _canonical_digest(model_usage),
        "outcome": outcome,
        "permission_denials": list(permission_denials),
    }
    body["receipt_digest"] = _canonical_digest(body)
    _validate_proposer_receipt(body)
    return body


def _injected_proposer_receipt(model: str) -> dict:
    return _build_proposer_receipt(
        source="test-injected", requested_model=model,
        actual_model="test-injected", input_tokens=0, output_tokens=0,
        model_usage={}, outcome="test-injected", permission_denials=())


def _validate_proposer_receipt(receipt: object) -> None:
    if isinstance(receipt, dict) and receipt.get("source") == "codex-cli":
        try:
            codex_headless.validate_codex_receipt(receipt)
        except codex_headless.CodexProposerFailure as exc:
            raise RuntimeError(
                f"Codex proposer receipt is invalid: {exc}") from exc
        return
    if not isinstance(receipt, dict) \
            or set(receipt) != _LEGACY_PROPOSER_RECEIPT_KEYS:
        raise RuntimeError("proposer model-usage receipt has invalid fields")
    if receipt["schema"] != LEGACY_CLAUDE_RECEIPT_SCHEMA:
        raise RuntimeError("proposer model-usage receipt schema differs")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_digest"}
    if receipt["receipt_digest"] != _canonical_digest(body):
        raise RuntimeError("proposer model-usage receipt digest does not reproduce")
    requested = receipt["requested_model"]
    actual = receipt["actual_model"]
    source = receipt["source"]
    outcome = receipt["outcome"]
    denials = receipt["permission_denials"]
    usage = receipt["model_usage"]
    input_tokens = receipt["input_tokens"]
    output_tokens = receipt["output_tokens"]
    if not isinstance(requested, str) or not requested \
            or not isinstance(actual, str) or not actual \
            or not isinstance(denials, list) or not isinstance(usage, dict) \
            or isinstance(input_tokens, bool) or not isinstance(input_tokens, int) \
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        raise RuntimeError("proposer model-usage receipt values are malformed")
    if receipt["model_usage_digest"] != _canonical_digest(usage):
        raise RuntimeError("proposer model-usage digest does not reproduce")
    if source == "claude-cli":
        if actual != requested or set(usage) != {actual} \
                or input_tokens < 0 or output_tokens < 0 \
                or input_tokens + output_tokens <= 0:
            raise RuntimeError("Claude model identity or positive usage differs")
        entry = usage[actual]
        if not isinstance(entry, dict) \
                or entry.get("inputTokens") != input_tokens \
                or entry.get("outputTokens") != output_tokens:
            raise RuntimeError("Claude token usage does not reproduce")
        if outcome not in {"success", "permission-denied"} \
                or bool(denials) != (outcome == "permission-denied"):
            raise RuntimeError("Claude permission outcome is inconsistent")
    elif source == "test-injected":
        if actual != "test-injected" or usage or input_tokens != 0 \
                or output_tokens != 0 or outcome != "test-injected" or denials:
            raise RuntimeError("injected proposer receipt is malformed")
    else:
        raise RuntimeError("unknown proposer receipt source")


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _identity_set_digest(identities: Sequence[str]) -> str:
    return _canonical_digest(sorted(set(identities)))


def _is_hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _is_prefixed_hex_digest(value: object) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") \
        and _is_hex_digest(value.removeprefix("sha256:"))


def _proposer_receipts_digest(receipts: Sequence[dict]) -> str:
    return _canonical_digest(list(receipts))


def _validate_record_proposer_evidence(
        record: "ProblemRecord", *, seen_threads: Optional[set[str]] = None,
        seen_event_streams: Optional[set[str]] = None) -> None:
    """Bind every production Codex turn to this problem and source/log chain."""
    if not isinstance(record.proposer_feedback, list) \
            or len(record.proposer_feedback) != record.attempts \
            or any(not isinstance(item, str)
                   or len(item.encode("utf-8")) > 10_000
                   for item in record.proposer_feedback) \
            or not record.proposer_feedback \
            or record.proposer_feedback[0] != "":
        raise RuntimeError("checkpoint proposer feedback trace is invalid")
    if not _is_hex_digest(record.baseline_log_digest) \
            or not _is_hex_digest(record.attempted_log_digest) \
            or not _is_prefixed_hex_digest(record.proposer_panel_set_digest):
        raise RuntimeError("checkpoint proposer input/output binding is invalid")
    codex_receipts = [
        receipt for receipt in record.proposer_receipts
        if receipt.get("source") == "codex-cli"
    ]
    if not codex_receipts:
        return
    if len(codex_receipts) != len(record.proposer_receipts):
        raise RuntimeError("checkpoint mixes Codex and non-Codex receipts")
    threads = seen_threads if seen_threads is not None else set()
    streams = seen_event_streams if seen_event_streams is not None else set()
    expected_source = record.baseline_source_digest
    expected_log = record.baseline_log_digest
    for index, receipt in enumerate(codex_receipts):
        expected_task = build_task(record.opaque_id, "")
        feedback = record.proposer_feedback[index]
        if feedback:
            expected_task += (
                "\nAUTHORITATIVE FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                + feedback + "\n")
        if receipt.get("task_digest") != _source_digest(expected_task):
            raise RuntimeError("checkpoint Codex task digest differs")
        if receipt.get("current_source_digest") != expected_source \
                or receipt.get("current_log_digest") != expected_log:
            raise RuntimeError("checkpoint Codex input chain differs")
        if receipt.get("panel_set_digest") != \
                record.proposer_panel_set_digest:
            raise RuntimeError("checkpoint Codex panel binding differs")
        thread_id = receipt.get("thread_id")
        event_digest = receipt.get("event_stream_digest")
        if thread_id in threads or event_digest in streams:
            raise RuntimeError("checkpoint reuses Codex turn identity evidence")
        threads.add(thread_id)
        streams.add(event_digest)
        expected_source = receipt.get("proposed_source_digest")
        expected_log = receipt.get("proposed_log_digest")
    if expected_source != record.attempted_source_digest \
            or expected_log != record.attempted_log_digest:
        raise RuntimeError("checkpoint final Codex output binding differs")


def _source_file_digest(module: object) -> str:
    path = getattr(module, "__file__", "")
    if not path or not os.path.isfile(path):
        raise RuntimeError("verifier module has no readable source file")
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _verifier_fingerprint() -> dict:
    """Bind replay to the exact selector/pricer code and numerical runtime."""
    body = {
        "schema": VERIFIER_FINGERPRINT_SCHEMA,
        "sources": {
            "bongard_arena.py": _source_file_digest(A),
            "predicate_pricing.py": _source_file_digest(predicate_price),
        },
        "selector_contract": {
            "selection_policy": A.PRICED_SELECTION_POLICY,
            "max_rule_atoms": A.MAX_RULE_ATOMS,
            "max_candidate_atoms": A.MAX_CANDIDATE_ATOMS,
            "call_cost": A.CALL_COST,
            "binding_cost": A.BINDING_COST,
            "authoritative_timeout_seconds":
                A.AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS,
            "resource_limits": A.verifier_resource_limit_policy(),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_cache_tag": getattr(sys.implementation, "cache_tag", ""),
            "python_hash_seed_env": os.environ.get("PYTHONHASHSEED", "random"),
            "python_hash_probes": [
                hash(f"bongard-unrestricted-replay/v3/{index}")
                for index in range(4)
            ],
            "byteorder": sys.byteorder,
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
            "scikit-image": _distribution_version("scikit-image"),
        },
    }
    body["fingerprint_digest"] = _canonical_digest(body)
    return body


def _validate_verifier_fingerprint(fingerprint: object) -> str:
    if not isinstance(fingerprint, dict):
        raise RuntimeError("checkpoint has no verifier fingerprint")
    expected = _verifier_fingerprint()
    if fingerprint != expected:
        raise RuntimeError(
            "checkpoint verifier fingerprint differs from the current runtime")
    return str(expected["fingerprint_digest"])


def _problem_index(opaque_id: str) -> int:
    if not isinstance(opaque_id, str) or not opaque_id.startswith("problem_"):
        raise RuntimeError("malformed opaque problem id")
    suffix = opaque_id[len("problem_"):]
    if not suffix or not suffix.isdigit():
        raise RuntimeError("malformed opaque problem id")
    index = int(suffix)
    if opaque_id != f"problem_{index:02d}":
        raise RuntimeError("opaque problem id is not canonically formatted")
    return index


def _source_trace_digest(records: Sequence[ProblemRecord]) -> str:
    """Hash every persisted record field, not a hand-maintained projection."""
    ordered = sorted(records, key=lambda record: _problem_index(record.opaque_id))
    return _canonical_digest([asdict(record) for record in ordered])


def _rule_atoms(rule: Optional[A.Rule]) -> List[dict]:
    if rule is None:
        return []
    return [
        {"name": atom.name, "op": atom.op, "threshold": atom.threshold}
        for atom in rule.atoms
    ]


def _definition_nodes(result: A.VerifyResult) -> List[dict]:
    receipt = result.definition_receipt
    if receipt is None:
        return []
    charged = receipt.charged_node_identities
    return [
        {
            "key": node.key,
            "identity": node.identity,
            "cost": node.cost,
            "charged": node.identity in charged,
        }
        for node in receipt.used_nodes
    ]


def _pricing_context(
        sharing_policy: str,
        paid_node_identities: Sequence[str],
        baseline_source_digest: str) -> dict:
    fingerprint = _verifier_fingerprint()
    body = {
        "schema": PRICING_CONTRACT_SCHEMA,
        "sharing_policy": sharing_policy,
        "selection_policy": A.PRICED_SELECTION_POLICY,
        "paid_node_identities": sorted(set(paid_node_identities)),
        "baseline_source_digest": baseline_source_digest,
        "verifier_fingerprint_digest": fingerprint["fingerprint_digest"],
    }
    body["context_digest"] = _canonical_digest(body)
    return body


_PRICING_CONTEXT_KEYS = frozenset({
    "schema", "sharing_policy", "selection_policy",
    "paid_node_identities", "baseline_source_digest",
    "verifier_fingerprint_digest", "context_digest",
})


def _validate_pricing_contract(contract: object) -> dict:
    """Validate the closed, current-version definition-pricing contract."""
    if not isinstance(contract, dict) or set(contract) != _PRICING_CONTEXT_KEYS:
        raise RuntimeError("pricing contract fields are invalid")
    if contract["schema"] != PRICING_CONTRACT_SCHEMA:
        raise RuntimeError("pricing contract schema differs")
    body = dict(contract)
    digest = body.pop("context_digest")
    if digest != _canonical_digest(body):
        raise RuntimeError("pricing context digest does not reproduce")
    paid = contract["paid_node_identities"]
    if contract["sharing_policy"] not in {P.SHARED, P.NO_SHARE} \
            or contract["selection_policy"] != A.PRICED_SELECTION_POLICY \
            or not isinstance(paid, list) \
            or any(not isinstance(item, str) or not item for item in paid) \
            or paid != sorted(set(paid)) \
            or not _is_hex_digest(contract["baseline_source_digest"]):
        raise RuntimeError("pricing contract values are invalid")
    expected_fingerprint = _verifier_fingerprint()["fingerprint_digest"]
    if contract["verifier_fingerprint_digest"] != expected_fingerprint:
        raise RuntimeError("pricing contract verifier fingerprint differs")
    return contract


def _write_pricing_contract(ws: str, contract: dict) -> str:
    _validate_pricing_contract(contract)
    path = os.path.join(ws, PRICING_CONTRACT_FILE)
    artifact_io.atomic_json(path, contract)
    return path


def _write_tester(ws: str) -> str:
    path = os.path.join(ws, "bongard_try.py")
    _atomic_text(path, TESTER.format(labdir=LAB_DIR))
    return path


def _verification_digest(
        result: A.VerifyResult, *, source_digest: str,
        pricing_context_digest: str,
        proposer_receipts_digest: str) -> str:
    return _canonical_digest({
        "source_digest": source_digest,
        "pricing_context_digest": pricing_context_digest,
        "proposer_receipts_digest": proposer_receipts_digest,
        "pricing_source_digest": result.pricing_source_digest,
        "paid_node_identities_digest": result.paid_node_identities_digest,
        "solved": result.solved,
        "heldout_accuracy": result.heldout_accuracy,
        "train_accuracy": result.train_accuracy,
        "predicate_errors": result.predicate_errors,
        "n_rotations": result.n_rotations,
        "rule_atoms": _rule_atoms(result.selected_rule),
        "fold_rule_atoms": [_rule_atoms(rule) for rule in result.fold_rules],
        "rule": result.rule,
        "rule_cost": result.rule_cost,
        "structure_cost": result.structure_cost,
        "definition_cost": result.definition_cost,
        "full_definition_cost": result.full_definition_cost,
        "used_definition_node_identities": list(
            result.used_definition_node_identities),
        "charged_definition_node_identities": list(
            result.charged_definition_node_identities),
        "reused_definition_node_identities": list(
            result.reused_definition_node_identities),
        "definition_nodes": _definition_nodes(result),
        "sharing_policy": result.sharing_policy,
        "selection_policy": result.selection_policy,
    })


def _no_share_repricing_digest(
        record: ProblemRecord, *, pricing_context_digest: str) -> str:
    return _canonical_digest({
        "schema": "bongard.no-share-held-fixed-repricing/v1",
        "source_verification_digest": record.source_verification_digest,
        "pricing_context_digest": pricing_context_digest,
        "opaque_id": record.opaque_id,
        "solved": record.solved,
        "accepted_source_digest": record.accepted_source_digest,
        "predicate_names": record.predicate_names,
        "rule_atoms": record.rule_atoms,
        "used_definition_nodes": record.used_definition_nodes,
        "definition_charge": record.definition_charge,
        "structure_charge": record.structure_charge,
        "total_charge": record.total_charge,
    })


def _validate_atom_list(value: object, field_name: str) -> List[A.Atom]:
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list")
    atoms: List[A.Atom] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "op", "threshold"}:
            raise RuntimeError(f"{field_name}[{index}] is malformed")
        name, op, threshold = item["name"], item["op"], item["threshold"]
        if not isinstance(name, str) or not name.startswith("p_") \
                or op not in {">=", "<="} \
                or isinstance(threshold, bool) \
                or not isinstance(threshold, (int, float)) \
                or not math.isfinite(float(threshold)):
            raise RuntimeError(f"{field_name}[{index}] is malformed")
        atoms.append(A.Atom(name, op, float(threshold)))
    return atoms


def _validate_record_rule_evidence(record: ProblemRecord) -> None:
    atoms = _validate_atom_list(record.rule_atoms, "rule_atoms")
    for fold_index, fold in enumerate(record.fold_rule_atoms):
        _validate_atom_list(fold, f"fold_rule_atoms[{fold_index}]")
    expected_names = sorted({atom.name for atom in atoms})
    if record.predicate_names != expected_names:
        raise RuntimeError("record predicate names disagree with structured rule")
    if atoms and A.Rule(atoms=tuple(atoms)).describe() != record.rule:
        raise RuntimeError("record formatted rule disagrees with structured atoms")
    if not atoms and record.rule not in {
            "CONST_True", "CONST_False", "PRICING_OR_LOAD_ERROR"}:
        raise RuntimeError("record constant/error rule is malformed")


def _expected_final_source(rep: Report) -> str:
    source = INITIAL_LIBRARY_SOURCE
    for record in sorted(
            rep.records, key=lambda item: _problem_index(item.opaque_id)):
        if record.solved:
            source = record.accepted_source
    return source


def _phase_binding_digest(binding: object, *, tag: str = "") -> str:
    if binding == {}:
        return ""
    try:
        P.validate_execution_binding(binding)  # type: ignore[arg-type]
    except P.PhaseDProtocolError as exc:
        raise RuntimeError(f"Phase execution binding is invalid: {exc}") from exc
    if tag and binding["execution_tag"] != tag:  # type: ignore[index]
        raise RuntimeError("Phase execution binding tag differs from report")
    if binding["track"] != "UNRESTRICTED":  # type: ignore[index]
        raise RuntimeError("Phase execution binding track differs")
    return str(binding["binding_digest"])  # type: ignore[index]


def _validate_priced_report(rep: Report) -> frozenset[str]:
    """Validate receipt/source history and reconstruct the paid-use ledger."""
    if rep.schema != REPORT_SCHEMA or rep.track != "UNRESTRICTED":
        raise RuntimeError("unsupported unrestricted checkpoint schema or track")
    phase_binding_digest = _phase_binding_digest(
        rep.phase_execution_binding, tag=rep.tag)
    history = rep.phase_execution_binding_history
    if not isinstance(history, list):
        raise RuntimeError("checkpoint Phase execution history must be a list")
    if phase_binding_digest:
        if not history or history[-1] != rep.phase_execution_binding:
            raise RuntimeError(
                "checkpoint Phase execution history lacks its active binding")
        previous_scale = 0
        for binding in history:
            digest = _phase_binding_digest(binding, tag=rep.tag)
            if not digest:
                raise RuntimeError("checkpoint Phase execution history is unbound")
            for name in (
                    "preregistration_digest", "execution_policy_digest",
                    "track", "condition", "execution_tag"):
                if binding[name] != rep.phase_execution_binding[name]:
                    raise RuntimeError(
                        "checkpoint Phase execution history mixes run families")
            if binding["scale"] <= previous_scale:
                raise RuntimeError(
                    "checkpoint Phase execution history scales are not increasing")
            previous_scale = binding["scale"]
        if len(rep.records) > rep.phase_execution_binding["scale"]:
            raise RuntimeError(
                "checkpoint has more records than its Phase execution scale")
    elif history:
        raise RuntimeError("unbound checkpoint carries Phase execution history")
    fingerprint_digest = _validate_verifier_fingerprint(
        rep.verifier_fingerprint)
    if bool(rep.corpus_digest) != bool(rep.corpus_bundle_digest):
        raise RuntimeError("checkpoint corpus and bundle identities are incomplete")
    if rep.sharing_policy == P.SHARED:
        if rep.condition not in {P.OBSERVED, P.SHUFFLED_SIDES} \
                or rep.label_policy != rep.condition \
                or rep.parent_source_trace_digest:
            raise RuntimeError("shared checkpoint experiment identity is invalid")
        if rep.condition == P.OBSERVED and rep.control_digest:
            raise RuntimeError("observed checkpoint carries a control identity")
        if rep.condition == P.SHUFFLED_SIDES and not rep.control_digest:
            raise RuntimeError("shuffled checkpoint has no control identity")
    elif rep.sharing_policy == P.NO_SHARE:
        if rep.condition != P.NO_SHARE or rep.label_policy != P.OBSERVED \
                or not _is_hex_digest(rep.parent_source_trace_digest) \
                or rep.control_digest:
            raise RuntimeError("no-share checkpoint is not a held-fixed replay")
    else:
        raise RuntimeError("unsupported checkpoint sharing policy")
    paid: set[str] = set()
    seen_codex_threads: set[str] = set()
    seen_codex_event_streams: set[str] = set()
    expected_ids = [f"problem_{index:02d}" for index in range(len(rep.records))]
    observed_ids = [record.opaque_id for record in rep.records]
    if observed_ids != expected_ids:
        raise RuntimeError("checkpoint problem records are not a contiguous prefix")
    current_source = INITIAL_LIBRARY_SOURCE
    current_log_digest = _source_digest("")
    for record in rep.records:
        if not isinstance(record.solved, bool):
            raise RuntimeError("checkpoint record solved flag must be boolean")
        if not isinstance(record.rule, str) \
                or isinstance(record.rule_cost, bool) \
                or not isinstance(record.rule_cost, (int, float)) \
                or not math.isfinite(float(record.rule_cost)) \
                or isinstance(record.marginal_C, bool) \
                or not isinstance(record.marginal_C, int) \
                or record.marginal_C < 0:
            raise RuntimeError("checkpoint record rule/cost evidence is invalid")
        if record.track != rep.track or record.condition != rep.condition \
                or record.sharing_policy != rep.sharing_policy \
                or record.label_policy != rep.label_policy:
            raise RuntimeError("checkpoint record experiment identity is inconsistent")
        if phase_binding_digest:
            expected_binding = next(
                (binding for binding in history
                 if _problem_index(record.opaque_id) < binding["scale"]),
                None,
            )
            if expected_binding is None \
                    or record.phase_execution_binding_digest != \
                    expected_binding["binding_digest"]:
                raise RuntimeError(
                    "checkpoint record Phase execution tranche differs")
        elif record.phase_execution_binding_digest:
            raise RuntimeError(
                "unbound checkpoint record claims Phase execution provenance")
        expected_status = (
            "SOLVED_UNRESTRICTED" if record.solved else
            VERIFIER_FAILURE_STATUS
            if record.rule == "PRICING_OR_LOAD_ERROR" else
            "UNSOLVED_UNRESTRICTED")
        if record.status != expected_status:
            raise RuntimeError("checkpoint record status disagrees with solved")
        if record.corpus_digest != rep.corpus_digest \
                or record.control_digest != rep.control_digest:
            raise RuntimeError("checkpoint record corpus/control identity is inconsistent")
        if bool(rep.corpus_digest) != bool(record.panel_set_digest):
            raise RuntimeError("checkpoint record panel identity is incomplete")
        if rep.corpus_digest and record.proposer_panel_set_digest != \
                record.panel_set_digest:
            raise RuntimeError(
                "checkpoint proposer panels differ from frozen corpus panels")
        if record.selection_policy != A.PRICED_SELECTION_POLICY:
            raise RuntimeError("checkpoint record selection policy differs")
        if record.verifier_fingerprint_digest != fingerprint_digest:
            raise RuntimeError("record verifier fingerprint differs from report")
        if isinstance(record.attempts, bool) or not isinstance(record.attempts, int) \
                or record.attempts < 1 or record.escalated != (record.attempts > 1) \
                or not isinstance(record.model, str) or not record.model:
            raise RuntimeError("checkpoint attempt/escalation evidence is invalid")
        if not isinstance(record.proposer_receipts, list) \
                or len(record.proposer_receipts) != record.attempts:
            raise RuntimeError("checkpoint proposer receipt count differs")
        for receipt in record.proposer_receipts:
            _validate_proposer_receipt(receipt)
        _validate_record_proposer_evidence(
            record,
            seen_threads=seen_codex_threads,
            seen_event_streams=seen_codex_event_streams,
        )
        if record.baseline_log_digest != current_log_digest:
            raise RuntimeError(
                "checkpoint baseline proposer log is not sequential")
        current_log_digest = record.attempted_log_digest
        if record.proposer_receipts[-1]["requested_model"] != record.model:
            raise RuntimeError("checkpoint final proposer model differs from receipt")
        for name, value in (
                ("heldout_accuracy", record.heldout_accuracy),
                ("train_accuracy", record.train_accuracy)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"checkpoint {name} is invalid")
        if isinstance(record.predicate_errors, bool) \
                or not isinstance(record.predicate_errors, int) \
                or not 0 <= record.predicate_errors <= 12 \
                or isinstance(record.n_rotations, bool) \
                or not isinstance(record.n_rotations, int) \
                or record.n_rotations != 36:
            raise RuntimeError("checkpoint verifier count evidence is invalid")
        _validate_record_rule_evidence(record)
        if rep.sharing_policy == P.SHARED \
                and record.source_verification_digest != record.verification_digest:
            raise RuntimeError("shared verification evidence was rewritten")
        if not _is_hex_digest(record.source_verification_digest) \
                or not _is_hex_digest(record.verification_digest):
            raise RuntimeError("record has no source verification evidence")
        baseline_digest = _source_digest(current_source)
        if record.baseline_source_digest != baseline_digest:
            raise RuntimeError("record baseline predicate source is not sequential")
        if _source_digest(record.attempted_source) != \
                record.attempted_source_digest:
            raise RuntimeError("attempted predicate source digest does not reproduce")
        pricing_paid = paid if rep.sharing_policy == P.SHARED else set()
        context = _pricing_context(
            rep.sharing_policy, sorted(pricing_paid), baseline_digest)
        if record.pricing_context_digest != context["context_digest"]:
            raise RuntimeError("record pricing context digest does not reproduce")
        if rep.sharing_policy == P.NO_SHARE and record.verification_digest != \
                _no_share_repricing_digest(
                    record, pricing_context_digest=context["context_digest"]):
            raise RuntimeError("no-share repricing digest does not reproduce")
        if not record.solved:
            if record.accepted_source or record.accepted_source_digest \
                    or record.used_definition_nodes \
                    or record.charged_definition_node_identities \
                    or record.reused_definition_node_identities \
                    or record.full_definition_cost \
                    or record.definition_charge or record.structure_charge \
                    or record.marginal_C or record.total_charge:
                raise RuntimeError("failed record carries admitted pricing evidence")
            continue
        if not record.accepted_source \
                or _source_digest(record.accepted_source) != \
                record.accepted_source_digest:
            raise RuntimeError("accepted predicate source digest does not reproduce")
        if record.attempted_source != record.accepted_source \
                or record.attempted_source_digest != record.accepted_source_digest:
            raise RuntimeError("solved record did not admit its verified source")
        model = predicate_price.build_pricing_model(record.accepted_source)
        if model.source_digest != record.accepted_source_digest:
            raise RuntimeError("pricing model source digest does not reproduce")
        receipt = (
            model.price(
                record.predicate_names, promoted_node_identities=paid)
            if rep.sharing_policy == P.SHARED
            else model.price_no_share(record.predicate_names)
        )
        observed_nodes = [
            {"key": node.key, "identity": node.identity, "cost": node.cost,
             "charged": node.identity in receipt.charged_node_identities}
            for node in receipt.used_nodes
        ]
        if observed_nodes != record.used_definition_nodes \
                or receipt.charged_cost != record.definition_charge \
                or receipt.full_cost != record.full_definition_cost \
                or [node.identity for node in receipt.charged_nodes] != \
                record.charged_definition_node_identities \
                or [node.identity for node in receipt.reused_nodes] != \
                record.reused_definition_node_identities:
            raise RuntimeError("stored definition-pricing receipt does not reproduce")
        if record.predicate_names != sorted(record.predicate_names) \
                or len(record.predicate_names) != len(set(record.predicate_names)):
            raise RuntimeError("stored predicate names are not canonical")
        expected_structure = len(record.rule_atoms) * (
            A.CALL_COST + A.BINDING_COST)
        if record.marginal_C != record.definition_charge \
                or record.structure_charge != expected_structure \
                or record.total_charge != \
                record.definition_charge + record.structure_charge \
                or record.rule_cost != record.total_charge:
            raise RuntimeError("stored rule/definition charges do not reproduce")
        paid.update(receipt.used_node_identities)
        current_source = record.accepted_source
    if rep.records and rep.source_trace_digest != _source_trace_digest(rep.records):
        raise RuntimeError("checkpoint source trace digest does not reproduce")
    return frozenset(paid)


def reprice_no_share(
        shared_report: Report, *, tag: str,
        max_problems: int = 0,
        phase_execution_binding: Optional[dict] = None) -> Report:
    """Derive the preregistered held-fixed no-share accounting control.

    The accepted sources, structured rules, risks, and outcomes are copied
    from an observed shared run.  Only the exact definition receipts are
    recomputed with prior-use discounts disabled; no proposer or reselection
    occurs.
    """
    _validate_priced_report(shared_report)
    if shared_report.sharing_policy != P.SHARED \
            or shared_report.condition != P.OBSERVED \
            or shared_report.label_policy != P.OBSERVED:
        raise ValueError("no-share requires an observed shared source trace")
    if not tag or tag == shared_report.tag:
        raise ValueError("no-share needs a distinct nonempty artifact tag")
    count = len(shared_report.records) if not max_problems else max_problems
    if count < 1 or count > len(shared_report.records):
        raise ValueError("no-share prefix length is outside the shared trace")
    source_records = shared_report.records[:count]
    parent_trace = _source_trace_digest(source_records)
    target_binding = (
        dict(phase_execution_binding)
        if phase_execution_binding is not None else {})
    target_binding_digest = _phase_binding_digest(target_binding, tag=tag)
    derived_records: List[ProblemRecord] = []
    current_source = INITIAL_LIBRARY_SOURCE
    for original in source_records:
        baseline_digest = _source_digest(current_source)
        if original.baseline_source_digest != baseline_digest:
            raise RuntimeError("shared source trace is not sequential")
        context = _pricing_context(P.NO_SHARE, (), baseline_digest)
        values = asdict(original)
        values.update({
            "condition": P.NO_SHARE,
            "label_policy": P.OBSERVED,
            "sharing_policy": P.NO_SHARE,
            "control_digest": "",
            "pricing_context_digest": context["context_digest"],
            "source_verification_digest": original.verification_digest,
            "verification_digest": "",
            "phase_execution_binding_digest": target_binding_digest,
        })
        if original.solved:
            model = predicate_price.build_pricing_model(
                original.accepted_source,
                filename=f"{original.opaque_id}/predicates.py",
            )
            receipt = model.price_no_share(original.predicate_names)
            nodes = [
                {"key": node.key, "identity": node.identity,
                 "cost": node.cost, "charged": True}
                for node in receipt.used_nodes
            ]
            values.update({
                "used_definition_nodes": nodes,
                "charged_definition_node_identities": [
                    node.identity for node in receipt.charged_nodes],
                "reused_definition_node_identities": [],
                "full_definition_cost": receipt.full_cost,
                "definition_charge": receipt.full_cost,
                "marginal_C": receipt.full_cost,
                "total_charge": receipt.full_cost + original.structure_charge,
                "rule_cost": receipt.full_cost + original.structure_charge,
            })
            current_source = original.accepted_source
        else:
            values.update({
                "used_definition_nodes": [],
                "charged_definition_node_identities": [],
                "reused_definition_node_identities": [],
                "full_definition_cost": 0,
                "definition_charge": 0,
                "marginal_C": 0,
                "structure_charge": 0.0,
                "total_charge": 0.0,
                "rule_cost": 0.0,
            })
        record = ProblemRecord(**values)
        record.verification_digest = _no_share_repricing_digest(
            record, pricing_context_digest=context["context_digest"])
        derived_records.append(record)
    report = Report(
        tag=tag,
        records=derived_records,
        track="UNRESTRICTED",
        condition=P.NO_SHARE,
        sharing_policy=P.NO_SHARE,
        corpus_digest=shared_report.corpus_digest,
        corpus_bundle_digest=shared_report.corpus_bundle_digest,
        control_digest="",
        schema=REPORT_SCHEMA,
        label_policy=P.OBSERVED,
        parent_source_trace_digest=parent_trace,
        verifier_fingerprint=_verifier_fingerprint(),
        phase_execution_binding=target_binding,
        phase_execution_binding_history=(
            [target_binding] if target_binding else []),
    )
    report.source_trace_digest = _source_trace_digest(report.records)
    _validate_priced_report(report)
    return report


def artifact_dir(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
        raise ValueError("artifact tag must be a simple 1-64 character name")
    return os.path.join(LAB_DIR, "agent_solutions", f"{tag}_predicates")


def _preflight_artifact_binding(
        tag: str, *, corpus_manifest: Optional[dict],
        corpus_bundle: Optional[dict], control_manifest: Optional[dict],
        problems: Sequence[A.Problem], condition: str,
        phase_execution_binding: Optional[dict] = None,
        phase_predecessor_execution_binding: Optional[dict] = None,
    phase_execution_binding_history: Optional[Sequence[dict]] = None,
        ) -> Optional[Report]:
    """Validate every existing destination before creating any binding file."""
    art = artifact_dir(tag)
    pending_state = _load_pending_promotion(art)
    pending = pending_state[0] if pending_state is not None else None
    prior = pending if pending is not None else _load_checkpoint(art)
    if prior is not None:
        if prior.tag != tag:
            raise RuntimeError(
                "checkpoint tag differs from its artifact directory")
        if corpus_manifest is None:
            if prior.corpus_digest:
                raise RuntimeError(
                    "checkpoint is corpus-bound but this run supplied no manifest")
        else:
            if prior.records and not prior.corpus_digest:
                raise RuntimeError(
                    "legacy checkpoint has no corpus identity; use a fresh tag")
            if prior.corpus_digest and prior.corpus_digest != \
                    corpus_manifest["corpus_digest"]:
                raise RuntimeError(
                    "checkpoint belongs to a different frozen corpus")
            if prior.corpus_bundle_digest and prior.corpus_bundle_digest != \
                    corpus_bundle["bundle_digest"]:
                raise RuntimeError(
                    "checkpoint belongs to different embedded corpus bytes")
            if prior.records and (
                    prior.condition != condition
                    or prior.sharing_policy != P.SHARED):
                raise RuntimeError(
                    "checkpoint belongs to a different experiment arm")
            if any(_problem_index(record.opaque_id) >= len(problems)
                   for record in prior.records):
                raise RuntimeError(
                    "active corpus prefix is shorter than the existing checkpoint")
            expected_control = (
                control_manifest["control_digest"]
                if control_manifest is not None else "")
            if prior.records and prior.control_digest != expected_control:
                raise RuntimeError(
                    "checkpoint belongs to a different experiment control")
        current = phase_execution_binding or {}
        predecessor = phase_predecessor_execution_binding or {}
        full_history = list(
            phase_execution_binding_history
            if phase_execution_binding_history is not None else
            ([current] if current else []))
        expected_history = None
        if prior.phase_execution_binding == current:
            expected_history = full_history
        elif predecessor and prior.phase_execution_binding == predecessor:
            expected_history = full_history[:-1]
        if expected_history is None:
            raise RuntimeError(
                "checkpoint Phase execution binding differs from this run")
        if prior.phase_execution_binding_history != expected_history:
            raise RuntimeError(
                "checkpoint Phase execution binding history differs from this run")
        if current:
            phase_policy = P.canonical_execution_policy(
                require_unrestricted_cli=True)
            for record in prior.records:
                P._validate_unrestricted_proposer_receipts(
                    asdict(record), record.opaque_id, phase_policy)
                if any(receipt.get("source") != "codex-cli"
                       for receipt in record.proposer_receipts):
                    raise RuntimeError(
                        "Phase checkpoint contains a test-injected receipt")

    expected_files: list[tuple[str, dict, Callable[[dict], None]]] = []
    if corpus_manifest is not None:
        expected_files.extend((
            (
                os.path.join(art, "corpus_manifest.json"),
                corpus_manifest,
                P.validate_corpus_manifest,
            ),
            (
                os.path.join(art, "corpus_panels.json"),
                corpus_bundle,
                lambda value: P.validate_corpus_bundle(
                    value, corpus_manifest),
            ),
        ))
    if control_manifest is not None:
        expected_files.append((
            os.path.join(art, "control_manifest.json"),
            control_manifest,
            lambda value: P.validate_shuffled_control_manifest(
                value, corpus_manifest),
        ))
    for path, expected, validator in expected_files:
        if not os.path.exists(path):
            continue
        existing = _read_required_json(path, os.path.basename(path))
        try:
            validator(existing)
        except (TypeError, ValueError, P.PhaseDProtocolError) as exc:
            raise RuntimeError(
                f"existing artifact binding is invalid: {path}") from exc
        if existing != expected:
            raise RuntimeError(
                f"artifact tag is bound to different {os.path.basename(path)}")
    control_path = os.path.join(art, "control_manifest.json")
    if control_manifest is None and os.path.exists(control_path):
        raise RuntimeError(
            "non-shuffled run cannot reuse a tag containing a control manifest")
    if prior is None and os.path.isdir(art):
        forbidden_without_checkpoint = {
            LIBRARY_FILE, LOG_FILE, "results.json", "README.md"}
        if forbidden_without_checkpoint.intersection(os.listdir(art)):
            raise RuntimeError(
                "artifact contains promoted state without a valid checkpoint")
    if pending is not None:
        # A promotion stages its full checkpoint and exact predicate log in
        # one atomic commit marker before touching the committed generation.
        # Recover either crash side only after all arm/corpus checks above and
        # a fresh scientific replay of that staged generation.
        if corpus_manifest is not None:
            replay_problems = _validate_report_protocol_evidence(
                pending, corpus_manifest, corpus_bundle, control_manifest)
        else:
            replay_problems = problems
        _cold_replay_report(pending, replay_problems)
        os.makedirs(art, exist_ok=True)
        _atomic_text(
            os.path.join(art, LIBRARY_FILE), _expected_final_source(pending))
        _atomic_bytes(os.path.join(art, LOG_FILE), pending_state[1])
        _save_checkpoint(art, pending)
        try:
            os.unlink(os.path.join(art, PENDING_CHECKPOINT_FILE))
        except FileNotFoundError:
            pass
    return prior


def bind_corpus_manifest_to_artifact(tag: str, manifest: dict) -> str:
    """Create the corpus binding, or refuse to redefine an existing tag."""
    P.validate_corpus_manifest(manifest)
    art = artifact_dir(tag)
    path = os.path.join(art, "corpus_manifest.json")
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
            P.validate_corpus_manifest(existing)
        except (OSError, json.JSONDecodeError, P.PhaseDProtocolError) as exc:
            raise RuntimeError(f"existing artifact corpus manifest is invalid: {exc}") \
                from exc
        if existing["corpus_digest"] != manifest["corpus_digest"]:
            raise RuntimeError("artifact tag is bound to a different corpus")
        return path
    if os.path.exists(os.path.join(art, CHECKPOINT_FILE)):
        raise RuntimeError(
            "legacy artifact has no corpus identity; use a fresh tag")
    os.makedirs(art, exist_ok=True)
    artifact_io.atomic_json(path, manifest)
    return path


def bind_control_manifest_to_artifact(
        tag: str, control_manifest: dict, corpus_manifest: dict) -> str:
    """Bind a tag to one preregistered shuffled-side replicate."""
    P.validate_shuffled_control_manifest(control_manifest, corpus_manifest)
    art = artifact_dir(tag)
    path = os.path.join(art, "control_manifest.json")
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
            P.validate_shuffled_control_manifest(existing, corpus_manifest)
        except (OSError, json.JSONDecodeError, P.PhaseDProtocolError) as exc:
            raise RuntimeError(f"existing artifact control manifest is invalid: {exc}") \
                from exc
        if existing["control_digest"] != control_manifest["control_digest"]:
            raise RuntimeError("artifact tag is bound to a different control")
        return path
    os.makedirs(art, exist_ok=True)
    artifact_io.atomic_json(path, control_manifest)
    return path


def bind_corpus_bundle_to_artifact(
        tag: str, corpus_bundle: dict, corpus_manifest: dict) -> str:
    """Persist replayable base panel bytes for solved and failed outcomes."""
    P.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    art = artifact_dir(tag)
    path = os.path.join(art, "corpus_panels.json")
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
            P.validate_corpus_bundle(existing, corpus_manifest)
        except (OSError, json.JSONDecodeError, P.PhaseDProtocolError) as exc:
            raise RuntimeError(f"existing artifact corpus bundle is invalid: {exc}") \
                from exc
        if existing["bundle_digest"] != corpus_bundle["bundle_digest"]:
            raise RuntimeError("artifact tag is bound to different corpus bytes")
        return path
    os.makedirs(art, exist_ok=True)
    artifact_io.atomic_json(path, corpus_bundle)
    return path


def _require_workspace_directory(ws: str, *, create: bool = False) -> None:
    """Require the workspace itself to be an actual directory, not a link."""
    try:
        before = os.lstat(ws)
    except FileNotFoundError:
        if not create:
            raise RuntimeError("proposer workspace does not exist")
        os.makedirs(ws, exist_ok=False)
        before = os.lstat(ws)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(
            "proposer workspace must be a non-symlink directory")
    if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
        raise RuntimeError("proposer workspace must be owned by the current user")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ws, flags)
    except OSError as exc:
        raise RuntimeError(
            "proposer workspace must be a non-symlink directory") from exc
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != \
                (after.st_dev, after.st_ino):
            raise RuntimeError("proposer workspace changed during validation")
    finally:
        os.close(descriptor)


def _private_default_workspace(tag: str, *, parent: Optional[str] = None) -> str:
    """Create a private scratch directory; callers retain it for diagnostics."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag) is None:
        raise ValueError("workspace tag must be a simple 1-64 character name")
    workspace = tempfile.mkdtemp(prefix=f"bongard_ws_{tag}_", dir=parent)
    info = os.lstat(workspace)
    if not stat.S_ISDIR(info.st_mode) \
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid()) \
            or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(
            "private proposer workspace must be owned mode-0700 storage")
    _require_workspace_directory(workspace)
    return workspace


def _require_regular_workspace_file(
        ws: str, name: str, *, required: bool) -> Optional[str]:
    if os.path.basename(name) != name:
        raise RuntimeError("workspace file name must be a basename")
    path = os.path.join(ws, name)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if required:
            raise RuntimeError(f"workspace {name} is missing")
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(
            f"workspace {name} must be a non-symlink, singly-linked regular file")
    return path


def _preflight_workspace_owned_paths(ws: str) -> None:
    _require_workspace_directory(ws)
    for name in WORKSPACE_CONTROL_FILES:
        _require_regular_workspace_file(ws, name, required=False)


def _require_problem_id(opaque_id: str) -> None:
    if re.fullmatch(r"problem_[0-9]{2,}", opaque_id) is None:
        raise RuntimeError(
            "current problem marker must contain a canonical opaque id")


def _atomic_bytes(path: str, value: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "wb", dir=os.path.dirname(path),
            prefix=f".{os.path.basename(path)}.", delete=False) as handle:
        temporary = handle.name
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_workspace_bytes(ws: str, name: str, value: bytes) -> str:
    _require_workspace_directory(ws)
    _require_regular_workspace_file(ws, name, required=False)
    path = os.path.join(ws, name)
    _atomic_bytes(path, value)
    _require_regular_workspace_file(ws, name, required=True)
    return path


def _atomic_workspace_text(ws: str, name: str, value: str) -> str:
    return _atomic_workspace_bytes(ws, name, value.encode("utf-8"))


def _snapshot_proposer_files(ws: str) -> Dict[str, bytes]:
    """Read both proposer-editable files through stable no-follow handles."""
    return {
        LIBRARY_FILE: _stable_workspace_file_bytes(
            ws, LIBRARY_FILE, predicate_price.MAX_SOURCE_UTF8_BYTES),
        LOG_FILE: _stable_workspace_file_bytes(
            ws, LOG_FILE, MAX_PROPOSER_LOG_UTF8_BYTES),
    }


def _stable_workspace_file_bytes(
        ws: str, name: str, maximum_bytes: int) -> bytes:
    """Read at most one bounded, stable, singly-linked workspace file."""
    path = _require_regular_workspace_file(ws, name, required=True)
    before = os.lstat(path)
    if before.st_size > maximum_bytes:
        raise RuntimeError(f"workspace {name} exceeds its byte limit")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        if (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != identity \
                or opened.st_nlink != 1:
            raise RuntimeError(f"workspace {name} changed during bounded read")
        payload = os.read(descriptor, maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise RuntimeError(f"workspace {name} exceeds its byte limit")
        after = os.fstat(descriptor)
        if identity != (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"workspace {name} changed during bounded read")
        if len(payload) != opened.st_size:
            raise RuntimeError(f"workspace {name} bounded read was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _read_workspace_predicate_source(ws: str) -> str:
    payload = _stable_workspace_file_bytes(
        ws, LIBRARY_FILE, predicate_price.MAX_SOURCE_UTF8_BYTES)
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("workspace predicates.py is not valid UTF-8") from exc
    if len(source) > predicate_price.MAX_SOURCE_CHARACTERS:
        raise RuntimeError("workspace predicates.py exceeds its character limit")
    return source


def _restore_proposer_files(ws: str, snapshots: Dict[str, bytes]) -> None:
    if set(snapshots) != {LIBRARY_FILE, LOG_FILE}:
        raise RuntimeError("proposer snapshot is incomplete")
    for name in (LIBRARY_FILE, LOG_FILE):
        _atomic_workspace_bytes(ws, name, snapshots[name])


def _atomic_workspace_copy(source: str, ws: str, name: str) -> str:
    """Copy a regular source by replacement, never through a destination link."""
    _require_workspace_directory(ws)
    _require_regular_workspace_file(ws, name, required=False)
    try:
        source_info = os.lstat(source)
    except FileNotFoundError as exc:
        raise RuntimeError(f"workspace seed source is missing: {source}") from exc
    if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
        raise RuntimeError(
            f"workspace seed source must be a singly-linked regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (source_info.st_dev, source_info.st_ino) != \
                (opened.st_dev, opened.st_ino):
            raise RuntimeError("workspace seed source changed during copy")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read()
    finally:
        os.close(descriptor)
    path = os.path.join(ws, name)
    _atomic_bytes(path, value)
    return path


def _atomic_text(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=os.path.dirname(path),
            prefix=f".{os.path.basename(path)}.", delete=False) as handle:
        temporary = handle.name
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_result_evidence(record: ProblemRecord) -> dict:
    return {
        "track": record.track,
        "condition": record.condition,
        "sharing_policy": record.sharing_policy,
        "corpus_digest": record.corpus_digest,
        "panel_set_digest": record.panel_set_digest,
        "control_digest": record.control_digest,
        "label_policy": record.label_policy,
        "selection_policy": record.selection_policy,
        "solved": record.solved,
        "status": record.status,
        "heldout_accuracy": record.heldout_accuracy,
        "train_accuracy": record.train_accuracy,
        "predicate_errors": record.predicate_errors,
        "n_rotations": record.n_rotations,
        "rule": record.rule,
        "rule_atoms": record.rule_atoms,
        "fold_rule_atoms": record.fold_rule_atoms,
        "predicate_names": record.predicate_names,
        "definition_charge": record.definition_charge,
        "full_definition_cost": record.full_definition_cost,
        "structure_charge": record.structure_charge,
        "total_charge": record.total_charge,
        "baseline_source_digest": record.baseline_source_digest,
        "accepted_source_digest": record.accepted_source_digest,
        "attempted_source_digest": record.attempted_source_digest,
        "pricing_context_digest": record.pricing_context_digest,
        "source_verification_digest": record.source_verification_digest,
        "verification_digest": record.verification_digest,
        "verifier_fingerprint_digest": record.verifier_fingerprint_digest,
    }


def _reconcile_results(
        rep: Report, existing: object, *,
        truth_problems: Optional[Sequence[A.Problem]] = None,
        corpus_manifest: Optional[dict] = None) -> Dict[str, dict]:
    """Rebuild scientific fields; retain ground truth only from a trusted row."""
    if not isinstance(existing, dict):
        existing = {}
    rebuilt: Dict[str, dict] = {}
    for record in rep.records:
        index = _problem_index(record.opaque_id)
        if truth_problems is not None:
            if index >= len(truth_problems):
                raise RuntimeError("results reconciliation lacks a live problem")
            truth = truth_problems[index]
            truth_fields = {
                "problem_id": truth.problem_id,
                "category": truth.category,
                "concept": truth.concept,
            }
        else:
            prior = existing.get(record.opaque_id)
            if not isinstance(prior, dict) or any(
                    key not in prior for key in ("problem_id", "category", "concept")):
                raise RuntimeError(
                    "results reconciliation lacks trusted ground-truth fields")
            truth_fields = {
                key: prior[key] for key in ("problem_id", "category", "concept")}
            if corpus_manifest is not None \
                    and truth_fields["category"] != \
                    corpus_manifest["problems"][index]["category"]:
                raise RuntimeError("results category differs from corpus manifest")
        rebuilt[record.opaque_id] = {
            **truth_fields,
            **_record_result_evidence(record),
        }
    return rebuilt


def derive_no_share_artifact(
        source_tag: str, target_tag: str, *, max_problems: int = 0,
        verbose: bool = True,
        phase_execution_binding: Optional[dict] = None,
        required_source_phase_execution_binding: Optional[dict] = None,
        ) -> Report:
    """Publish an offline no-share reprice of one frozen observed artifact."""
    source_dir = artifact_dir(source_tag)
    source_report = _load_checkpoint(source_dir)
    if source_report is None:
        raise RuntimeError("shared source artifact has no checkpoint")
    if source_report.tag != source_tag:
        raise RuntimeError("shared checkpoint tag differs from its artifact tag")
    if required_source_phase_execution_binding is not None \
            and source_report.phase_execution_binding != \
            required_source_phase_execution_binding:
        raise RuntimeError(
            "no-share source lacks the preregistered primary execution binding")
    manifest_path = os.path.join(source_dir, "corpus_manifest.json")
    bundle_path = os.path.join(source_dir, "corpus_panels.json")
    results_path = os.path.join(source_dir, "results.json")
    try:
        with open(manifest_path) as handle:
            corpus_manifest = json.load(handle)
        with open(bundle_path) as handle:
            corpus_bundle = json.load(handle)
        with open(results_path) as handle:
            source_results = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "no-share source artifact lacks frozen corpus/results evidence") from exc
    P.validate_corpus_manifest(corpus_manifest)
    P.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    if source_report.corpus_digest != corpus_manifest["corpus_digest"] \
            or source_report.corpus_bundle_digest != corpus_bundle["bundle_digest"]:
        raise RuntimeError("shared checkpoint differs from its frozen corpus evidence")
    source_results = _reconcile_results(
        source_report,
        source_results,
        corpus_manifest=corpus_manifest,
    )
    artifact_io.atomic_json(results_path, source_results)
    derived = reprice_no_share(
        source_report, tag=target_tag, max_problems=max_problems,
        phase_execution_binding=phase_execution_binding)
    replay_problems = _validate_report_protocol_evidence(
        derived, corpus_manifest, corpus_bundle, None)
    _cold_replay_report(derived, replay_problems)
    target_dir = artifact_dir(target_tag)
    bind_corpus_manifest_to_artifact(target_tag, corpus_manifest)
    bind_corpus_bundle_to_artifact(target_tag, corpus_bundle, corpus_manifest)
    existing = _load_checkpoint(target_dir)
    if existing is not None and existing.tag != target_tag:
        raise RuntimeError("target checkpoint tag differs from its artifact tag")
    if existing is not None and existing.to_json() != derived.to_json():
        raise RuntimeError("target tag is bound to a different no-share trace")

    final_source = INITIAL_LIBRARY_SOURCE
    derived_results = _reconcile_results(
        derived,
        {record.opaque_id: source_results[record.opaque_id]
         for record in derived.records},
        corpus_manifest=corpus_manifest,
    )
    for record in derived.records:
        if record.solved:
            final_source = record.accepted_source

    _atomic_text(os.path.join(target_dir, LIBRARY_FILE), final_source)
    source_log = os.path.join(source_dir, LOG_FILE)
    if os.path.exists(source_log):
        _atomic_text(os.path.join(target_dir, LOG_FILE), _read(source_log))
    _save_checkpoint(target_dir, derived)
    artifact_io.atomic_json(
        os.path.join(target_dir, "results.json"), derived_results)
    _atomic_text(
        os.path.join(target_dir, "README.md"),
        f"# {target_tag} held-fixed no-share artifact\n\n"
        f"Derived offline from `{source_tag}` without proposer calls or "
        "candidate reselection. Exact accepted predicate definitions are "
        "repaid once per solved problem; call and binding costs are unchanged.\n\n"
        f"- Problems solved: {derived.solved}/{len(derived.records)}\n"
        f"- Definition charge: {derived.total_definition_charge}\n"
        f"- Structure charge: {derived.total_structure_charge}\n"
        f"- Total charge: {derived.total_charge}\n"
        f"- Parent source trace: `{derived.parent_source_trace_digest}`\n",
    )
    if verbose:
        print(
            f"derived held-fixed no-share artifact: {target_dir} "
            f"({derived.solved}/{len(derived.records)}, "
            f"C={derived.total_charge})")
    return derived


def _load_preregistration(
        path: str, *, corpus_manifest: Optional[dict] = None) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            preregistration = json.load(handle)
        P.validate_preregistration(
            preregistration, corpus_manifest=corpus_manifest)
    except (OSError, json.JSONDecodeError, P.PhaseDProtocolError) as exc:
        raise RuntimeError(f"invalid Phase D preregistration: {exc}") from exc
    return preregistration


def _validate_preregistered_arm(
        preregistration: dict, arm_id: str, *, corpus_digest: str,
        condition: str, sharing_policy: str, scale: int,
        control_manifest: Optional[dict] = None,
        ladder: Optional[Sequence[str]] = None,
        minutes: Optional[int] = None,
        infra_wait_seconds: Optional[int] = None,
        max_infra_waits: Optional[int] = None,
        restore_wip_context: Optional[bool] = None,
        execution_tag: Optional[str] = None) -> dict:
    P.validate_preregistration(preregistration)
    unrestricted_policy = preregistration["execution_policy"]["unrestricted"]
    if ladder is not None and list(ladder) != \
            unrestricted_policy["proposer_ladder"]:
        raise RuntimeError("proposer ladder differs from preregistered policy")
    if minutes is not None and minutes != \
            unrestricted_policy["minutes_per_attempt"]:
        raise RuntimeError(
            "minutes per attempt differ from preregistered policy")
    if infra_wait_seconds is not None and infra_wait_seconds != \
            unrestricted_policy["infrastructure_retry_wait_seconds"]:
        raise RuntimeError(
            "infrastructure retry wait differs from preregistered policy")
    if max_infra_waits is not None and max_infra_waits != \
            unrestricted_policy["maximum_infrastructure_retries_per_rung"]:
        raise RuntimeError(
            "infrastructure retry count differs from preregistered policy")
    if restore_wip_context is not None and restore_wip_context is not \
            unrestricted_policy["restore_wip_context"]:
        raise RuntimeError(
            "WIP restoration policy differs from preregistration")
    matches = [
        arm for arm in preregistration["arms"] if arm["arm_id"] == arm_id]
    if len(matches) != 1:
        raise RuntimeError("requested Phase D arm is not preregistered")
    arm = matches[0]
    if execution_tag is not None and execution_tag != arm["execution_tag"]:
        raise RuntimeError(
            "artifact tag differs from the preregistered execution tag")
    report_condition = "primary" if condition == P.OBSERVED else condition
    if preregistration["corpus_digest"] != corpus_digest \
            or arm["track"] != "UNRESTRICTED" \
            or arm["condition"] != report_condition \
            or arm["label_policy"] != (
                P.OBSERVED if condition == P.NO_SHARE else condition) \
            or arm["sharing_policy"] != sharing_policy \
            or arm["scale"] != scale:
        raise RuntimeError("runner arguments differ from the preregistered arm")
    if condition == P.SHUFFLED_SIDES:
        if control_manifest is None \
                or arm["replicate"] != control_manifest["replicate"] \
                or preregistration["shuffled_sides"]["seed"] != \
                control_manifest["seed"] \
                or arm.get("control_digest") != \
                control_manifest["control_digest"]:
            raise RuntimeError("shuffled control differs from preregistered replicate")
    elif arm["replicate"] is not None or control_manifest is not None \
            or arm.get("control_digest", ""):
        raise RuntimeError("non-shuffled arm cannot carry control assignment")
    return arm


def _validate_preregistered_scale_transition(
        preregistration: dict, arm: dict,
        checkpoint: Optional[Report]) -> None:
    """Require exact predecessor completion before adaptive scale growth."""
    if arm["condition"] == P.NO_SHARE:
        return
    scales = preregistration["scales"]
    try:
        scale_index = scales.index(arm["scale"])
    except ValueError as exc:
        raise RuntimeError("preregistered arm scale is not canonical") from exc
    predecessor = 0 if scale_index == 0 else scales[scale_index - 1]
    if checkpoint is None and scale_index != 0:
        raise RuntimeError(
            "only the first preregistered scale may start without checkpoint")
    completed = len(checkpoint.records) if checkpoint is not None else 0
    if completed > arm["scale"]:
        raise RuntimeError("active Phase scale would shrink its checkpoint")
    if scale_index > 0 and completed < predecessor:
        raise RuntimeError(
            "active Phase scale requires the complete immediate predecessor")


def _validate_phase_run_configuration(
        *, tag: str, ladder: Sequence[str], minutes: int,
        infra_wait_seconds: int, max_infra_waits: int,
        restore_wip: bool, phase_execution_binding: Optional[dict],
        phase_predecessor_execution_binding: Optional[dict]) \
        -> tuple[dict, dict]:
    """Validate current/predecessor provenance before any runner write."""
    current = (
        dict(phase_execution_binding)
        if phase_execution_binding is not None else {})
    predecessor = (
        dict(phase_predecessor_execution_binding)
        if phase_predecessor_execution_binding is not None else {})
    current_digest = _phase_binding_digest(current, tag=tag)
    predecessor_digest = _phase_binding_digest(predecessor, tag=tag)
    if predecessor_digest and not current_digest:
        raise RuntimeError("Phase predecessor binding has no current binding")
    if current_digest:
        policy = P.canonical_execution_policy(
            require_unrestricted_cli=True)["unrestricted"]
        if list(ladder) != policy["proposer_ladder"] \
                or minutes != policy["minutes_per_attempt"] \
                or infra_wait_seconds != \
                policy["infrastructure_retry_wait_seconds"] \
                or max_infra_waits != \
                policy["maximum_infrastructure_retries_per_rung"] \
                or restore_wip != policy["restore_wip_context"]:
            raise RuntimeError(
                "runner arguments differ from bound Phase execution policy")
    if predecessor_digest:
        for name in (
                "preregistration_digest", "execution_policy_digest", "track",
                "condition", "execution_tag"):
            if predecessor[name] != current[name]:
                raise RuntimeError(
                    "Phase predecessor binding is from a different run family")
        if predecessor["scale"] >= current["scale"]:
            raise RuntimeError("Phase predecessor binding scale is not smaller")
    return current, predecessor


def publish_phase_d_track_report(
        report: Report, preregistration: dict, arm_id: str, *,
        control_manifest: Optional[dict] = None,
        allow_test_injected_receipts: bool = False) -> str:
    arm = _validate_preregistered_arm(
        preregistration,
        arm_id,
        corpus_digest=report.corpus_digest,
        condition=report.condition,
        sharing_policy=report.sharing_policy,
        scale=len(report.records),
        control_manifest=control_manifest,
        execution_tag=report.tag,
    )
    expected_execution_binding = P.execution_binding(
        preregistration, arm_id)
    expected_binding_history = P.execution_binding_family(
        preregistration, arm)
    if report.phase_execution_binding != expected_execution_binding \
            or report.phase_execution_binding_history != \
            expected_binding_history:
        raise RuntimeError(
            "Phase D publication requires the exact preregistered execution "
            "binding")
    for record in report.records:
        for receipt in record.proposer_receipts:
            _validate_proposer_receipt(receipt)
            if receipt["source"] != "codex-cli" \
                    and not allow_test_injected_receipts:
                raise RuntimeError(
                    "Phase D publication requires real Codex CLI model receipts")
    protocol_source_trace = f"sha256:{report.source_trace_digest}"
    protocol_parent_trace = (
        f"sha256:{report.parent_source_trace_digest}"
        if report.parent_source_trace_digest else "")
    records: List[dict] = []
    for record in report.records:
        value = asdict(record)
        value["runner_condition"] = value["condition"]
        value["condition"] = arm["condition"]
        value["label_policy"] = arm["label_policy"]
        value["sharing_policy"] = arm["sharing_policy"]
        records.append(value)
    track_report = P.build_track_report(
        preregistration,
        arm_id=arm_id,
        records=records,
        report_source_trace_digest=protocol_source_trace,
        parent_source_trace_digest=protocol_parent_trace,
    )
    filename = arm_id.replace(":", "__") + ".json"
    path = os.path.join(
        artifact_dir(report.tag), "track_reports", filename)
    if artifact_io.create_json_once(path, track_report):
        return path
    try:
        existing = _load_stable_json(path, "existing Phase D track report")
        P.validate_track_report(existing, preregistration)
    except (RuntimeError, P.PhaseDProtocolError) as exc:
        raise RuntimeError("existing Phase D track report is invalid") from exc
    if existing != track_report:
        raise RuntimeError("artifact tag is bound to a different Phase D arm report")
    return path


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _load_stable_json(path: str, description: str) -> object:
    """Read strict bounded JSON from one stable, no-follow regular file."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(
                f"{description} is not a singly-linked regular file")
        if before.st_size > MAX_PHASE_JSON_BYTES:
            raise RuntimeError(f"{description} exceeds its byte limit")
        blocks = []
        remaining = MAX_PHASE_JSON_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns,
        )
        if len(payload) > MAX_PHASE_JSON_BYTES:
            raise RuntimeError(f"{description} exceeds its byte limit")
        if identity(before) != identity(after) \
                or identity(after) != identity(current) \
                or len(payload) != after.st_size:
            raise RuntimeError(f"{description} changed while being read")
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cannot load {description}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_required_json(path: str, description: str) -> dict:
    try:
        value = _load_stable_json(path, description)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint lacks valid {description}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint {description} must be a mapping")
    return value


def _validate_report_protocol_evidence(
        rep: Report, corpus_manifest: dict, corpus_bundle: dict,
        control_manifest: Optional[dict]) -> Sequence[A.Problem]:
    P.validate_corpus_manifest(corpus_manifest)
    P.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    if rep.corpus_digest != corpus_manifest["corpus_digest"] \
            or rep.corpus_bundle_digest != corpus_bundle["bundle_digest"]:
        raise RuntimeError("checkpoint differs from its frozen corpus evidence")
    base_problems = P.problems_from_corpus_bundle(
        corpus_bundle, corpus_manifest)
    if rep.condition == P.SHUFFLED_SIDES:
        if control_manifest is None:
            raise RuntimeError("shuffled checkpoint lacks its control manifest")
        P.validate_shuffled_control_manifest(
            control_manifest, corpus_manifest)
        if rep.control_digest != control_manifest["control_digest"]:
            raise RuntimeError("checkpoint differs from its control manifest")
        reproduced = P.build_shuffled_sides_control(
            base_problems,
            corpus_manifest,
            seed=control_manifest["seed"],
            replicate=control_manifest["replicate"],
        )
        if reproduced.manifest["control_digest"] != \
                control_manifest["control_digest"]:
            raise RuntimeError("shuffled control assignments do not reproduce")
        replay_problems: Sequence[A.Problem] = reproduced.problems
        expected_panel_digests = [
            entry["controlled_panel_set_digest"]
            for entry in control_manifest["problems"]]
    else:
        if control_manifest is not None or rep.control_digest:
            raise RuntimeError("non-shuffled checkpoint carries control evidence")
        replay_problems = base_problems
        expected_panel_digests = [
            entry["panel_set_digest"] for entry in corpus_manifest["problems"]]
    for record in rep.records:
        index = _problem_index(record.opaque_id)
        if index >= len(expected_panel_digests) \
                or record.panel_set_digest != expected_panel_digests[index] \
                or record.corpus_digest != corpus_manifest["corpus_digest"]:
            raise RuntimeError(
                f"{record.opaque_id} panel/corpus identity differs")
    return replay_problems


def _validate_artifact_checkpoint(directory: str, rep: Report) -> None:
    """Validate promoted bytes and cold replay any corpus-bound checkpoint."""
    expected_source = _expected_final_source(rep)
    library_path = os.path.join(directory, LIBRARY_FILE)
    if not os.path.isfile(library_path) or _read(library_path) != expected_source:
        raise RuntimeError(
            "artifact predicates.py differs from the final accepted source")
    expected_log_digest = (
        rep.records[-1].attempted_log_digest
        if rep.records else _source_digest(""))
    log_bytes = _stable_workspace_file_bytes(
        directory, LOG_FILE, MAX_PROPOSER_LOG_UTF8_BYTES)
    if hashlib.sha256(log_bytes).hexdigest() != expected_log_digest:
        raise RuntimeError(
            "artifact predicates_log.md differs from the proposer log trace")
    if not rep.corpus_digest:
        return
    corpus_manifest = _read_required_json(
        os.path.join(directory, "corpus_manifest.json"), "corpus manifest")
    corpus_bundle = _read_required_json(
        os.path.join(directory, "corpus_panels.json"), "corpus panel bundle")
    control_path = os.path.join(directory, "control_manifest.json")
    control_manifest = (
        _read_required_json(control_path, "control manifest")
        if os.path.exists(control_path) else None)
    replay_problems = _validate_report_protocol_evidence(
        rep, corpus_manifest, corpus_bundle, control_manifest)
    _cold_replay_report(rep, replay_problems)


def _save_checkpoint(directory: str, rep: Report) -> None:
    if not rep.verifier_fingerprint:
        rep.verifier_fingerprint = _verifier_fingerprint()
    rep.source_trace_digest = _source_trace_digest(rep.records)
    _validate_priced_report(rep)
    artifact_io.atomic_json(
        os.path.join(directory, CHECKPOINT_FILE), rep.to_json())


def _report_from_checkpoint_data(data: object) -> Report:
    """Parse and reproduce one closed checkpoint mapping."""
    if not isinstance(data, dict):
        raise RuntimeError("checkpoint root must be a mapping")
    if set(data) != _REPORT_KEYS:
        raise RuntimeError("checkpoint top-level schema keys are invalid")
    if data.get("schema") != REPORT_SCHEMA:
        raise RuntimeError(
            "legacy checkpoint lacks replay-complete predicate evidence; "
            "use a fresh tag")
    raw_records = data.get("records")
    if not isinstance(raw_records, list) or any(
            not isinstance(record, dict) or set(record) != _RECORD_KEYS
            for record in raw_records):
        raise RuntimeError("checkpoint record schema keys are invalid")
    try:
        records = [ProblemRecord(**record) for record in raw_records]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint record schema is invalid") from exc
    report = Report(
        tag=data.get("tag", ""),
        records=records,
        track=data.get("track", "UNRESTRICTED"),
        condition=data.get("condition", P.OBSERVED),
        sharing_policy=data.get("sharing_policy", P.SHARED),
        corpus_digest=data.get("corpus_digest", ""),
        corpus_bundle_digest=data.get("corpus_bundle_digest", ""),
        control_digest=data.get("control_digest", ""),
        schema=data.get("schema", REPORT_SCHEMA),
        label_policy=data.get("label_policy", data.get("condition", P.OBSERVED)),
        source_trace_digest=data.get("source_trace_digest", ""),
        parent_source_trace_digest=data.get("parent_source_trace_digest", ""),
        verifier_fingerprint=data.get("verifier_fingerprint", {}),
        phase_execution_binding=data.get("phase_execution_binding", {}),
        phase_execution_binding_history=data.get(
            "phase_execution_binding_history", []),
    )
    _validate_priced_report(report)
    stored_paid = data.get("paid_node_identities", report.paid_node_identities)
    if list(stored_paid) != report.paid_node_identities:
        raise RuntimeError("checkpoint paid-definition ledger does not reproduce")
    for key, expected in (
        ("solved", report.solved),
        ("total_marginal_C", report.total_marginal_C),
        ("total_definition_charge", report.total_definition_charge),
        ("total_structure_charge", report.total_structure_charge),
        ("total_charge", report.total_charge),
        ("free_energy", report.free_energy),
    ):
        if key in data and data[key] != expected:
            raise RuntimeError(f"checkpoint {key} does not reproduce")
    return report


def _load_checkpoint(
        directory: str, *, filename: str = CHECKPOINT_FILE,
        validate_artifact: bool = True) -> Optional[Report]:
    path = os.path.join(directory, filename)
    if not os.path.lexists(path):
        return None
    try:
        data = _load_stable_json(path, "checkpoint")
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint exists but is unreadable or invalid") from exc
    report = _report_from_checkpoint_data(data)
    if validate_artifact:
        _validate_artifact_checkpoint(directory, report)
    return report


def _pending_promotion_payload(rep: Report, log_bytes: bytes) -> dict:
    """Build one atomically publishable, replay-complete commit marker."""
    if len(log_bytes) > MAX_PROPOSER_LOG_UTF8_BYTES:
        raise RuntimeError("workspace predicates_log.md exceeds its byte limit")
    try:
        log_text = log_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "workspace predicates_log.md is not valid UTF-8") from exc
    log_digest = hashlib.sha256(log_bytes).hexdigest()
    expected_log_digest = (
        rep.records[-1].attempted_log_digest
        if rep.records else _source_digest(""))
    if log_digest != expected_log_digest:
        raise RuntimeError(
            "workspace predicate log differs from the proposer log trace")
    body = {
        "schema": PENDING_PROMOTION_SCHEMA,
        "report": rep.to_json(),
        "predicates_log": log_text,
        "predicates_log_digest": log_digest,
    }
    body["pending_digest"] = _canonical_digest(body)
    return body


def _load_pending_promotion(
        directory: str) -> Optional[tuple[Report, bytes]]:
    """Load one self-contained staged promotion, failing closed on damage."""
    path = os.path.join(directory, PENDING_CHECKPOINT_FILE)
    if not os.path.lexists(path):
        return None
    try:
        data = _load_stable_json(path, "pending promotion")
    except RuntimeError as exc:
        raise RuntimeError(
            "pending promotion exists but is unreadable or invalid") from exc
    if not isinstance(data, dict) or set(data) != _PENDING_PROMOTION_KEYS:
        raise RuntimeError("pending promotion schema keys are invalid")
    if data.get("schema") != PENDING_PROMOTION_SCHEMA:
        raise RuntimeError("pending promotion schema differs")
    body = {key: value for key, value in data.items()
            if key != "pending_digest"}
    try:
        if data.get("pending_digest") != _canonical_digest(body):
            raise RuntimeError("pending promotion digest does not reproduce")
        log_text = data.get("predicates_log")
        if not isinstance(log_text, str):
            raise RuntimeError("pending promotion predicate log is malformed")
        log_bytes = log_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "pending promotion predicate log is not valid UTF-8") from exc
    if len(log_bytes) > MAX_PROPOSER_LOG_UTF8_BYTES:
        raise RuntimeError("pending promotion predicate log exceeds its byte limit")
    log_digest = hashlib.sha256(log_bytes).hexdigest()
    if data.get("predicates_log_digest") != log_digest:
        raise RuntimeError("pending promotion predicate log digest differs")
    report = _report_from_checkpoint_data(data.get("report"))
    expected_log_digest = (
        report.records[-1].attempted_log_digest
        if report.records else _source_digest(""))
    if log_digest != expected_log_digest:
        raise RuntimeError(
            "pending promotion predicate log differs from its report")
    return report, log_bytes


def seed_workspace_from_artifact(
        tag: str, ws: str, verbose: bool = True, *,
        prevalidated_report: Optional[Report] = None,
        checkpoint_preflighted: bool = False) -> Optional[Report]:
    """Scratch is disposable; the promoted artifact is the source of truth.

    Before overwriting, any in-flight scratch state that differs from the
    artifact (an attempt interrupted by power/credit loss) is preserved as a
    WIP snapshot -- the same never-lose-live-context discipline as gkm_legs."""
    _preflight_workspace_owned_paths(ws)
    art = artifact_dir(tag)
    try:
        artifact_info = os.lstat(art)
    except FileNotFoundError:
        artifact_info = None
    if artifact_info is not None:
        if not stat.S_ISDIR(artifact_info.st_mode):
            raise RuntimeError(
                "artifact seed directory must be a non-symlink directory")
        for name in PROMOTED_FILES:
            source = os.path.join(art, name)
            try:
                source_info = os.lstat(source)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
                raise RuntimeError(
                    f"artifact seed {name} must be a non-symlink, singly-linked "
                    "regular file")
    rep = (prevalidated_report if checkpoint_preflighted
           else _load_checkpoint(art))
    if rep is None:
        return None
    if rep.tag != tag:
        raise RuntimeError("checkpoint tag differs from its artifact directory")
    ws_lib = os.path.join(ws, LIBRARY_FILE)
    if os.path.exists(ws_lib) and _read_workspace_predicate_source(ws) != \
            predicate_price.read_predicate_source(
                os.path.join(art, LIBRARY_FILE)):
        marker = _require_regular_workspace_file(
            ws, "current_problem.txt", required=False)
        oid = (_read(marker).strip() if marker is not None else "") or "preseed"
        if oid != "preseed":
            _require_problem_id(oid)
        snapshot_wip(tag, ws, f"interrupted_{oid}", verbose=verbose)
    for name in PROMOTED_FILES:
        src = os.path.join(art, name)
        if os.path.exists(src):
            _atomic_workspace_copy(src, ws, name)
    if verbose:
        print(f"seeded workspace from artifact: {art} "
              f"(solved={rep.solved}, C={rep.total_charge})")
    return rep


def promote_verified_artifact(tag: str, ws: str, rep: Report,
                              results: Dict[str, dict],
                              verbose: bool = True,
                              corpus_manifest: Optional[dict] = None,
                              control_manifest: Optional[dict] = None,
                              corpus_bundle: Optional[dict] = None) -> bool:
    """Publish the current verified library state. Gated on the taint check;
    verification itself is re-run by the caller (pure function = replay)."""
    assert_workspace_not_tainted(ws)
    expected_source = _expected_final_source(rep)
    workspace_library = os.path.join(ws, LIBRARY_FILE)
    if not os.path.isfile(workspace_library) \
            or _read_workspace_predicate_source(ws) != expected_source:
        raise RuntimeError(
            "workspace predicate source differs from the replayed final source")
    expected_log_digest = (
        rep.records[-1].attempted_log_digest
        if rep.records else _source_digest(""))
    workspace_log_bytes = _stable_workspace_file_bytes(
        ws, LOG_FILE, MAX_PROPOSER_LOG_UTF8_BYTES)
    if hashlib.sha256(workspace_log_bytes).hexdigest() != expected_log_digest:
        raise RuntimeError(
            "workspace predicate log differs from the proposer log trace")
    art = artifact_dir(tag)
    if corpus_manifest is not None:
        bind_corpus_manifest_to_artifact(tag, corpus_manifest)
        if corpus_bundle is None:
            raise RuntimeError("corpus-bound promotion requires embedded panel bytes")
        bind_corpus_bundle_to_artifact(tag, corpus_bundle, corpus_manifest)
    if control_manifest is not None:
        if corpus_manifest is None:
            raise RuntimeError("control promotion requires a base corpus manifest")
        bind_control_manifest_to_artifact(
            tag, control_manifest, corpus_manifest)
    os.makedirs(art, exist_ok=True)
    _save_checkpoint(ws, rep)
    pending_payload = _pending_promotion_payload(rep, workspace_log_bytes)
    artifact_io.atomic_json(
        os.path.join(art, PENDING_CHECKPOINT_FILE), pending_payload)
    # Publish immutable bytes captured/reconstructed before the commit marker,
    # never a second read of a proposer-writable path after replay.
    _atomic_text(os.path.join(art, LIBRARY_FILE), expected_source)
    _atomic_bytes(os.path.join(art, LOG_FILE), workspace_log_bytes)
    _save_checkpoint(art, rep)
    # Ground truth stays harness-side: results.json exists ONLY in the
    # artifact dir, never in the workspace.
    artifact_io.atomic_json(os.path.join(art, "results.json"), results)
    _atomic_text(
        os.path.join(art, "README.md"),
            f"# {tag} predicate-library artifact\n\n"
            "Latest verified predicate-library state promoted by "
            "`bongard_legs.py`. Every terminal verdict is cold-replayed from "
            "its exact per-problem `attempted_source` snapshot and embedded "
            "panels; `predicates.py` is the final accepted library snapshot.\n\n"
            f"- Problems solved: {rep.solved}/{len(rep.records)}\n"
            f"- Exact definition charge: {rep.total_definition_charge}\n"
            f"- Call/binding charge: {rep.total_structure_charge}\n"
            f"- Total charge: {rep.total_charge}\n"
            f"- F = {rep.free_energy:.3f}\n\n"
            "Per-problem novelty:\n\n"
            + "\n".join(f"- {r.opaque_id}: solved={r.solved} "
                        f"marginal_C={r.marginal_C} model={r.model}"
                        for r in rep.records) + "\n")
    try:
        os.unlink(os.path.join(art, PENDING_CHECKPOINT_FILE))
    except FileNotFoundError:
        pass
    if verbose:
        print(f"promoted artifact: {art}")
    return True


def _restore_wip_context(tag: str, ws: str, opaque_id: str,
                         verbose: bool = True) -> int:
    """Copy the newest WIP snapshot's proposer files (notes, probe scripts)
    for this problem back into the workspace -- the ARC restore-WIP idiom.
    Promoted files and panels are never restored (the artifact is the
    verified source of truth); newer scratch files are never clobbered."""
    _require_problem_id(opaque_id)
    _require_workspace_directory(ws)
    restored = 0
    skip = set(PROMOTED_FILES) | {
        "bongard_try.py", "current_problem.txt", PRICING_CONTRACT_FILE}
    artifact = artifact_dir(tag)
    try:
        artifact_info = os.lstat(artifact)
    except FileNotFoundError:
        return 0
    if not stat.S_ISDIR(artifact_info.st_mode):
        raise RuntimeError("WIP artifact root is not a real directory")
    wip_root = os.path.join(artifact, "wip_context")
    try:
        wip_info = os.lstat(wip_root)
    except FileNotFoundError:
        return 0
    if not stat.S_ISDIR(wip_info.st_mode):
        raise RuntimeError("WIP context root is not a real directory")
    for kind in (opaque_id, f"interrupted_{opaque_id}"):
        base = os.path.join(wip_root, kind)
        try:
            base_info = os.lstat(base)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(base_info.st_mode):
            raise RuntimeError(f"WIP context {kind} is not a real directory")
        snaps = sorted(os.listdir(base))
        if not snaps:
            continue
        src = os.path.join(base, snaps[-1])
        source_dir_info = os.lstat(src)
        if not stat.S_ISDIR(source_dir_info.st_mode):
            raise RuntimeError("WIP snapshot is not a real directory")
        for name in sorted(os.listdir(src)):
            if name in skip or name.startswith("problem_"):
                continue
            s, d = os.path.join(src, name), os.path.join(ws, name)
            source_info = os.lstat(s)
            if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
                raise RuntimeError(
                    f"WIP source {name} is not a singly-linked regular file")
            try:
                destination_info = os.lstat(d)
            except FileNotFoundError:
                destination_info = None
            if destination_info is not None:
                if not stat.S_ISREG(destination_info.st_mode) \
                        or destination_info.st_nlink != 1:
                    raise RuntimeError(
                        f"WIP destination {name} is not a singly-linked regular file")
                if destination_info.st_mtime >= source_info.st_mtime:
                    continue
            _atomic_workspace_copy(s, ws, name)
            restored += 1
    if verbose and restored:
        print(f"restored {restored} WIP file(s) for {opaque_id}")
    return restored


def _prune_stale_problem_dirs(ws: str, keep_oid: str) -> None:
    """Only the current problem's panels are visible to the proposer; stale
    panel dirs from earlier problems waste its budget and blur the
    information boundary. Only exact, inspected panel files are removed;
    unknown content is never silently deleted."""
    _require_problem_id(keep_oid)
    _require_workspace_directory(ws)
    for name in os.listdir(ws):
        if re.fullmatch(r"problem_[0-9]{2,}", name) is None:
            continue
        path = os.path.join(ws, name)
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"{name} panel directory must be a non-symlink directory")
        if name != keep_oid:
            A.remove_panel_directory(ws, name)


def snapshot_wip(tag: str, ws: str, opaque_id: str, verbose: bool = True) -> str:
    """Preserve a failed attempt's workspace files (including the reverted
    library candidate) without admitting them."""
    if re.fullmatch(
            r"(?:problem_[0-9]{2,}|interrupted_(?:problem_[0-9]{2,}|preseed))",
            opaque_id) is None:
        raise RuntimeError("WIP snapshot id is not canonical")
    _preflight_workspace_owned_paths(ws)
    sources = []
    for name in sorted(os.listdir(ws)):
        path = os.path.join(ws, name)
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(
                f"cannot snapshot non-regular or multiply-linked workspace "
                f"entry {name}")
        sources.append((name, path))
    dst = os.path.join(artifact_dir(tag), "wip_context", opaque_id,
                       time.strftime("%Y%m%dT%H%M%S")
                       + f"_{time.time_ns()}")
    os.makedirs(dst, exist_ok=False)
    for name, path in sources:
        _atomic_workspace_copy(path, dst, name)
    if verbose:
        print(f"saved WIP context: {dst}")
    return dst


def git_checkpoint(tag: str, rep: Report, verbose: bool = True) -> None:
    """Best-effort commit+push of the promoted artifact after each problem,
    so a power/credit interruption never loses more than the in-flight
    attempt even if the machine is lost. Failures (offline, races) are
    logged and ignored -- the run must not die on git."""
    art = artifact_dir(tag)
    repo = os.path.abspath(os.path.join(LAB_DIR, "..", ".."))
    msg = (f"[auto] bongard crack {tag}: solved={rep.solved}/"
           f"{len(rep.records)} C={rep.total_charge}")
    try:
        subprocess.run(["git", "-C", repo, "add", art], check=True,
                       capture_output=True, timeout=60)
        commit = subprocess.run(["git", "-C", repo, "commit", "-m", msg],
                                capture_output=True, timeout=60)
        if commit.returncode == 0:
            subprocess.run(["git", "-C", repo, "push", "origin", "master"],
                           capture_output=True, timeout=120)
            if verbose:
                print(f"git checkpoint pushed: solved={rep.solved}")
    except Exception as exc:  # never let git kill the run
        if verbose:
            print(f"git checkpoint skipped: {exc}")


def interleave_corpus(basic: List[A.Problem], abstract: List[A.Problem],
                      period: int = 5) -> List[A.Problem]:
    """Deterministic curriculum order: one abstract problem every `period`
    slots until abstract is exhausted, then the remaining basic. Stable
    under raising --max-problems: the first N slots never change, so resume
    by opaque index stays aligned as the corpus prefix grows."""
    out: List[A.Problem] = []
    b, a = list(basic), list(abstract)
    while b or a:
        if a and (len(out) % period == period - 1 or not b):
            out.append(a.pop(0))
        elif b:
            out.append(b.pop(0))
    return out


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------

def _predicate_capability_prompt_summary() -> str:
    """Render the full positive capability manifest as canonical JSON."""
    manifest = predicate_price.predicate_capability_manifest()
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_task(
        opaque_id: str, tester_cmd: str,
        sharing_policy: str = P.SHARED) -> str:
    if sharing_policy != P.SHARED:
        raise ValueError(
            "no-share is a held-fixed replay control, not a proposer condition")
    # Kept in the API for compatibility with offline callers. The proposer has
    # no Bash tool, and leaking the parent interpreter path serves no purpose.
    _ = tester_cmd
    _require_problem_id(opaque_id)
    capability_summary = _predicate_capability_prompt_summary()
    return (
        PRECONCEPTIONS
        + "\nThe twelve labeled panel PNGs are attached directly to this turn;"
        " ink is dark on white. No experiment workspace, tester, repository,"
        " file, shell, network, plugin, browser, or sub-agent tool is available.\n\n"
        "OUTCOME:\n"
        "1. Infer the shortest natural rule separating all positive panels\n"
        "   from all negative panels.\n"
        "2. Return a complete EXTENDED `predicates.py`: module-level pure\n"
        "   functions\n"
        "   `p_<name>(panel) -> float | bool` measuring properties of a single\n"
        "   panel. The exact positive capability manifest (canonical JSON) is:\n   "
        + capability_summary + "\n"
        "   Only manifest-allowlisted builtins, calls, values, attributes, and\n"
        "   bounded resource patterns are accepted. Predicates must not read\n"
        "   files or use dynamic namespaces. Validation is fail-closed over the\n"
        "   WHOLE persistent source snapshot: one disallowed construct anywhere\n"
        "   rejects every predicate before execution.\n"
        "3. Refactor near-duplicates into reusable measurements and return a\n"
        "   complete updated `predicates_log.md` describing the recurring\n"
        "   pattern. There is no same-turn test rerun.\n"
        "4. Only the outer harness evaluates the structured proposal after\n"
        "   this turn. It privately composes a bounded minimal rule and runs\n"
        "   rotated leave-one-out; you never write or run that rule. Feedback\n"
        "   may appear on a later attempt. Do not claim evaluation you cannot\n"
        "   perform.\n\n"
        "GROW A LIBRARY (this is the point -- minimise novelty):\n"
        "- `predicates.py` is a persistent SHARED library that carries over to\n"
        "  later problems. REUSE existing predicates where possible; add as FEW\n"
        "  new ones as possible. Exact definitions already USED by an earlier\n"
        "  accepted rule are paid; unused or changed code is charged on first\n"
        "  use. Selection minimizes risk, then transitive definition + call +\n"
        "  binding cost (conditional MDL).\n"
        "- Preserve existing accepted measurements unless a genuine shared\n"
        "  refactor is required; the entire library is validated together.\n\n"
        "CONSTRAINTS: reason only from the attached panels and supplied current\n"
        "library/log. Pixel hashes, exact templates, coordinate fingerprints, or\n"
        "any lookup keyed to these 12 panel identities are forbidden. Because\n"
        "you see all panels before defining predicates, rotated leave-one-out\n"
        "does NOT detect such memorization; it would invalidate the evidence.\n")


def _validate_proposer_workspace(
        ws: str, opaque_id: Optional[str] = None) -> str:
    """Validate every path the proposer can read or edit."""
    _preflight_workspace_owned_paths(ws)
    marker = _require_regular_workspace_file(
        ws, "current_problem.txt", required=True)
    current = _read(marker).strip()
    _require_problem_id(current)
    if opaque_id is not None and current != opaque_id:
        raise RuntimeError("current problem marker differs from active problem")
    _require_regular_workspace_file(ws, LIBRARY_FILE, required=True)
    _require_regular_workspace_file(ws, LOG_FILE, required=True)
    library_info = os.lstat(os.path.join(ws, LIBRARY_FILE))
    log_info = os.lstat(os.path.join(ws, LOG_FILE))
    if library_info.st_size > predicate_price.MAX_SOURCE_UTF8_BYTES:
        raise RuntimeError("workspace predicates.py exceeds its byte limit")
    if log_info.st_size > MAX_PROPOSER_LOG_UTF8_BYTES:
        raise RuntimeError("workspace predicates_log.md exceeds its byte limit")
    A.validate_panel_directory(ws, current)
    return current


def _proposer_allowed_tools(opaque_id: str) -> List[str]:
    _require_problem_id(opaque_id)
    readable = [
        f"Read(./{LIBRARY_FILE})",
        f"Read(./{LOG_FILE})",
        f"Read(./{opaque_id})",
    ]
    readable.extend(
        f"Read(./{opaque_id}/{name})"
        for name in A.canonical_panel_filenames())
    return readable + [
        f"Edit(./{LIBRARY_FILE})",
        f"Edit(./{LOG_FILE})",
    ]


def _strict_claude_result(stdout: str, requested_model: str) -> ProposerOutcome:
    if not isinstance(stdout, str) or not stdout \
            or len(stdout.encode("utf-8")) > 2_000_000:
        raise ProposerInfrastructureFailure(
            "Claude Code returned an empty or oversized JSON result")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            stdout,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ProposerInfrastructureFailure(
            f"Claude Code returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict) \
            or payload.get("type") != "result" \
            or payload.get("subtype") != "success" \
            or payload.get("is_error") is not False \
            or not isinstance(payload.get("result"), str):
        raise ProposerInfrastructureFailure(
            "Claude Code returned a non-success result envelope")
    denials = payload.get("permission_denials")
    if not isinstance(denials, list):
        raise ProposerInfrastructureFailure(
            "Claude Code result lacks permission-denial evidence")
    model_usage = payload.get("modelUsage")
    if not isinstance(model_usage, dict) or set(model_usage) != {requested_model}:
        raise ProposerInfrastructureFailure(
            "Claude Code actual model differs from the requested model")
    usage = model_usage[requested_model]
    if not isinstance(usage, dict):
        raise ProposerInfrastructureFailure(
            "Claude Code model usage entry is malformed")
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) \
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) \
            or input_tokens < 0 or output_tokens < 0 \
            or input_tokens + output_tokens <= 0:
        raise ProposerInfrastructureFailure(
            "Claude Code returned no positive token usage")
    outcome = "permission-denied" if denials else "success"
    receipt = _build_proposer_receipt(
        source="claude-cli", requested_model=requested_model,
        actual_model=requested_model, input_tokens=input_tokens,
        output_tokens=output_tokens, model_usage=model_usage,
        outcome=outcome, permission_denials=denials)
    return ProposerOutcome(payload["result"], receipt)


def claude_propose(task: str, ws: str, model: str, minutes: int = 15,
                   verbose: bool = True) -> ProposerOutcome:
    """Run Claude Code with a fail-closed workspace-only tool surface.

    The model can inspect the supplied panels and edit the predicate library,
    but cannot launch a shell, access the network, traverse outside ``ws``,
    load project/user plugins, or bypass permissions.  The harness performs
    authoritative evaluation after the process exits.
    """
    opaque_id = _validate_proposer_workspace(ws)
    allowed_tools = _proposer_allowed_tools(opaque_id)
    denied_tools = ["Write", "Bash", "WebFetch", "WebSearch", "Agent"]
    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": allowed_tools,
            "deny": denied_tools,
        },
    }
    cmd = ["claude", "-p", task,
           "--safe-mode",
           "--disable-slash-commands",
           "--no-chrome",
           "--no-session-persistence",
           "--strict-mcp-config", "--mcp-config", "{}",
           "--settings", json.dumps(settings, sort_keys=True),
           "--tools", "Read,Edit",
           "--allowedTools", *allowed_tools,
           "--disallowedTools", *denied_tools,
           "--permission-mode", "dontAsk",
           "--model", model,
           "--output-format", "json"]
    if verbose:
        print(f"invoking {model} proposer in {ws} (up to {minutes} min)")
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                              timeout=minutes * 60)
        if verbose:
            print("=== proposer transcript (tail) ===")
            print((proc.stdout or "")[-1500:])
        if proc.returncode != 0:
            detail = ((proc.stderr or proc.stdout or "").strip()[-500:]
                      or "no diagnostic output")
            raise ProposerInfrastructureFailure(
                f"Claude Code exited {proc.returncode}: {detail}")
        return _strict_claude_result(proc.stdout or "", model)
    except subprocess.TimeoutExpired as exc:
        raise ProposerInfrastructureFailure(
            f"Claude Code timed out after {minutes} min") from exc
    except OSError as exc:
        raise ProposerInfrastructureFailure(
            f"Claude Code could not be launched: {exc}") from exc


def codex_propose(task: str, ws: str, model: str, minutes: int = 15,
                  verbose: bool = True) -> ProposerOutcome:
    """Run the production schema-only Codex proposer and apply its response.

    Codex never receives ``ws``.  The transport copies only the twelve PNGs
    into a separate private view; current source/log bytes travel in the
    prompt, and this function applies both returned texts only after the turn,
    receipt, panel view, schema, and unchanged harness workspace validate.
    """
    opaque_id = _validate_proposer_workspace(ws)
    before = _snapshot_proposer_files(ws)
    try:
        current_source = before[LIBRARY_FILE].decode("utf-8", errors="strict")
        current_log = before[LOG_FILE].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProposerInfrastructureFailure(
            "Codex proposer workspace text is not valid UTF-8") from exc
    panel_dir = os.path.join(ws, opaque_id)
    panel_paths = [
        os.path.abspath(os.path.join(panel_dir, f"{side}_{index}.png"))
        for side in ("pos", "neg") for index in range(6)
    ]
    try:
        proposal = codex_headless.run_codex_proposer(
            task,
            panel_paths,
            current_source,
            current_log,
            model=model,
            minutes=minutes,
            verbose=verbose,
            reasoning_effort=codex_headless.DEFAULT_REASONING_EFFORT,
        )
    except codex_headless.CodexProposerFailure as exc:
        raise ProposerInfrastructureFailure(str(exc)) from exc
    receipt = proposal.receipt.to_dict()
    _validate_proposer_receipt(receipt)
    expected_bindings = {
        "task_digest": _source_digest(task),
        "current_source_digest": _source_digest(current_source),
        "current_log_digest": _source_digest(current_log),
        "proposed_source_digest": _source_digest(
            proposal.predicates_source),
        "proposed_log_digest": _source_digest(proposal.predicates_log),
        "panel_view_digest": codex_headless.ordered_panel_view_digest(
            panel_paths),
        "panel_set_digest": codex_headless.semantic_panel_set_digest(
            panel_paths),
        "input_digest": codex_headless.predicate_proposer_input_digest(
            task, current_source, current_log, panel_paths),
        "structured_output_digest": (
            codex_headless.predicate_proposer_output_digest(
                proposal.predicates_source,
                proposal.predicates_log,
                proposal.rationale,
            )
        ),
    }
    if any(receipt.get(name) != value
           for name, value in expected_bindings.items()):
        raise ProposerInfrastructureFailure(
            "Codex receipt differs from the exact proposer input/output")
    if _snapshot_proposer_files(ws) != before:
        try:
            _restore_proposer_files(ws, before)
        except BaseException as restore_exc:
            raise RuntimeError(
                "isolated Codex workspace-change rollback failed") \
                from restore_exc
        raise ProposerInfrastructureFailure(
            "harness workspace changed during isolated Codex turn")
    candidate_text = (
        proposal.predicates_source + "\n" + proposal.predicates_log).lower()
    for marker in SOURCE_TAINT_MARKERS:
        if marker in candidate_text:
            raise WorkspaceTainted(
                "Codex structured proposal contains forbidden dataset/"
                f"ground-truth marker: {marker}")
    try:
        _atomic_workspace_text(ws, LIBRARY_FILE, proposal.predicates_source)
        _atomic_workspace_text(ws, LOG_FILE, proposal.predicates_log)
        after = _snapshot_proposer_files(ws)
        if after != {
                LIBRARY_FILE: proposal.predicates_source.encode("utf-8"),
                LOG_FILE: proposal.predicates_log.encode("utf-8"),
        }:
            raise RuntimeError("applied Codex proposal bytes differ")
    except BaseException as exc:
        try:
            _restore_proposer_files(ws, before)
        except BaseException as restore_exc:
            raise RuntimeError(
                "Codex proposal application and rollback both failed") \
                from restore_exc
        raise ProposerInfrastructureFailure(
            f"could not apply Codex structured proposal: {exc}") from exc
    return ProposerOutcome(
        transcript=proposal.rationale,
        receipt=receipt,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _verify_workspace(
        ws: str, problem: A.Problem, pricing_contract: dict) -> A.VerifyResult:
    lib = os.path.join(ws, LIBRARY_FILE)
    if not os.path.exists(lib):
        source = ""
    else:
        source = _read_workspace_predicate_source(ws)
    return _verify_source_snapshot(
        source, problem, pricing_contract, filename=lib)


def _verification_failure(sharing_policy: str) -> A.VerifyResult:
    return A.VerifyResult(
        False, 0.0, 0.0, "PRICING_OR_LOAD_ERROR", 0.0, 12, 36,
        sharing_policy=sharing_policy,
        selection_policy=A.PRICED_SELECTION_POLICY)


def _current_virtual_memory_bytes() -> int:
    """Measure this process's mapped virtual size on supported platforms."""
    system = platform.system()
    if system == "Linux":
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open("/proc/self/statm", flags)
        try:
            raw = os.read(descriptor, 256)
        finally:
            os.close(descriptor)
        fields_value = raw.split()
        if not fields_value:
            raise RuntimeError("Linux virtual-memory probe returned no fields")
        pages = int(fields_value[0])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        measured = pages * page_size
    elif system == "Darwin":
        class TimeValue(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int32),
                ("microseconds", ctypes.c_int32),
            ]

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", TimeValue),
                ("system_time", TimeValue),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        mach_task_self = library.mach_task_self
        mach_task_self.argtypes = []
        mach_task_self.restype = ctypes.c_uint32
        task_info = library.task_info
        task_info.argtypes = [
            ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(
            ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        status = task_info(
            mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count))
        if status != 0:
            raise RuntimeError(
                f"Darwin virtual-memory probe failed with status {status}")
        measured = int(info.virtual_size)
    else:
        raise RuntimeError(
            f"unsupported verifier resource-limit platform {system!r}")
    if measured <= 0:
        raise RuntimeError("virtual-memory probe returned a nonpositive size")
    return measured


def _set_child_resource_limit(kind: int, target: int) -> int:
    """Install the exact fingerprinted ceiling or fail before verification."""
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise RuntimeError("child resource target must be a positive integer")
    soft, hard = resource.getrlimit(kind)
    for existing in (soft, hard):
        if existing != resource.RLIM_INFINITY:
            if isinstance(existing, bool) or not isinstance(existing, int) \
                    or existing < 0:
                raise RuntimeError("existing child resource limit is invalid")
            if existing < target:
                raise RuntimeError(
                    "inherited child resource limit is below the "
                    "fingerprinted target")
    resource.setrlimit(kind, (target, target))
    return target


def _apply_verifier_resource_limits() -> dict:
    """Install CPU and memory ceilings inside the fresh verifier child."""
    policy = A.verifier_resource_limit_policy()
    cpu_limit = policy["child_cpu_limit_seconds"]
    parent_timeout = policy["parent_wall_timeout_seconds"]
    if isinstance(cpu_limit, bool) or not isinstance(cpu_limit, int) \
            or not 0 < cpu_limit < parent_timeout:
        raise RuntimeError("child CPU limit must be below the parent timeout")
    virtual_size = _current_virtual_memory_bytes()
    memory_cap = virtual_size + policy["memory_headroom_bytes"]
    applied = {
        "RLIMIT_CPU": _set_child_resource_limit(resource.RLIMIT_CPU, cpu_limit),
        "RLIMIT_AS": _set_child_resource_limit(resource.RLIMIT_AS, memory_cap),
        "RLIMIT_DATA": _set_child_resource_limit(
            resource.RLIMIT_DATA, memory_cap),
    }
    return {
        "virtual_memory_bytes": virtual_size,
        "derived_memory_cap_bytes": memory_cap,
        "applied": applied,
    }


class _PredicateLineBudgetExceeded(RuntimeError):
    """Internal control signal for deterministic proposer-code metering."""


def _predicate_line_event_trace(filename: str, maximum_events: int):
    """Trace only frames compiled from the immutable predicate source."""
    if not isinstance(filename, str) or not filename \
            or isinstance(maximum_events, bool) \
            or not isinstance(maximum_events, int) or maximum_events <= 0:
        raise RuntimeError("predicate line-event budget is invalid")
    events = 0

    def local_trace(frame, event, argument):
        nonlocal events
        if event == "line":
            events += 1
            if events > maximum_events:
                raise _PredicateLineBudgetExceeded(
                    "predicate Python line-event budget exceeded")
        return local_trace

    def dispatch(frame, event, argument):
        if event == "call" and frame.f_code.co_filename == filename:
            return local_trace
        return None

    return dispatch


def _verification_worker(
        sender: object, source: str, problem: A.Problem,
        pricing_contract: dict, filename: str) -> None:
    """Fresh-process verifier target; never expose exception text as evidence."""
    try:
        _apply_verifier_resource_limits()
        line_limit = A.verifier_resource_limit_policy()[
            "predicate_python_line_event_limit"]
        sys.settrace(_predicate_line_event_trace(filename, line_limit))
        try:
            result = A.verify_priced_source(
                source,
                problem,
                sharing_policy=pricing_contract["sharing_policy"],
                paid_node_identities=pricing_contract["paid_node_identities"],
                filename=filename,
            )
        finally:
            sys.settrace(None)
        sender.send((True, result))  # type: ignore[attr-defined]
    except BaseException:
        try:
            sender.send((False, None))  # type: ignore[attr-defined]
        except BaseException:
            pass
    finally:
        try:
            sender.close()  # type: ignore[attr-defined]
        except BaseException:
            pass


def _verify_source_snapshot(
        source: str, problem: A.Problem, pricing_contract: dict,
        *, filename: str = "<replayed_predicates>") -> A.VerifyResult:
    _validate_pricing_contract(pricing_contract)
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context(
        "fork" if "fork" in methods else "spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_verification_worker,
        args=(sender, source, problem, pricing_contract, filename),
        daemon=True,
    )
    try:
        process.start()
        sender.close()
        # Drain the one-way pipe while the child is alive.  Joining first can
        # deadlock when a valid definition receipt exceeds the OS pipe buffer:
        # the child waits for a reader while the parent waits for child exit.
        deadline = time.monotonic() + AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS
        message = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if receiver.poll(min(0.1, max(0.0, remaining))):
                message = receiver.recv()
                break
            if not process.is_alive():
                break
        process.join(2.0 if message is not None else 0.0)
        if message is None and process.is_alive():
            process.terminate()
            process.join(2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(2.0)
            return _verification_failure(pricing_contract["sharing_policy"])
        if message is not None:
            if process.is_alive():
                process.terminate()
                process.join(2.0)
            ok, result = message
            if ok and isinstance(result, A.VerifyResult):
                return result
        return _verification_failure(pricing_contract["sharing_policy"])
    except Exception:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        return _verification_failure(pricing_contract["sharing_policy"])
    finally:
        receiver.close()


def _assert_replayed_result(
        record: ProblemRecord, result: A.VerifyResult, *,
        pricing_context_digest: str, source_digest: str,
        compare_admitted_pricing: bool) -> None:
    observed_digest = _verification_digest(
        result,
        source_digest=source_digest,
        pricing_context_digest=pricing_context_digest,
        proposer_receipts_digest=_proposer_receipts_digest(
            record.proposer_receipts),
    )
    if observed_digest != record.source_verification_digest:
        raise RuntimeError(
            f"{record.opaque_id} cold-replay verification digest differs")
    common_pairs = (
        ("solved", record.solved, result.solved),
        ("heldout_accuracy", record.heldout_accuracy,
         result.heldout_accuracy),
        ("train_accuracy", record.train_accuracy, result.train_accuracy),
        ("rule", record.rule, result.rule),
        ("predicate_errors", record.predicate_errors,
         result.predicate_errors),
        ("n_rotations", record.n_rotations, result.n_rotations),
        ("predicate_names", record.predicate_names,
         list(result.predicate_names)),
        ("rule_atoms", record.rule_atoms, _rule_atoms(result.selected_rule)),
        ("fold_rule_atoms", record.fold_rule_atoms,
         [_rule_atoms(rule) for rule in result.fold_rules]),
    )
    for name, stored, replayed in common_pairs:
        if stored != replayed:
            raise RuntimeError(
                f"{record.opaque_id} cold-replay {name} differs")
    if compare_admitted_pricing:
        if record.rule_cost != result.rule_cost:
            raise RuntimeError(
                f"{record.opaque_id} cold-replay rule cost differs")
        if record.solved:
            pricing_pairs = (
                ("definition nodes", record.used_definition_nodes,
                 _definition_nodes(result)),
                ("charged definitions",
                 record.charged_definition_node_identities,
                 list(result.charged_definition_node_identities)),
                ("reused definitions",
                 record.reused_definition_node_identities,
                 list(result.reused_definition_node_identities)),
                ("definition charge", record.definition_charge,
                 result.definition_cost),
                ("full definition cost", record.full_definition_cost,
                 result.full_definition_cost),
                ("structure charge", record.structure_charge,
                 result.structure_cost),
            )
            for name, stored, replayed in pricing_pairs:
                if stored != replayed:
                    raise RuntimeError(
                        f"{record.opaque_id} cold-replay {name} differs")


def _cold_replay_report(
        rep: Report, problems: Sequence[A.Problem]) -> None:
    """Replay every terminal record in source order from exact candidate bytes."""
    _validate_priced_report(rep)
    if len(problems) < len(rep.records):
        raise RuntimeError("cold replay has fewer panels than checkpoint records")
    current_source = INITIAL_LIBRARY_SOURCE
    source_paid: set[str] = set()
    for record in rep.records:
        index = _problem_index(record.opaque_id)
        baseline_digest = _source_digest(current_source)
        source_context = _pricing_context(
            P.SHARED, sorted(source_paid), baseline_digest)
        if rep.sharing_policy == P.SHARED \
                and source_context["context_digest"] != \
                record.pricing_context_digest:
            raise RuntimeError(
                f"{record.opaque_id} replay pricing context differs")
        if record.status == VERIFIER_FAILURE_STATUS:
            # A verifier failure is a specific canonical zero-admission
            # result, not a bucket for arbitrary unsolved outcomes.  Check
            # both the stored sentinel and exact fresh reproduction so a
            # normal failure or solution cannot be downgraded by resealing
            # local JSON.
            _assert_replayed_result(
                record,
                _verification_failure(source_context["sharing_policy"]),
                pricing_context_digest=source_context["context_digest"],
                source_digest=record.attempted_source_digest,
                compare_admitted_pricing=rep.sharing_policy == P.SHARED,
            )
        result = _verify_source_snapshot(
            record.attempted_source,
            problems[index],
            source_context,
            filename=f"{record.opaque_id}/attempted_predicates.py",
        )
        if record.status == VERIFIER_FAILURE_STATUS:
            _assert_replayed_result(
                record,
                result,
                pricing_context_digest=source_context["context_digest"],
                source_digest=record.attempted_source_digest,
                compare_admitted_pricing=rep.sharing_policy == P.SHARED,
            )
        else:
            _assert_replayed_result(
                record,
                result,
                pricing_context_digest=source_context["context_digest"],
                source_digest=record.attempted_source_digest,
                compare_admitted_pricing=rep.sharing_policy == P.SHARED,
            )
        if record.solved:
            source_paid.update(result.used_definition_node_identities)
            current_source = record.accepted_source


def run(problems: Sequence[A.Problem], tag: str = "logo",
        ws: Optional[str] = None,
        propose_fn: Callable[[str, str, str, int], object] = None,
        ladder: Sequence[str] = DEFAULT_LADDER,
        minutes: int = 15, verbose: bool = True,
        git_checkpoints: bool = False,
        infra_wait_seconds: int = DEFAULT_INFRA_WAIT_SECONDS,
        max_infra_waits: int = DEFAULT_MAX_INFRA_WAITS,
        restore_wip: bool = True,
        corpus_manifest: Optional[dict] = None,
        corpus_bundle: Optional[dict] = None,
        condition: str = P.OBSERVED,
        control_manifest: Optional[dict] = None,
        base_problems: Optional[Sequence[A.Problem]] = None,
        phase_execution_binding: Optional[dict] = None,
        phase_predecessor_execution_binding: Optional[dict] = None,
        phase_execution_binding_history: Optional[Sequence[dict]] = None) \
        -> Report:
    """PROPOSE -> VERIFY -> DEBRIEF over a problem sequence, with structural
    admission and bounded Codex retry escalation. Resumable: solved problems in the
    promoted artifact are not re-run."""
    active_phase_binding, predecessor_phase_binding = \
        _validate_phase_run_configuration(
            tag=tag, ladder=ladder, minutes=minutes,
            infra_wait_seconds=infra_wait_seconds,
            max_infra_waits=max_infra_waits,
            restore_wip=restore_wip,
            phase_execution_binding=phase_execution_binding,
            phase_predecessor_execution_binding=(
                phase_predecessor_execution_binding),
        )
    expected_phase_history = list(
        phase_execution_binding_history
        if phase_execution_binding_history is not None else
        ([active_phase_binding] if active_phase_binding else []))
    if active_phase_binding:
        if not expected_phase_history \
                or expected_phase_history[-1] != active_phase_binding \
                or (predecessor_phase_binding and (
                    len(expected_phase_history) < 2
                    or expected_phase_history[-2] != predecessor_phase_binding)):
            raise RuntimeError(
                "Phase execution binding history differs from current/predecessor")
        for binding in expected_phase_history:
            P.validate_execution_binding(binding)
    elif expected_phase_history:
        raise RuntimeError("unbound run carries Phase execution history")
    if condition not in {P.OBSERVED, P.SHUFFLED_SIDES}:
        if condition == P.NO_SHARE:
            raise ValueError(
                "no-share is derived from a held-fixed shared source trace; "
                "it cannot launch a fresh proposer run")
        raise ValueError("unsupported unrestricted experiment condition")
    if not ladder:
        raise ValueError("proposer ladder must contain at least one model")
    if isinstance(infra_wait_seconds, bool) \
            or not isinstance(infra_wait_seconds, int) \
            or infra_wait_seconds < 0 \
            or isinstance(max_infra_waits, bool) \
            or not isinstance(max_infra_waits, int) \
            or max_infra_waits < 0:
        raise ValueError("infrastructure retry controls must be non-negative integers")
    if corpus_manifest is not None:
        if corpus_bundle is None:
            raise ValueError("frozen corpus run requires embedded corpus_bundle")
        P.validate_corpus_bundle(corpus_bundle, corpus_manifest)
        if condition == P.OBSERVED:
            if control_manifest is not None:
                raise ValueError("observed condition must not carry a control manifest")
            P.assert_corpus_prefix_matches(corpus_manifest, problems)
        else:
            if control_manifest is None:
                raise ValueError("shuffled-side condition requires a control manifest")
            if base_problems is None:
                raise ValueError("shuffled-side condition requires base_problems")
            P.assert_corpus_matches(corpus_manifest, base_problems)
            P.assert_shuffled_control_prefix_matches(
                control_manifest, corpus_manifest, problems)
    elif corpus_bundle is not None or control_manifest is not None \
            or condition != P.OBSERVED:
        raise ValueError("controlled run requires a frozen base corpus manifest")
    preflight_prior = _preflight_artifact_binding(
        tag,
        corpus_manifest=corpus_manifest,
        corpus_bundle=corpus_bundle,
        control_manifest=control_manifest,
        problems=problems,
        condition=condition,
        phase_execution_binding=active_phase_binding,
        phase_predecessor_execution_binding=predecessor_phase_binding,
        phase_execution_binding_history=expected_phase_history,
    )
    ws = ws if ws is not None else _private_default_workspace(tag)
    _require_workspace_directory(ws, create=True)
    _preflight_workspace_owned_paths(ws)
    if corpus_manifest is not None:
        bind_corpus_manifest_to_artifact(tag, corpus_manifest)
        bind_corpus_bundle_to_artifact(tag, corpus_bundle, corpus_manifest)
        if control_manifest is not None:
            bind_control_manifest_to_artifact(
                tag, control_manifest, corpus_manifest)
    propose = propose_fn or (lambda task, w, model, mins:
                             codex_propose(task, w, model, mins, verbose))
    prior = seed_workspace_from_artifact(
        tag, ws, verbose=verbose,
        prevalidated_report=preflight_prior,
        checkpoint_preflighted=True)
    if prior is not None and prior.tag != tag:
        raise RuntimeError("checkpoint tag differs from its artifact directory")
    rep = prior if prior is not None else Report(
        tag=tag, verifier_fingerprint=_verifier_fingerprint(),
        phase_execution_binding=active_phase_binding,
        phase_execution_binding_history=expected_phase_history)
    prior_phase_binding = dict(rep.phase_execution_binding)
    if active_phase_binding and prior_phase_binding != active_phase_binding:
        if prior_phase_binding != predecessor_phase_binding:
            raise RuntimeError(
                "checkpoint is not bound to the immediate Phase predecessor")
        rep.phase_execution_binding_history = expected_phase_history
    elif not active_phase_binding:
        rep.phase_execution_binding_history = []
    rep.phase_execution_binding = active_phase_binding
    if corpus_manifest is None:
        if rep.corpus_digest:
            raise RuntimeError(
                "checkpoint is corpus-bound but this run supplied no manifest")
    else:
        digest = corpus_manifest["corpus_digest"]
        if rep.records and not rep.corpus_digest:
            raise RuntimeError(
                "legacy checkpoint has no corpus identity; use a fresh tag")
        if rep.corpus_digest and rep.corpus_digest != digest:
            raise RuntimeError("checkpoint belongs to a different frozen corpus")
        if rep.corpus_bundle_digest and rep.corpus_bundle_digest != \
                corpus_bundle["bundle_digest"]:
            raise RuntimeError("checkpoint belongs to different embedded corpus bytes")
        if rep.records and (rep.condition != condition
                            or rep.sharing_policy != P.SHARED):
            raise RuntimeError("checkpoint belongs to a different experiment arm")
        if any(_problem_index(record.opaque_id) >= len(problems)
               for record in rep.records):
            raise RuntimeError(
                "active corpus prefix is shorter than the existing checkpoint")
        rep.corpus_digest = digest
        rep.corpus_bundle_digest = corpus_bundle["bundle_digest"]
        rep.track = "UNRESTRICTED"
        rep.condition = condition
        rep.label_policy = condition
        rep.sharing_policy = P.SHARED
        rep.control_digest = (
            control_manifest["control_digest"]
            if control_manifest is not None else "")
        control_path = os.path.join(artifact_dir(tag), "control_manifest.json")
        if condition == P.OBSERVED and os.path.exists(control_path):
            raise RuntimeError(
                "observed run cannot reuse a tag containing a control manifest")
    rep.schema = REPORT_SCHEMA
    rep.track = "UNRESTRICTED"
    rep.condition = condition
    rep.label_policy = condition
    rep.sharing_policy = P.SHARED
    rep.control_digest = (
        control_manifest["control_digest"]
        if control_manifest is not None else "")
    if not rep.verifier_fingerprint:
        rep.verifier_fingerprint = _verifier_fingerprint()
    paid_node_identities = set(_validate_priced_report(rep))
    if corpus_manifest is not None:
        replay_problems = _validate_report_protocol_evidence(
            rep, corpus_manifest, corpus_bundle, control_manifest)
        _cold_replay_report(rep, replay_problems)
    else:
        _cold_replay_report(rep, problems)
    # Scientific outcomes are terminal. Retrying a prior failure after later
    # library growth would expose future definitions and invalidate ordering.
    done = {record.opaque_id for record in rep.records}

    art = artifact_dir(tag)
    results_path = os.path.join(art, "results.json")
    existing_results: object = {}
    if os.path.exists(results_path):
        try:
            existing_results = json.loads(_read(results_path) or "{}")
        except json.JSONDecodeError:
            existing_results = {}
    truth_problems = base_problems if base_problems is not None else problems
    results = _reconcile_results(
        rep,
        existing_results,
        truth_problems=truth_problems,
        corpus_manifest=corpus_manifest,
    )
    if rep.records:
        artifact_io.atomic_json(results_path, results)

    _write_tester(ws)
    lib_path = os.path.join(ws, LIBRARY_FILE)
    if _require_regular_workspace_file(
            ws, LIBRARY_FILE, required=False) is None:
        _atomic_workspace_text(ws, LIBRARY_FILE, INITIAL_LIBRARY_SOURCE)
    if _require_regular_workspace_file(ws, LOG_FILE, required=False) is None:
        _atomic_workspace_text(ws, LOG_FILE, "")

    tester_cmd = f"{sys.executable} bongard_try.py"
    for k, problem in enumerate(problems):
        truth_problem = base_problems[k] if base_problems is not None else problem
        oid = f"problem_{k:02d}"
        if oid in done:
            continue
        _prune_stale_problem_dirs(ws, oid)
        if restore_wip:
            _restore_wip_context(tag, ws, oid, verbose=verbose)
        A.write_panels(ws, problem, oid)
        _atomic_workspace_text(ws, "current_problem.txt", oid)
        lib_before = _read_workspace_predicate_source(ws)
        baseline_source_digest = _source_digest(lib_before)
        log_before = _stable_workspace_file_bytes(
            ws, LOG_FILE, MAX_PROPOSER_LOG_UTF8_BYTES).decode(
                "utf-8", errors="strict")
        baseline_log_digest = _source_digest(log_before)
        proposer_panel_set_digest = semantic_replay.panel_set_digest(
            semantic_replay.panel_records_from_problem(problem))
        pricing_contract = _pricing_context(
            P.SHARED, sorted(paid_node_identities), baseline_source_digest)
        _write_pricing_contract(ws, pricing_contract)

        result = None
        model_used = ladder[0]
        attempts = 0
        rung = 0
        infra_waits = 0
        infra_stop = False
        prior_harness_feedback = ""
        proposer_receipts: List[dict] = []
        proposer_feedback: List[str] = []
        while rung < len(ladder):
            model = ladder[rung]
            _validate_proposer_workspace(ws, oid)
            # Restore the canonical tester context before every adaptive round.
            _write_tester(ws)
            _write_pricing_contract(ws, pricing_contract)
            attempt_task = build_task(oid, tester_cmd, P.SHARED)
            if prior_harness_feedback:
                attempt_task += (
                    "\nAUTHORITATIVE FEEDBACK FROM THE PREVIOUS ATTEMPT:\n"
                    + prior_harness_feedback + "\n")
            _validate_proposer_workspace(ws, oid)
            pre_call_files = _snapshot_proposer_files(ws)
            infrastructure_failure = None
            try:
                raw_outcome = propose(attempt_task, ws, model, minutes)
            except ProposerInfrastructureFailure as exc:
                raw_outcome = None
                infrastructure_failure = str(exc)
            finally:
                # Validate injected and production proposers alike before any
                # transcript handling, retry, source read, or harness write.
                try:
                    _validate_proposer_workspace(ws, oid)
                except BaseException:
                    # A regular oversized/invalid edit is safely recoverable;
                    # linked replacements remain untouched if restoration is
                    # itself refused by the path validator.
                    try:
                        _restore_proposer_files(ws, pre_call_files)
                    except BaseException:
                        pass
                    raise
            if infrastructure_failure is not None:
                _restore_proposer_files(ws, pre_call_files)
                infra_waits += 1
                if infra_waits > max_infra_waits:
                    print(f"{oid}: proposer infrastructure still failing after "
                          f"{max_infra_waits} waits; stopping run (resumable)")
                    infra_stop = True
                    break
                if verbose:
                    print(f"{oid}: proposer infra failure "
                          f"({infra_waits}/{max_infra_waits}); retrying same "
                          f"rung in {infra_wait_seconds}s"
                          + (f": {infrastructure_failure}"
                             if infrastructure_failure else ""))
                time.sleep(infra_wait_seconds)
                continue
            if isinstance(raw_outcome, ProposerOutcome):
                outcome = raw_outcome
                _validate_proposer_receipt(outcome.receipt)
                if outcome.receipt["requested_model"] != model:
                    raise RuntimeError(
                        "proposer receipt requested a different ladder model")
            elif raw_outcome is None or isinstance(raw_outcome, str):
                outcome = ProposerOutcome(
                    raw_outcome or "", _injected_proposer_receipt(model))
            else:
                raise RuntimeError("proposer returned an unsupported result type")
            attempts += 1
            model_used = model
            rung += 1
            proposer_receipts.append(outcome.receipt)
            proposer_feedback.append(prior_harness_feedback)
            if outcome.receipt["outcome"] == "permission-denied":
                _restore_proposer_files(ws, pre_call_files)
                assert_workspace_not_tainted(ws)
                attempted_source_snapshot = _read_workspace_predicate_source(ws)
                # A denied edit leaves the exact baseline source in place.
                # Verify that source instead of inventing an administrative
                # sentinel: an already-admitted library may legitimately
                # solve the next problem, and ordinary baseline failures have
                # ordinary replayable risk evidence.
                result = _verify_source_snapshot(
                    attempted_source_snapshot,
                    problem,
                    pricing_contract,
                    filename=lib_path,
                )
                prior_harness_feedback = "PROPOSER_PERMISSION_DENIED"
                if verbose:
                    print(f"{oid} attempt {attempts} ({model}): "
                          "permission denied; consuming rung")
                if result.solved:
                    break
                continue
            assert_workspace_not_tainted(ws)
            _write_pricing_contract(ws, pricing_contract)
            attempted_source_snapshot = _read_workspace_predicate_source(ws)
            result = _verify_source_snapshot(
                attempted_source_snapshot,
                problem,
                pricing_contract,
                filename=lib_path,
            )
            if verbose:
                print(f"{oid} attempt {attempts} ({model}): {result.result_line()}")
            if result.solved:
                break
            prior_harness_feedback = result.result_line()
        if infra_stop:
            # No verdict: the exact pre-call library and log were restored.
            # Retain this private/caller workspace for a resumable relaunch.
            break

        lib_after = attempted_source_snapshot
        log_after = _stable_workspace_file_bytes(
            ws, LOG_FILE, MAX_PROPOSER_LOG_UTF8_BYTES).decode(
                "utf-8", errors="strict")
        if _read_workspace_predicate_source(ws) != lib_after:
            raise RuntimeError(
                "predicate source changed after authoritative verification")
        candidate_source_digest = _source_digest(lib_after)
        verification_digest = (
            _verification_digest(
                result,
                source_digest=candidate_source_digest,
                pricing_context_digest=pricing_contract["context_digest"],
                proposer_receipts_digest=_proposer_receipts_digest(
                    proposer_receipts),
            ) if result is not None else ""
        )
        if result is not None and result.pricing_source_digest \
                and result.pricing_source_digest != candidate_source_digest:
            raise RuntimeError(
                "verifier priced different predicate source bytes")
        if result is not None and result.solved:
            if not result.pricing_source_digest:
                raise RuntimeError(
                    "solved verifier result has no priced source identity")
            marginal = result.definition_cost
            definition_nodes = _definition_nodes(result)
            accepted_source = lib_after
            structure_charge = result.structure_cost
        else:
            # Structural admission: failed attempts do not grow the library.
            snapshot_wip(tag, ws, oid, verbose=verbose)
            _atomic_workspace_text(ws, LIBRARY_FILE, lib_before)
            marginal = 0
            definition_nodes = []
            accepted_source = ""
            structure_charge = 0.0

        record = ProblemRecord(
            opaque_id=oid,
            solved=bool(result and result.solved),
            heldout_accuracy=result.heldout_accuracy if result else 0.0,
            rule=result.rule if result else "",
            rule_cost=result.rule_cost if result else 0.0,
            marginal_C=marginal,
            model=model_used,
            attempts=attempts,
            escalated=attempts > 1,
            proposer_receipts=proposer_receipts,
            proposer_feedback=proposer_feedback,
            proposer_panel_set_digest=proposer_panel_set_digest,
            baseline_log_digest=baseline_log_digest,
            attempted_log_digest=_source_digest(log_after),
            phase_execution_binding_digest=(
                active_phase_binding.get("binding_digest", "")),
            status=(
                "SOLVED_UNRESTRICTED"
                if result is not None and result.solved else
                VERIFIER_FAILURE_STATUS
                if result is not None
                and result.rule == "PRICING_OR_LOAD_ERROR" else
                "UNSOLVED_UNRESTRICTED"),
            track="UNRESTRICTED",
            condition=condition,
            sharing_policy=P.SHARED,
            corpus_digest=(
                corpus_manifest["corpus_digest"] if corpus_manifest else ""),
            panel_set_digest=(
                control_manifest["problems"][k][
                    "controlled_panel_set_digest"]
                if control_manifest is not None else
                corpus_manifest["problems"][k]["panel_set_digest"]
                if corpus_manifest else ""),
            control_digest=(
                rep.control_digest),
            label_policy=condition,
            selection_policy=A.PRICED_SELECTION_POLICY,
            baseline_source_digest=baseline_source_digest,
            attempted_source_digest=candidate_source_digest,
            attempted_source=lib_after,
            accepted_source_digest=(
                candidate_source_digest if accepted_source else ""),
            accepted_source=accepted_source,
            predicate_names=(
                list(result.predicate_names)
                if result is not None else []),
            rule_atoms=(
                _rule_atoms(result.selected_rule)
                if result is not None else []),
            used_definition_nodes=definition_nodes,
            charged_definition_node_identities=(
                list(result.charged_definition_node_identities)
                if result is not None and result.solved else []),
            reused_definition_node_identities=(
                list(result.reused_definition_node_identities)
                if result is not None and result.solved else []),
            full_definition_cost=(
                result.full_definition_cost
                if result is not None and result.solved else 0),
            definition_charge=marginal,
            structure_charge=structure_charge,
            total_charge=marginal + structure_charge,
            pricing_context_digest=pricing_contract["context_digest"],
            verification_digest=verification_digest,
            source_verification_digest=verification_digest,
            train_accuracy=result.train_accuracy if result else 0.0,
            predicate_errors=result.predicate_errors if result else 12,
            n_rotations=result.n_rotations if result else 36,
            fold_rule_atoms=(
                [_rule_atoms(rule) for rule in result.fold_rules]
                if result is not None else []),
            verifier_fingerprint_digest=rep.verifier_fingerprint[
                "fingerprint_digest"],
        )
        rep.records = [r for r in rep.records if r.opaque_id != oid] + [record]
        rep.records.sort(key=lambda r: _problem_index(r.opaque_id))
        results[oid] = {
            "problem_id": truth_problem.problem_id,
            "category": truth_problem.category,
            "concept": truth_problem.concept,
            **_record_result_evidence(record),
        }
        if record.solved:
            paid_node_identities.update(
                node["identity"] for node in record.used_definition_nodes)
        rep.source_trace_digest = _source_trace_digest(rep.records)
        _validate_priced_report(rep)
        replay_input = (
            _validate_report_protocol_evidence(
                rep, corpus_manifest, corpus_bundle, control_manifest)
            if corpus_manifest is not None else problems)
        _cold_replay_report(rep, replay_input)
        _save_checkpoint(ws, rep)
        promote_verified_artifact(
            tag, ws, rep, results, verbose=verbose,
            corpus_manifest=corpus_manifest,
            control_manifest=control_manifest,
            corpus_bundle=corpus_bundle)
        if git_checkpoints:
            git_checkpoint(tag, rep, verbose=verbose)

    if verbose:
        print(f"=== {tag}: solved {rep.solved}/{len(rep.records)} | "
              f"definition_C={rep.total_definition_charge} | "
              f"structure_C={rep.total_structure_charge} | "
              f"F={rep.free_energy:.3f} ===")
        print("definition-charge trace: "
              + ", ".join(f"{r.opaque_id.split('_')[1]}:{r.definition_charge}"
                          for r in rep.records))
    return rep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the replay-gated unrestricted Bongard track")
    parser.add_argument(
        "--dataset", default=os.path.join(
            LAB_DIR, "..", "..", "downloads", "Bongard-LOGO"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--source", choices=P.SOURCES, default="basic")
    parser.add_argument("--tag", default="logo")
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument(
        "--infra-wait-seconds", type=int,
        default=DEFAULT_INFRA_WAIT_SECONDS)
    parser.add_argument(
        "--max-infra-waits", type=int,
        default=DEFAULT_MAX_INFRA_WAITS)
    parser.add_argument(
        "--ladder", default=",".join(DEFAULT_LADDER),
        help="comma-separated exact proposer model ladder")
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument(
        "--condition", choices=(P.OBSERVED, P.SHUFFLED_SIDES),
        default=P.OBSERVED)
    parser.add_argument("--control-seed", type=int, default=20260805)
    parser.add_argument("--control-replicate", type=int, default=0)
    parser.add_argument("--no-share-from", default="")
    parser.add_argument("--preregistration", default="")
    parser.add_argument("--arm-id", default="")
    parser.add_argument("--git-checkpoint", action="store_true")
    parser.add_argument("--clean-resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    cli = parser.parse_args()
    dataset = cli.dataset
    limit, seed, source = cli.limit, cli.seed, cli.source
    tag, minutes = cli.tag, cli.minutes
    infra_wait_seconds = cli.infra_wait_seconds
    max_infra_waits = cli.max_infra_waits
    if infra_wait_seconds < 0 or max_infra_waits < 0:
        raise SystemExit("infrastructure retry controls must be non-negative")
    ladder: Sequence[str] = tuple(
        item.strip() for item in cli.ladder.split(",") if item.strip())
    max_problems = cli.max_problems
    git_checkpoints = cli.git_checkpoint
    clean_resume = cli.clean_resume
    prepare_only = cli.prepare_only
    condition = cli.condition
    control_seed, control_replicate = cli.control_seed, cli.control_replicate
    no_share_from = cli.no_share_from
    preregistration_path, arm_id = cli.preregistration, cli.arm_id
    if bool(preregistration_path) != bool(arm_id):
        raise SystemExit(
            "--preregistration and --arm-id must be supplied together")
    if no_share_from:
        if prepare_only or condition not in {P.OBSERVED, P.NO_SHARE}:
            raise SystemExit(
                "--no-share-from cannot be combined with corpus preparation "
                "or shuffled labels")
        preregistration = None
        no_share_execution_binding = None
        primary_execution_binding = None
        if preregistration_path:
            source_manifest = _read_required_json(
                os.path.join(
                    artifact_dir(no_share_from), "corpus_manifest.json"),
                "no-share source corpus manifest",
            )
            preregistration = _load_preregistration(
                preregistration_path, corpus_manifest=source_manifest)
            source_report = _load_checkpoint(artifact_dir(no_share_from))
            if source_report is None:
                raise SystemExit("no-share source artifact has no checkpoint")
            derived_scale = (
                max_problems if max_problems else len(source_report.records))
            no_share_arm = _validate_preregistered_arm(
                preregistration,
                arm_id,
                corpus_digest=source_report.corpus_digest,
                condition=P.NO_SHARE,
                sharing_policy=P.NO_SHARE,
                scale=derived_scale,
                execution_tag=tag,
            )
            primary_matches = [
                arm for arm in preregistration["arms"]
                if arm["track"] == "UNRESTRICTED"
                and arm["condition"] == "primary"
                and arm["scale"] == no_share_arm["scale"]
            ]
            if len(primary_matches) != 1 \
                    or no_share_from != primary_matches[0]["execution_tag"]:
                raise SystemExit(
                    "--no-share-from differs from the preregistered primary "
                    "execution tag")
            no_share_execution_binding = P.execution_binding(
                preregistration, arm_id)
            primary_execution_binding = P.execution_binding(
                preregistration, primary_matches[0]["arm_id"])
        derived = derive_no_share_artifact(
            no_share_from, tag, max_problems=max_problems, verbose=True,
            phase_execution_binding=no_share_execution_binding,
            required_source_phase_execution_binding=(
                primary_execution_binding))
        if preregistration is not None:
            path = publish_phase_d_track_report(
                derived, preregistration, arm_id)
            print(f"published preregistered track report: {path}")
        raise SystemExit(0)
    all_problems = P.sample_corpus(
        dataset,
        limit_per_source=limit,
        seed=seed,
        source=source,
    )
    corpus_manifest = P.build_corpus_manifest(
        all_problems,
        source=source,
        seed=seed,
        limit_per_source=limit,
        dataset_revision=P.dataset_revision(dataset),
        dataset_inputs_digest=P.dataset_content_digest(dataset),
    )
    corpus_bundle = P.build_corpus_bundle(all_problems, corpus_manifest)
    control_manifest = None
    experiment_problems = all_problems
    if condition == P.SHUFFLED_SIDES:
        control = P.build_shuffled_sides_control(
            all_problems,
            corpus_manifest,
            seed=control_seed,
            replicate=control_replicate,
        )
        experiment_problems = list(control.problems)
        control_manifest = control.manifest
    elif condition != P.OBSERVED:
        raise SystemExit(
            f"--condition must be {P.OBSERVED!r} or {P.SHUFFLED_SIDES!r}")
    problems = experiment_problems
    if max_problems:
        problems = problems[:max_problems]
    print(f"frozen corpus: {corpus_manifest['corpus_digest']} | "
          f"condition {condition} | "
          f"active prefix {len(problems)}/{len(all_problems)} "
          f"({sum(1 for p in problems if p.category == 'basic')} basic, "
          f"{sum(1 for p in problems if p.category == 'abstract')} abstract)")
    preregistration = None
    active_phase_binding = None
    predecessor_phase_binding = None
    binding_family: list[dict] = []
    if preregistration_path:
        preregistration = _load_preregistration(
            preregistration_path, corpus_manifest=corpus_manifest)
        active_arm = _validate_preregistered_arm(
            preregistration, arm_id,
            corpus_digest=corpus_manifest["corpus_digest"],
            condition=condition,
            sharing_policy=P.SHARED,
            scale=len(problems),
            control_manifest=control_manifest,
            ladder=ladder,
            minutes=minutes,
            infra_wait_seconds=infra_wait_seconds,
            max_infra_waits=max_infra_waits,
            restore_wip_context=not clean_resume,
            execution_tag=tag)
        _validate_preregistered_scale_transition(
            preregistration, active_arm,
            _load_checkpoint(artifact_dir(tag)))
        binding_family = P.execution_binding_family(
            preregistration, active_arm)
        active_phase_binding = binding_family[-1]
        if len(binding_family) > 1:
            predecessor_phase_binding = binding_family[-2]
    if prepare_only:
        _preflight_artifact_binding(
            tag,
            corpus_manifest=corpus_manifest,
            corpus_bundle=corpus_bundle,
            control_manifest=control_manifest,
            problems=problems,
            condition=condition,
            phase_execution_binding=active_phase_binding,
            phase_predecessor_execution_binding=predecessor_phase_binding,
            phase_execution_binding_history=binding_family,
        )
        path = bind_corpus_manifest_to_artifact(tag, corpus_manifest)
        bind_corpus_bundle_to_artifact(tag, corpus_bundle, corpus_manifest)
        if control_manifest is not None:
            bind_control_manifest_to_artifact(
                tag, control_manifest, corpus_manifest)
        print(f"prepared corpus manifest without proposer invocation: {path}")
        raise SystemExit(0)
    report = run(
        problems, tag=tag, ladder=ladder, minutes=minutes,
        infra_wait_seconds=infra_wait_seconds,
        max_infra_waits=max_infra_waits,
        git_checkpoints=git_checkpoints, restore_wip=not clean_resume,
        corpus_manifest=corpus_manifest, corpus_bundle=corpus_bundle,
        condition=condition,
        control_manifest=control_manifest,
        base_problems=(all_problems if control_manifest is not None else None),
        phase_execution_binding=active_phase_binding,
        phase_predecessor_execution_binding=predecessor_phase_binding,
        phase_execution_binding_history=binding_family)
    if preregistration is not None:
        path = publish_phase_d_track_report(
            report, preregistration, arm_id,
            control_manifest=control_manifest)
        print(f"published preregistered track report: {path}")
