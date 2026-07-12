"""Lean Messages-API proposer for the Bongard crack (the gkm_api_agent rung).

The next rung DOWN the proposer ladder from the headless Claude Code agent:
no tools, no shell -- just a Messages-API conversation loop. The model sees
the 12 panels as images plus the current shared library, and must reply with
a complete new `predicates.py`; the LOOP (not the model) runs the verifier
and feeds the RESULT line back. Iterate until solved or budget.

Two prompt variants for the stage-1.5 A/B (bongard_crack_plan.md Section 9):

  current        straight to predicates (implicit description)
  describe_first mandatory human-like panel descriptions + a candidate
                 one-sentence rule BEFORE any code (language as an
                 inductive-bias channel); descriptions are logged

The boundary holds: descriptions are hypothesis generation and articulation
only. The verified object remains deterministic p_*(panel) code; no VLM call
ever sits inside a predicate.

Factory `api_propose(variant)` returns a propose_fn compatible with
`bongard_legs.run`, so the whole orchestration (admission, marginal C, WIP,
taint, git checkpoints, infra-failure guardrails) is reused unchanged.
"""
from __future__ import annotations

import base64
import glob
import ast
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_arena as A

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
KEY_FILE = os.path.join(REPO_ROOT, "ANTHROPIC_API_KEY.env.local")

MODEL_MAP = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}
VERIFY_TIMEOUT_S = 45.0
TAINT_MARKER = "TAINTED_ASSISTANT_FABRICATION"
NO_VALID_CODE_MARKER = "NO_VALID_PREDICATES_CODE"
SEMANTIC_DRIFT_MARKER = "SEMANTIC_DRIFT_REJECTED"

SEMANTIC_STOPWORDS = {
    "about", "after", "against", "being", "between", "class", "classes",
    "code", "concept", "contain", "contains", "drawn", "each", "from",
    "have", "image", "images", "into", "itself", "measurement",
    "measurements", "negative", "negatives", "object", "objects", "panel",
    "panels", "positive", "positives", "predicate", "predicates", "shape",
    "shapes", "their", "there", "these", "those", "through", "while",
    "with", "without",
}

SEMANTIC_GROUPS = {
    "symmetry": {
        "asymmetric", "asymmetry", "axis", "axes", "lopsided", "mirror",
        "mirrored", "reflect", "reflected", "reflection", "regular", "self",
        "symmetric", "symmetry", "transpose", "transposed",
    },
    "circle_arc": {
        "arc", "arcs", "circle", "circular", "curve", "curved", "curvature",
        "round", "rounded",
    },
    "topology": {
        "branch", "cross", "crossing", "hole", "holes", "intersect",
        "intersecting", "intersection", "self", "skeleton", "topological",
    },
}


@dataclass
class SemanticCVResult:
    accepted: bool
    cv_accuracy: float
    train_accuracy: float
    cv_errors: int
    train_errors: int
    n_examples: int
    rule: str
    threshold: float
    fold_threshold_min: float
    fold_threshold_max: float
    max_cv_errors: int
    max_train_errors: int
    predicate_errors: int

    def result_line(self) -> str:
        return (
            f"SEMANTIC_CV accepted={self.accepted} cv={self.cv_accuracy:.3f} "
            f"train={self.train_accuracy:.3f} "
            f"cv_errors={self.cv_errors}/{self.n_examples} "
            f"train_errors={self.train_errors}/{self.n_examples} "
            f"budget=cv<={self.max_cv_errors},train<={self.max_train_errors} "
            f"rule=\"{self.rule}\" "
            f"threshold={self.threshold:.4g} "
            f"fold_t=[{self.fold_threshold_min:.4g},{self.fold_threshold_max:.4g}] "
            f"predicate_errors={self.predicate_errors}"
        )


@dataclass
class ReplyPayload:
    semantic_text: str
    code: str


DESCRIBE_FIRST_INSTRUCTION = """\
Before writing code, name the semantic concept as a structural invariant, not
as a pixel statistic. Think categorically:
- objects are drawn panels modulo nuisance morphisms such as translation,
  scale, rotation, stroke thickness, sampling noise, and small drawing error;
- a semantic proposal is an invariant or relation on that quotient object;
- code is a measurement functor from panels to an ordered numeric space that
  should factor through the quotient, up to drawing tolerance;
- threshold selection is the harness choosing a cut in that ordered space.

Use 2-4 candidate semantic invariants, each phrased as the smallest number of
most-general statements that might separate positive from negative panels.
Map those semantic proposals to continuous code measurements. The code and
cross-validation feedback must trim the proposal set down firmly to one. After
verifier feedback, keep the surviving semantic object fixed and improve only
its measurement functor; do not replace it with a different invariant such as
aspect ratio merely because that has a better threshold."""

REPLY_FORMAT = """\
Reply with exactly one fenced Python-literal dictionary and no other Markdown.
Use ```python or ```dict. The dictionary must pair one categorical semantic
description with its complete code realization directly:
```python
{
  'semantic': {
    'object': 'what quotient object/class of drawings is being compared',
    'nuisance_morphisms': [
      'translation', 'scale', 'rotation/reflection if irrelevant',
      'stroke thickness', 'small drawing noise'
    ],
    'positive_structure': 'the invariant/relation positives have',
    'negative_structure': 'the contrasting invariant/relation negatives lack',
    'measurement_functor': 'p_name: panel -> raw numeric measure of that invariant',
    'order': 'low means more positive, or high means more positive',
    'final_rule': 'semantic rule without numeric overfitting'
  },
  'code': r'''
COMPLETE new predicates.py source
'''
}
```

The dictionary may alternatively be a one-entry mapping from a semantic
statement string to the code string, but the same invariant holds: the
semantic object and executable code are one paired object. The code must
contain module-level pure functions
`p_<name>(panel) -> float | bool` over a 128x128 uint8 array, ink=1;
numpy/math/scipy only; deterministic; no file or network access. Keep existing
library predicates unless they are broken, but the semantic measurement for
this problem must be implemented as a new or modified `p_*` function in the
returned code. Use ASCII only.

How the verifier scores you: predicates should return RAW continuous
measurements. The verifier selects comparison thresholds itself, then
RE-SELECTS them under rotated leave-one-out; a measurement therefore only
survives if the two classes are separated by a WIDE margin, not merely
separated. Never bake a pass/fail cutoff into a predicate. If a semantic
measurement separates thinly, do not abandon the concept for pixel
statistics -- normalize the measurement so it is comparable across panels
(invariant to size, position, stroke length), which is what widens the
margin. For example, prefer a reusable raw measurement like circle-fit
residual over a predicate hard-coded to residual < 0.01; the harness will
cross-validate whether there is a stable common cutoff."""


def load_api_key(path: str = KEY_FILE) -> str:
    text = open(path).read().strip()
    return text.split("=", 1)[1].strip() if "=" in text.split("\n")[0] and text.startswith("ANTHROPIC") else text


def extract_code(reply: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    candidates = []
    for block in blocks:
        code = block.strip() + "\n"
        if "def p_" not in code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        candidates.append(code)
    return candidates[-1] if candidates else None


def _dict_objects_from_reply(reply: str) -> List[dict]:
    fenced = re.findall(
        r"```(?:json|python|dict)?\s*\n(.*?)```",
        reply,
        re.DOTALL | re.IGNORECASE,
    )
    candidates = fenced if fenced else [reply]
    payloads = []
    for raw in candidates:
        raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _string_list(value) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _semantic_text_from_value(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    categorical_fields = [
        ("object", "OBJECT"),
        ("quotient_object", "OBJECT"),
        ("nuisance_morphisms", "NUISANCE_MORPHISMS"),
        ("morphisms", "NUISANCE_MORPHISMS"),
        ("positive_structure", "POSITIVE_STRUCTURE"),
        ("negative_structure", "NEGATIVE_STRUCTURE"),
        ("invariant", "INVARIANT"),
        ("relation", "RELATION"),
        ("measurement_functor", "MEASUREMENT_FUNCTOR"),
        ("realization", "MEASUREMENT_FUNCTOR"),
        ("order", "ORDER"),
        ("final_rule", "FINAL_SEMANTIC_RULE"),
        ("final_semantic_rule", "FINAL_SEMANTIC_RULE"),
    ]
    lines = []
    seen_labels = set()
    for key, label in categorical_fields:
        if key not in value:
            continue
        items = _string_list(value.get(key))
        if not items:
            continue
        if label not in seen_labels:
            lines.append(f"{label}:")
            seen_labels.add(label)
        lines.extend(f"- {item}" for item in items)
    proposals = _string_list(value.get("proposals"))
    distillation = _string_list(value.get("distillation"))
    cv = str(value.get("cv_interpretation", "")).strip()
    final = str(value.get("final_rule", "") or value.get("final_semantic_rule", "")).strip()
    if proposals:
        lines.append("SEMANTIC_PROPOSALS:")
        lines.extend(f"- {item}" for item in proposals)
    if distillation:
        lines.append("DISTILLATION:")
        lines.extend(f"- {item}" for item in distillation)
    if cv:
        lines.append("CV_INTERPRETATION:")
        lines.append(cv)
    if final and "FINAL_SEMANTIC_RULE" not in seen_labels:
        lines.append("FINAL_SEMANTIC_RULE:")
        lines.append(final)
    return "\n".join(lines).strip()


def _function_sources(code: str) -> dict[str, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    sources = {}
    lines = code.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("p_"):
            if node.end_lineno is None:
                continue
            source = "\n".join(lines[node.lineno - 1:node.end_lineno]).strip()
            sources[node.name] = source
    return sources


def changed_predicate_names(before_code: str, after_code: str) -> List[str]:
    before = _function_sources(before_code)
    after = _function_sources(after_code)
    new_names = [name for name in after if name not in before]
    if new_names:
        return sorted(new_names)
    names = [name for name, source in after.items()
             if before.get(name) != source]
    return sorted(names)


def _stem_semantic_token(token: str) -> str:
    if token == "axes":
        return "axis"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def semantic_terms(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z]+", text.replace("_", " ").lower())
    terms = {
        _stem_semantic_token(tok)
        for tok in raw
        if len(tok) >= 4 and tok not in SEMANTIC_STOPWORDS
    }
    for group, words in SEMANTIC_GROUPS.items():
        if terms & {_stem_semantic_token(w) for w in words}:
            terms.add(f"concept:{group}")
    return terms


def semantic_matches_anchor(anchor: str, candidate: str,
                            predicate_names: Sequence[str] = ()) -> bool:
    """Heuristic guardrail against switching semantic objects mid-attempt.

    The model may improve the measurement functor for the current semantic
    object, but it must not respond to failed CV by replacing that object with
    an unrelated invariant that happens to threshold better. This is not
    problem-specific: the anchor is created from the model's own first
    semantic proposal in this proposer call.
    """
    anchor_terms = semantic_terms(anchor)
    candidate_terms = semantic_terms(candidate + " " + " ".join(predicate_names))
    if not anchor_terms or not candidate_terms:
        return True
    if any(t.startswith("concept:") and t in candidate_terms for t in anchor_terms):
        return True
    shared = {t for t in (anchor_terms & candidate_terms)
              if not t.startswith("concept:")}
    required = 1 if len(anchor_terms) <= 3 else 2
    return len(shared) >= required


def extract_reply_payload(reply: str) -> Optional[ReplyPayload]:
    """Extract one paired semantics+code JSON answer.

    The proposer is only allowed to score code that arrived in the same
    structured payload as the semantic concept. This prevents a malformed
    semantic/code attempt from being replaced by stale or unrelated library
    code just because a Python fence elsewhere parses.
    """
    valid = []
    for payload in _dict_objects_from_reply(reply):
        if "code" in payload or "semantic" in payload:
            code = payload.get("code", payload.get("predicates_py"))
            semantic_value = payload.get("semantic")
        elif len(payload) == 1:
            semantic_value, code = next(iter(payload.items()))
        else:
            code = payload.get("predicates_py")
            semantic_value = payload.get("semantic")
        if not isinstance(code, str):
            continue
        code = code.strip() + "\n"
        if "def p_" not in code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        semantic_text = _semantic_text_from_value(semantic_value)
        if not semantic_text:
            proposals = _string_list(payload.get("semantic_proposals"))
            distillation = _string_list(payload.get("distillation"))
            cv = str(payload.get("cv_interpretation", "")).strip()
            final = str(payload.get("final_semantic_rule", "")).strip()
            legacy = {
                "proposals": proposals,
                "distillation": distillation,
                "cv_interpretation": cv,
                "final_rule": final,
            }
            semantic_text = _semantic_text_from_value(legacy)
        if not semantic_text:
            continue
        valid.append(ReplyPayload(semantic_text + "\n", code))
    return valid[-1] if valid else None


def extract_semantic_text(reply: str) -> str:
    texts = []
    for payload in _dict_objects_from_reply(reply):
        semantic_text = _semantic_text_from_value(payload.get("semantic"))
        if not semantic_text:
            proposals = _string_list(payload.get("semantic_proposals"))
            distillation = _string_list(payload.get("distillation"))
            cv = str(payload.get("cv_interpretation", "")).strip()
            final = str(payload.get("final_semantic_rule", "")).strip()
            semantic_text = _semantic_text_from_value({
                "proposals": proposals,
                "distillation": distillation,
                "cv_interpretation": cv,
                "final_rule": final,
            })
        if semantic_text:
            texts.append(semantic_text)
    return texts[-1] if texts else ""


def no_tools_fabrication_reason(reply: str) -> str:
    """Detect assistant-authored fake tool/verifier transcripts.

    This Bongard API loop gives the model no tools. Real verifier output is
    appended by this harness after code extraction, so any tool transcript or
    RESULT line inside the assistant reply is fabricated model text.
    """
    checks = [
        (r"(?mi)^\s*\*\*Tool call:", "fake Markdown tool call"),
        (r"(?mi)^\s*\*\*Tool result:", "fake Markdown tool result"),
        (r"(?mi)^\s*RESULT\s+solved\s*=", "self-reported verifier RESULT"),
        (r"(?i)\bloo_acc\s*=", "self-reported leave-one-out score"),
    ]
    reasons = [reason for pattern, reason in checks if re.search(pattern, reply)]
    return "; ".join(reasons)


def _panel_blocks(pdir: str) -> List[dict]:
    blocks: List[dict] = []
    for path in sorted(glob.glob(os.path.join(pdir, "*.png"))):
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        blocks.append({"type": "text", "text": os.path.basename(path) + ":"})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": data}})
    return blocks


def _problem_from_ws(ws: str) -> A.Problem:
    oid = open(os.path.join(ws, "current_problem.txt")).read().strip()
    pdir = os.path.join(ws, oid)
    pos = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "pos_*.npy")))]
    neg = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "neg_*.npy")))]
    return A.Problem("current", "?", "?", pos, neg)


def _best_threshold(col: np.ndarray, labels: np.ndarray,
                    op: str) -> tuple[float, float]:
    uniq = np.unique(col)
    if len(uniq) < 2:
        return float(col[0]) if len(col) else 0.0, float((labels == labels[0]).mean())
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0
    best = None
    for t in thresholds:
        pred = col >= t if op == ">=" else col <= t
        acc = float((pred == labels).mean())
        key = (acc, -abs(float(t)))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1], best[0][0]


def semantic_cv_select(values: np.ndarray, names: List[str], labels: np.ndarray,
                       predicate_errors: int = 0, max_cv_errors: int = 1,
                       max_train_errors: int = 0) -> SemanticCVResult:
    """Select a one-measurement semantic predicate by leave-one-image-out CV.

    The code is treated as a noisy realization of a semantic concept: choose a
    threshold from each one-image-held-out training fold, require the full
    visible set to stay within a train-error budget, and accept a small,
    explicit heldout-error budget rather than pretending approximate drawings
    must be threshold-perfect.
    """
    best = None
    for j, name in enumerate(names):
        col = values[:, j]
        if len(np.unique(col)) < 2:
            continue
        for op in (">=", "<="):
            full_t, train_acc = _best_threshold(col, labels, op)
            fold_thresholds = []
            correct = 0
            for held in range(len(labels)):
                mask = np.array([i != held for i in range(len(labels))])
                t, _fold_train = _best_threshold(col[mask], labels[mask], op)
                fold_thresholds.append(t)
                pred = col[held] >= t if op == ">=" else col[held] <= t
                correct += int(pred == labels[held])
            cv_acc = correct / len(labels) if len(labels) else 0.0
            cv_errors = int(len(labels) - correct)
            train_errors = int(round(len(labels) * (1.0 - train_acc)))
            folds = np.asarray(fold_thresholds, dtype=float)
            span = float(np.ptp(folds)) if len(folds) else 0.0
            rule = f"{name}{op}{full_t:.4g}"
            key = (-cv_errors, -train_errors, -span, cv_acc, train_acc, rule)
            if best is None or key > best[0]:
                best = (key, rule, full_t, folds, cv_acc, train_acc,
                        cv_errors, train_errors)
    if best is None:
        return SemanticCVResult(
            False, 0.0, 0.0, len(labels), len(labels), len(labels),
            "NO_VARYING_MEASUREMENT", 0.0, 0.0, 0.0,
            max_cv_errors, max_train_errors,
            predicate_errors)
    (_key, rule, threshold, folds, cv_acc, train_acc,
     cv_errors, train_errors) = best
    accepted = bool(
        train_errors <= max_train_errors and cv_errors <= max_cv_errors)
    return SemanticCVResult(
        accepted,
        float(cv_acc),
        float(train_acc),
        int(cv_errors),
        int(train_errors),
        int(len(labels)),
        rule,
        float(threshold),
        float(folds.min()) if len(folds) else float(threshold),
        float(folds.max()) if len(folds) else float(threshold),
        int(max_cv_errors),
        int(max_train_errors),
        predicate_errors,
    )


def _semantic_cv_ws_direct(ws: str, max_cv_errors: int = 1,
                           max_train_errors: int = 0,
                           allowed_names: Optional[Sequence[str]] = None) -> SemanticCVResult:
    problem = _problem_from_ws(ws)
    try:
        preds = A.load_predicates(os.path.join(ws, "predicates.py"))
    except Exception:
        return SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12, "LOAD_ERROR", 0.0, 0.0, 0.0,
            max_cv_errors, max_train_errors, 36)
    if allowed_names is not None:
        allowed = set(allowed_names)
        preds = {name: fn for name, fn in preds.items() if name in allowed}
        if not preds:
            return SemanticCVResult(
                False, 0.0, 0.0, 12, 12, 12,
                "NO_CHANGED_SEMANTIC_MEASUREMENT", 0.0, 0.0, 0.0,
                max_cv_errors, max_train_errors, 0)
    panels = [p for p, _ in problem.panels()]
    labels = np.array([lab for _, lab in problem.panels()])
    values, names, errors = A.predicate_values(preds, panels)
    return semantic_cv_select(
        values, names, labels, errors, max_cv_errors, max_train_errors)


def _semantic_cv_worker(queue, ws: str, max_cv_errors: int,
                        max_train_errors: int,
                        allowed_names: Optional[Sequence[str]]) -> None:
    try:
        queue.put(("ok", _semantic_cv_ws_direct(
            ws, max_cv_errors, max_train_errors, allowed_names)))
    except BaseException as exc:
        queue.put(("error", repr(exc)))


def semantic_cv_ws_with_timeout(ws: str, timeout: float = VERIFY_TIMEOUT_S,
                                max_cv_errors: int = 1,
                                max_train_errors: int = 0,
                                allowed_names: Optional[Sequence[str]] = None) -> SemanticCVResult:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_semantic_cv_worker,
        args=(queue, ws, max_cv_errors, max_train_errors, allowed_names))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12,
            f"SEMANTIC_CV_TIMEOUT:{timeout:.1f}s", 0.0, 0.0, 0.0,
            max_cv_errors, max_train_errors, 36)
    if queue.empty():
        return SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12,
            f"SEMANTIC_CV_ERROR:exit={proc.exitcode}", 0.0, 0.0, 0.0,
            max_cv_errors, max_train_errors, 36)
    status, payload = queue.get()
    if status == "error":
        return SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12,
            f"SEMANTIC_CV_ERROR:{payload}", 0.0, 0.0, 0.0,
            max_cv_errors, max_train_errors, 36)
    return payload


def atomic_cv_report(values: np.ndarray, names: List[str],
                     labels: np.ndarray, max_lines: int = 8) -> str:
    """One-predicate leave-one-image-out threshold stability diagnostics.

    This is the same cross-validation geometry used by semantic admission,
    exposed in a form the next proposer turn can use for semantic
    distillation: which raw measurement supports a common cutoff, and which
    one is just a brittle number.
    """
    rows = []
    for j, name in enumerate(names):
        col = values[:, j]
        if len(np.unique(col)) < 2:
            continue
        for op in (">=", "<="):
            full_t, full_acc = _best_threshold(col, labels, op)
            correct = 0
            total = 0
            fold_thresholds = []
            for held in range(len(labels)):
                mask = np.array([r != held for r in range(len(labels))])
                t, _train_acc = _best_threshold(col[mask], labels[mask], op)
                fold_thresholds.append(t)
                pred = col[held] >= t if op == ">=" else col[held] <= t
                correct += int(pred == labels[held])
                total += 1
            cv_acc = correct / total if total else 0.0
            folds = np.asarray(fold_thresholds, dtype=float)
            pos = col[labels == 1]
            neg = col[labels == 0]
            rows.append((
                -cv_acc,
                -full_acc,
                float(np.ptp(folds)) if len(folds) else 0.0,
                name,
                op,
                cv_acc,
                full_acc,
                full_t,
                float(folds.min()) if len(folds) else full_t,
                float(folds.max()) if len(folds) else full_t,
                float(pos.min()), float(pos.max()),
                float(neg.min()), float(neg.max()),
            ))
    if not rows:
        return "\nATOMIC_CV: no varying predicate measurements available."
    rows.sort()
    lines = [
        "\nATOMIC_CV: best single-measurement common-cutoff candidates "
        "(higher cv is better; narrow fold_t range means the semantic "
        "measurement is threshold-stable):"
    ]
    for row in rows[:max_lines]:
        (_neg_cv, _neg_full, _span, name, op, cv_acc, full_acc, full_t,
         fold_min, fold_max, pos_min, pos_max, neg_min, neg_max) = row
        lines.append(
            f"- {name}{op}: cv={cv_acc:.3f} full={full_acc:.3f} "
            f"full_t={full_t:.4g} fold_t=[{fold_min:.4g},{fold_max:.4g}] "
            f"pos=[{pos_min:.4g},{pos_max:.4g}] neg=[{neg_min:.4g},{neg_max:.4g}]"
        )
    return "\n".join(lines)


def diagnose_workspace(ws: str, result: SemanticCVResult) -> str:
    """Compact semantic-CV counterexamples for the next proposer turn."""
    if result.rule.startswith(("LOAD_ERROR", "SEMANTIC_CV_ERROR",
                               "SEMANTIC_CV_TIMEOUT")):
        return ""
    try:
        problem = _problem_from_ws(ws)
        preds = A.load_predicates(os.path.join(ws, "predicates.py"))
        panels = [p for p, _ in problem.panels()]
        labels = np.array([lab for _, lab in problem.panels()])
        values, names, _errors = A.predicate_values(preds, panels)
    except Exception as exc:
        return f"\nDIAGNOSTIC unavailable: {exc!r}"

    panel_names = [f"pos_{i}" for i in range(len(problem.pos))]
    panel_names += [f"neg_{i}" for i in range(len(problem.neg))]
    lines = [f"\nDIAGNOSTIC: selected={result.rule}"]
    if result.rule in {"NO_VARYING_MEASUREMENT", "LOAD_ERROR"}:
        lines.append("No varying semantic measurement was available.")
        lines.append(atomic_cv_report(values, names, labels))
        return "\n".join(lines)

    match = re.match(r"(.+?)(>=|<=)([-+0-9.eE]+)$", result.rule)
    if match:
        name, op, threshold_s = match.groups()
        threshold = float(threshold_s)
        if name in names:
            col = values[:, names.index(name)]
            pred = col >= threshold if op == ">=" else col <= threshold
            misses = [
                f"{panel_names[i]} expected={'pos' if labels[i] else 'neg'} "
                f"value={col[i]:.4g}"
                for i in range(len(labels)) if pred[i] != labels[i]
            ]
            if misses:
                lines.append("full-threshold misses: " + "; ".join(misses[:4]))
            else:
                lines.append(
                    "full visible threshold separates all panels; remaining "
                    "risk is fold-threshold movement on heldout images.")
            pos = col[labels == 1]
            neg = col[labels == 0]
            lines.append(
                f"class ranges for {name}: pos [{pos.min():.4g}, {pos.max():.4g}] "
                f"neg [{neg.min():.4g}, {neg.max():.4g}] threshold={threshold:.4g}")
    if result.train_errors > result.max_train_errors:
        lines.append(
            "The full visible set is not cleanly separated; improve the raw "
            "measurement for the same semantic concept.")
    elif result.cv_errors > result.max_cv_errors:
        lines.append(
            "The semantic measurement is close but exceeds the heldout error "
            "budget. Normalize this same semantic measurement so it has a more "
            "stable common cutoff.")
    lines.append(atomic_cv_report(values, names, labels))
    return "\n".join(lines)


class APICallTimeout(TimeoutError):
    """Raised when one model call exceeds the harness hard wall."""


def _api_call_direct(model_id: str, max_tokens: int,
                     messages: List[dict], timeout: float) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key())
    reply = client.messages.create(
        model=model_id, max_tokens=max_tokens, messages=messages,
        timeout=timeout)
    return "".join(b.text for b in reply.content
                   if getattr(b, "type", "") == "text")


def _api_call_cli() -> None:
    in_path, out_path = sys.argv[2], sys.argv[3]
    try:
        with open(in_path, encoding="utf-8") as f:
            payload = json.load(f)
        text = _api_call_direct(
            payload["model_id"],
            int(payload["max_tokens"]),
            payload["messages"],
            float(payload["timeout"]),
        )
        result = {"status": "ok", "text": text}
    except BaseException as exc:
        result = {"status": "error", "error": repr(exc)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def _create_message_text(model_id: str, max_tokens: int, messages: List[dict],
                         timeout: float,
                         client_factory: Optional[Callable] = None) -> str:
    if client_factory is not None:
        client = client_factory()
        reply = client.messages.create(
            model=model_id, max_tokens=max_tokens, messages=messages,
            timeout=timeout)
        return "".join(b.text for b in reply.content
                       if getattr(b, "type", "") == "text")

    with tempfile.TemporaryDirectory(prefix="bongard_api_call_") as td:
        in_path = os.path.join(td, "request.json")
        out_path = os.path.join(td, "response.json")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": model_id,
                "max_tokens": max_tokens,
                "messages": messages,
                "timeout": timeout,
            }, f)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--api-call-worker", in_path, out_path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise APICallTimeout(
                f"API subprocess timed out after {timeout + 10.0:.1f}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"API subprocess exited {proc.returncode}: {proc.stderr[-1000:]}")
        if not os.path.exists(out_path):
            raise RuntimeError("API subprocess produced no response file")
        with open(out_path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("error", "unknown API error"))
        return str(payload.get("text", ""))


def api_propose(variant: str = "current", max_turns: int = 8,
                max_tokens: int = 8000, per_call_timeout: float = 90.0,
                max_cv_errors: int = 1, max_train_errors: int = 0,
                client_factory: Callable = None,
                verbose: bool = True) -> Callable[[str, str, str, int], Optional[str]]:
    """Build a propose_fn for bongard_legs.run.

    `client_factory` is injectable for offline tests; default builds a real
    anthropic.Anthropic client with the repo-local key."""
    assert variant in ("current", "describe_first")

    def propose(task: str, ws: str, model: str, minutes: int) -> Optional[str]:
        model_id = MODEL_MAP.get(model, model)
        oid = open(os.path.join(ws, "current_problem.txt")).read().strip()
        lib_path = os.path.join(ws, "predicates.py")
        library = open(lib_path).read() if os.path.exists(lib_path) else ""
        semantic_anchor = ""

        intro = [{"type": "text", "text": task}]
        intro += _panel_blocks(os.path.join(ws, oid))
        parts = ["Current shared library `predicates.py`:\n```python\n"
                 + library + "\n```"]
        if variant == "describe_first":
            parts.append(DESCRIBE_FIRST_INSTRUCTION)
        parts.append(REPLY_FORMAT)
        intro.append({"type": "text", "text": "\n\n".join(parts)})

        messages = [{"role": "user", "content": intro}]
        transcript: List[str] = []
        verified_code = False
        deadline = time.time() + minutes * 60
        for turn in range(max_turns):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                timeout = max(1.0, min(per_call_timeout, remaining))
                text = _create_message_text(
                    model_id, max_tokens, messages, timeout, client_factory)
            except Exception as exc:
                transcript.append(f"API failure: {exc}")
                break
            transcript.append(text)
            taint_reason = no_tools_fabrication_reason(text)
            if taint_reason:
                transcript.append(f"{TAINT_MARKER}: {taint_reason}")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content":
                              "Your previous reply fabricated tool or verifier "
                              "output. This API loop has no tools; only the "
                              "harness runs verification. Reply with exactly "
                              "one paired semantics/code dictionary."}]
                continue
            payload = extract_reply_payload(text)
            if payload is None:
                semantic_only = extract_semantic_text(text)
                if semantic_only:
                    if semantic_anchor and not semantic_matches_anchor(
                            semantic_anchor, semantic_only):
                        retry = (
                            f"{SEMANTIC_DRIFT_MARKER}. You named a different "
                            "semantic object/invariant instead of realizing "
                            "the one already proposed in this attempt. Double "
                            "down on the existing semantic object and provide "
                            "valid code for it.\n\nSEMANTIC OBJECT TO REALIZE:\n"
                            + semantic_anchor
                            + "\n\n" + REPLY_FORMAT
                        )
                    else:
                        if not semantic_anchor:
                            semantic_anchor = semantic_only
                        retry = (
                            "You named a semantic concept but did not provide valid "
                            "paired executable code for it. Double down on this "
                            "semantic object and realize it in code; do not switch "
                            "to an unrelated invariant.\n\nSEMANTIC OBJECT TO REALIZE:\n"
                            + semantic_anchor
                            + "\n\nReply with exactly one dictionary containing "
                            "'semantic' and syntactically valid complete 'code'. "
                            + REPLY_FORMAT
                        )
                else:
                    retry = (
                        "No valid paired semantics/code dictionary was found. "
                        "The answer must be exactly one fenced Python-literal "
                        "dictionary with 'semantic' and syntactically valid "
                        "complete 'code'. " + REPLY_FORMAT
                    )
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": retry}]
                continue
            candidate_names = changed_predicate_names(library, payload.code)
            if not candidate_names:
                semantic_only = payload.semantic_text.strip()
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content":
                              "The semantic/code dictionary did not introduce "
                              "or modify any p_* measurement for this semantic. "
                              "Double down on the stated semantic and implement "
                              "it as a new or modified raw continuous predicate; "
                              "old library predicates cannot be selected as "
                              "evidence for a new semantic claim.\n\nSEMANTIC "
                              "TO REALIZE:\n" + semantic_only + "\n\n" + REPLY_FORMAT}]
                continue
            if semantic_anchor and not semantic_matches_anchor(
                    semantic_anchor, payload.semantic_text, candidate_names):
                transcript.append(
                    f"{SEMANTIC_DRIFT_MARKER}: locked={semantic_anchor!r} "
                    f"proposed={payload.semantic_text.strip()!r} "
                    f"predicates={candidate_names!r}")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content":
                              f"{SEMANTIC_DRIFT_MARKER}. You changed the "
                              "semantic object/invariant after CV feedback. "
                              "Keep the same quotient object and invariant; "
                              "only improve the measurement functor. Do not "
                              "replace it with an unrelated thresholdable "
                              "proxy.\n\nCURRENT SEMANTIC OBJECT:\n"
                              + semantic_anchor + "\n\n" + REPLY_FORMAT}]
                continue
            if not semantic_anchor:
                semantic_anchor = payload.semantic_text.strip()
            with open(os.path.join(ws, f"semantic_candidate_names_{oid}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(candidate_names, f)
            if variant == "describe_first":
                with open(os.path.join(ws, f"descriptions_{oid}.md"), "a") as f:
                    f.write(payload.semantic_text + "---\n")
            with open(lib_path, "w") as f:
                f.write(payload.code)
            result = semantic_cv_ws_with_timeout(
                ws,
                max_cv_errors=max_cv_errors,
                max_train_errors=max_train_errors,
                allowed_names=candidate_names,
            )
            verified_code = True
            transcript.append(result.result_line())
            if verbose:
                print(f"  api-turn {turn + 1}: {result.result_line()}")
            if result.accepted:
                break
            detail = diagnose_workspace(ws, result)
            if result.rule.startswith("SEMANTIC_CV_TIMEOUT"):
                feedback = (
                    result.result_line()
                    + "\nToo slow. Use simpler bounded measurements. Reply with full predicates.py."
                    + detail)
            else:
                feedback = (
                    result.result_line()
                    + "\nThe semantic object/invariant is now fixed for this "
                    "proposer call. Do not switch invariants. Improve the raw "
                    "measurement functor for the same semantic object, use the ATOMIC_CV "
                    "threshold-stability report to reject brittle realizations "
                    "of that same concept, then reply with one paired "
                    "dictionary containing the same semantic object and "
                    "its complete code."
                    + detail)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": feedback}]
        if not verified_code:
            transcript.append(
                f"{NO_VALID_CODE_MARKER}: no syntactically valid complete "
                "predicates.py block was verified in this proposer call")
        full = "\n\n".join(transcript)
        with open(os.path.join(ws, f"api_transcript_{oid}.md"), "a") as f:
            f.write(full + "\n\n=====\n\n")
        return full

    return propose


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--api-call-worker":
    _api_call_cli()
