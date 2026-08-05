"""Cofibered semantic-cone proposer interface.

The real experiment uses an LLM proposer that returns typed cone IR only,
through a forced structured-output boundary (an Anthropic tool call or the
isolated Codex CLI's output schema) — free-text JSON parsing is gone, so
malformed output and truncation become explicit retryable states instead of
run-killing parse errors.  The proposer does not return final classifier code.
Static proposals are allowed only in unit tests and are labeled as fixtures.

The proposer is the ONLY source of semantic novelty: it names the semantic
terms, declares the witness requirements, and generates gluing
(cofibration) requests for composite structure.  The harness never supplies
concept-specific structure; it only type-checks, audits coverage, verifies
and prices.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import codex_proposer as codex_headless
from semantic_ir import SemanticHypothesis
from semantic_legs import default_registry, is_witness_codomain

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KEY_FILE = os.path.join(REPO_ROOT, "ANTHROPIC_API_KEY.env.local")
MODEL_MAP = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}
CODEX_DEFAULT_MODEL = codex_headless.DEFAULT_CODEX_MODEL

TOOL_NAME = "submit_semantic_cones"
MAX_TOKENS_CAP = 16000
SEMANTIC_PROPOSER_RECEIPT_SCHEMA = \
    "bongard.semantic-proposer-model-receipt/v1"


@dataclass(frozen=True)
class ProposalBundle:
    problem_id: str
    hypotheses: tuple[SemanticHypothesis, ...]
    raw_text: str
    proposer_kind: str
    parse_error: str = ""
    model_receipts: tuple[dict, ...] = ()


def _model_receipt(message: Any, requested_model: str) -> dict:
    actual_model = getattr(message, "model", None)
    usage = getattr(message, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    stop_reason = getattr(message, "stop_reason", None)
    if not isinstance(actual_model, str) or not actual_model \
            or actual_model != requested_model \
            or isinstance(input_tokens, bool) \
            or not isinstance(input_tokens, int) or input_tokens < 0 \
            or isinstance(output_tokens, bool) \
            or not isinstance(output_tokens, int) or output_tokens < 0 \
            or input_tokens + output_tokens <= 0 \
            or not isinstance(stop_reason, str) or not stop_reason:
        raise RuntimeError(
            "Anthropic response lacks exact positive model-usage evidence")
    body = {
        "schema": SEMANTIC_PROPOSER_RECEIPT_SCHEMA,
        "source": "anthropic-messages-api",
        "requested_model": requested_model,
        "actual_model": actual_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    body["receipt_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return body


class CofiberedProposer(Protocol):
    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        ...

    def refine(self, problem_id: str, feedback: str) -> ProposalBundle:
        ...


def _load_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    text = open(KEY_FILE, encoding="utf-8").read().strip()
    if text.startswith("ANTHROPIC") and "=" in text:
        return text.split("=", 1)[1].strip()
    return text


_MORPH_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "scope": {"type": "string"},
        "expected_effect": {"type": "string", "enum": ["preserve"]},
    },
    "required": ["name", "scope", "expected_effect"],
}

_GLUING_SCHEMA = {
    "type": "object",
    "description": (
        "A gluing (cofibration) request: source_node is glued into "
        "target_node along the declared interface. Verified mechanically "
        "per panel, up to ID renaming and numeric tolerance."
    ),
    "properties": {
        "name": {"type": "string"},
        "source_node": {"type": "string"},
        "target_node": {"type": "string"},
        "source_type": {"type": "string"},
        "target_type": {"type": "string"},
        "interface_fields": {"type": "array", "minItems": 1,
                             "items": {"type": "string"}},
        "added_fields": {"type": "array", "minItems": 1,
                         "items": {"type": "string"}},
        "attachment_leg": {"type": "string"},
        "preserved_invariants": {"type": "array", "items": {"type": "string"}},
        "projection_leg": {"type": "string"},
    },
    "required": [
        "name", "source_node", "target_node", "source_type", "target_type",
        "interface_fields", "added_fields", "attachment_leg",
    ],
}

HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Human-like semantic invariant/relation, not a raw pixel statistic.",
                    },
                    "polarity": {"type": "string", "enum": ["positive_satisfies"]},
                    "diagram": {
                        "type": "object",
                        "properties": {
                            "edges": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string"},
                                        "call": {
                                            "type": "object",
                                            "properties": {
                                                "leg_name": {"type": "string"},
                                                "args": {"type": "array",
                                                         "items": {"type": "string"}},
                                            },
                                            "required": ["leg_name", "args"],
                                        },
                                    },
                                    "required": ["target", "call"],
                                },
                            },
                        },
                        "required": ["edges"],
                    },
                    "score_node": {
                        "type": "string",
                        "description": "Node whose leg codomain is Measurement.",
                    },
                    "order": {"type": "string",
                              "enum": ["low_positive", "high_positive"]},
                    "semantic_requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Every rich term and its score operator (more/fewer, "
                            "no, exact cardinality) from the description."),
                    },
                    "witness_requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Witness types the diagram must produce and the score must depend on.",
                    },
                    "relations": {"type": "array", "items": {"type": "string"}},
                    "cofibrations": {"type": "array", "items": _GLUING_SCHEMA},
                    "preservation_morphisms": {
                        "type": "array", "minItems": 1, "items": _MORPH_SCHEMA},
                },
                "required": ["hypothesis_id", "description", "polarity", "diagram",
                             "score_node", "order", "semantic_requirements",
                             "witness_requirements", "preservation_morphisms"],
            },
        },
    },
    "required": ["hypotheses"],
}


def _leg_lines() -> str:
    lines = []
    for contract in default_registry().contracts():
        domain = ", ".join(contract.domain)
        vocab = f"  (covers: {', '.join(contract.proxy_for)})" if contract.proxy_for else ""
        measurement = (
            f"  [measurement: {contract.measurement_kind}]"
            if contract.measurement_kind else "")
        directions = (
            "  [directions: "
            + ", ".join(f"{term}={direction}"
                        for term, direction in contract.proxy_directions)
            + "]"
            if contract.proxy_directions else "")
        absences = (
            "  [semantic absence: " + ", ".join(contract.failure_modes) + "]"
            if contract.failure_modes else "")
        indeterminate = (
            "  [indeterminate: "
            + ", ".join(contract.indeterminate_modes) + "]"
            if contract.indeterminate_modes else "")
        lines.append(
            f"- {contract.name}: {domain} -> {contract.codomain}"
            f"{measurement}{vocab}{directions}{absences}{indeterminate}")
    return "\n".join(lines)


def _witness_type_lines() -> str:
    types = sorted(t for t in default_registry().terminal_types()
                   if is_witness_codomain(t))
    return ", ".join(types)


def build_prompt(
        problem_id: str, *,
        submission_instruction: str =
        "Submit 3 to 8 hypotheses through the tool.") -> str:
    return PROMPT_TEMPLATE.format(
        problem_id=problem_id,
        submission_instruction=submission_instruction,
        legs=_leg_lines(),
        witness_types=_witness_type_lines(),
    )


def hypotheses_from_tool_input(data: Any) -> tuple[tuple[SemanticHypothesis, ...], str]:
    """Parse the structured tool input; one bad hypothesis never kills the bundle."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return (), "tool input is a non-JSON string"
    if not isinstance(data, dict):
        return (), f"tool input is {type(data).__name__}, expected object"
    items = data.get("hypotheses", [])
    # Models occasionally stringify the array despite the schema; the API
    # does not enforce input_schema, so unwrap it here instead of failing.
    if isinstance(items, str):
        try:
            decoded = json.loads(items)
        except json.JSONDecodeError:
            return (), "hypotheses is a non-JSON string"
        items = decoded.get("hypotheses", decoded) if isinstance(decoded, dict) else decoded
    if not isinstance(items, list):
        return (), "hypotheses is not a list"
    if not 3 <= len(items) <= 8:
        return (), (
            f"hypotheses count is {len(items)}, expected between 3 and 8")
    hypotheses: list[SemanticHypothesis] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    required = {
        "hypothesis_id", "description", "polarity", "diagram", "score_node", "order",
        "semantic_requirements", "witness_requirements",
        "preservation_morphisms",
    }
    for i, item in enumerate(items):
        try:
            if isinstance(item, str):
                item = json.loads(item)
            item = dict(item)
            missing = sorted(required - set(item))
            if missing:
                raise ValueError("missing required fields: " + ", ".join(missing))
            for field_name in (
                    "semantic_requirements", "witness_requirements",
                    "preservation_morphisms"):
                if not isinstance(item[field_name], list):
                    raise TypeError(f"{field_name} must be a list")
            for morph_index, morph in enumerate(item["preservation_morphisms"]):
                if not isinstance(morph, dict):
                    raise TypeError(
                        f"preservation_morphisms[{morph_index}] must be an object")
                missing_morph = {
                    "name", "scope", "expected_effect"} - set(morph)
                if missing_morph:
                    raise ValueError(
                        f"preservation_morphisms[{morph_index}] missing fields: "
                        + ", ".join(sorted(missing_morph)))
            hypothesis = SemanticHypothesis.from_dict(item)
            if hypothesis.hypothesis_id in seen_ids:
                raise ValueError(
                    f"duplicate hypothesis_id {hypothesis.hypothesis_id!r}")
            seen_ids.add(hypothesis.hypothesis_id)
            hypotheses.append(hypothesis)
        except Exception as exc:  # recorded, surfaced in feedback next round
            errors.append(f"hypothesis[{i}]: {type(exc).__name__}: {exc}")
    return tuple(hypotheses), "; ".join(errors)


class AnthropicCofiberedProposer:
    def __init__(self, model: str = "sonnet", max_tokens: int = 8000) -> None:
        self.model = MODEL_MAP.get(model, model)
        self.max_tokens = max_tokens
        self._conversations: dict[str, list[dict]] = {}
        self._last_tool_use_id: dict[str, str | None] = {}

    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        content: list[dict] = [{"type": "text", "text": build_prompt(problem_id)}]
        for path in panel_paths:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            })
        self._conversations[problem_id] = [{"role": "user", "content": content}]
        self._last_tool_use_id[problem_id] = None
        return self._request(problem_id)

    def refine(self, problem_id: str, feedback: str) -> ProposalBundle:
        messages = self._conversations.get(problem_id)
        if messages is None:
            raise RuntimeError(f"refine() before propose() for {problem_id}")
        tool_use_id = self._last_tool_use_id.get(problem_id)
        if tool_use_id:
            content: list[dict] = [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": feedback,
            }]
        else:
            content = [{"type": "text", "text": feedback}]
        messages.append({"role": "user", "content": content})
        return self._request(problem_id)

    # ------------------------------------------------------------------

    def _request(self, problem_id: str) -> ProposalBundle:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for LLM proposals") from exc

        client = anthropic.Anthropic(api_key=_load_api_key())
        messages = self._conversations[problem_id]
        max_tokens = self.max_tokens
        msg = None
        model_receipts: list[dict] = []
        for _truncation_retry in range(3):
            msg = self._create_with_retries(client, messages, max_tokens)
            self._require_response_model(msg)
            model_receipts.append(_model_receipt(msg, self.model))
            if msg.stop_reason != "max_tokens":
                break
            max_tokens = min(max_tokens * 2, MAX_TOKENS_CAP)
        assert msg is not None

        tool_use_id = None
        tool_input = None
        text_parts: list[str] = []
        assistant_content: list[dict] = []
        for block in msg.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_use_id = block.id
                tool_input = block.input
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })

        if tool_input is None:
            # Truncated or refused: keep the conversation replayable by not
            # appending a dangling assistant turn.
            self._last_tool_use_id[problem_id] = None
            reason = ("response truncated at max_tokens after retries"
                      if msg.stop_reason == "max_tokens"
                      else f"no tool_use block (stop_reason={msg.stop_reason})")
            return ProposalBundle(problem_id, (), "\n".join(text_parts),
                                  "anthropic", reason,
                                  tuple(model_receipts))

        messages.append({"role": "assistant", "content": assistant_content})
        self._last_tool_use_id[problem_id] = tool_use_id
        hypotheses, parse_error = hypotheses_from_tool_input(tool_input)
        raw = json.dumps(tool_input, indent=2)
        if text_parts:
            raw = "\n".join(text_parts) + "\n" + raw
        return ProposalBundle(
            problem_id, hypotheses, raw, "anthropic", parse_error,
            tuple(model_receipts))

    def _require_response_model(self, message: Any) -> None:
        """Reject provider routing that changes the preregistered model.

        ``MODEL_MAP`` turns the convenient CLI labels into concrete provider
        identifiers before the request.  The Messages response independently
        names the model that served it; accepting a missing or different value
        would let an alias, gateway, or fallback silently change the Phase D
        treatment while the checkpoint continued to claim the requested one.
        """
        actual = getattr(message, "model", None)
        if not isinstance(actual, str) or not actual:
            raise RuntimeError(
                "Anthropic response omitted its concrete provider model")
        if actual != self.model:
            raise RuntimeError(
                "Anthropic response model differs from the requested concrete "
                f"model ({actual!r} != {self.model!r})")

    def _create_with_retries(self, client, messages: list[dict], max_tokens: int):
        import anthropic

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                return client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                    tools=[{
                        "name": TOOL_NAME,
                        "description": "Submit typed semantic cone hypotheses.",
                        "input_schema": HYPOTHESES_SCHEMA,
                    }],
                    tool_choice={"type": "tool", "name": TOOL_NAME},
                )
            except (anthropic.APIConnectionError, anthropic.RateLimitError,
                    anthropic.InternalServerError) as exc:
                last_exc = exc
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"proposer API failed after retries: {last_exc}")


CODEX_SUBMISSION_INSTRUCTION = (
    "Return 3 to 8 hypotheses as the final JSON object required by the "
    "attached output schema. There is no tool call in this transport: return "
    "only the schema-conforming object, with no classifier code or prose "
    "outside its fields."
)


@dataclass
class _CodexSemanticContext:
    panel_paths: tuple[str, ...]
    payloads: list[dict]
    feedback: list[str]


class CodexCofiberedProposer:
    """Schema-only semantic proposer over the isolated headless Codex runner.

    Every round is a new ephemeral Codex turn with the same twelve direct image
    attachments.  Refinement history travels only in the prompt, so this class
    does not weaken :mod:`codex_proposer`'s private image view, read-only
    sandbox, disabled-tool surface, or exact input/output receipt binding.
    """

    def __init__(
            self,
            model: str = CODEX_DEFAULT_MODEL,
            *,
            minutes: int = 15,
            reasoning_effort: str = codex_headless.DEFAULT_REASONING_EFFORT,
            verbose: bool = True,
            executable: str = "codex") -> None:
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self.executable = executable
        self._contexts: dict[str, _CodexSemanticContext] = {}

    @staticmethod
    def _base_prompt(problem_id: str) -> str:
        return build_prompt(
            problem_id,
            submission_instruction=CODEX_SUBMISSION_INSTRUCTION,
        )

    @staticmethod
    def _payload_text(payload: dict) -> str:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def _prompt(
            self, problem_id: str, context: _CodexSemanticContext,
            pending_feedback: str | None = None) -> str:
        feedback = list(context.feedback)
        if pending_feedback is not None:
            feedback.append(pending_feedback)
        if len(feedback) not in {len(context.payloads) - 1,
                                 len(context.payloads)}:
            raise RuntimeError("Codex semantic refinement history is inconsistent")

        parts = [self._base_prompt(problem_id)]
        for index, payload in enumerate(context.payloads):
            parts.extend((
                f"\nPRIOR STRUCTURED PROPOSAL ROUND {index} "
                "(untrusted candidate data):\n<prior_proposal>",
                self._payload_text(payload),
                "</prior_proposal>",
            ))
            if index < len(feedback):
                parts.extend((
                    "AUTHORITATIVE VERIFIER FEEDBACK FOR THAT ROUND:\n"
                    "<verifier_feedback>",
                    feedback[index],
                    "</verifier_feedback>",
                ))
        if context.payloads:
            parts.append(
                "Return a complete replacement hypothesis bundle that responds "
                "to the verifier feedback; do not return a patch or commentary.")
        return "\n".join(parts)

    def _request(
            self, problem_id: str, context: _CodexSemanticContext,
            *, pending_feedback: str | None = None) \
            -> tuple[ProposalBundle, dict]:
        result = codex_headless.run_codex_structured(
            self._prompt(problem_id, context, pending_feedback),
            context.panel_paths,
            HYPOTHESES_SCHEMA,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            verbose=self.verbose,
            executable=self.executable,
        )
        payload = dict(result.payload)
        hypotheses, parse_error = hypotheses_from_tool_input(payload)
        raw_text = self._payload_text(payload)
        receipt = result.receipt.to_dict()
        # ``run_codex_structured`` already validates the receipt. Keep every
        # field intact: semantic artifacts can therefore reproduce the exact
        # prompt, image view, schema, structured output and event stream.
        return ProposalBundle(
            problem_id=problem_id,
            hypotheses=hypotheses,
            raw_text=raw_text,
            proposer_kind="codex",
            parse_error=parse_error,
            model_receipts=(receipt,),
        ), payload

    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        context = _CodexSemanticContext(tuple(panel_paths), [], [])
        bundle, payload = self._request(problem_id, context)
        context.payloads.append(payload)
        self._contexts[problem_id] = context
        return bundle

    def refine(self, problem_id: str, feedback: str) -> ProposalBundle:
        context = self._contexts.get(problem_id)
        if context is None:
            raise RuntimeError(f"refine() before propose() for {problem_id}")
        if not isinstance(feedback, str):
            raise TypeError("Codex semantic verifier feedback must be text")
        bundle, payload = self._request(
            problem_id, context, pending_feedback=feedback)
        context.feedback.append(feedback)
        context.payloads.append(payload)
        return bundle


def make_cofibered_proposer(
        kind: str,
        model: str | None = None,
        *,
        max_tokens: int = 8000,
        codex_minutes: int = 15,
        codex_reasoning_effort: str =
        codex_headless.DEFAULT_REASONING_EFFORT,
        verbose: bool = True,
        codex_executable: str = "codex") -> CofiberedProposer:
    """Construct a live semantic proposer without changing the IR contract."""
    if kind == "anthropic":
        return AnthropicCofiberedProposer(model or "sonnet", max_tokens)
    if kind == "codex":
        return CodexCofiberedProposer(
            model or CODEX_DEFAULT_MODEL,
            minutes=codex_minutes,
            reasoning_effort=codex_reasoning_effort,
            verbose=verbose,
            executable=codex_executable,
        )
    raise ValueError(f"unknown cofibered proposer kind {kind!r}")


class StaticFixtureProposer:
    """Test-only proposer; do not use for reported experiments."""

    def __init__(self, hypotheses: tuple[SemanticHypothesis, ...]) -> None:
        self.hypotheses = hypotheses

    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        return ProposalBundle(problem_id, self.hypotheses,
                              json.dumps({"hypotheses": [h.to_dict() for h in self.hypotheses]}),
                              "static_fixture")

    def refine(self, problem_id: str, feedback: str) -> ProposalBundle:
        return self.propose(problem_id, [])


PROMPT_TEMPLATE = """\
You are proposing typed semantic cones for a Bongard problem, not writing a
classifier.  The goal is to recover the human-like semantic description that
separates the two sides, with executable typed evidence for every rich term
you name.

Problem id: {problem_id}

You see 12 images: the first six are positive, the next six are negative.
{submission_instruction}

Semantics first: write the invariant as a human would state it, then list
  every content-bearing term in that prose claim (including exclusions you
  mention) in semantic_requirements, the witness types that must carry it in
  witness_requirements, and build a typed diagram whose final Measurement
  score actually depends on those witnesses. The compiler rejects prose terms
  omitted from the structured declaration.

Declaration format (enforced):
- semantic_requirements: short semantic terms
  ("open curve", "triangle", "no crossing", "higher object count"), NOT
  sentences. Include every absence, cardinal, or comparative operator from
  the description; the compiler rejects a prose/structured mismatch. Every content word
  must be carried by a witness type, a leg in the score's dependency path,
  a leg's "covers" vocabulary below, or a declared gluing. Relative words
  must agree with order; `no` means exactly zero on every positive; an exact
  number means that exact count on every positive. A fitted threshold cannot
  reinterpret any of those absolute claims. Bind each comparator to the
  literal metric the final score executes: line length is not line residual,
  and aspect is not occupancy. Use plain integer words/digits; unsupported
  signed, decimal, exponent, or symbolic-comparator syntax fails closed.
- witness_requirements: EXACT witness type names only, chosen from:
  {witness_types}.
  Measurement is not a witness type. For absence claims ("no crossing") do
  NOT demand that witness in witness_requirements — its absence is the claim.
  Prefer an honest counting measurement when one exists; direct absence of
  the final typed witness is also executable.

Typed legs available (domain -> codomain; "covers" lists the vocabulary the
leg's contract discharges):
{legs}

Rules the harness enforces mechanically:
- The diagram starts from the node "panel" (type Panel).  Each edge binds a
  new node: {{"target": "scene", "call": {{"leg_name": "parse_scene",
  "args": ["panel"]}}}}.
- score_node must have codomain Measurement; order says which side is
  positive (low_positive: positives have LOW scores). Directional aliases
  shown in the leg catalog are composed with comparisons/negation and enforced
  against order. Every calibrated term must bind to this final Measurement;
  an upstream witness cannot silently discharge a second scalar claim.
- Witness-producing legs return a typed absence when the claimed structure is
  absent; the verifier can execute literal witness presence/absence rather
  than treating it as a crash. Counting measurements (contact_count,
  intersection_count, part_count, object_count) return 0 honestly. Exact,
  bounded, binary, conjunction, and witness-presence claims become fixed
  decision rules in every support/LOO/transform check; only explicit relative
  comparisons fit a threshold.
- A rich term may never be discharged by a scalar proxy. Metric-only legs
  such as bbox_aspect and bbox_occupancy cover only their literal metrics;
  categorical terms such as thin/filled need an absolute typed predicate.
  Otherwise the harness
  returns MISSING_LEG: that is a useful outcome, do not avoid it by
  weakening the description.
- Composite structure is expressed as gluings: use the cofibrations field to
  declare that source_node is glued into target_node along an interface
  (e.g. a part glued into a part graph at a contact).  Gluings are verified
  on positive panels up to ID renaming and numeric tolerance. The source must
  be extracted from the target (declare and execute projection_leg) unless it
  is otherwise load-bearing; attachment_leg must consume target_node, produce
  a typed relation witness, and feed the final score. Interface and added
  fields must be nonempty and disjoint. If a needed arrow does not exist,
  name it anyway; the harness reports MISSING_LEG.
- preservation_morphisms are executed: translate/rotate/reflect are applied
  to the panels and the cone's decision must be invariant. Each entry must
  explicitly set scope="panel" and expected_effect="preserve". Declare only
  true invariances of your semantic claim.
- Do not mention dataset concepts or filenames.  Do not return code.
"""
