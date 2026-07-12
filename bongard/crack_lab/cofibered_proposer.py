"""Cofibered semantic-cone proposer interface.

The real experiment uses an LLM proposer that returns typed cone IR only,
through a forced structured-output tool call — free-text JSON parsing is
gone, so malformed output and truncation become explicit retryable states
instead of run-killing parse errors.  The proposer does not return final
classifier code.  Static proposals are allowed only in unit tests and are
labeled as fixtures.

The proposer is the ONLY source of semantic novelty: it names the semantic
terms, declares the witness requirements, and generates gluing
(cofibration) requests for composite structure.  The harness never supplies
concept-specific structure; it only type-checks, audits coverage, verifies
and prices.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from semantic_ir import SemanticHypothesis
from semantic_legs import default_registry

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KEY_FILE = os.path.join(REPO_ROOT, "ANTHROPIC_API_KEY.env.local")
MODEL_MAP = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}

TOOL_NAME = "submit_semantic_cones"
MAX_TOKENS_CAP = 16000


@dataclass(frozen=True)
class ProposalBundle:
    problem_id: str
    hypotheses: tuple[SemanticHypothesis, ...]
    raw_text: str
    proposer_kind: str
    parse_error: str = ""


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
        "expected_effect": {"type": "string", "enum": ["preserve", "violate"]},
    },
    "required": ["name"],
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
        "interface_fields": {"type": "array", "items": {"type": "string"}},
        "added_fields": {"type": "array", "items": {"type": "string"}},
        "attachment_leg": {"type": "string"},
        "preserved_invariants": {"type": "array", "items": {"type": "string"}},
        "projection_leg": {"type": "string"},
    },
    "required": ["name", "source_node", "target_node"],
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
                        "description": "Every rich term the description claims.",
                    },
                    "witness_requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Witness types the diagram must produce and the score must depend on.",
                    },
                    "relations": {"type": "array", "items": {"type": "string"}},
                    "cofibrations": {"type": "array", "items": _GLUING_SCHEMA},
                    "preservation_morphisms": {
                        "type": "array", "minItems": 1, "items": _MORPH_SCHEMA},
                    "contrast_interventions": {
                        "type": "array", "items": _MORPH_SCHEMA},
                },
                "required": ["hypothesis_id", "description", "diagram",
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
        lines.append(f"- {contract.name}: {domain} -> {contract.codomain}{vocab}")
    return "\n".join(lines)


def _witness_type_lines() -> str:
    types = sorted(t for t in default_registry().terminal_types()
                   if t.endswith("Witness"))
    return ", ".join(types)


def build_prompt(problem_id: str) -> str:
    return PROMPT_TEMPLATE.format(problem_id=problem_id, legs=_leg_lines(),
                                  witness_types=_witness_type_lines())


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
    hypotheses: list[SemanticHypothesis] = []
    errors: list[str] = []
    for i, item in enumerate(items):
        try:
            if isinstance(item, str):
                item = json.loads(item)
            hypotheses.append(SemanticHypothesis.from_dict(dict(item)))
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
        for _truncation_retry in range(3):
            msg = self._create_with_retries(client, messages, max_tokens)
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
                                  "anthropic", reason)

        messages.append({"role": "assistant", "content": assistant_content})
        self._last_tool_use_id[problem_id] = tool_use_id
        hypotheses, parse_error = hypotheses_from_tool_input(tool_input)
        raw = json.dumps(tool_input, indent=2)
        if text_parts:
            raw = "\n".join(text_parts) + "\n" + raw
        return ProposalBundle(problem_id, hypotheses, raw, "anthropic", parse_error)

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
Submit 3 to 8 hypotheses through the tool.

Semantics first: write the invariant as a human would state it, then list
every rich term of that claim in semantic_requirements, the witness types
that must carry it in witness_requirements, and build a typed diagram whose
final Measurement score actually depends on those witnesses.

Declaration format (enforced):
- semantic_requirements: short structural noun terms, 1-3 words each
  ("open curve", "triangle", "crossing"), NOT sentences.  Every content word
  must be carried by a witness type, a leg in the score's dependency path,
  a leg's "covers" vocabulary below, or a declared gluing; quantity words
  (number, count, high, low, ...) are threshold content and are ignored.
- witness_requirements: EXACT witness type names only, chosen from:
  {witness_types}.
  Measurement is not a witness type.  For absence claims ("no crossing")
  do NOT demand the witness type — its absence is the claim; score it with
  a counting measurement instead.

Typed legs available (domain -> codomain; "covers" lists the vocabulary the
leg's contract discharges):
{legs}

Rules the harness enforces mechanically:
- The diagram starts from the node "panel" (type Panel).  Each edge binds a
  new node: {{"target": "scene", "call": {{"leg_name": "parse_scene",
  "args": ["panel"]}}}}.
- score_node must have codomain Measurement; order says which side is
  positive (low_positive: positives have LOW scores).
- Witness-producing legs raise an explicit error when the claimed structure
  is absent; that is honest evidence, not a bug.  For absence claims use the
  counting measurements (contact_count, intersection_count, part_count,
  object_count, contour_closedness) which return 0 instead of raising.
- A rich term may never be discharged by a scalar proxy (bbox/fill/aspect/
  closure/symmetry_residual are legal only when the leg's own contract names
  your term, e.g. "open"/"closed" for closure_ratio).  Otherwise the harness
  returns MISSING_LEG: that is a useful outcome, do not avoid it by
  weakening the description.
- Composite structure is expressed as gluings: use the cofibrations field to
  declare that source_node is glued into target_node along an interface
  (e.g. a part glued into a part graph at a contact).  Gluings are verified
  per panel up to ID renaming and numeric tolerance.  If the attachment leg
  you need does not exist, name it anyway; the harness reports it as
  MISSING_LEG so the missing arrow becomes visible.
- preservation_morphisms are executed: translate/rotate/reflect are applied
  to the panels and the cone's decision must be invariant.  Declare only
  true invariances of your semantic claim.
- Do not mention dataset concepts or filenames.  Do not return code.
"""
