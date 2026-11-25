"""Shared checks for explicit proposer protocol and containment violations."""
from __future__ import annotations

import json
import re


_INVALID_TURN = re.compile(
    r"\b(?:turn|generation|attempt|lineage)\s+(?:(?:is|was|has been|as)\s+)?invalid(?:ated)?\b"
    r"|\binvalidat(?:es?|ed|ing)\s+(?:and\s+terminat(?:es?|ed|ing)\s+)?"
    r"(?:(?:this|the|my|entire|whole|proposer|discovery)\s+){0,5}"
    r"(?:turn|generation|attempt|lineage)\b"
)
_OWN_ACTION = re.compile(
    r"(?:^|[.!?;:\n]\s*)(?:i|(?:(?:a|my|the)\s+)?probe|(?:my|the)\s+solver)"
    r"\s+(?:had\s+)?(?:attempted|issued|sent|called|supplied|generated|produced)\b"
)
_OWN_BOUNDARY_FAULT = re.compile(
    r"(?:^|[.!?;:\n]\s*)i\s+(?:violated\b|found\s+a\s+boundary\s+violation\b)"
)
_OWN_BOUNDARY_ATTEMPT = re.compile(
    r"(?:^|[.!?;:\n]\s*)(?:(?:a|my|the)\s+)?"
    r"(?:(?:shell\s+)?(?:heredoc|here-document)|(?:shell\s+)?command|probe|compiler|tool)"
    r"\s+(?:had\s+)?(?:attempted|tried)\b[^.!?;\n]*"
    r"\b(?:outside\s+(?:(?:the|my|its|declared)\s+){0,3}workspace"
    r"|(?:at|in|to)\s+(?:an?\s+)?(?:uncontrolled|external|unauthori[sz]ed)\s+location)\b"
)
_TEMP_WRITES = tuple(re.compile(pattern, re.MULTILINE) for pattern in (
    r"(?<![\w-])-o\s*['\"]?\s*,?\s*['\"]?\s*/(?:private/)?(?:var/)?tmp/",
    r">{1,2}\s*['\"]?/(?:private/)?(?:var/)?tmp/",
    r"(?:open|Path)\(\s*['\"]/(?:private/)?(?:var/)?tmp/[^'\"]+['\"]\s*"
    r"(?:,\s*['\"][wax]|\)\s*\.(?:write_text|write_bytes|open)\()",
))


def external_temp_write(text: str) -> bool:
    """Recognize explicit output paths escaping the declared workspace."""
    return any(pattern.search(text) for pattern in _TEMP_WRITES)


def codex_reported_violations(text: str) -> set[str]:
    """Read actual self-invalidation reports, excluding policy hypotheticals."""
    violations: set[str] = set()
    for raw in text.splitlines():
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        message = item.get("text")
        if not isinstance(message, str):
            continue
        message = message.lower()
        if not _INVALID_TURN.search(message):
            continue
        coordinate_fault = (
            "out-of-frame" in message or "out of frame" in message
            or (("outside" in message or "coordinate" in message)
                and ("0..63" in message or "64x64" in message))
        )
        malformed_action = "malformed" in message and (
            "action" in message or re.search(r"\(\s*-?\d+\s*,", message)
        )
        if _OWN_ACTION.search(message) and (coordinate_fault or malformed_action):
            violations.add("public_action_protocol_violation")
        if (
            _OWN_BOUNDARY_ATTEMPT.search(message)
            and ("workspace" in message or "clean-room boundary" in message
                 or "filesystem boundary" in message)
        ) or (
            _OWN_BOUNDARY_FAULT.search(message) and "workspace" in message and (
                "outside" in message or "workspace-only boundary" in message
                or "workspace boundary" in message
            )
        ):
            violations.add("workspace_boundary_violation")
    return violations
