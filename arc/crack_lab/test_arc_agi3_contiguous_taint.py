from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from arc.crack_lab import arc_agi3_codex_app_server_transport as Transport
from arc.crack_lab import arc_agi3_contiguous_conformance as Conformance
from arc.crack_lab import arc_agi3_contiguous_taint as Taint


STATE_ROOT = "/private/contiguous/state/attempt-1"
NEUTRAL_CWD = "/private/contiguous/neutral/attempt-1"
FRONTIER_PROMPT = "receipt-bound public frontier"


def _live_canaries(
    *,
    sibling_value: str = "6" * 64,
) -> tuple[Taint.LiveCanary, ...]:
    values = {
        "repository": "1" * 64,
        "home": "2" * 64,
        "environment": "3" * 64,
        "auth_source": "4" * 64,
        "controller_control_root": "5" * 64,
        "sibling_lane": sibling_value,
    }
    return tuple(
        Taint.LiveCanary(
            category=category,
            location_name=f"fixture:{category}",
            value=values[category],
        )
        for category in Taint.CONTROLLER_CANARY_CATEGORIES
    )


def _policy(
    *,
    secret_sentinels: tuple[str, ...] = (),
) -> Taint.AppServerScanPolicy:
    return Taint.AppServerScanPolicy(
        state_root=STATE_ROOT,
        neutral_cwd=NEUTRAL_CWD,
        model="gpt-5.6-sol",
        model_provider="openai",
        reasoning_effort="max",
        thread_mode="new",
        resume_thread_id=None,
        prompt_sha256=hashlib.sha256(
            FRONTIER_PROMPT.encode("utf-8")
        ).hexdigest(),
        hard_safety_seconds=Taint.APP_SERVER_HARD_SAFETY_SECONDS,
        max_auth_refreshes=Taint.MAX_AUTH_REFRESHES,
        secret_sentinels=secret_sentinels,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_chain(
    path: Path,
    events: list[tuple[str, object]],
) -> None:
    previous: str | None = None
    rows: list[bytes] = []
    for sequence, (direction, payload) in enumerate(events, 1):
        body = {
            "schema": 1,
            "sequence": sequence,
            "previous_digest": previous,
            "direction": direction,
            "payload": payload,
        }
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        rows.append(_canonical({**body, "digest": digest}))
        previous = digest
    path.write_bytes(b"\n".join(rows) + b"\n")


def _request(
    request_id: str,
    method: str,
    params: object,
) -> tuple[str, object]:
    return (
        "client_request",
        {"id": request_id, "method": method, "params": params},
    )


def _response(
    request_id: str,
    result: object,
) -> tuple[str, object]:
    return ("server_response", {"id": request_id, "result": result})


def test_general_taint_dependency_is_control_bound_and_syntax_aware() -> None:
    repository = Path(__file__).resolve().parents[2]
    logical_path = "arc/audit_submission_taint.py"
    assert Taint.Transport is Transport
    assert logical_path in Conformance.CONTROL_CONTRACT_FILES
    assert (
        Path(Taint.GeneralTaint.__file__).resolve()
        == repository / logical_path
    )

    observational_command = (
        "/bin/zsh -lc \"python - <<'PY'\n"
        "bridges = {1}\n"
        "pegs = {2}\n"
        "occupied = bridges|pegs\n"
        "PY\n"
        "awk '/^top$/{section=\\\"top\\\"} "
        "section==\\\"top\\\"{print \\\"top\\\"}' observations.txt\""
    )
    assert Taint._scan_text_full(
        observational_command,
        execution_surface=True,
    ) == []

    python_process_command = (
        "/bin/zsh -lc \"python - <<'PY'\n"
        "import subprocess\n"
        "subprocess.run(['ps', '-ef'])\n"
        "PY\""
    )
    assert "host_process_introspection" in Taint._scan_text_full(
        python_process_command,
        execution_surface=True,
    )
    assert "host_process_introspection" in Taint._scan_text_full(
        "/bin/zsh -lc 'top'",
        execution_surface=True,
    )


def _valid_events() -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    policy = _policy()
    critical_config = {
        "model": policy.model,
        "model_provider": policy.model_provider,
        "model_reasoning_effort": policy.reasoning_effort,
        "approval_policy": "never",
        "sandbox_mode": "read-only",
        "web_search": "disabled",
    }
    disabled_features = {
        name: False
        for name in Taint.AUTHORITY_FEATURE_DENYLIST
    }
    config_origins = {
        **{
            name: {"source": "lane-config"}
            for name in critical_config
        },
        **{
            f"features.{name}": {"source": "lane-config"}
            for name in disabled_features
        },
    }
    result_by_method: dict[str, object] = {
        "initialize": {
            "codexHome": STATE_ROOT,
            "platformFamily": "unix",
            "platformOs": "macos",
            "userAgent":
                "gkm-arc-agi3-contiguous/0.145.0 (test)",
        },
        "account/login/start": {"type": "chatgptAuthTokens"},
        "account/read": {
            "account": {
                "email": "sentinel@example.invalid",
                "planType": "unknown",
                "type": "chatgpt",
            },
            "requiresOpenaiAuth": True,
        },
        "account/rateLimits/read": {
            "rateLimitsByLimitId": {
                "codex": {
                    "planType": "team",
                    "credits": {
                        "hasCredits": True,
                        "unlimited": True,
                        "balance": None,
                    },
                    "spendControlReached": False,
                },
            },
        },
        "model/list": {
            "data": [
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "supportedReasoningEfforts": [
                        {
                            "description": effort,
                            "reasoningEffort": effort,
                        }
                        for effort in (
                            "low",
                            "medium",
                            "high",
                            "xhigh",
                            "max",
                        )
                    ],
                }
            ],
            "nextCursor": None,
        },
        "modelProvider/capabilities/read": {
            "imageGeneration": True,
            "namespaceTools": True,
            "webSearch": True,
        },
        "config/read": {
            "config": {
                **critical_config,
                "apps": None,
                "features": disabled_features,
                "marketplaces": {},
                "mcp_servers": {},
                "plugins": {},
                "tools": None,
            },
            "layers": [
                {
                    "config": {
                        **critical_config,
                        "features": disabled_features,
                    },
                    "name": {
                        "file": f"{STATE_ROOT}/config.toml",
                        "type": "user",
                    },
                },
                {
                    "config": {},
                    "name": {
                        "file": "/etc/codex/config.toml",
                        "type": "system",
                    },
                },
            ],
            "origins": config_origins,
        },
        "skills/list": {
            "data": [
                {
                    "cwd": NEUTRAL_CWD,
                    "errors": [],
                    "skills": [
                        {
                            "enabled": False,
                            "name": name,
                            "path":
                                f"{STATE_ROOT}/skills/.system/{name}/SKILL.md",
                            "scope": "system",
                        }
                        for name in Transport.DISABLED_SYSTEM_SKILLS
                    ],
                }
            ]
        },
        "hooks/list": {
            "data": [
                {
                    "cwd": NEUTRAL_CWD,
                    "errors": [],
                    "hooks": [],
                    "warnings": [],
                }
            ]
        },
        "plugin/list": {
            "featuredPluginIds": [],
            "marketplaceLoadErrors": [],
            "marketplaces": [],
        },
        "app/list": {"data": [], "nextCursor": None},
        "experimentalFeature/list": {
            "data": [
                {"enabled": False, "name": name}
                for name in sorted(Taint.AUTHORITY_FEATURE_DENYLIST)
            ],
            "nextCursor": None,
        },
        "mcpServerStatus/list": {
            "data": [],
            "nextCursor": None,
        },
    }
    for index, method in enumerate(
        Transport.PREFLIGHT_REQUEST_SEQUENCE,
        1,
    ):
        request_id = f"preflight-{index}"
        if method == "account/login/start":
            params = {
                "type": "chatgptAuthTokens",
                "accessToken": "REDACTED",
                "chatgptAccountId": "REDACTED",
            }
        else:
            params = dict(
                Taint._expected_preflight_params(method, policy)
                or {}
            )
        result = result_by_method[method]
        events.extend(
            (
                _request(request_id, method, params),
                _response(request_id, result),
            )
        )
        if method == "initialize":
            events.append(
                (
                    "client_notification",
                    {"method": "initialized", "params": {}},
                )
            )
        elif method == "account/login/start":
            events.extend(
                (
                    (
                        "server_notification",
                        {
                            "method": "account/login/completed",
                            "params": {
                                "error": None,
                                "loginId": None,
                                "success": True,
                            },
                            "emittedAtMs": 1,
                        },
                    ),
                    (
                        "server_notification",
                        {
                            "method": "account/updated",
                            "params": {
                                "authMode": "chatgptAuthTokens",
                                "planType": "unknown",
                            },
                            "emittedAtMs": 1,
                        },
                    ),
                )
            )

    events.extend(
        (
            _request(
                "thread-request",
                "thread/start",
                {
                    "allowProviderModelFallback": False,
                    "approvalPolicy": "never",
                    "baseInstructions": Transport.BASE_INSTRUCTIONS,
                    "cwd": NEUTRAL_CWD,
                    "developerInstructions":
                        Transport.DEVELOPER_INSTRUCTIONS,
                    "dynamicTools":
                        list(Transport.DYNAMIC_TOOL_SPECS),
                    "environments": [],
                    "ephemeral": False,
                    "experimentalRawEvents": False,
                    "historyMode": "paginated",
                    "model": "gpt-5.6-sol",
                    "modelProvider": "openai",
                    "runtimeWorkspaceRoots": [NEUTRAL_CWD],
                    "sandbox": "read-only",
                    "selectedCapabilityRoots": [],
                },
            ),
            _response(
                "thread-request",
                {"thread": {"id": "thread-1"}},
            ),
            (
                "server_notification",
                {
                    "method": "thread/started",
                    "params": {"thread": {"id": "thread-1"}},
                },
            ),
            _request(
                "turn-request",
                "turn/start",
                {
                    "approvalPolicy": "never",
                    "cwd": NEUTRAL_CWD,
                    "effort": "max",
                    "environments": [],
                    "input": [
                        {
                            "text": FRONTIER_PROMPT,
                            "text_elements": [],
                            "type": "text",
                        }
                    ],
                    "model": "gpt-5.6-sol",
                    "runtimeWorkspaceRoots": [NEUTRAL_CWD],
                    "sandboxPolicy": {
                        "networkAccess": False,
                        "type": "readOnly",
                    },
                    "threadId": "thread-1",
                },
            ),
            _response(
                "turn-request",
                {"turn": {"id": "turn-1"}},
            ),
            (
                "server_notification",
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1"},
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "item/started",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "arguments": {"path": "solver.py"},
                            "id": "call-1",
                            "status": "inProgress",
                            "tool": "workspace_read",
                            "type": "dynamicToolCall",
                        },
                    },
                },
            ),
            (
                "server_request",
                {
                    "id": "server-request-1",
                    "method": "item/tool/call",
                    "params": {
                        "arguments": {"path": "solver.py"},
                        "callId": "call-1",
                        "threadId": "thread-1",
                        "tool": "workspace_read",
                        "turnId": "turn-1",
                    },
                },
            ),
            (
                "client_response",
                {
                    "id": "server-request-1",
                    "result": {
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": "declared own-workspace content",
                            }
                        ],
                        "success": True,
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "arguments": {"path": "solver.py"},
                            "id": "call-1",
                            "status": "completed",
                            "success": True,
                            "tool": "workspace_read",
                            "type": "dynamicToolCall",
                        },
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1"},
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                    },
                },
            ),
        )
    )
    return events


def _scan_events(
    tmp_path: Path,
    events: list[tuple[str, object]],
    *,
    policy: Taint.AppServerScanPolicy | None = None,
) -> Taint.ScanRecord:
    path = tmp_path / "app_server.jsonl"
    _write_chain(path, events)
    return Taint.scan_app_server_jsonl(
        path, policy=policy or _policy()
    )


def _response_for_method(
    events: list[tuple[str, object]],
    method: str,
) -> dict[str, Any]:
    request_id = next(
        payload["id"]
        for direction, payload in events
        if direction == "client_request"
        and isinstance(payload, dict)
        and payload.get("method") == method
    )
    return next(
        payload
        for direction, payload in events
        if direction == "server_response"
        and isinstance(payload, dict)
        and payload.get("id") == request_id
    )


def test_complete_exact_lifecycle_and_tool_pairing_passes(
    tmp_path: Path,
) -> None:
    record = _scan_events(tmp_path, _valid_events())
    assert record.evidence_kind == "app_server_jsonl"
    assert record.hits == ()
    assert record.size > 0


def test_login_must_be_redacted(tmp_path: Path) -> None:
    events = _valid_events()
    login = next(
        payload
        for direction, payload in events
        if direction == "client_request"
        and isinstance(payload, dict)
        and payload.get("method") == "account/login/start"
    )
    login["params"]["accessToken"] = "raw-secret"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="redacted",
    ):
        _scan_events(tmp_path, events)


@pytest.mark.parametrize(
    ("method", "replacement"),
    (
        (
            "account/login/start",
            {"type": "apiKey"},
        ),
        (
            "account/read",
            {
                "account": {"type": "apiKey"},
                "requiresOpenaiAuth": False,
            },
        ),
        (
            "modelProvider/capabilities/read",
            {
                "imageGeneration": True,
                "namespaceTools": False,
                "webSearch": True,
            },
        ),
    ),
)
def test_security_preflight_responses_fail_closed(
    tmp_path: Path,
    method: str,
    replacement: object,
) -> None:
    events = _valid_events()
    _response_for_method(events, method)["result"] = replacement
    with pytest.raises(Taint.ContiguousTaintError):
        _scan_events(tmp_path, events)


def test_model_and_config_are_attempt_bound(tmp_path: Path) -> None:
    events = _valid_events()
    model = _response_for_method(
        events, "model/list"
    )["result"]["data"][0]
    model["model"] = "gpt-5.6-terra"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="model",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    config = _response_for_method(
        events, "config/read"
    )["result"]["config"]
    config["model_provider"] = "fallback"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="config",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    config = _response_for_method(
        events, "config/read"
    )["result"]["config"]
    config["features"]["browser_use"] = True
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="config",
    ):
        _scan_events(tmp_path, events)


@pytest.mark.parametrize(
    "method",
    (
        "skills/list",
        "hooks/list",
        "plugin/list",
        "app/list",
        "experimentalFeature/list",
        "mcpServerStatus/list",
    ),
)
def test_authority_inventories_reject_enabled_or_nonempty_rows(
    tmp_path: Path,
    method: str,
) -> None:
    events = _valid_events()
    result = _response_for_method(events, method)["result"]
    if method == "skills/list":
        result["data"][0]["skills"][0]["enabled"] = True
    elif method == "hooks/list":
        result["data"][0]["hooks"] = [{"name": "surprise"}]
    elif method == "plugin/list":
        result["marketplaces"] = [{"name": "surprise"}]
    elif method == "app/list":
        result["data"] = [{"name": "surprise"}]
    elif method == "experimentalFeature/list":
        result["data"][0]["enabled"] = True
    else:
        result["data"] = [{"name": "surprise"}]
    with pytest.raises(Taint.ContiguousTaintError):
        _scan_events(tmp_path, events)


def test_preflight_and_turn_request_documents_are_exact(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    model_request = next(
        payload
        for direction, payload in events
        if direction == "client_request"
        and isinstance(payload, dict)
        and payload.get("method") == "model/list"
    )
    model_request["params"]["includeHidden"] = False
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="parameters",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    thread_request = next(
        payload
        for direction, payload in events
        if direction == "client_request"
        and isinstance(payload, dict)
        and payload.get("method") == "thread/start"
    )
    thread_request["params"]["allowProviderModelFallback"] = True
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="security",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    turn_request = next(
        payload
        for direction, payload in events
        if direction == "client_request"
        and isinstance(payload, dict)
        and payload.get("method") == "turn/start"
    )
    turn_request["params"]["sandboxPolicy"]["networkAccess"] = True
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="security",
    ):
        _scan_events(tmp_path, events)


def test_notification_timestamp_is_retained_and_monotone(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    updated = next(
        payload
        for direction, payload in events
        if direction == "server_notification"
        and isinstance(payload, dict)
        and payload.get("method") == "account/updated"
    )
    updated["emittedAtMs"] = 0
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="emittedAtMs",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    updated = next(
        payload
        for direction, payload in events
        if direction == "server_notification"
        and isinstance(payload, dict)
        and payload.get("method") == "account/updated"
    )
    updated["emittedAtMs"] = True
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="emittedAtMs",
    ):
        _scan_events(tmp_path, events)


def test_server_stderr_is_a_release_hit(tmp_path: Path) -> None:
    events = _valid_events()
    events.insert(
        1,
        (
            "server_stderr",
            "unexpected app-server diagnostic",
        ),
    )
    record = _scan_events(tmp_path, events)
    assert "app_server_stderr" in record.hits


def test_live_credential_sentinel_scan_covers_all_response_bytes(
    tmp_path: Path,
) -> None:
    sentinel = "unique-live-token-sentinel"
    events = _valid_events()
    model = _response_for_method(
        events, "model/list"
    )["result"]["data"][0]
    model["description"] = sentinel
    record = _scan_events(
        tmp_path,
        events,
        policy=_policy(secret_sentinels=(sentinel,)),
    )
    assert record.hits == ("credential_sentinel_exposure",)


def _insert_auth_refresh(
    events: list[tuple[str, object]],
    *,
    request_id: str = "auth-refresh-1",
) -> None:
    insertion = next(
        index
        for index, (direction, payload) in enumerate(events)
        if direction == "server_notification"
        and isinstance(payload, dict)
        and payload.get("method") == "item/started"
    )
    events[insertion:insertion] = [
        (
            "server_request",
            {
                "id": request_id,
                "method": "account/chatgptAuthTokens/refresh",
                "params": {
                    "previousAccountId": "REDACTED",
                    "reason": "unauthorized",
                },
            },
        ),
        (
            "client_response",
            {
                "id": request_id,
                "result": {
                    "accessToken": "REDACTED",
                    "chatgptAccountId": "REDACTED",
                },
            },
        ),
    ]


def test_one_redacted_midturn_auth_refresh_is_admitted(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    _insert_auth_refresh(events)
    assert _scan_events(tmp_path, events).hits == ()


def test_auth_refresh_is_redacted_and_bounded(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    _insert_auth_refresh(events)
    response = next(
        payload
        for direction, payload in events
        if direction == "client_response"
        and isinstance(payload, dict)
        and payload.get("id") == "auth-refresh-1"
    )
    response["result"]["accessToken"] = "raw-token"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="redacted",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    for index in range(1, 8):
        _insert_auth_refresh(
            events,
            request_id=f"auth-refresh-{index}",
        )
    assert _scan_events(tmp_path, events).hits == ()

    _insert_auth_refresh(events, request_id="auth-refresh-8")
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="over-budget",
    ):
        _scan_events(tmp_path, events)


def test_auth_refresh_budget_is_derived_from_hard_safety_ceiling() -> None:
    values = {
        **asdict(_policy()),
        "max_auth_refreshes": 6,
    }
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="policy",
    ):
        Taint.AppServerScanPolicy(**values)


def test_preflight_order_and_cardinality_are_exact(tmp_path: Path) -> None:
    events = _valid_events()
    first_account = next(
        index
        for index, (_, payload) in enumerate(events)
        if isinstance(payload, dict)
        and payload.get("method") == "account/login/start"
    )
    events[first_account], events[first_account + 4] = (
        events[first_account + 4],
        events[first_account],
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="order|notification",
    ):
        _scan_events(tmp_path, events)


def test_mcp_and_unknown_notifications_fail_closed(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    turn_started = next(
        index
        for index, (_, payload) in enumerate(events)
        if isinstance(payload, dict)
        and payload.get("method") == "turn/started"
    )
    events.insert(
        turn_started + 1,
        (
            "server_notification",
            {
                "method": "mcpServer/startupStatus/updated",
                "params": {"name": "forbidden"},
            },
        ),
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="unknown",
    ):
        _scan_events(tmp_path, events)


def test_dynamic_tool_request_requires_exact_pair_and_binding(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    events = [
        event
        for event in events
        if not (
            event[0] == "client_response"
            and isinstance(event[1], dict)
            and event[1].get("id") == "server-request-1"
        )
    ]
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="pending|unresolved|binding|completion",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    tool_request = next(
        payload
        for direction, payload in events
        if direction == "server_request"
    )
    tool_request["params"]["threadId"] = "other-thread"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="binding",
    ):
        _scan_events(tmp_path, events)


def test_tool_result_media_is_text_only(tmp_path: Path) -> None:
    events = _valid_events()
    response = next(
        payload
        for direction, payload in events
        if direction == "client_response"
    )
    response["result"]["contentItems"] = [
        {"type": "inputImage", "imageUrl": "data:image/png;base64,AA=="}
    ]
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="media|schema",
    ):
        _scan_events(tmp_path, events)


def test_item_lifecycle_and_type_are_exact(tmp_path: Path) -> None:
    events = _valid_events()
    started = next(
        payload
        for direction, payload in events
        if direction == "server_notification"
        and isinstance(payload, dict)
        and payload.get("method") == "item/started"
    )
    started["params"]["item"]["type"] = "commandExecution"
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="item",
    ):
        _scan_events(tmp_path, events)

    events = _valid_events()
    events = [
        event
        for event in events
        if not (
            event[0] == "server_notification"
            and isinstance(event[1], dict)
            and event[1].get("method") == "item/started"
        )
    ]
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="binding|completion",
    ):
        _scan_events(tmp_path, events)


def test_duplicate_keys_extra_envelope_fields_and_partial_eof_fail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "app_server.jsonl"
    _write_chain(path, _valid_events())
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[0] = rows[0].replace(
        '"schema":1',
        '"schema":1,"schema":1',
        1,
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="malformed",
    ):
        Taint.scan_app_server_jsonl(path, policy=_policy())

    events = _valid_events()
    _write_chain(path, events)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    first["extra"] = True
    body = dict(first)
    body.pop("digest")
    first["digest"] = hashlib.sha256(_canonical(body)).hexdigest()
    remaining = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text(
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        + "\n"
        + "\n".join(remaining)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="schema",
    ):
        Taint.scan_app_server_jsonl(path, policy=_policy())

    _write_chain(path, events)
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="LF-terminated",
    ):
        Taint.scan_app_server_jsonl(path, policy=_policy())


def test_structured_tool_error_is_scanned_and_retained_as_evidence(
    tmp_path: Path,
) -> None:
    events = _valid_events()
    response = next(
        payload
        for direction, payload in events
        if direction == "client_response"
    )
    response.pop("result")
    response["error"] = {
        "code": -32000,
        "message": "declared tool failure",
    }
    record = _scan_events(tmp_path, events)
    assert record.hits == ()


def test_generic_jsonl_also_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "backend.jsonl"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="malformed",
    ):
        Taint.scan_jsonl(path, evidence_kind="backend_jsonl")


def test_complete_controller_state_scan_binds_every_exact_file(
    tmp_path: Path,
) -> None:
    state = (tmp_path / "state").resolve()
    (state / "sessions").mkdir(parents=True)
    (state / "config.json").write_text(
        '{"theme":"dark"}\n', encoding="utf-8"
    )
    (state / "sessions" / "state.db").write_bytes(
        b"\x00sqlite-like-public-state\xff"
    )
    inventory = Transport.inventory_controller_state(state)

    observed = Taint.scan_controller_state(
        state,
        inventory=inventory,
        canaries=_live_canaries(),
    )

    assert observed.status == "CLEAN"
    assert observed.hits == ()
    assert observed.canary_count == 6
    assert observed.canary_occurrences == 0
    assert observed.tree_sha256 == inventory.tree_sha256
    assert observed.inventory_sha256 == inventory.inventory_sha256
    assert tuple(record.path for record in observed.records) == (
        "config.json",
        "sessions/state.db",
    )
    assert observed.as_receipt()["records"][1]["sha256"] == (
        inventory.files[1][1]
    )


@pytest.mark.parametrize(
    ("payload", "expected_hit"),
    (
        (b"prefix-" + b"6" * 64 + b"-suffix",
         "controller_state_canary_exposure"),
        (b"cached path: wa30.py", "hidden_source_or_prior_solution"),
        (b"\x00HOME=/Users/example\x00", "host_environment_marker"),
        (b"solver inspected env._game", "direct_private_runtime"),
    ),
)
def test_controller_state_scan_detects_state_only_taint_and_canaries(
    tmp_path: Path,
    payload: bytes,
    expected_hit: str,
) -> None:
    state = (tmp_path / "state").resolve()
    state.mkdir()
    (state / "opaque.db").write_bytes(payload)
    inventory = Transport.inventory_controller_state(state)

    observed = Taint.scan_controller_state(
        state,
        inventory=inventory,
        canaries=_live_canaries(
            sibling_value="6" * 64
        ),
    )

    assert observed.status == "TAINT"
    assert expected_hit in observed.hits
    if expected_hit == "controller_state_canary_exposure":
        assert observed.canary_occurrences == 1


def test_controller_state_scan_rejects_post_inventory_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = (tmp_path / "state").resolve()
    state.mkdir()
    target = state / "state.json"
    target.write_text('{"state":1}\n', encoding="utf-8")
    inventory = Transport.inventory_controller_state(state)
    original_inventory = Transport.inventory_controller_state

    def mutate_then_inventory(root: Path, *args: Any, **kwargs: Any):
        target.write_text('{"state":2}\n', encoding="utf-8")
        return original_inventory(root, *args, **kwargs)

    monkeypatch.setattr(
        Transport, "inventory_controller_state", mutate_then_inventory
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="inventory changed",
    ):
        Taint.scan_controller_state(
            state,
            inventory=inventory,
            canaries=_live_canaries(),
        )


def test_live_canaries_require_all_six_unique_crypto_markers() -> None:
    complete = _live_canaries()
    assert len(Taint.validate_live_canaries(complete)) == 6
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="incomplete or malformed",
    ):
        Taint.validate_live_canaries(complete[:-1])
    malformed = (
        Taint.LiveCanary(
            category="repository",
            location_name="fixture:repository",
            value="not-a-cryptographic-marker",
        ),
        *complete[1:],
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="incomplete or malformed",
    ):
        Taint.validate_live_canaries(malformed)


@pytest.mark.parametrize(
    "category",
    Taint.CONTROLLER_CANARY_CATEGORIES,
)
def test_retained_scan_detects_each_canary_category(
    tmp_path: Path,
    category: str,
) -> None:
    host = (tmp_path / "host").resolve()
    output = (tmp_path / "output").resolve()
    host.mkdir()
    output.mkdir()
    canaries = _live_canaries()
    selected = next(
        item for item in canaries if item.category == category
    )
    (output / "artifact.bin").write_bytes(
        b"public-prefix\x00"
        + selected.value.encode("ascii")
        + b"\x00public-suffix"
    )

    observed = Taint.scan_retained_canary_roots(
        {"host_evidence": host, "proposer_output": output},
        canaries=canaries,
    )

    assert observed.status == "TAINT"
    assert observed.canary_occurrences == 1
    assert observed.hits == (
        f"containment_canary_exposure:{category}",
    )


def test_retained_scan_binds_clean_disjoint_roots_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    host = (tmp_path / "host").resolve()
    output = (tmp_path / "output").resolve()
    host.mkdir()
    output.mkdir()
    (host / "bridge.jsonl").write_text(
        '{"event":"public"}\n', encoding="utf-8"
    )
    (output / "solver.py").write_text(
        "def solve(): return True\n", encoding="utf-8"
    )

    observed = Taint.scan_retained_canary_roots(
        {"host_evidence": host, "proposer_output": output},
        canaries=_live_canaries(),
    )

    assert observed.status == "CLEAN"
    assert observed.hits == ()
    assert observed.canary_occurrences == 0
    assert tuple(row[0] for row in observed.root_inventories) == (
        "host_evidence",
        "proposer_output",
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="overlap",
    ):
        Taint.scan_retained_canary_roots(
            {"host_evidence": host, "nested": host},
            canaries=_live_canaries(),
        )


def test_canary_commit_reveal_is_exact_and_values_are_not_in_scan_receipt(
    tmp_path: Path,
) -> None:
    host = (tmp_path / "host").resolve()
    output = (tmp_path / "output").resolve()
    host.mkdir()
    output.mkdir()
    canaries = _live_canaries()
    scan = Taint.scan_retained_canary_roots(
        {"host_evidence": host, "proposer_output": output},
        canaries=canaries,
    )
    encoded_receipt = _canonical(scan.as_receipt())
    assert all(
        item.value.encode("ascii") not in encoded_receipt
        for item in canaries
    )

    reveal = Taint.build_live_canary_reveal(canaries)
    reconstructed = Taint.validate_live_canary_reveal(
        reveal,
        expected_commitments=scan.canary_commitments,
    )
    assert reconstructed == Taint.validate_live_canaries(canaries)

    tampered = json.loads(json.dumps(reveal))
    tampered["canaries"][0]["value"] = "f" * 64
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="commitment mismatch",
    ):
        Taint.validate_live_canary_reveal(
            tampered,
            expected_commitments=scan.canary_commitments,
        )


def test_retained_scan_rejects_post_inventory_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = (tmp_path / "host").resolve()
    output = (tmp_path / "output").resolve()
    host.mkdir()
    output.mkdir()
    target = host / "bridge.jsonl"
    target.write_text('{"state":1}\n', encoding="utf-8")
    original_inventory = Transport.inventory_controller_state
    calls = 0

    def mutate_on_second_inventory(
        root: Path, *args: Any, **kwargs: Any
    ):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text('{"state":2}\n', encoding="utf-8")
        return original_inventory(root, *args, **kwargs)

    monkeypatch.setattr(
        Transport,
        "inventory_controller_state",
        mutate_on_second_inventory,
    )
    with pytest.raises(
        Taint.ContiguousTaintError,
        match="changed during canary scan",
    ):
        Taint.scan_retained_canary_roots(
            {"host_evidence": host, "proposer_output": output},
            canaries=_live_canaries(),
        )
