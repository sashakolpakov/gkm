"""Data-only glue proposals through the existing Codex login; no API keys copied.

The CLI runs in an empty temporary directory with ambient config and tool
features disabled. Any tool event invalidates the attempt. This is a local
experiment, not the ARC production container's stronger containment claim.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path


MODEL = "gpt-5.6-sol"
DISABLED = (
    "shell_tool", "unified_exec", "code_mode", "code_mode_only", "code_mode_host",
    "code_mode_buffered_exec", "multi_agent", "multi_agent_v2", "enable_fanout",
    "apps", "plugins", "hooks", "memories", "js_repl", "computer_use", "browser_use",
    "tool_search", "skill_search", "remote_control", "image_generation",
    "standalone_web_search",
)
DISABLED_HOST_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; "
    "enable `features.code_mode_host` and install `codex-code-mode-host`."
)


def schema():
    integer = {"type": "integer"}
    rule = {"type": "object", "additionalProperties": False,
            "properties": {"state": integer, "observation": integer,
                           "next_state": integer,
                           "actions": {"type": "array", "items": integer}},
            "required": ["state", "observation", "actions", "next_state"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"library_hash": {"type": "string"}, "fresh_states": integer,
                           "ports": {"type": "array", "items": integer},
                           "rules": {"type": "array", "items": rule}},
            "required": ["library_hash", "fresh_states", "ports", "rules"]}


def command(workdir, schema_path):
    args = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
            "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--color", "never",
            "--model", MODEL, "--cd", str(workdir), "--output-schema", str(schema_path),
            "-c", 'model_provider="openai"', "-c", 'model_reasoning_effort="medium"',
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", "memories.use_memories=false", "-c", "memories.generate_memories=false"]
    for feature in DISABLED:
        args += ["--disable", feature]
    args += ["-c", 'skills.config=[{name="imagegen",enabled=false},'
             '{name="openai-docs",enabled=false},{name="skill-creator",enabled=false},'
             '{name="skill-installer",enabled=false}]', "-"]
    return args


def extract(events):
    messages = []
    completed = False
    usage = None
    started = False
    for event in events:
        if event.get("type") == "turn.started":
            started = True
        if event.get("type") in {"error", "turn.failed"}:
            raise RuntimeError("Codex turn failed; inspect the local event receipt")
        item = event.get("item")
        if item is not None:
            if not started and item.get("type") == "error" and item.get("message") == DISABLED_HOST_NOTICE:
                continue
            if item.get("type") not in {"agent_message", "reasoning"}:
                raise RuntimeError("tool or unexpected item invalidates the proposal")
            if event["type"] == "item.completed" and item["type"] == "agent_message":
                messages.append(item["text"])
        if event.get("type") == "turn.completed":
            completed = True
            usage = event.get("usage")
    if not completed or len(messages) != 1:
        raise RuntimeError("expected one complete structured Codex answer")
    return json.loads(messages[0]), usage


def _rss_kib(process_group):
    result = subprocess.run(["ps", "-axo", "pgid=,rss="], capture_output=True,
                            text=True, timeout=5, check=True)
    return sum(int(parts[1]) for line in result.stdout.splitlines()
               if len(parts := line.split()) == 2 and int(parts[0]) == process_group)


def propose(prompt, artifact_dir: Path, timeout=120, memory_mib=768, response_schema=None):
    artifact_dir.mkdir(parents=True, exist_ok=False)
    raw_prompt = json.dumps(prompt, sort_keys=True)
    (artifact_dir / "request.json").write_text(raw_prompt)
    schema_path = artifact_dir.resolve() / "schema.json"
    schema_path.write_text(json.dumps(schema() if response_schema is None else response_schema))
    # Avoid accidental API-key billing; the existing managed Codex login is used.
    child_env = dict(os.environ)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        child_env.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="gkm-transduction-proposer-") as neutral:
        args = command(neutral, schema_path)
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=neutral, env=child_env,
                                start_new_session=True)
        selector = selectors.DefaultSelector()
        streams = {"stdout": bytearray(), "stderr": bytearray()}
        peak_rss = 0
        started = last_check = time.monotonic()
        try:
            proc.stdin.write(raw_prompt.encode())
            proc.stdin.close()
            for name in streams:
                selector.register(getattr(proc, name), selectors.EVENT_READ, name)
            while selector.get_map():
                now = time.monotonic()
                if now - started > timeout:
                    raise RuntimeError("Codex proposal exceeded wall-time limit")
                if now - last_check >= 1:
                    peak_rss = max(peak_rss, _rss_kib(proc.pid))
                    last_check = now
                    if peak_rss > memory_mib * 1024:
                        raise RuntimeError("Codex process group exceeded memory limit")
                for key, _ in selector.select(0.2):
                    data = os.read(key.fileobj.fileno(), 16384)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    streams[key.data].extend(data)
                    if len(streams[key.data]) > 2 * 1024 * 1024:
                        raise RuntimeError("Codex transcript exceeded byte limit")
            if proc.wait(timeout=5) != 0:
                raise RuntimeError("Codex exited unsuccessfully; see local receipt")
            events = [json.loads(line) for line in streams["stdout"].splitlines() if line.strip()]
            result, usage = extract(events)
            (artifact_dir / "receipt.json").write_text(json.dumps({
                "model": MODEL, "transport": "codex exec; existing managed login",
                "argv": args, "seconds": time.monotonic() - started,
                "peak_process_group_rss_kib": peak_rss, "usage": usage,
                "tool_events": 0,
            }, indent=2))
            return result
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
            selector.close()
            proc.stdout.close()
            proc.stderr.close()
            (artifact_dir / "events.jsonl").write_bytes(streams["stdout"])
            (artifact_dir / "stderr.log").write_bytes(streams["stderr"])
