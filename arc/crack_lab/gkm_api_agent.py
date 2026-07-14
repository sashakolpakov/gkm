"""Messages-API agentic proposer: a lean write/run/fix loop over the Anthropic API.

Third proposer backend for gkm_legs.py (after the Claude Code CLI and opencode/codex),
billed against ANTHROPIC_API_KEY Console credits -- a separate pool from the CLI
subscription, so it survives a CLI credit-out.

The loop is deliberately minimal: the model gets the SAME 7-sentence task as every
other backend (no extra prose -- see FINDINGS R-PROMPT-MINIMALISM) plus two
Anthropic-defined, client-executed tools:

  * bash (bash_20250124)                 -- runs in the workspace, output truncated
  * text editor (text_editor_20250728)   -- view/create/str_replace/insert, confined
                                            to the workspace directory

Everything the agent says and does is appended to proposer_last.log in the workspace
so gkm_legs' credit-out markers, transient-failure retry, and forensics work
unchanged. The `client` is injectable for offline unit tests.
"""
from __future__ import annotations
import os
import re
import subprocess
import time
from typing import Optional

MODEL_ALIASES = {
    None: "claude-sonnet-4-6",       # cheap path default (R-SONNET)
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}

MAX_TOKENS = 16000
BASH_TIMEOUT_S = 600        # gkm_try.py BFS verifications can take minutes
TOOL_OUTPUT_CAP = 20000     # chars per tool result kept in context
API_TRANSIENT_RETRIES = 6
API_TRANSIENT_BACKOFF_S = (20, 45, 90, 180, 300, 300)
BLOCKED_ATTEMPTS_LOG = "blocked_attempts.log"

TOOLS = [
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
]

SYSTEM = ("You are a coding agent working headlessly in the directory {ws}. "
          "Use the bash and text editor tools to complete the task; iterate until "
          "it is done. There are no background jobs or wakeups: finish the work "
          "within this session. Never read game implementation source: do not inspect "
          "environment_files, game source .py files outside the workspace, or prior "
          "solutions. Discover mechanics only by running the arena and observing frames.")

ARC_GAME_SOURCE_NAMES = tuple(
    f"{game}.py" for game in (
        "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
        "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
        "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
        "wa30",
    )
)

FORBIDDEN_SOURCE_PATTERNS = (
    "environment_files",
    "/environment_files/",
    "agent_solutions/",
    "/agent_solutions/",
) + ARC_GAME_SOURCE_NAMES


def _forbidden_source_reference(text: str) -> Optional[str]:
    """Return the forbidden marker when a tool command tries to read source/history."""
    lowered = text.lower()
    if re.search(r"\.\s*_(?:game|env)\b|__dict__|inspect\.getsource", lowered):
        return "private game/runtime introspection"
    for marker in FORBIDDEN_SOURCE_PATTERNS:
        if marker.lower() in lowered:
            return marker
    if re.search(r"find\s+(?:/|\.{2}|~).*-(?:name|path)\s+['\"]?\*?(?:wa30|ls20)\*?", lowered):
        return "filesystem search for game source"
    if re.search(r"(?:cat|sed|grep|rg|head|tail|less|more|python\d*\s+-c).*?(?:wa30|ls20)\.py", lowered, re.S):
        return "direct game source file access"
    if re.search(r"(?:cat|sed|grep|rg|head|tail|less|more).*?(?:prior|previous).*solution", lowered, re.S):
        return "prior solution history access"
    return None


def _workspace_forbidden_reference(ws: str) -> Optional[str]:
    """Scan agent-authored workspace text before running shell commands.

    The model can write a probe script and then execute it, so checking only the
    shell command is not enough. This scan is intentionally conservative: any
    workspace file containing source/history markers taints the attempt before
    it can be run.
    """
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache"}]
        for name in files:
            if name == BLOCKED_ATTEMPTS_LOG:
                continue
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    marker = _forbidden_source_reference(f.read())
            except OSError:
                continue
            if marker:
                return f"{marker} in workspace file {os.path.relpath(path, ws)}"
    return None


def _cap(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_CAP:
        return text
    return text[:TOOL_OUTPUT_CAP // 2] + "\n...[output truncated]...\n" + text[-TOOL_OUTPUT_CAP // 2:]


def _tool_env() -> dict:
    """Pass normal process env to tools, minus credentials they do not need."""
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")
    return {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in secret_markers)
    }


def _redact_secrets(text: str) -> str:
    out = text
    for key, value in os.environ.items():
        if value and key not in _tool_env():
            out = out.replace(value, "[REDACTED]")
    out = re.sub(r"(ANTHROPIC_API_KEY\s*=\s*)\S+", r"\1[REDACTED]", out)
    return out


def _record_blocked_attempt(ws: str, tool: str, payload) -> None:
    """Keep rejected tool input for audit without treating it as executed access."""
    path = os.path.join(ws, BLOCKED_ATTEMPTS_LOG)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{time.time():.6f}] {tool}: {payload!r}\n")


def _transient_api_failure(ex: Exception) -> bool:
    text = f"{type(ex).__name__}: {ex}".lower()
    markers = (
        "credit balance",
        "credits",
        "quota",
        "rate limit",
        "overloaded",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "529",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in markers)


def _run_bash(ws: str, command: str) -> tuple[str, bool]:
    forbidden = _forbidden_source_reference(command)
    if forbidden:
        return (f"forbidden source/history access blocked: {forbidden}. "
                "Use arena clones and frame observations only."), True
    forbidden = _workspace_forbidden_reference(ws)
    if forbidden:
        return (f"forbidden source/history access blocked: {forbidden}. "
                "Remove the tainted workspace content and use arena observations only."), True
    try:
        r = subprocess.run(["bash", "-lc", command], cwd=ws, capture_output=True,
                           text=True, timeout=BASH_TIMEOUT_S, env=_tool_env())
    except subprocess.TimeoutExpired:
        return f"command timed out after {BASH_TIMEOUT_S}s", True
    out = (r.stdout or "") + (("\n--- STDERR ---\n" + r.stderr) if r.stderr else "")
    if r.returncode != 0:
        out += f"\n[exit code {r.returncode}]"
    return _cap(_redact_secrets(out)) or "(no output)", r.returncode != 0


def _safe_path(ws: str, path: str) -> str:
    """Resolve an editor path and confine it to the workspace."""
    forbidden = _forbidden_source_reference(path)
    if forbidden:
        raise ValueError(f"forbidden source/history access blocked: {forbidden}")
    p = os.path.realpath(os.path.join(ws, path) if not os.path.isabs(path) else path)
    root = os.path.realpath(ws)
    if p != root and not p.startswith(root + os.sep):
        raise ValueError(f"path {path} is outside the workspace")
    return p


def _run_editor(ws: str, inp: dict) -> tuple[str, bool]:
    try:
        cmd = inp["command"]
        path = _safe_path(ws, inp["path"])
        if cmd == "view":
            if os.path.isdir(path):
                return "\n".join(sorted(os.listdir(path))), False
            with open(path) as f:
                lines = f.readlines()
            rng = inp.get("view_range")
            if rng:
                lo, hi = rng[0], (len(lines) if rng[1] == -1 else rng[1])
                lines = lines[lo - 1:hi]
                start = lo
            else:
                start = 1
            return _cap("".join(f"{i + start}: {ln}" for i, ln in enumerate(lines))), False
        if cmd == "create":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(inp["file_text"])
            return f"created {path}", False
        if cmd == "str_replace":
            with open(path) as f:
                text = f.read()
            n = text.count(inp["old_str"])
            if n != 1:
                return f"str_replace needs exactly 1 match of old_str, found {n}", True
            with open(path, "w") as f:
                f.write(text.replace(inp["old_str"], inp.get("new_str", ""), 1))
            return f"edited {path}", False
        if cmd == "insert":
            with open(path) as f:
                lines = f.readlines()
            at = inp["insert_line"]
            lines[at:at] = [inp["insert_text"] if inp["insert_text"].endswith("\n")
                            else inp["insert_text"] + "\n"]
            with open(path, "w") as f:
                f.writelines(lines)
            return f"inserted into {path}", False
        return f"unsupported command {cmd}", True
    except (KeyError, ValueError, OSError) as ex:
        return f"error: {ex}", True


def run_agent(ws: str, task: str, model: Optional[str] = None, minutes: int = 40,
              client=None, log_path: Optional[str] = None) -> str:
    """Drive the agentic loop until the model stops calling tools or the time
    budget runs out. Returns the transcript (also written to ``log_path``)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model_id = MODEL_ALIASES.get(model, model)
    log_path = log_path or os.path.join(ws, "proposer_last.log")
    deadline = time.monotonic() + minutes * 60
    messages = [{"role": "user", "content": task}]
    log = open(log_path, "w")

    def emit(text: str) -> None:
        log.write(text + "\n")
        log.flush()

    try:
        while True:
            if time.monotonic() > deadline:
                emit(f"[api proposer hit {minutes}min budget; verifying partial work]")
                break
            for attempt in range(API_TRANSIENT_RETRIES + 1):
                try:
                    response = client.messages.create(
                        model=model_id,
                        max_tokens=MAX_TOKENS,
                        thinking={"type": "adaptive"},
                        system=SYSTEM.format(ws=ws),
                        tools=TOOLS,
                        messages=messages,
                    )
                    break
                except Exception as ex:  # typed errors stringify with their cause
                    if attempt < API_TRANSIENT_RETRIES and _transient_api_failure(ex):
                        delay = API_TRANSIENT_BACKOFF_S[
                            min(attempt, len(API_TRANSIENT_BACKOFF_S) - 1)
                        ]
                        emit(_redact_secrets(
                            f"API transient error; retry {attempt + 1}/"
                            f"{API_TRANSIENT_RETRIES} after {delay}s: "
                            f"{type(ex).__name__}: {ex}"
                        ))
                        if time.monotonic() + delay > deadline:
                            emit("[api retry skipped: proposer time budget exhausted]")
                            emit(_redact_secrets(f"API Error: {type(ex).__name__}: {ex}"))
                            response = None
                            break
                        time.sleep(delay)
                        continue
                    emit(_redact_secrets(f"API Error: {type(ex).__name__}: {ex}"))
                    response = None
                    break
            if response is None:
                break
            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text:
                    emit(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)
            if response.stop_reason != "tool_use" or not tool_uses:
                if response.stop_reason not in ("end_turn", "tool_use"):
                    emit(f"[stopped: {response.stop_reason}]")
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                if tu.name == "bash":
                    if tu.input.get("restart"):
                        out, err = "(stateless shell; nothing to restart)", False
                        emit("$ [restart ignored: stateless shell]")
                    else:
                        cmd = tu.input.get("command", "")
                        out, err = _run_bash(ws, cmd)
                        if out.startswith("forbidden source/history access blocked:"):
                            _record_blocked_attempt(ws, "bash", cmd)
                            emit(f"$ [BLOCKED: {out.split('.', 1)[0]}]")
                        else:
                            emit(f"$ {cmd}")
                else:
                    out, err = _run_editor(ws, tu.input)
                    if "forbidden source/history access blocked:" in out:
                        _record_blocked_attempt(ws, "editor", dict(tu.input))
                        emit(f"[edit BLOCKED: {out.split('.', 1)[0]}]")
                    else:
                        emit(f"[edit] {tu.input.get('command')} {tu.input.get('path')}")
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out, "is_error": err})
            messages.append({"role": "user", "content": results})
    finally:
        log.close()
    return open(log_path).read()
