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

TOOLS = [
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
]

SYSTEM = ("You are a coding agent working headlessly in the directory {ws}. "
          "Use the bash and text editor tools to complete the task; iterate until "
          "it is done. There are no background jobs or wakeups: finish the work "
          "within this session.")


def _cap(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_CAP:
        return text
    return text[:TOOL_OUTPUT_CAP // 2] + "\n...[output truncated]...\n" + text[-TOOL_OUTPUT_CAP // 2:]


def _run_bash(ws: str, command: str) -> tuple[str, bool]:
    try:
        r = subprocess.run(["bash", "-lc", command], cwd=ws, capture_output=True,
                           text=True, timeout=BASH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"command timed out after {BASH_TIMEOUT_S}s", True
    out = (r.stdout or "") + (("\n--- STDERR ---\n" + r.stderr) if r.stderr else "")
    if r.returncode != 0:
        out += f"\n[exit code {r.returncode}]"
    return _cap(out) or "(no output)", r.returncode != 0


def _safe_path(ws: str, path: str) -> str:
    """Resolve an editor path and confine it to the workspace."""
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
            try:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    thinking={"type": "adaptive"},
                    system=SYSTEM.format(ws=ws),
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as ex:  # typed errors stringify with their cause
                emit(f"API Error: {type(ex).__name__}: {ex}")
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
                    else:
                        cmd = tu.input.get("command", "")
                        emit(f"$ {cmd}")
                        out, err = _run_bash(ws, cmd)
                else:
                    emit(f"[edit] {tu.input.get('command')} {tu.input.get('path')}")
                    out, err = _run_editor(ws, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out, "is_error": err})
            messages.append({"role": "user", "content": results})
    finally:
        log.close()
    return open(log_path).read()
