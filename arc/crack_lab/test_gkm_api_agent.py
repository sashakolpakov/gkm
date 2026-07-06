"""Offline tests for the Messages-API proposer loop (gkm_api_agent).

The Anthropic client is faked: scripted responses drive the loop, so the tool
execution, transcript writing, workspace confinement, and loop termination can
be checked without credits or network.
"""
import os
import gkm_api_agent as G


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_loop_runs_tools_then_stops(tmp_path):
    ws = str(tmp_path)
    responses = [
        _Resp([_Block(type="text", text="probing"),
               _Block(type="tool_use", id="t1", name="bash",
                      input={"command": "echo marker42 > probe.txt && cat probe.txt"})],
              "tool_use"),
        _Resp([_Block(type="tool_use", id="t2", name="str_replace_based_edit_tool",
                      input={"command": "create", "path": "players.py",
                             "file_text": "def play_level_1(env):\n    pass\n"})],
              "tool_use"),
        _Resp([_Block(type="text", text="done: wrote play_level_1")], "end_turn"),
    ]
    client = _FakeClient(responses)
    transcript = G.run_agent(ws, "task text", model="sonnet", minutes=5, client=client)

    # bash ran in the workspace; editor wrote the file
    assert open(os.path.join(ws, "probe.txt")).read().strip() == "marker42"
    assert "play_level_1" in open(os.path.join(ws, "players.py")).read()
    # transcript captured text, commands, and landed in proposer_last.log
    assert "probing" in transcript
    assert "$ echo marker42" in transcript
    assert "done: wrote play_level_1" in transcript
    assert transcript == open(os.path.join(ws, "proposer_last.log")).read()
    # tool results went back in a single user message per assistant turn
    third_call = client.messages.calls[2]
    assert third_call["messages"][-1]["role"] == "user"
    assert third_call["messages"][-1]["content"][0]["type"] == "tool_result"
    assert third_call["model"] == "claude-sonnet-4-6"


def test_editor_confined_to_workspace(tmp_path):
    out, err = G._run_editor(str(tmp_path), {"command": "view", "path": "/etc/passwd"})
    assert err and "outside the workspace" in out


def test_api_error_is_logged_not_raised(tmp_path):
    class _Boom:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("Connection error")
    transcript = G.run_agent(str(tmp_path), "task", client=_Boom(), minutes=1)
    assert "API Error" in transcript
