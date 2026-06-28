"""Proposer = the REAL Claude Code agent, uncrippled: given (a) the context the GKM
process DISCOVERED by interaction (avatar/carrier/region/mechanic/toggle, grounded by
the probe -- not hand-coded), (b) the human-preconception priors, and (c) TOOLS plus
a tester, so it can WRITE a solve(env) program, RUN it against the real arena, see it
fail, and FIX it -- the same iterate-and-verify loop a human (or Claude in a chat) uses.

The earlier `--proposer=claude` was one-shot blind text (no tools, no testing, no
discovered context); that is why it underperformed -- same model, blindfolded. Here
the agent runs its own loop; the GKM layer still verifies + prices the final program
by free energy on the game.

    python gkm_solve_agent.py [--game=wa30] [--model=NAME] [--minutes=40]
"""
from __future__ import annotations
import os
import subprocess
import sys
import time

import gkm_arena as A
import gkm_crack as GC


def discovered_context(game: str) -> str:
    """Run the GKM discovery (probe, no hand-coding) to ground the game's semantics,
    then state them as context for the proposer."""
    C = GC.CarryConnector.build(game, use_llm=False, verbose=False)
    B = C.B
    return (
        f"DISCOVERED BY INTERACTION (grounded, trust these): the avatar (you) is colour "
        f"{B.avatar}; movable carriers are colour {B.carrier}; the target container's "
        f"interior is colour {B.region}; background is colour {B.background}. The "
        f"manipulation mechanic is {B.mechanic}: action {B.toggle} ATTACHES the carrier "
        f"you are facing (then moving carries it) and, pressed again, RELEASES it. A "
        f"carrier counts as delivered when it rests inside the container region and is "
        f"not attached. Movement actions translate the avatar; you cannot push carriers "
        f"by walking into them -- you must attach+carry+release. Later levels may add a "
        f"WALL splitting the board and an autonomous orange HELPER agent that also "
        f"carries (it acts on every move you make) -- and a PURPLE adversary you can "
        f"remove by attaching to it; confirm these by experiment on clones.")


TESTER = '''import importlib.util, sys
sys.path.insert(0, {labdir!r})
import gkm_arena as A
spec = importlib.util.spec_from_file_location("solution", "solution.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
levels, path, err = A.run_program({game!r}, m.solve)
ok = A.validate({game!r}, path, levels) if path else False
print(f"RESULT levels={{levels}} moves={{len(path)}} replay_ok={{ok}} err={{err}}")
'''


def build_task(game: str, context: str) -> str:
    return (
        A.PRECONCEPTIONS + "\n\n" + context + "\n\n" + A.API +
        "\n\nWORKFLOW (you have Bash/Write/Read/Edit tools in this directory):\n"
        "1. Write your program to `solution.py` here (it must define solve(env)).\n"
        "2. Test it by running `python gkm_try.py` and read its RESULT line "
        "(levels / moves / replay_ok / err).\n"
        "3. ITERATE: write tiny experiment scripts to learn the mechanics on clones "
        "(e.g. confirm the attach->carry->release loop actually moves a carrier; once "
        "you are next to an object, TRY ALL actions on it and observe), fix bugs, "
        "improve the strategy, and re-run the tester. Keep going until RESULT shows "
        f"levels>=1 for {game} (push for 2 then 3 if you can).\n"
        "4. Leave your BEST working solve(env) in `solution.py`. Then print the final "
        "RESULT line you achieved.\n"
        "IMPORTANT: if `solution.py` already clears some levels, do NOT restart -- "
        "READ it and EXTEND it to clear the NEXT level. solve(env) keeps playing as "
        "levels advance, so handle each level in turn. Later levels add an autonomous "
        "HELPER (orange) that also carries (it moves on every move you make) and a WALL "
        "that splits the board so YOU cannot reach the container -- then RELAY: carry a "
        "carrier to the wall boundary, release it where the helper can take it, step "
        "away. Discover these by experiment; push to level 2 then level 3.\n"
        "Note: env.clone()/step are ~300/s; use bounded per-subgoal search, not "
        "exhaustive global search. The move budget per level is limited.")


def run(game="wa30", model=None, minutes=40, verbose=True):
    ws = f"/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad/gkm_ws_{game}"
    os.makedirs(ws, exist_ok=True)
    labdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(ws, "gkm_try.py"), "w") as fh:
        fh.write(TESTER.format(labdir=labdir, game=game))
    # seed an empty solution so the tester never crashes on import before the agent writes
    sol = os.path.join(ws, "solution.py")
    if not os.path.exists(sol):
        with open(sol, "w") as fh:
            fh.write("def solve(env):\n    return\n")

    context = discovered_context(game)
    if verbose:
        print("DISCOVERED CONTEXT handed to the proposer:\n " + context + "\n")
    task = build_task(game, context)

    cmd = ["claude", "-p", task,
           "--allowedTools", "Bash", "Read", "Write", "Edit",
           "--dangerously-skip-permissions",
           "--add-dir", labdir,
           "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    if verbose:
        print(f"invoking Claude agent (tools + tester) in {ws} ... (up to {minutes} min)\n")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                              timeout=minutes * 60)
        if verbose:
            print("=== agent transcript (tail) ===")
            print((proc.stdout or "")[-2000:])
            if proc.stderr:
                print("=== stderr (tail) ===\n" + proc.stderr[-800:])
    except subprocess.TimeoutExpired:
        print(f"agent timed out after {minutes} min")

    # GKM verification: run the agent's final program on a fresh game + price it
    code = open(sol).read()
    solve, cerr = A._compile(code)
    if solve is None:
        print(f"\n=== {game}: agent's solution.py did not compile: {cerr} ===")
        return {"levels": 0, "compiled": False}
    levels, path, rerr = A.run_program(game, solve)
    ok = A.validate(game, path, levels) if path else False
    F = A.free_energy(levels, code, path) if ok else float("inf")
    print(f"\n=== {game}: AGENT-WRITTEN solve() -> level {levels} | replay-validated={ok} "
          f"| moves={len(path)} | F={F:.3f} | wrote in {time.time()-t0:.0f}s ===")
    if path:
        print(f"PATH={path}")
    return {"levels": levels, "path": path, "validated": ok, "F": F, "code": code}


if __name__ == "__main__":
    game, model, minutes = "wa30", None, 40
    for a in sys.argv[1:]:
        if a.startswith("--game="): game = a.split("=", 1)[1]
        elif a.startswith("--model="): model = a.split("=", 1)[1]
        elif a.startswith("--minutes="): minutes = int(a.split("=", 1)[1])
    run(game=game, model=model, minutes=minutes)
