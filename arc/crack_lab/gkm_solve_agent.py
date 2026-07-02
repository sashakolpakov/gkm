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


def discovered_context(game: str) -> str:
    """GAME-AGNOSTIC: run the GKM directed probe (gkm_discovery, no hand-coding) to
    ground whatever this game affords -- the controllable avatar and its movement
    actions/vectors. Hand that, and ONLY that, to the proposer; the manipulation
    semantics, the goal, and any remaining mechanics are for the agent to discover
    from frames. (Earlier versions also passed the probe's named manipulation verbs
    -- 'push' / 'pick_up_and_carry' -- but those names come from a HAND-CODED verb
    vocabulary in gkm_discovery, which pre-tells the game's nature. Retired: the
    agent must name its own mechanics.)"""
    import gkm_discovery as D
    try:
        verbs, effects, w = D.discover(game, use_llm=False, verbose=False)
    except Exception as ex:
        return ("DISCOVERY: directed probing found NO clear controllable avatar "
                f"({str(ex)[:80]}). Discover the avatar, the mechanics, and the goal "
                "entirely from the raw frames yourself.")
    mv = {a: tuple(v) for a, v in sorted(w.movement.items())}
    return (
        f"DISCOVERED BY INTERACTION (grounded -- trust these, confirm the rest): the "
        f"controllable avatar is colour {w.anchor_color}; movement actions {sorted(w.movement)} "
        f"translate it with pixel vectors {mv}; the remaining action(s) "
        f"{list(w.effects)} did something other than move the avatar -- what they do "
        f"is for YOU to determine by experiment. These were found by probing each "
        f"action on clones. The action numbering may be NON-standard. You must still "
        f"DISCOVER the goal/win-condition (from how the reward changes) and any "
        f"further mechanics, objects, barriers, or other agents by experiment on "
        f"clones.")


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
        "levels advance, so handle each level in turn. Later levels typically ESCALATE "
        "(more objects, new barriers, and possibly other autonomous agents that act on "
        "every move you make -- some helpful, some adversarial). Discover each level's "
        "structure by experiment on clones and ADAPT your strategy per level.\n"
        "\nGROW A LEG LIBRARY (this is the point -- minimise novelty on later levels):\n"
        "- Keep a persistent `legs.py` of small, well-named, reusable skills (legs): "
        "e.g. perception helpers, `go_adjacent_facing(env,cell)`, "
        "`carry_object_to(env,obj,dest)`, `relay_across(env,obj,boundary)`, "
        "`bfs(...)`. `solve(env)` and per-level players IMPORT from `legs.py` and "
        "COMPOSE these legs; they should contain little logic of their own.\n"
        "- On the EARLY levels you will invent most legs (you are still learning the "
        "rules). On LATER levels, prefer to RECOGNISE that a level is an earlier one in "
        "a different geometric configuration but semantically the same, and solve it by "
        "COMPOSING existing legs, adding as FEW new legs as possible. The novelty should "
        "live in the COMBINATION, not in new legs -- iterate on the composition, not the "
        "legs. (Free energy rewards this: a reused leg is already paid for; only NEW "
        "structure costs.)\n"
        "- After you clear each level, DEBRIEF: compare this level's player to the "
        "previous levels', refactor any repeated code into a shared leg in `legs.py` (so "
        "it is written once), and jot the recurring composition pattern in "
        "`legs_log.md`. Re-run `gkm_try.py` to confirm behaviour is unchanged.\n"
        "Note: env.clone()/step are ~300/s; use bounded per-subgoal search, not "
        "exhaustive global search. The move budget per level is limited. Push as far "
        "through the levels as you can (they go up to 9).")


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
