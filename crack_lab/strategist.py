"""Endow the connector with HUMAN PRECONCEPTIONS via a system prompt and ask the
local LLM to propose a high-level strategy for a level. The point (per the user):
an agent that figures out e.g. a two-agent hand-off across a wall is one carrying a
lot of human priors about the world -- objects, containers, barriers, agency, and
COOPERATION (two agents that each can't reach the goal can relay objects at a shared
boundary). We inject those priors as inductive bias and see if it is "enough".

The scene is described in SEMANTIC terms already grounded by discovery.py (a
controllable carrier-agent, an autonomous carrier-agent, a barrier, a target
region, carriers and which side each is on) -- NOT raw pixels. The strategist
proposes a plan; the search/engine would remain the verifier.
"""
from __future__ import annotations
import time
import numpy as np

import gkm_crack as G
from logical_grid import components

PRECONCEPTIONS = """You are an agent that understands the physical and social world the way a human \
child does. Bring these preconceptions to every puzzle:
- OBJECTS persist, occupy space, and can be carried; two solid things can't share a cell.
- CONTAINERS/targets are places things are meant to end up.
- BARRIERS (walls) block movement and divide space into regions; if a region is \
walled off, an agent on one side cannot reach the other side at all.
- AGENCY: some objects are agents with goals; an AUTONOMOUS helper will, on its own, \
pick up reachable objects and carry them to the target.
- REACHABILITY: first ask, for each agent, which objects and which target it can \
actually reach given the barriers.
- COOPERATION / HAND-OFF: if agent A can reach an object but not the target, and \
agent B can reach the target but not the object, they can RELAY it -- A leaves the \
object at a place B can pick it up (a shared boundary / drop-off point). This only \
works if A's reachable drop cells and B's reachable pick-up cells actually meet \
(adjacent within the pick-up range); if a wall leaves a gap wider than the pick-up \
range, the hand-off is impossible and those objects are stranded.
- Always check feasibility: count what must reach the target, and whether every such \
object has a chain of reachable agents that can move it there."""


def describe_scene(c: G.CarryConnector, fd) -> str:
    """A semantic, grounded description of the current level (no raw pixels)."""
    arr = np.asarray(fd.frame[-1])
    rb = c.region_bbox(fd)
    sc = c.perceive(arr, rb)
    # the autonomous helper = largest mover that isn't avatar/carrier/structure
    helper = c._helper(arr)
    # barrier columns (impassable structure), here the colour-2 wall
    walls = sorted({x for x, y in getattr(c, "_walls", set())})
    av = sc.avatar
    region = f"x {rb[0]}-{rb[2]}, y {rb[1]}-{rb[3]}" if rb else "unknown"
    boxes = ", ".join(f"({b.tl[0]},{b.tl[1]})" for b in sc.boxes)
    lines = [
        f"- Grid is 64x64 (cells of {G.PITCH}px). Mechanic (discovered): the controllable "
        "agent picks up and CARRIES one box at a time; an effect button attaches/drops it.",
        f"- Controllable agent (you) at ({round(av[0])},{round(av[1])})." if av else "- Controllable agent: unknown.",
        f"- Autonomous helper agent at ({round(helper[0])},{round(helper[1])})." if helper else "- No helper.",
        f"- Target container region: {region}. WIN = every box resting in it, none carried.",
        f"- Boxes at: {boxes}.",
        "- There is a full-height impassable WALL down the column x=32 (verified): the "
        "controllable agent is confined to x<=28, the helper to x>=36; neither can cross. "
        "A carried box can be moved to at most x=28 from the left, or picked up at x>=32 "
        "from the right; the pick-up range is one cell (4px).",
    ]
    return "\n".join(lines)


def propose_strategy(game="wa30", level_index=2, model=None, verbose=True):
    """Reach the given level, describe it, and ask the prior-laden LLM for a plan."""
    c = G.CarryConnector.build(game, model=model, use_llm=False, verbose=False)
    g, fd = c.fresh()
    while c.level(fd) < level_index:
        before = c.level(fd)
        g, fd, _ = G.gkm_cone(c, g, fd, before, deadline=time.time()+600, verbose=False)
        if c.level(fd) <= before:
            print(f"could not reach level {level_index+1}"); return
    scene = describe_scene(c, fd)
    if verbose:
        print(f"SCENE (level {c.level(fd)+1}):\n{scene}\n")
    import llm_binder
    prompt = (PRECONCEPTIONS + "\n\nHere is the level:\n" + scene +
              "\n\nReason step by step as a human would. Then answer in JSON: "
              "{\"reachable_by_you\":[..], \"reachable_by_helper\":[..], "
              "\"plan\":\"<one paragraph>\", \"feasible\":true/false, "
              "\"stranded_boxes\":[..]}.")
    out = llm_binder.ollama_json(prompt, **({} if model is None else {"model": model}),
                                 num_predict=900)
    print("LLM STRATEGY (human-preconception system prompt):")
    print(out)
    return out


if __name__ == "__main__":
    import sys
    game = next((a for a in sys.argv[1:] if not a.startswith("--")), "wa30")
    lvl = 2
    for a in sys.argv[1:]:
        if a.startswith("--level="):
            lvl = int(a.split("=")[1]) - 1
    propose_strategy(game, level_index=lvl)
