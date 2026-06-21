"""LLM binding pass (local, offline via ollama): read the symbolic scene off the
connector and BIND abstract, named verb-templates to THIS game's concrete colours
/regions. This is the game-agnostic 'read-off' — the LLM does the grounding; the
deterministic search verifies against levels_completed. No internet; GPU/local.

The abstract verbs are generic (no colour/game constants). The LLM picks which
verbs apply and binds their params to the colours actually present. A generic
goal_fn TEMPLATE per verb (built here in code) turns a binding into the search's
heuristic — so even the goal_fn carries no game-specific constant; only the
LLM-chosen colours flow in.
"""
from __future__ import annotations
import json, urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from lab import arc

OLLAMA = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3-coder:30b"

VERB_LIBRARY = """\
Available abstract verbs (generic; bind their params to colour NUMBERS present):
- transport(carrier, region): move pieces of colour `carrier` so they come to sit
  inside the area of colour `region`. Success = the region's cells get occupied by
  the carrier colour (a locked delivery). Use when there's a container (a ring of
  one colour around an interior of another) plus movable pieces.
- reach(target): move the controllable avatar onto colour `target`. Use for
  navigation/maze games with a goal marker.
- fill(region): drive the count of colour `region` down to zero (its cells get
  consumed/covered as the objective is met).
- clear(region): same as fill — minimise colour `region`.
- toggle(obj): interact (ACTION5) to change the state of colour `obj`.
- align(group, template): arrange movable colour `group` to match a shown
  template region `template`.
"""


def ollama_json(prompt: str, model: str = DEFAULT_MODEL, num_predict: int = 400, timeout: float = 180.0) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "format": "json", "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    try:
        return json.loads(resp.get("response", "{}"))
    except Exception:
        return {}


def describe_scene(scene: Dict, avatar_color: Optional[int], win_levels: int) -> str:
    conts = "; ".join(f"ring={r} interior={i}" for (r, i, _c) in scene["containers"]) or "none"
    movers = ", ".join(f"colour {c}({s} cells)" for (c, _ce, s) in scene["movable_objects"][:10]) or "none"
    counts = ", ".join(f"{c}:{n}" for c, n in sorted(scene["colour_counts"].items()))
    return (f"avatar_colour: {avatar_color}\n"
            f"containers (ring around interior): {conts}\n"
            f"movable objects: {movers}\n"
            f"colour cell-counts: {counts}\n"
            f"structure/background colours: {scene['structure_colours']}\n"
            f"available actions: {scene['available_actions']} (5=interact)\n"
            f"levels to win: {win_levels}")


@dataclass
class BoundVerb:
    name: str
    verb: str
    params: Dict[str, int]
    rationale: str = ""
    goal_fn: Optional[Callable] = field(default=None, repr=False)


def bind(scene: Dict, avatar_color: Optional[int], win_levels: int, model: str = DEFAULT_MODEL) -> List[BoundVerb]:
    """Ask the local LLM to bind verbs to this game's colours."""
    prompt = (
        "You analyse a grid puzzle game. Objects are connected regions of a colour "
        "(integers 0-15; 0 = background). You are given a symbolic description of the "
        "current frame. Propose the most likely OBJECTIVE as a short ranked list of "
        "bound verbs, choosing from the library and binding params to the colour "
        "numbers that are actually present.\n\n"
        f"{VERB_LIBRARY}\n"
        "SCENE:\n" + describe_scene(scene, avatar_color, win_levels) + "\n\n"
        'Return ONLY JSON: {"goals":[{"name": "<short semantic name>", '
        '"verb": "<one of the library verbs>", "params": {<param>:<colour int>, ...}, '
        '"rationale": "<one sentence>"}]} ranked best-first, at most 3 goals.'
    )
    out = ollama_json(prompt, model=model)
    goals = out.get("goals", []) if isinstance(out, dict) else []
    bound = []
    for g in goals:
        if not isinstance(g, dict) or "verb" not in g:
            continue
        params = {k: int(v) for k, v in (g.get("params") or {}).items() if str(v).lstrip("-").isdigit()}
        bound.append(BoundVerb(name=str(g.get("name", g["verb"])), verb=str(g["verb"]),
                               params=params, rationale=str(g.get("rationale", ""))))
    return bound


# ---------------------------------------------------------------------------
# Generic goal_fn templates: turn a binding into a search heuristic. No game
# constants — only the LLM-chosen colours. lower = closer.
# ---------------------------------------------------------------------------

def _footprint(frame, colour):
    return [(x, y) for y in range(len(frame)) for x in range(len(frame[0])) if frame[y][x] == colour]


def _centroid(frame, colour):
    cs = [(x, y) for y in range(len(frame)) for x in range(len(frame[0])) if frame[y][x] == colour]
    if not cs:
        return None
    return (sum(p[0] for p in cs) / len(cs), sum(p[1] for p in cs) / len(cs))


def attach_goal_fn(bv: BoundVerb, frame0, avatar_color: Optional[int]) -> BoundVerb:
    """Build bv.goal_fn(frame)->float from the verb type + LLM bindings + the
    level's start frame (for fixed footprints)."""
    v, p = bv.verb, bv.params
    if v == "transport" and "carrier" in p and "region" in p:
        carrier, region = p["carrier"], p["region"]
        fp = _footprint(frame0, region)

        def gf(arr, fp=fp, carrier=carrier, region=region):
            locked = sum(1 for (x, y) in fp if arr[y][x] == carrier)   # genuine deliveries
            empty = sum(1 for (x, y) in fp if arr[y][x] == region)     # unfilled slots
            return -locked + 0.05 * empty                              # locked-primary
        bv.goal_fn = gf
    elif v in ("fill", "clear") and "region" in p:
        region = p["region"]

        def gf(arr, region=region):
            return sum(1 for row in arr for val in row if val == region)
        bv.goal_fn = gf
    elif v == "reach" and ("target" in p or "obj" in p) and avatar_color is not None:
        target = p.get("target", p.get("obj"))

        def gf(arr, target=target, av=avatar_color):
            a, t = _centroid(arr, av), _centroid(arr, target)
            return 1e6 if (a is None or t is None) else abs(a[0] - t[0]) + abs(a[1] - t[1])
        bv.goal_fn = gf
    else:  # toggle / align / unbound -> generic novelty (make non-structure change)
        skip = {0, avatar_color}

        def gf(arr, skip=skip):
            return -len({(x, y) for y in range(len(arr)) for x in range(len(arr[0]))
                         if arr[y][x] and arr[y][x] not in skip})
        bv.goal_fn = gf
    return bv


if __name__ == "__main__":  # smoke test on wa30's real scene
    import sys
    from lab import make_env
    from bfs_crack import detect_avatar_color
    import proposer
    game = sys.argv[1] if len(sys.argv) > 1 else "wa30"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    e = make_env(game)(); s = e.reset()
    av, _, _ = detect_avatar_color(make_env(game))
    scene = proposer.scene_summary(s.frame, e.available_actions)
    print(f"{game} scene: containers={scene['containers']} avatar={av}")
    for bv in bind(scene, av, s.win_levels, model=model):
        print(f"  LLM-BOUND [{bv.verb}] {bv.name}: params={bv.params}  ({bv.rationale})")
