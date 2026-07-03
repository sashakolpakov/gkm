"""LLM-driven ANCHOR CONNECTOR (game-specific seam, but not hand-coded).

The general engine knows only: quotient the renderer into cofibrant objects
(logical_grid) and VERIFY a proposal against interaction (cofibrant.identify_anchor).
It carries no game semantics. The game-specific part -- WHICH cofibrant object is
the action anchor ("the avatar, whatever it is") -- lives here in a connector that
is LLM-DRIVEN:

    PROPOSE (local LLM)  reads a compact symbolic scene + a short directed-probe
                         effect summary and proposes ranked anchor candidates
                         {colour, channel, why}. Catches anchors blind
                         exploration never elicits (maze sprite, symbol cursor).
    VERIFY  (interaction) accepts a proposal ONLY if that object actually responds
                         to actions distinctively + consistently. The LLM never
                         replaces the signal; it only orders the search.
    FALL BACK            if no proposal verifies (or no LLM), use the algorithmic
                         anchor. Honest about which path won.

Reuses the ollama seam from llm_binder. See SPEC_logical_cofibrant.md.
"""
from __future__ import annotations
import copy
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from logical_grid import Grid, components, objects
from cofibrant import Anchor, identify_anchor, identify_all_anchors

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5",
        6: "ACTION6", 7: "ACTION7"}


# --------------------------- directed probe ---------------------------------

def directed_probe(make_env, actions=(1, 2, 3, 4, 5), repeats=4):
    """From reset, apply each action `repeats` times (fresh clone per action).
    Surfaces the anchor far better than a random walk in mazes / gated games.
    Returns (sequences, start_frame), sequences = {action: [(before, after), ...]}
    CONTIGUOUS from reset (so per-component continuity tracking holds)."""
    from arcengine import ActionInput, GameAction as EA
    e = make_env(); e.reset(); g0 = copy.deepcopy(e._env._game)
    start = np.asarray(g0.perform_action(ActionInput(id=EA.RESET), raw=True).frame[-1])
    sequences = {}
    for a in actions:
        g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
        seq = []
        for _ in range(repeats):
            before = np.asarray(fd.frame[-1])
            fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
            seq.append((before, np.asarray(fd.frame[-1])))
            if str(fd.state).endswith(("GAME_OVER", "WIN")):
                break
        sequences[a] = seq
    return sequences, start


# --------------------------- symbolic summaries -----------------------------

def scene_summary(arr: np.ndarray, grid: Grid) -> str:
    palette = [int(c) for c in np.unique(arr) if c != 0]
    lines = []
    for c in palette:
        comps = components(arr, c)
        sizes = sorted(len(k) for k in comps)
        lines.append(f"  colour {c}: {len(comps)} object(s), pixel-sizes {sizes[:8]}")
    return (f"logical grid {grid.pitch}px cells; colours present {palette}\n"
            + "\n".join(lines))


def effect_summary(sequences, grid: Grid) -> str:
    """Per-colour, the BEST-MOVING component's per-action displacement (per-component,
    so a small avatar inside a multi-component colour is surfaced to the LLM)."""
    from cofibrant import _comp_cells, score_seed
    start = next(iter(sequences.values()))[0][0]
    palette = [int(c) for c in np.unique(start) if c != 0]
    rows = []
    for c in palette:
        seeds = _comp_cells(start, grid, c)
        best = None
        for seed in seeds:
            vectors, consistency, moved = score_seed(sequences, grid, c, seed)
            nmoved = sum(1 for a in (1, 2, 3, 4) if moved.get(a))
            if best is None or nmoved > best[0]:
                best = (nmoved, vectors, moved)
        ncomp = len(seeds)
        if best and best[0] > 0:
            tags = " ".join(f"A{a}:{best[1][a]}" if best[2].get(a) else f"A{a}:-"
                            for a in sorted(best[1]))
            rows.append(f"  colour {c} ({ncomp} comp): best-mover {tags}")
        else:
            rows.append(f"  colour {c} ({ncomp} comp): no component moves")
    return "\n".join(rows)


# --------------------------- the connector ----------------------------------

@dataclass
class AnchorResult:
    anchor: Optional[Anchor]
    source: str            # 'llm-verified' | 'algorithmic' | 'none'
    llm_ranked: List[dict]
    rejected: List[str]    # LLM picks that failed verification


def propose_anchor_llm(scene: str, effects: str, actions, model: str = "qwen3-coder:30b") -> List[dict]:
    """Local LLM proposes ranked anchor candidates. Reuses llm_binder.ollama_json."""
    try:
        from llm_binder import ollama_json
    except Exception:
        return []
    prompt = (
        "You analyse a grid puzzle. The player's ACTIONS act through ONE object -- "
        "the ANCHOR (an avatar/cursor/handle). It need not move: its effect channel "
        "may be 'move' (a navigated avatar), 'activity' (a cursor/tile that changes "
        "in place), or 'count' (something that appears/vanishes). Identify the anchor.\n\n"
        f"SCENE:\n{scene}\n\nWHAT EACH ACTION DID (directed probe):\n{effects}\n\n"
        f"available actions: {list(actions)}\n\n"
        "Return JSON {\"ranked\":[{\"color\":int,\"channel\":\"move|activity|count\","
        "\"why\":str}, ...]} best-first. Prefer the object actions affect most "
        "distinctively. If the probe shows no motion, a small unique sprite in a maze "
        "is still likely the avatar -- propose it."
    )
    out = ollama_json(prompt, model=model)
    ranked = out.get("ranked") if isinstance(out, dict) else None
    return ranked if isinstance(ranked, list) else []


def verify_anchor(color: int, sequences, grid: Grid, min_score: float = 0.4) -> Optional[Anchor]:
    """Accept the proposed colour as anchor ONLY if interaction confirms SOME
    component of it responds distinctively+consistently (per-component scorer)."""
    a = identify_anchor(sequences, grid, candidate_colors=[color])
    return a if (a is not None and a.score >= min_score) else None


class AnchorConnector:
    """propose (LLM) -> verify (interaction) -> fall back (algorithmic)."""

    def __init__(self, model: str = "qwen3-coder:30b", min_score: float = 0.4,
                 use_llm: bool = True):
        self.model = model; self.min_score = min_score; self.use_llm = use_llm

    def identify(self, make_env, actions=(1, 2, 3, 4, 5)) -> AnchorResult:
        sequences, start = directed_probe(make_env, actions=actions)
        grid = Grid.infer(start)
        ranked, rejected = [], []
        if self.use_llm:
            ranked = propose_anchor_llm(scene_summary(start, grid),
                                        effect_summary(sequences, grid), actions, self.model)
            for cand in ranked:
                try:
                    col = int(cand.get("color"))
                except Exception:
                    continue
                v = verify_anchor(col, sequences, grid, self.min_score)
                if v is not None:
                    return AnchorResult(anchor=v, source="llm-verified",
                                        llm_ranked=ranked, rejected=rejected)
                rejected.append(f"colour {col} ({cand.get('channel','?')}) - unverified")
        algo = identify_anchor(sequences, grid)
        if algo is not None and algo.score >= self.min_score:
            return AnchorResult(anchor=algo, source="algorithmic",
                                llm_ranked=ranked, rejected=rejected)
        return AnchorResult(anchor=algo, source="none", llm_ranked=ranked, rejected=rejected)

    def identify_all(self, make_env, actions=(1, 2, 3, 4, 5)) -> Tuple[List[Anchor], Grid]:
        """Find ALL steerable components --- every component that responds to
        directional actions with distinct, consistent movement.

        Returns (anchors, grid) where *anchors* is the list of all
        independently steerable objects (empty if none). Uses the algorithmic
        probe only (no LLM round-trip) so the result is deterministic and
        game-agnostic."""
        sequences, start = directed_probe(make_env, actions=actions)
        grid = Grid.infer(start)
        return identify_all_anchors(sequences, grid), grid
