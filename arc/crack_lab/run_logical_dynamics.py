"""Runner: learn + VERIFY a logical-cell move-rule on a real ARC-AGI-3 game.

Pipeline (game-agnostic): collect random-walk transitions -> infer the logical
grid -> identify the cofibrant avatar by action-response (shape-invariant) ->
learn the controls (constant floor, data-only rule) and the structured rule
(vectors + wall-blocking + push), partition fit by the data -> report held-out
fidelity for each + the recovered NON-STANDARD move-rule + ASCII.

Usage:  python3 run_logical_dynamics.py [game=wa30] [n_transitions=600]
"""
from __future__ import annotations
import copy, random, sys
import numpy as np

from lab import make_env
import priors
from logical_grid import Grid, to_logical, render_logical, objects
from cofibrant import identify_avatar, probe_avatar
from dynamics_model import (LState, MoveRule, fidelity, constant_floor, DataRule,
                            fit_structured_rule)

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
DIR = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT", (0, 0): "stay"}


def collect(game, n, actions=(1, 2, 3, 4, 5), seed=0):
    from arcengine import ActionInput, GameAction as EA
    e = make_env(game)(); e.reset(); g0 = copy.deepcopy(e._env._game)
    g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    rng = random.Random(seed); trans = []
    for _ in range(n):
        before = np.asarray(fd.frame[-1]); a = rng.choice(list(actions))
        fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        trans.append((before, a, np.asarray(fd.frame[-1])))
        if str(fd.state).endswith(("GAME_OVER", "WIN")):
            g = copy.deepcopy(g0); fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
    return trans, g0


def analyze(game, n=600, verbose=True):
    trans, g0 = collect(game, n)
    a0 = trans[0][0]
    grid = Grid.infer(a0)
    struct = set(priors.structure_colours(a0.tolist()))
    palette = [int(c) for c in np.unique(a0) if c != 0]
    dynamic = [c for c in palette if c not in struct]          # non-bg, non-floor

    avatar = identify_avatar(trans, grid)
    if avatar is None:
        if verbose:
            print(f"[{game}] NO cofibrant avatar found (not grid-navigational?). "
                  f"palette={palette} struct={sorted(struct)} grid={grid}")
        return None
    box_dynamic = [c for c in dynamic if c != avatar.color]

    split = int(len(trans) * 0.7); train, test = trans[:split], trans[split:]
    floor = constant_floor(test, grid, avatar.color, box_dynamic, [])
    data = DataRule.learn(train, grid, avatar.color, box_dynamic)
    data_f = data.fidelity(test, grid, avatar.color)
    struct_rule = fit_structured_rule(train, grid, avatar.color, box_dynamic)
    struct_f = fidelity(struct_rule, test, grid, avatar.color)

    if verbose:
        print(f"\n========== {game} ==========")
        print(f"grid={grid}  palette={palette}  floor(struct)={sorted(struct)}")
        print(f"cofibrant avatar: colour {avatar.color}  score={avatar.score:.2f}")
        print("verified move-rule (action -> logical vector):")
        for a in sorted(avatar.vectors):
            v = avatar.vectors[a]
            print(f"   ACTION{a}: {v} {DIR.get(v,'?'):5s}  conf={avatar.confidence.get(a,0):.2f}")
        std = all(avatar.vectors.get(a) == d for a, d in
                  zip((1, 2, 3, 4), [(0, -1), (1, 0), (0, 1), (-1, 0)]))
        print(f"   -> {'STANDARD' if std else 'NON-STANDARD'} action mapping")
        print(f"\nstructured rule: walls={sorted(struct_rule.wall_colors)} "
              f"boxes={sorted(struct_rule.box_colors)} push={struct_rule.push}")
        print("\nFIDELITY (held-out exact avatar+box cells):")
        print(f"   constant floor : {floor}")
        print(f"   data-only rule : {data_f}")
        print(f"   structured rule: {struct_f}   <- LLM-proposable form")
        print(f"\nlogical grid {to_logical(a0, grid).shape} (avatar cell {avatar and LState.of(a0, grid, avatar.color, box_dynamic, struct_rule.wall_colors).avatar}):")
        st0 = LState.of(a0, grid, avatar.color, box_dynamic, struct_rule.wall_colors)
        print(render_logical(to_logical(a0, grid), mark=st0.avatar))

    return dict(game=game, grid=grid, avatar=avatar, floor=floor, data=data_f,
                structured=struct_f, rule=struct_rule, box_colors=box_dynamic,
                train=train, test=test)


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "wa30"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    analyze(game, n)


if __name__ == "__main__":
    main()
