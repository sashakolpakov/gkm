"""Run the game-agnostic cone_engine on wa30 with an LLM-supplied Connector.
LLM supplies roles (once); the engine (cofibration search, real-step verified)
runs per level, rebuilding the connector from each level's start frame."""
from __future__ import annotations
import copy, sys, time
import numpy as np
from lab import arc, make_env
from arcengine import ActionInput, GameAction as EA
import priors, proposer, llm_binder, cone_engine
from llm_connector import DeliveryConnector

GAME = sys.argv[1] if len(sys.argv) > 1 else "wa30"
MODEL = sys.argv[2] if len(sys.argv) > 2 else llm_binder.DEFAULT_MODEL
TIME_CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


def step(g, a):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def frame_of(fd):
    return np.asarray(fd.frame[-1]).tolist()


def level_of(fd):
    return fd.levels_completed


def llm_roles(game_env_factory):
    """LLM supplies (carrier, region): the colour that fills a slot on delivery,
    and the container interior. Uses the bind+variant-expansion (the LLM may swap
    carrier/region; we constrain region->real interior, carrier->a mover)."""
    e = game_env_factory(); s = e.reset()
    avatar, _, _ = priors.detect_avatar_color(game_env_factory)
    scene = proposer.scene_summary(s.frame, e.available_actions)
    bound = llm_binder.bind(scene, avatar, s.win_levels, model=MODEL)
    interiors = {i for (_, i, _c) in scene["containers"]}
    nonstruct = set(scene["colour_counts"]) - set(scene["structure_colours"])
    for bv in bound:
        if bv.verb == "transport":
            c, r = bv.params.get("carrier"), bv.params.get("region")
            for (cc, rr) in [(c, r), (r, c)]:
                if rr in interiors and cc in nonstruct and cc != rr:
                    return cc, rr, avatar
    # fall back to the detected container if the LLM didn't give a usable transport
    if scene["containers"]:
        ring, interior, _ = scene["containers"][0]
        return ring, interior, avatar
    return None, None, avatar


def main():
    factory = make_env(GAME)
    print(f"{GAME}: asking local LLM ({MODEL}) for roles ...")
    t0 = time.time()
    carrier, region, avatar = llm_roles(factory)
    print(f"LLM roles ({time.time()-t0:.0f}s): carrier(delivers)={carrier} region(container)={region} avatar={avatar}")
    if carrier is None:
        print("no usable roles; abort."); return

    e = factory(); win_levels = e.reset().win_levels
    root_game = copy.deepcopy(e._env._game)
    root_fd = root_game.perform_action(ActionInput(id=EA.RESET), raw=True)
    actions = [a for a in (e.available_actions or [1, 2, 3, 4, 5]) if 1 <= a <= 5]
    cfg = cone_engine.EngineConfig(leg_budget=12000, leg_depth=55, max_partials=150, time_cap=TIME_CAP)

    game, fd, level = root_game, root_fd, 0
    while level < win_levels:
        frame = frame_of(fd)
        footprint = [(x, y) for y in range(len(frame)) for x in range(len(frame[0]))
                     if frame[y][x] == region]
        conn = DeliveryConnector(carrier, region, avatar, footprint, actions)
        print(f"LEVEL {level+1}: region slots={len(footprint)} -- engine cofibration search "
              f"({time.time()-t0:.0f}s)")
        res = cone_engine.solve_level(step, frame_of, level_of, conn, game, fd, level, cfg)
        if res[0] != "win":
            print(f"  L{level+1} NOT cracked by the engine ({time.time()-t0:.0f}s)")
            break
        _, game, fd = res
        level = level_of(fd)
        print(f"  *** LEVEL {level} CLEARED by the cofibration engine ({time.time()-t0:.0f}s)")

    print(f"\nRESULT engine: {GAME} levels={level}/{win_levels} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
