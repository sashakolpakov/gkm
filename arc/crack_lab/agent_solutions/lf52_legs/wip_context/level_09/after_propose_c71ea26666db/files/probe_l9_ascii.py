"""One symbolic rail map for the verified second level-9 region."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))
    for _ in range(int(os.environ.get("L9_ASCII_RIGHT", "0"))):
        safe_step(env, 4)
    frame = arr(env.frame())
    symbols = {
        0: " ", 1: "o", 5: "#", 7: ">", 9: "-",
        10: ".", 11: "C", 12: "c", 14: "P", 15: "F",
    }
    print("legend o=hole #=wall -=rail C/c=carrier P=peg F/>=fixed_bridge")
    for row in range(8, 56):
        print(f"{row:02d} " + "".join(symbols.get(int(value), "?")
                                      for value in frame[row]))


arena.run_program("lf52", probe)
