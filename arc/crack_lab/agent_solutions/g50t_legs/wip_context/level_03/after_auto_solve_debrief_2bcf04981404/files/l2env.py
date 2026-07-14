import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

def get_l2_env():
    """Return a fresh env advanced to start of level 2 (levels_completed==1)."""
    ck = json.load(open("checkpoint.json"))
    path1 = ck["final_path"]
    env = A.Arena(game='g50t')
    for a in path1:
        env.step(a)
    assert env.levels_completed == 1, env.levels_completed
    return env
