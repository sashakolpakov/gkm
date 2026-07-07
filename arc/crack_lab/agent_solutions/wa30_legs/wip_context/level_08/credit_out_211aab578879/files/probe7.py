import importlib.util, json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
import perception as P

def get_env_at_L7():
    env = A.Arena('wa30')
    with open("checkpoint.json") as f:
        ck = json.load(f)
    # replay final path only until levels_completed==6
    for a in ck["final_path"]:
        if env.levels_completed >= 6:
            break
        env.step(a)
    return env

if __name__ == "__main__":
    env = get_env_at_L7()
    print("levels_completed:", env.levels_completed, "terminal:", env.terminal())
    f = env.frame()
    print("shape", f.shape)
    print("colors:", P.color_counts(f))

def show(env):
    f = env.frame()
    for r in range(64):
        print("".join("{:x}".format(int(v)) for v in f[r]))
