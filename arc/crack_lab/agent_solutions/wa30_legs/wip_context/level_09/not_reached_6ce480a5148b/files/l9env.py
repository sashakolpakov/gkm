import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

def get_l9():
    ck = json.load(open("checkpoint.json"))
    env = A.Arena('wa30')
    for a in ck["final_path"]:
        env.step(a)
    return env
