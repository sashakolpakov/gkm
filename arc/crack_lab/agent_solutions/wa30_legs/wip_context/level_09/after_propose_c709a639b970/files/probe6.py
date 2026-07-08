import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

_CK = json.load(open('checkpoint.json'))

def fresh_at_level6(budget=5_000_000):
    env = A.Arena('wa30', _budget=A._Budget(budget))
    for a in _CK['final_path']:
        env.step(a)
    return env
