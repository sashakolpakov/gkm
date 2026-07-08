import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
_p=json.load(open('base_shrunk.json'))
_base=None
def base():
    global _base
    if _base is None:
        e=A.Arena('wa30', _budget=A._Budget(600*400))
        for a in _p[:227]: e.step(a)
        _base=e
    return _base
def fresh():
    return base().clone()
