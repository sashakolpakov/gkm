import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck=json.load(open('checkpoint.json'))
_baselen=len(ck['final_path'])
_base=None
def base():
    global _base
    if _base is None:
        e=A.Arena('wa30', _budget=A._Budget(600*400))
        for a in ck['final_path']:
            e.step(a)
        _base=e
    return _base
def fresh():
    return base().clone()
def run(strategy, verbose=False):
    e=fresh()
    try:
        strategy(e)
    except Exception as ex:
        if verbose:
            import traceback; traceback.print_exc()
    moves=len(e.path)-_baselen
    return e.levels_completed, moves, e.terminal()
