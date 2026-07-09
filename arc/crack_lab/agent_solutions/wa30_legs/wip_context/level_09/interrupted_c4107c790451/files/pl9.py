import sys, json; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck = json.load(open('checkpoint.json'))
def fresh():
    env=A.Arena('wa30', _budget=A._Budget(600*400))
    for a in ck['final_path']:
        env.step(a)
    return env
def classify(blk):
    s=set(blk.flatten().tolist()); s.discard(1)
    if not s: return '..'
    d={frozenset({5}):'##',frozenset({7}):'==',frozenset({2}):'22',
       frozenset({4}):'44',frozenset({12}):'CC',frozenset({15}):'MM',
       frozenset({9}):'99',frozenset({0}):'00',frozenset({0,14}):'AV',
       frozenset({14}):'ee',frozenset({4,9}):'4c',frozenset({2,9}):'2c',
       frozenset({1,2}):'22',frozenset({1,7}):'==',frozenset({5,7}):'#7'}
    fs=frozenset(s)
    return d.get(fs, '?'+''.join(str(x) for x in sorted(s)))
def show(f):
    for r in range(16):
        print(f'{r:2d} '+' '.join(classify(f[r*4:r*4+4,c*4:c*4+4]) for c in range(16)))
def grid(f):
    return [[classify(f[r*4:r*4+4,c*4:c*4+4]) for c in range(16)] for r in range(16)]
def find(f,tag):
    g=grid(f); return [(r,c) for r in range(16) for c in range(16) if g[r][c]==tag]
if __name__=='__main__':
    env=fresh(); show(np.asarray(env.frame()))
