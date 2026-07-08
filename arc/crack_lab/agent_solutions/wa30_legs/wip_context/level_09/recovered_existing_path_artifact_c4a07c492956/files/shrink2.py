import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=json.load(open('base_shrunk.json'))
def ok(p):
    try: return A.validate('wa30', p, 8)
    except Exception: return False
assert ok(path)
t0=time.time()
# greedy single deletions, repeat passes
passn=0
while time.time()-t0<1500:
    passn+=1
    any_cut=False
    i=0
    while i<len(path):
        cand=path[:i]+path[i+1:]
        if ok(cand):
            path=cand; any_cut=True
        else:
            i+=1
    print('pass',passn,'len',len(path),'%.0fs'%(time.time()-t0),flush=True)
    json.dump(path, open('base_shrunk.json','w'))
    if not any_cut: break
print('final', len(path))
