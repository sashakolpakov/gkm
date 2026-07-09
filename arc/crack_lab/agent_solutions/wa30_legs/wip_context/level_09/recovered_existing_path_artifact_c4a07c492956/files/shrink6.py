import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=json.load(open('base2.json'))
def ok(p):
    try: return A.validate('wa30', p, 8)
    except Exception: return False
t0=time.time()
W=40

i=150
while i<421 and time.time()-t0<1500:
    found=False
    for j in range(i+1, min(i+W, len(path))):
        cand=path[:i]+path[i+1:j]+path[j+1:]
        if ok(cand):
            path=cand
            print('pair cut (%d,%d) -> %d'%(i,j,len(path)),flush=True)
            json.dump(path,open('base2.json','w'))
            found=True; break
    if not found: i+=1
json.dump(path,open('base2.json','w'))
print('done', len(path), '%.0fs'%(time.time()-t0))
