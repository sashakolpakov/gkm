import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=json.load(open('base2.json'))
def ok(p):
    try: return A.validate('wa30', p, 8)
    except Exception: return False
t0=time.time()
# runs first
improved=True
while improved and time.time()-t0<400:
    improved=False
    runs=[]; i=0
    while i<len(path):
        j=i
        while j<len(path) and path[j]==path[i]: j+=1
        runs.append((i,j-i,path[i])); i=j
    for start,length,act in sorted(runs,key=lambda r:-r[1]):
        if length<2: continue
        for cut in [length,length//2,2,1]:
            if cut<1 or cut>length: continue
            cand=path[:start]+path[start+cut:]
            if ok(cand):
                path=cand; improved=True
                print('cut %d x%d at %d -> %d'%(cut,act,start,len(path)),flush=True)
                break
        if improved: break
json.dump(path,open('base2.json','w'))
print('runs done', len(path))
# single deletions
i=0
while i<len(path) and time.time()-t0<1200:
    cand=path[:i]+path[i+1:]
    if ok(cand):
        path=cand
        print('del at %d -> %d'%(i,len(path)),flush=True)
    else:
        i+=1
json.dump(path,open('base2.json','w'))
print('final', len(path))
