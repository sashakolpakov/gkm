import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

ck=json.load(open('checkpoint.json'))
path=list(ck['final_path'])
TARGET_LV=8

def ok(p):
    try:
        return A.validate('wa30', p, TARGET_LV)
    except Exception:
        return False

assert ok(path)
t0=time.time()
# pass 1: try deleting runs of identical actions (esp idles), longest first
improved=True
while improved and time.time()-t0<600:
    improved=False
    # find runs
    runs=[]
    i=0
    while i<len(path):
        j=i
        while j<len(path) and path[j]==path[i]: j+=1
        runs.append((i,j-i,path[i]))
        i=j
    # try trimming runs from the end of the run, biggest cuts first
    for start,length,act in sorted(runs, key=lambda r:-r[1]):
        if length<2: continue
        for cut in [length, length//2, 2, 1]:
            if cut<1 or cut>length: continue
            cand=path[:start]+path[start+cut:]
            if ok(cand):
                path=cand
                improved=True
                print('cut run of %d x action %d at %d -> len %d'%(cut,act,start,len(path)), flush=True)
                break
        if improved: break
print('after run-pass', len(path), '%.1fs'%(time.time()-t0))
json.dump(path, open('base_shrunk.json','w'))
