import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=json.load(open('base_shrunk.json'))
def ok(p):
    try: return A.validate('wa30', p, 8)
    except Exception: return False
t0=time.time()
# adjacent-pair and triple deletions
for width in (2,3,4):
    i=0
    while i<len(path):
        cand=path[:i]+path[i+width:]
        if ok(cand):
            path=cand
            print('cut %d at %d -> %d'%(width,i,len(path)),flush=True)
        else:
            i+=1
    print('width',width,'done len',len(path),'%.0fs'%(time.time()-t0),flush=True)
    json.dump(path,open('base_shrunk.json','w'))
