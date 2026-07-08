import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=json.load(open('base_shrunk.json'))
def ok(p):
    try: return A.validate('wa30', p, 8)
    except Exception: return False
t0=time.time()
W=40
i=0
while i<len(path):
    found=False
    for j in range(i+1, min(i+W, len(path))):
        cand=path[:i]+path[i+1:j]+path[j+1:]
        if ok(cand):
            path=cand
            print('cut pair (%d,%d) -> %d'%(i,j,len(path)), flush=True)
            json.dump(path, open('base_shrunk2.json','w'))
            found=True
            break
    if not found:
        i+=1
    if time.time()-t0>3000:
        print('time out'); break
json.dump(path, open('base_shrunk2.json','w'))
print('final', len(path), '%.0fs'%(time.time()-t0))
