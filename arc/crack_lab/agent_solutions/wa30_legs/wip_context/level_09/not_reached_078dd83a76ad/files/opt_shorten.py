import sys, json, random, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

ck=json.load(open('checkpoint.json'))
PATH=ck['final_path']
BOUNDS=[(1,0,27),(2,27,81),(3,81,161),(4,161,230),(5,230,354),(6,354,401),(7,401,466),(8,466,588)]

def make_prefix(start):
    env=A.Arena('wa30', _budget=A._Budget(10**8))
    for a in PATH[:start]:
        env.step(a)
    return env

def evaluate(prefix, seg, level):
    """Replay seg on a clone; return completion length (moves used) or None."""
    c=prefix.clone()
    for i,a in enumerate(seg):
        if c.terminal(): return None
        try: c.step(a)
        except Exception: return None
        if c.levels_completed>=level:
            return i+1
    return None

def shorten(level, start, end, seconds=60, seed=0):
    rnd=random.Random(seed)
    prefix=make_prefix(start)
    best=list(PATH[start:end])
    n=evaluate(prefix,best,level)
    assert n is not None
    best=best[:n]
    t0=time.time()
    trials=0
    while time.time()-t0<seconds:
        trials+=1
        cand=list(best)
        k=rnd.choice([1,1,1,2,2,3])
        for _ in range(k):
            if not cand: break
            i=rnd.randrange(len(cand))
            del cand[i]
        m=evaluate(prefix,cand,level)
        if m is not None and m<len(best):
            best=cand[:m]
    return best,trials

if __name__=='__main__':
    lvl=int(sys.argv[1]); secs=float(sys.argv[2]) if len(sys.argv)>2 else 60
    for (L,s,e) in BOUNDS:
        if L==lvl:
            best,tr=shorten(L,s,e,secs)
            print(f'L{L}: {e-s} -> {len(best)}  ({tr} trials)')
            json.dump(best, open(f'seg_L{L}.json','w'))
