import sys, json, random, time, os
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

ck=json.load(open('checkpoint.json'))
PATH=ck['final_path']
BOUNDS={1:(0,27),2:(27,81),3:(81,161),4:(161,230),5:(230,354),6:(354,401),7:(401,466),8:(466,588),9:(588,588)}

def make_prefix(start):
    env=A.Arena('wa30', _budget=A._Budget(10**8))
    for a in PATH[:start]:
        env.step(a)
    return env

def evaluate(prefix, seg, level):
    c=prefix.clone()
    for i,a in enumerate(seg):
        if c.terminal(): return None
        try: c.step(a)
        except Exception: return None
        if c.levels_completed>=level:
            return i+1
    return None

def search(level, seconds=180, seed=1):
    rnd=random.Random(seed)
    start,end=BOUNDS[level]
    prefix=make_prefix(start)
    fn=f'seg_L{level}.json'
    cur=json.load(open(fn)) if os.path.exists(fn) else list(PATH[start:end])
    n=evaluate(prefix,cur,level); cur=cur[:n]
    best=list(cur)
    t0=time.time(); trials=0
    while time.time()-t0<seconds:
        trials+=1
        cand=list(cur)
        op=rnd.random()
        if op<0.5:
            w=rnd.randint(1,4); i=rnd.randrange(max(1,len(cand)-w+1))
            del cand[i:i+w]
        elif op<0.8:
            i=rnd.randrange(len(cand)); cand[i]=rnd.choice([1,2,3,4,5])
        elif op<0.9:
            i=rnd.randrange(len(cand)-1); cand[i],cand[i+1]=cand[i+1],cand[i]
        else:
            i=rnd.randrange(len(cand)); cand.insert(i, rnd.choice([1,2,3,4,5]))
        m=evaluate(prefix,cand,level)
        if m is None: continue
        cand=cand[:m]
        if len(cand)<len(best):
            best=list(cand); cur=list(cand)
        elif len(cand)<=len(cur):
            cur=list(cand)  # sideways
    json.dump(best, open(fn,'w'))
    print(f'L{level}: {end-start} -> {len(best)} ({trials} trials)')

if __name__=='__main__':
    search(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 1)
