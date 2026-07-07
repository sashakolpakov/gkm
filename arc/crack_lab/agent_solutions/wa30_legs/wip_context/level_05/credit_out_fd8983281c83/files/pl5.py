import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players, legs

def get_env_at_level(target):
    env = A.Arena('wa30')
    while env.levels_completed < target and not env.terminal():
        k = env.levels_completed + 1
        fn = getattr(players, f'play_level_{k}', None)
        if fn is None: break
        before = env.levels_completed
        fn(env)
        if env.levels_completed <= before:
            break
    return env

CH='.1234567890abcdef'
def show(f):
    f=np.array(f)
    for r in f:
        print(''.join(CH[int(v)] if int(v)<16 else '?' for v in r))

def diff(a,b):
    a,b=np.array(a),np.array(b)
    ys,xs=np.where(a!=b)
    return [(int(y),int(x),int(a[y,x]),int(b[y,x])) for y,x in zip(ys,xs)]

if __name__=='__main__':
    env=get_env_at_level(4)
    print("levels",env.levels_completed,"terminal",env.terminal())
    show(env.frame())

def probe_actions():
    base=get_env_at_level(4)
    f0=base.frame()
    for a in [1,2,3,4,5]:
        c=base.clone()
        c.step(a)
        d=diff(f0,c.frame())
        print("action",a,"ndiff",len(d),d[:12])

def probe2():
    base=get_env_at_level(4)
    frames={}
    for a in [1,2,3,4,5]:
        c=base.clone(); c.step(a); frames[a]=np.array(c.frame())
    # compare each action to action 5 (baseline "use")
    for a in [1,2,3,4]:
        ys,xs=np.where(frames[a]!=frames[5])
        print("act",a,"vs5:",[(int(y),int(x),int(frames[5][y,x]),int(frames[a][y,x])) for y,x in zip(ys,xs)])

def objmap(f):
    f=np.array(f)
    # find 4x4 boxes with border color b and core 9
    res={}
    for b in [4,12,14,2,9,13,11]:
        ys,xs=np.where(f==b)
        res[b]=len(ys)
    return res

def watch_use(n=12):
    env=get_env_at_level(4)
    print("lvl",env.levels_completed)
    for i in range(n):
        f0=np.array(env.frame())
        env.step(5)
        f1=np.array(env.frame())
        ys,xs=np.where(f0!=f1)
        # summarize courier (12) bbox
        c=np.where(f1==12)
        cb=(c[0].min(),c[0].max(),c[1].min(),c[1].max()) if len(c[0]) else None
        print(f"step{i} lvl={env.levels_completed} ndiff={len(ys)} courier12_bbox={cb}")

def boxes4(f):
    f=np.array(f); out=[]
    ys,xs=np.where(f==4)
    seen=set()
    for y,x in zip(ys,xs):
        if (y,x) in seen: continue
        if (y==0 or f[y-1,x]!=4) and (x==0 or f[y,x-1]!=4):
            out.append((int(y),int(x)))
    return sorted(out)

def watch_use2(n=30):
    env=get_env_at_level(4)
    for i in range(n):
        env.step(5)
        f=np.array(env.frame())
        print(f"s{i} lvl={env.levels_completed} n4={int((f==4).sum())} boxes={boxes4(f)}")

def avpos(f):
    f=np.array(f); c=np.where(f==14)
    if not len(c[0]): return None
    return (int(c[0].min()),int(c[0].max()),int(c[1].min()),int(c[1].max()))

def test_push():
    # move avatar down to box below (box at rows56-59 cols44-47), avatar rows37-39 cols44-47
    env=get_env_at_level(4)
    print("av0",avpos(env.frame()))
    # step down until adjacent to box
    for i in range(6):
        f0=np.array(env.frame())
        env.step(2)  # down
        f1=np.array(env.frame())
        print("down",i,"av",avpos(f1),"n4",int((f1==4).sum()),"moved",avpos(f0)!=avpos(f1))

def boxes4b(f):
    f=np.array(f); out=[]
    ys,xs=np.where(f==4)
    for y,x in zip(ys,xs):
        if y>=62: continue
        if (y==0 or f[y-1,x]!=4) and (x==0 or f[y,x-1]!=4):
            out.append((int(y),int(x)))
    return sorted(set(out))

def test_push2():
    env=get_env_at_level(4)
    print("boxes",boxes4b(env.frame()),"av",avpos(env.frame()))
    seq=[2,2,2,2,5,2]
    for a in seq:
        env.step(a)
        f=env.frame()
        print("act",a,"av",avpos(f),"boxes",boxes4b(f))

def nav_show(seq,label=""):
    env=get_env_at_level(4)
    print(label,"start av",avpos(env.frame()),"boxes",boxes4b(env.frame()))
    for a in seq:
        env.step(a)
        f=env.frame()
        print(" a",a,"av",avpos(f),"boxes",boxes4b(f))
    return env

# navigate to below box(48,52): right2, down to rows52-54
if __name__=='__main__':
    pass

def run_seq(seq, verbose=False):
    env=get_env_at_level(4)
    for a in seq:
        if env.terminal(): break
        env.step(a)
        if verbose:
            f=np.array(env.frame())
            print('a',a,'lvl',env.levels_completed,'av',avpos(f),'b4',boxes4b(f),'n3',int((f==3).sum()))
    f=np.array(env.frame())
    print('END lvl',env.levels_completed,'terminal',env.terminal(),'av',avpos(f),'boxes4',boxes4b(f),'n3',int((f==3).sum()))
    return env

def allboxes(f):
    # return list of top-left of any 4x4 box (color 4 or 3 border)
    f=np.array(f); out=[]
    for col in (4,3):
        ys,xs=np.where(f==col)
        for y,x in zip(ys,xs):
            if y>=62: continue
            if (y==0 or f[y-1,x]!=col) and (x==0 or f[y,x-1]!=col):
                out.append((int(y),int(x),col))
    return sorted(out)

def av_tl(f):
    p=avpos(f)
    if p is None: return None
    return (p[0],p[2])  # min row(of14), min col

def try_move(env,a):
    f0=av_tl(env.frame()); env.step(a); f1=av_tl(env.frame()); return f0,f1

def goto(env, target_tl, max_steps=40):
    # greedy move avatar TL toward target_tl (row,col)
    for _ in range(max_steps):
        r,c=av_tl(env.frame())
        tr,tc=target_tl
        if (r,c)==(tr,tc): return True
        acts=[]
        if r>tr: acts.append(1)
        if r<tr: acts.append(2)
        if c>tc: acts.append(3)
        if c<tc: acts.append(4)
        moved=False
        for a in acts:
            r0,c0=av_tl(env.frame())
            env.step(a)
            r1,c1=av_tl(env.frame())
            if (r1,c1)!=(r0,c0):
                moved=True; break
        if not moved:
            # try any axis to unstick
            for a in [1,2,3,4]:
                r0,c0=av_tl(env.frame()); env.step(a); r1,c1=av_tl(env.frame())
                if (r1,c1)!=(r0,c0): moved=True; break
            if not moved: return False
    return av_tl(env.frame())==tuple(target_tl)

def remove_box(env, box_tl, face):
    # position avatar adjacent on `face` side, then bump+USE until box gone
    by,bx=box_tl
    if face=='up': tgt=(by-4,bx); dir=2
    elif face=='down': tgt=(by+4,bx); dir=1
    elif face=='left': tgt=(by,bx-4); dir=4
    else: tgt=(by,bx+4); dir=3
    goto(env,tgt)
    for _ in range(4):
        env.step(dir); env.step(5)
        # check box gone
        f=np.array(env.frame())
        if f[by,bx] not in (3,4): return True
    f=np.array(env.frame())
    return f[by,bx] not in (3,4)

def solve_collect():
    env=get_env_at_level(4)
    order=[((56,44),'up'),((52,60),'left'),((48,52),'up'),((8,56),'down'),((4,48),'down')]
    for box,face in order:
        ok=remove_box(env,box,face)
        f=np.array(env.frame())
        print('box',box,'ok',ok,'lvl',env.levels_completed,'remaining',allboxes(f))
    print('FINAL lvl',env.levels_completed)
