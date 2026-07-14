import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def bod(f):
    r9=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15]
    r2=[b.bbox for b in P.connected_components(f,colors=[2])]
    return "9="+str(r9)+" 2="+str(r2)+" leg="+''.join(str(int(v)) for v in f[1,1:8])
def step(c,a,label):
    c.step(a); print(label, P.ACTION_NAME[a], bod(np.asarray(c.frame())))
def prog(env):
    c=env.clone()
    print("init", bod(np.asarray(c.frame())))
    step(c,2,"1")
    step(c,5,"2")
    step(c,4,"3")
    step(c,4,"4")
    step(c,5,"5")  # USE again (phase B->A)
    step(c,2,"6")
    step(c,5,"7")  # USE (A->B) from (14,14) again
    raise SystemExit
A.run_program('g50t', prog)
