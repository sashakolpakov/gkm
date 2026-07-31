import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr
import model as M

def read_state(env):
    f = arr(env.frame())
    return {(i,j): int(f[M.ROWS[i], M.COLS[j]]) for (i,j) in M.CELLS}

def probe(env):
    for _ in range(5): env.step(6, 5, 32)
    st = read_state(env)
    print("start tokens", len(st), "colors", sorted(set(st.values())))
    print("b cells", [c for c,v in st.items() if v==11])
    seq = [M.ORDER[(i*i+5*i+2) % 6] for i in range(40)]
    cl = env.clone()
    sim = dict(st)
    bad = 0
    for n,b in enumerate(seq):
        cl.step(6, *M.px(b))
        sim = M.apply_state(sim, b)
        real = read_state(cl)
        if real != sim:
            bad += 1
            if bad <= 2:
                print("MISMATCH at", n, b, {k:(sim[k],real[k]) for k in sim if sim[k]!=real[k]})
    print("steps", len(seq), "mismatches", bad, "lvl", cl.levels_completed)
A.run_program("lp85", probe)
