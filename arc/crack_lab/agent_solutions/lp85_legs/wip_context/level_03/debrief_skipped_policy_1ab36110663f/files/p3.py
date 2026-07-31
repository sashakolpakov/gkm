import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr, frame_delta

ROWS = [17,20,23,26,29,32,35,38,41,44]
COLS = [17,20,23,26,29,32,35,38,41,44]

def grid(env):
    f = arr(env.frame())
    g = {}
    for i,r in enumerate(ROWS):
        for j,c in enumerate(COLS):
            v = int(f[r,c])
            if v != 4:
                g[(i,j)] = v
    return g

def show(g):
    out = []
    for i in range(10):
        out.append("".join(("%x"%g[(i,j)]) if (i,j) in g else "." for j in range(10)))
    return out

BUT = {"A_L":(20,17),"A_R":(39,17),"B_L":(14,26),"B_R":(48,26),"C_L":(14,35),"C_R":(48,35)}

def probe(env):
    print("row1 @L0:", "".join("%x"%v for v in arr(env.frame())[1]))
    for _ in range(5):
        env.step(6, 5, 32)
    print("row1 @L1:", "".join("%x"%v for v in arr(env.frame())[1]))
    base = grid(env)
    print("BASE"); [print("  ",l) for l in show(base)]
    bf = env.frame()
    for name,(x,y) in BUT.items():
        cl = env.clone(); cl.step(6,x,y)
        d = frame_delta(bf, cl.frame())
        g2 = grid(cl)
        print(name, (x,y), "delta_cells", d["count"], "bbox", d["bbox"], "lvl", cl.levels_completed)
        for l in show(g2): print("   ", l)
    # non-button control click
    for name,(x,y) in {"empty_panel":(30,48),"outside":(5,5),"lvl1btn":(5,32)}.items():
        cl = env.clone(); cl.step(6,x,y)
        d = frame_delta(bf, cl.frame())
        print("ctrl", name, (x,y), "delta", d["count"], d["bbox"], "lvl", cl.levels_completed)
A.run_program("lp85", probe)
