import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import model as M

SEQ = ['A_R','C_R','A_R','A_R','A_R','C_R','C_R','C_R']

def probe(env):
    for _ in range(5): env.step(6, 5, 32)
    base = env.levels_completed
    cl = env.clone()
    for n,b in enumerate(SEQ):
        cl.step(6, *M.px(b))
        print(n+1, b, "lvl", cl.levels_completed, "term", cl.terminal())
    print("RESULT lvl", cl.levels_completed, "gain", cl.levels_completed-base)
A.run_program("lp85", probe)
