import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    base=np.asarray(env.frame())
    # step(6) alone in various states
    c=env.clone(); c.step(2)  # move first
    b=np.asarray(c.frame()).copy()
    c.step(6)
    print("step6 after a move delta", int((b!=np.asarray(c.frame())).sum()))
    # try step(6, r, c) with r,c as maze coords for goal center and switch
    for args in [(6,50,44),(6,44,50),(6,8,38),(6,38,8),(6,52,46),(6,46,52)]:
        cc=env.clone()
        try:
            cc.step(*args)
            print(args,"delta",int((base!=np.asarray(cc.frame())).sum()))
        except Exception as e:
            print(args,"ERR",e)
    # try step with 3 coord args
    for args in [(6,1,1,1),(6,50,44,9)]:
        cc=env.clone()
        try:
            cc.step(*args); print(args,"ok delta",int((base!=np.asarray(cc.frame())).sum()))
        except Exception as e:
            print(args,"ERR",e)
    raise SystemExit
A.run_program('g50t', prog)
