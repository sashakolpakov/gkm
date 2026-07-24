import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
import perception as P

def main(env):
    base=np.asarray(env.frame()).copy()
    targets = {
      'avatar_rc':(20,40),'avatar_cr':(40,20),
      'dock':(19,14),'dock_cr':(14,19),
      'ff1':(51,15),'ff1_cr':(15,51),
      'checker_fill':(50,25),'checker_hole':(50,30),
      'checker_cr':(25,50),
    }
    for name,(x,y) in targets.items():
        c=env.clone(); c.step(6,x,y)
        d=P.frame_delta(base,c.frame())
        print(name,(x,y),"count",d['count'],"lvl",c.levels_completed)
A.run_program('sc25', main)
