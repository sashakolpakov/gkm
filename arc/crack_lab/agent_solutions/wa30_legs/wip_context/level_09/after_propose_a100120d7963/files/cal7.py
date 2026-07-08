import sys; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players, numpy as np
env=A.Arena('wa30', _budget=A._Budget(50000*400))
for k in range(1,7):
    getattr(players,f'play_level_{k}')(env)
    print('after L%d levels=%d'%(k,env.levels_completed))
assert env.levels_completed==6
np.save('l7start.npy', np.asarray(env.frame()).copy())
prev=[np.asarray(env.frame()).copy()]; cap={'b':None,'a':None}
orig=env.step
def cs(a):
    bl=env.levels_completed; r=orig(a)
    if env.levels_completed>bl and cap['b'] is None:
        cap['b']=prev[0].copy(); cap['a']=np.asarray(env.frame()).copy()
    prev[0]=np.asarray(env.frame()).copy(); return r
env.step=cs
players.play_level_7(env)
print('after L7 levels=',env.levels_completed)
if cap['b'] is not None:
    np.save('l7_before.npy',cap['b'])
    print('captured')
