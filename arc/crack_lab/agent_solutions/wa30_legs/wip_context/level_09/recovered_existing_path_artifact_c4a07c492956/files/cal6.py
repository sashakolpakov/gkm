import sys; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players, numpy as np
env=A.Arena('wa30', _budget=A._Budget(50000*400))
for k in range(1,6):
    getattr(players,f'play_level_{k}')(env)
assert env.levels_completed==5
np.save('l6start.npy', np.asarray(env.frame()).copy())
prev=[np.asarray(env.frame()).copy()]; cap={'b':None}
orig=env.step
def cs(a):
    bl=env.levels_completed; r=orig(a)
    if env.levels_completed>bl and cap['b'] is None:
        cap['b']=prev[0].copy()
    prev[0]=np.asarray(env.frame()).copy(); return r
env.step=cs
players.play_level_6(env)
print('after L6 levels=',env.levels_completed)
if cap['b'] is not None: np.save('l6_before.npy',cap['b']); print('captured')
