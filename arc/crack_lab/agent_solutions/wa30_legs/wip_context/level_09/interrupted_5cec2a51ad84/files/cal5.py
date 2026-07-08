import sys; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players, numpy as np
env=A.Arena('wa30', _budget=A._Budget(50000*400))
# play levels 1-4 to reach level 5 start
for k in range(1,5):
    fn=getattr(players,f'play_level_{k}')
    before=env.levels_completed
    fn(env)
    print('after L%d play, levels=%d'%(k,env.levels_completed))
assert env.levels_completed==4
# Now at level 5 start. Snapshot, then play with capture of win transition.
f5start=np.asarray(env.frame()).copy()
np.save('l5start.npy',f5start)
# manually run play_level_5 but capture frame just before win: wrap step
prev=[f5start.copy()]
cap={'before':None,'after':None}
orig=env.step
def cs(a):
    b=env.levels_completed
    r=orig(a)
    if env.levels_completed>b and cap['before'] is None:
        cap['before']=prev[0].copy()
        cap['after']=np.asarray(env.frame()).copy()
    prev[0]=np.asarray(env.frame()).copy()
    return r
env.step=cs
players.play_level_5(env)
print('after L5 play levels=',env.levels_completed)
if cap['before'] is not None:
    np.save('l5_before.npy',cap['before']); np.save('l5_after.npy',cap['after'])
    print('captured win transition')
