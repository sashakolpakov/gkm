import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import legs
env=A.Arena('g50t')
print("avatar_tl start:",legs.avatar_tl(env))
rp,reach=legs._move_explore(env)
print("reward_path:",rp,"reach size:",len(reach),"positions:",sorted(reach.keys()))
