import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
env=A.Arena('g50t')
f=env.frame()
print("   "+''.join(str(c%10) for c in range(64)))
for r in range(64):
    row=''.join(('.' if v==0 else str(int(v))) for v in f[r])
    print(f"{r:2d} {row}")
