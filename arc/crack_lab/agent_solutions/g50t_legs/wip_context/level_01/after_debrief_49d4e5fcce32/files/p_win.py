import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
path=[4,4,4,4,5, 2,2,2,2,2,2,2, 4,4,4,4,4]
e=A.Arena('g50t')
prev=None
for i,a in enumerate(path):
    e.step(a)
    print(i+1,"act",a,"levels",e.levels_completed,"terminal",e.terminal())
