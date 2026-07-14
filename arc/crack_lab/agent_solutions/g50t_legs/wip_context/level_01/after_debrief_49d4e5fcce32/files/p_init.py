import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
import perception as P

env = A.Arena('g50t')
print("actions:", env.actions)
print("levels:", env.levels_completed, "terminal:", env.terminal())
f = env.frame()
print("shape:", f.shape, "colors:", P.color_counts(f))
# print grid compactly
for row in f:
    print(''.join(str(int(v)) for v in row))
