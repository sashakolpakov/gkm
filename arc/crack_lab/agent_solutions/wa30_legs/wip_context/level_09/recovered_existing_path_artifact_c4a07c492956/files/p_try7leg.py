import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
base=env.levels_completed; n0=len(env.path)
# top container target cells (interior-ish). Container cells R2-3 C11-14.
try:
    legs.clear_frozen_mover_then_fill(env, targets=[(2,11),(2,12),(3,11),(3,12)])
except Exception as e:
    print("EXC",type(e).__name__,e)
print("moves used",len(env.path)-n0,"level",env.levels_completed,"terminal",env.terminal())
