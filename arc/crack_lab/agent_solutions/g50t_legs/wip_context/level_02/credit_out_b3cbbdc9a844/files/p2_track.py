from l2env import get_l2_env
import numpy as np
from legs import _components

def blobs9(f):
    return _components(f==9)

env = get_l2_env()
f0 = np.asarray(env.frame())
# avatar is the 9-blob near rows26-30 col50-54. list all 9 comps
comps = blobs9(f0)
print("num 9-comps", len(comps))
for tr,tc,cells in comps:
    print("  tl",(tr,tc),"size",len(cells))
