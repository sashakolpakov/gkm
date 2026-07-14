from l2env import get_l2_env
import perception as P
import numpy as np
env = get_l2_env()
d = P.action_deltas(env)
for a,info in d.items():
    print("action",a,"count",info["count"],"bbox",info["bbox"])
    for s in info["samples"][:12]:
        print("   ",s)
