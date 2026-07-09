import l9env, perception as P
import numpy as np
def blobs(f,cols):
    return [ (o['color'],o['bbox']) for o in P.object_candidates(f,min_area=3) if o['color'] in cols]
env=l9env.get_l9()
# move right toward box at cols56-59
print("start avatar/box:", blobs(np.asarray(env.frame()),{14,4}))
seq=[4,4,4,4,4]  # right 5 times = 20 cells, avatar col32->52
for i,a in enumerate(seq):
    env.step(a)
f=np.asarray(env.frame())
print("after right x5:", [b for b in blobs(f,{14,4,0}) if b[1][1]>50 or b[0]==14])
