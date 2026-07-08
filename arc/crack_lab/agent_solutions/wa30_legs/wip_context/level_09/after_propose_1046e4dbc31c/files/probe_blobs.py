import l9env, perception as P
env = l9env.get_l9()
f = env.frame()
objs = P.object_candidates(f, min_area=4)
for o in objs:
    print(o['color'], 'bbox', o['bbox'], 'size', o['size'], 'area', o['area'])
