import l9env, perception as P
env = l9env.get_l9()
d = P.action_deltas(env)
for a, info in d.items():
    print(P.ACTION_NAME[a], 'count', info['count'], 'bbox', info['bbox'])
    for s in info['samples'][:12]:
        print('   ', s)
