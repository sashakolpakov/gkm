import l9env, json
for fn in ["best_l9_final.json","l9_result_path.json","best_l9_path.json"]:
    try:
        p=json.load(open(fn))
    except Exception as e:
        print(fn,"ERR",e); continue
    env=l9env.get_l9()
    start=env.levels_completed
    for a in p:
        if env.terminal(): break
        env.step(a)
    print(fn,"len",len(p),"start",start,"->",env.levels_completed,"terminal",env.terminal())
