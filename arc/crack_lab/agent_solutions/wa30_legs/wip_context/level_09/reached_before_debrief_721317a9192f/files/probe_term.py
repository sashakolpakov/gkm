import l9env
import numpy as np
env=l9env.get_l9()
for t in range(90):
    if env.terminal():
        print("terminal at t",t,"lvl",env.levels_completed); break
    lvl=env.levels_completed
    env.step(5)
    if env.levels_completed!=lvl:
        print("LEVEL CHANGE at t",t,"->",env.levels_completed)
else:
    print("no terminal in 90, lvl",env.levels_completed)
