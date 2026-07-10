import sys
sys.path.insert(0, '/Users/sasha/gkm/bongard/crack_lab')
import glob, os
import numpy as np
import bongard_arena as A

ws = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.join(ws, open(os.path.join(ws, "current_problem.txt")).read().strip())
pos = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "pos_*.npy")))]
neg = [np.load(p) for p in sorted(glob.glob(os.path.join(pdir, "neg_*.npy")))]
problem = A.Problem("current", "?", "?", pos, neg)
preds = A.load_predicates(os.path.join(ws, "predicates.py"))
print(A.verify(preds, problem).result_line())
