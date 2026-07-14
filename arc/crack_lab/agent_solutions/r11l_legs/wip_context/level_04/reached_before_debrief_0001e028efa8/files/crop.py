import numpy as np, sys
f = np.load('l4_frame.npy')
r0,r1,c0,c1 = [int(x) for x in sys.argv[1:5]]
print("cols:", ''.join(str(c%10) for c in range(c0,c1)))
for r in range(r0,r1):
    print(''.join('{:X}'.format(int(v)) for v in f[r,c0:c1]), r)
