import numpy as np
f = np.load('l4_frame.npy')
for r in range(64):
    print(''.join('{:X}'.format(int(v)) for v in f[r]))
