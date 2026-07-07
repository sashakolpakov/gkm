from probe5 import *
env = get_env()
c = env.clone()
# avatar rows36-39 cols44-47. Box B6 at (56,44). Move down 16 to be adjacent above it.
for i in range(16):
    c.step(2)
f = np.array(c.frame())
ys,xs = np.where(f==14)
print("avatar 14 cells rows", ys.min(), ys.max(), "cols", xs.min(), xs.max())
c.step(5)  # grab
f2 = np.array(c.frame())
diff(f, f2)
