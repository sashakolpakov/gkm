import numpy as np

def avatar(f):
    f = np.array(f)
    ys, xs = np.where(f == 14)
    if len(ys) == 0:
        return None
    # avatar is 4x4 region including 0 strip; use 14+0 combined bbox
    y0, x0 = ys.min(), xs.min()
    y1, x1 = ys.max(), xs.max()
    # expand to 4x4 by checking 0s adjacent
    if y1 - y0 < 3:  # 0-strip is horizontal
        if y0 > 0 and (np.array(f)[y0-1, x0:x1+1] == 0).all():
            y0 -= 1
        else:
            y1 += 1
    if x1 - x0 < 3:
        if x0 > 0 and (np.array(f)[y0:y1+1, x0-1] == 0).all():
            x0 -= 1
        else:
            x1 += 1
    return (y0, x0)

def objects(f, border):
    """find 4x4-ish blocks with given border colour; return set of top-left corners"""
    f = np.array(f)
    out = []
    ys, xs = np.where(f == border)
    seen = set()
    for y, x in zip(ys, xs):
        if (y, x) in seen:
            continue
        # candidate top-left if cell up-left isn't border
        if (y == 0 or f[y-1, x] != border) and (x == 0 or f[y, x-1] != border):
            out.append((y, x))
    return out

def timer(f):
    f = np.array(f)
    return int((f[63] == 7).sum())
