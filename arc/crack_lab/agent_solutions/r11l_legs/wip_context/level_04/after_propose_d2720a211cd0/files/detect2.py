import numpy as np
f=np.load('l4_frame.npy')
NON=[0,1,3,5,6]
def isolated_ring_cells(f):
    # cells of a color c (c not in {0,1,3,5,6}) whose 4-neighbors are only bg(5) or same color? 
    # ring outline cells touch only bg. Use: cell color c not bg, and all 4-neighbors in {5} (touch only bg)
    out=[]
    for r,c in np.argwhere(~np.isin(f,[5])):
        col=int(f[r,c])
        if col in (0,1,3,6): continue
        nb=set()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<64 and 0<=cc<64: nb.add(int(f[rr,cc]))
        if nb<= {5}:
            out.append((int(r),int(c),col))
    return out
cells=isolated_ring_cells(f)
# cluster by proximity
clusters=[]
for (r,c,col) in cells:
    placed=False
    for cl in clusters:
        if any(abs(r-rr)<=5 and abs(c-cc)<=5 for rr,cc,_ in cl):
            cl.append((r,c,col)); placed=True; break
    if not placed: clusters.append([(r,c,col)])
# merge overlapping clusters iteratively
merged=True
while merged:
    merged=False
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            a,b=clusters[i],clusters[j]
            if any(abs(r1-r2)<=5 and abs(c1-c2)<=5 for r1,c1,_ in a for r2,c2,_ in b):
                clusters[i]=a+b; clusters.pop(j); merged=True; break
        if merged: break
for cl in clusters:
    rs=[r for r,c,_ in cl]; cs=[c for r,c,_ in cl]; cols=set(x for _,_,x in cl)
    print("RING center",(round(np.mean(rs)),round(np.mean(cs))),"colors",sorted(cols),"n",len(cl))
