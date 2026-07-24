import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
def show(f,r,c):
    for row in f[r[0]:r[1]]:
        print(''.join(str(int(v)) for v in row[c[0]:c[1]]))
env=A.Arena('g50t')
f=env.frame()
print("== top-left legend rows0-7 cols0-10 ==")
show(f,(0,7),(0,10))
print("== top-right 8 target rows7,15 cols37,45 ==")
show(f,(7,16),(37,46))
print("== bottom-right 9box rows47,58 cols40,52 ==")
show(f,(47,58),(40,52))
print("== mid 8 bar rows38,43 cols13,45 ==")
show(f,(38,43),(13,45))
