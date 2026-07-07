from probe5 import *
env = get_env()
c = env.clone()
for i in range(16): c.step(2)
c.step(5)
c.step(1); c.step(1)  # rows up: avatar 50-53? then carry
# get avatar to rows 28-31 while box below rows 32-35
# avatar currently? move up until avatar top row 28
f=np.array(c.frame()); ys,xs=np.where(f==14); print("av rows",ys.min(),ys.max(),"cols",xs.min(),xs.max())
while True:
    f=np.array(c.frame()); ys,xs=np.where(f==14)
    if ys.min()<=28: break
    c.step(1)
f=np.array(c.frame()); ys,xs=np.where(f==14); print("av now",ys.min(),ys.max(),xs.min(),xs.max())
# now move left as far as possible
for i in range(10):
    prev=np.array(c.frame())
    c.step(3)
    cur=np.array(c.frame())
    ys,xs=np.where(cur==14)
    print("left",i,"av cols",xs.min(),xs.max(), "levels",c.levels_completed)
    if (prev==cur).all():
        print("blocked"); break
show(c.frame())
