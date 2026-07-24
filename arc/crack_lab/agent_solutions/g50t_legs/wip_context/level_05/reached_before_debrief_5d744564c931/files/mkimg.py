import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from PIL import Image
pal={0:(20,20,20),1:(220,50,50),2:(50,120,220),5:(90,90,90),8:(230,200,40),9:(40,200,90)}
def prog(env):
    f=np.asarray(env.frame())
    img=np.zeros((64,64,3),np.uint8)
    for k,v in pal.items(): img[f==k]=v
    Image.fromarray(img).resize((512,512),Image.NEAREST).save("frame0.png")
    prog.d=True
A.run_program('g50t', prog)
print("saved")
