import numpy as np
from PIL import Image
pal={0:(0,0,0),1:(230,230,230),2:(80,160,80),4:(200,120,0),5:(60,60,60),7:(240,220,40),9:(160,90,200),12:(0,140,220),14:(220,0,0),15:(0,0,220)}
f=np.load('/tmp/l8_frame.npy')
img=np.zeros((64,64,3),dtype=np.uint8)
for c,rgb in pal.items(): img[f==c]=rgb
Image.fromarray(img).resize((512,512),Image.NEAREST).save('/tmp/l8.png')
print("saved")
