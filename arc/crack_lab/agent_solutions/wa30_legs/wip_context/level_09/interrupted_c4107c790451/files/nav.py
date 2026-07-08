import numpy as np
def pos(f,color):
    ys,xs=np.where(f==color)
    return None if len(ys)==0 else (int(ys.min())//4,int(xs.min())//4)
def avc(f):
    return pos(f,14)
def cellmap(f):
    lines=[]
    for R in range(16):
        row=''
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=set(int(v) for v in np.unique(blk))
            if u=={1}: ch='.'
            elif u=={5}: ch='#'
            elif u=={7}: ch='_'
            elif 15 in u: ch='T'
            elif 14 in u or 0 in u: ch='A'
            elif u=={2}: ch='2'
            elif 4 in u and 9 in u: ch='b'
            elif 2 in u and 9 in u: ch='B'
            elif 9 in u: ch='9'
            else: ch='?'
            row+=ch
        lines.append('%2d %s'%(R,row))
    return '\n'.join(lines)
