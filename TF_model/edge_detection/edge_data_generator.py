#%%
import cv2
import numpy as np
from numpy.core.fromnumeric import resize
from numpy.random import rand

def createTransform(scale=1, angle=0, x=0, y=0, p1=0, p2=0):
    angle = angle/180*np.pi
    return np.array([[scale*np.cos(angle), -scale*np.sin(angle), x ],
                    [scale*np.sin(angle),  scale*np.cos(angle),  y ],
                    [p1, p2, 1]])

def random_perspective_placing(fg, bg):
    bh, bw, _ = bg.shape
    h, w, _ = fg.shape
    dx, dy = np.random.normal(0, 2/3, size=2)
    p1, p2 = np.random.normal(0, 0.4/3, size=2)
    a = np.random.rand()*360
    s = np.random.normal(0.3, 0.2/3)
    bx = np.random.normal(0, bw/3*0.2)
    by = np.random.normal(0, bh/3*0.2)

    DocCent = createTransform(x=-w/2, y=-h/2)
    AS = createTransform(angle=a, scale=2/h)
    T = createTransform(x=dx, y=dy)
    P = createTransform(p1=p1, p2=p2)
    S = createTransform(scale=h/2*s)
    BGCent = createTransform(x=bw/2+bx, y=bh/2+by)
    perspective_trans = BGCent@S@P@T@AS@DocCent

    return cv2.warpPerspective(fg, perspective_trans, (bw, bh), bg, borderMode=cv2.BORDER_TRANSPARENT) 




while(1):
    docName = np.floor(rand()*14)
    doc = cv2.imread('./raw_dataset/docs/{name:.0f}.jpg'.format(name=docName))
    bg = cv2.imread('./raw_dataset/0.jpg')
    result = random_perspective_placing(doc, bg)
    result = cv2.resize(result, None, fx=0.5, fy=0.5)
    cv2.imshow('0', result)
    key = cv2.waitKey(100)
    if key != -1 and key != 255:
        break
cv2.destroyAllWindows()

# %%
