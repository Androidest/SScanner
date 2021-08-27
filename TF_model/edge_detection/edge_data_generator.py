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

    sample = cv2.warpPerspective(fg, perspective_trans, (bw, bh), bg, borderMode=cv2.BORDER_TRANSPARENT) 
    
    black = np.zeros((bh, bw))
    pts = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
    pts = perspective_trans @ pts
    pts = pts[:2, :] / pts[2, :]
    pts = np.int32(pts.T)
    ground_truth = cv2.polylines(black, [pts], isClosed=True, color=(1,1,1), thickness=2)

    return sample, ground_truth

def random_cut(img, w, h):
    ih, iw, _ = bg.shape
    r1 = int(rand() * (ih-h))
    r2 = r1 + h
    c1 = int(rand() * (iw-w))
    c2 = c1 + w
    return img[r1:r2, c1:c2]

while(1):
    docName = np.floor(rand()*14)
    w, h = 1080, 1080
    doc = cv2.imread('./raw_dataset/docs/{name:.0f}.jpg'.format(name=docName))
    bg = cv2.imread('./raw_dataset/0.jpg')
    bg = random_cut(bg, w, h)
    sample, ground_truth = random_perspective_placing(doc, bg)

    sample = cv2.resize(sample, None, fx=0.5, fy=0.5)
    ground_truth = cv2.resize(ground_truth, None, fx=0.5, fy=0.5)

    cv2.imshow('0', sample)
    cv2.imshow('1', ground_truth)
    key = cv2.waitKey(100)
    if key != -1 and key != 255:
        break
cv2.destroyAllWindows()

# %%
