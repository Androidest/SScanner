#%%
import cv2
import numpy as np
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
    s = np.random.normal(0.4, 0.25 /3)
    bx = np.random.normal(0, bw/3*0.2)
    by = np.random.normal(0, bh/3*0.2)

    DocCent = createTransform(x=-w/2, y=-h/2)
    AS = createTransform(angle=a, scale=2/h)
    T = createTransform(x=dx, y=dy)
    P = createTransform(p1=p1, p2=p2)
    S = createTransform(scale=bh*s)
    BGCent = createTransform(x=bw/2+bx, y=bh/2+by)
    perspective_trans = BGCent@S@P@T@AS@DocCent

    input = cv2.warpPerspective(fg, perspective_trans, (bw, bh), bg, borderMode=cv2.BORDER_TRANSPARENT) 
    
    black = np.zeros((bh, bw))
    pts = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
    pts = perspective_trans @ pts
    pts = pts[:2, :] / pts[2, :]
    pts = np.int32(pts.T)
    output = cv2.polylines(black, [pts], isClosed=True, color=255, thickness=2)

    return input, output.astype(np.uint8)

def random_cut(img, w, h):
    ih, iw, _ = img.shape
    r1 = int(rand() * (ih-h))
    r2 = r1 + h
    c1 = int(rand() * (iw-w))
    c2 = c1 + w
    return img[r1:r2, c1:c2]

def generate_edge_data(w, h):
    docName = np.floor(rand()*25)
    bgName = np.floor(rand()*397)
    doc = cv2.imread('./raw_dataset/docs/{name:.0f}.jpg'.format(name=docName))
    bg = cv2.imread('./raw_dataset/backgrounds/{name:.0f}.jpg'.format(name=bgName))
    bg = random_cut(bg, w, h)
    return random_perspective_placing(doc, bg)

def test(w, h):
    while(1):
        input, output = generate_edge_data(w, h)
        cv2.imshow('input', input)
        cv2.imshow('output', output)

        key = cv2.waitKey(10)
        if key != -1 and key != 255:
            break
    cv2.destroyAllWindows()

def read_dataset(numb):
    for i in range(numb):
        x = cv2.imread("./dataset/x/{i}.jpg".format(i=i))
        y = cv2.imread("./dataset/y/{i}.jpg".format(i=i))
        cv2.imshow('input', x)
        cv2.imshow('output', y)

        key = cv2.waitKey(500)
        if key != -1 and key != 255:
            break
    cv2.destroyAllWindows()

def generate(w, h, start, numb): 
    for i in range(start, start+numb):
        input, output = generate_edge_data(w, h)
        cv2.imwrite("./dataset/x/{i}.jpg".format(i=i), input)
        cv2.imwrite("./dataset/y/{i}.jpg".format(i=i), output)

        print('Generating Images: {i}.jpg'.format(i=i))
        key = cv2.waitKey(1)
        if key != -1 and key != 255:
            cv2.destroyAllWindows()
            return
    print('finished')
    
# generate(w=512, h=512, start=3000, numb=1)
# read_dataset(numb=3000)
