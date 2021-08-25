#%%

import cv2
import numpy as np

bg = cv2.imread('./raw_dataset/0.jpg')
doc = cv2.imread('./raw_dataset/1.jpg')
doc = cv2.cvtColor(doc, cv2.COLOR_BGR2RGB)
w, h, _ = doc.shape

M = cv2.getRotationMatrix2D((w/2, h/2), 10, .5)
doc = cv2.warpAffine(doc, M, (w,h))
cv2.imshow('0', doc)

while(1):
    key = cv2.waitKey(1)
    if key != -1 and key != 255:
        break
cv2.destroyAllWindows()
# %%
