# %%
from edge_data_generator import generate_edge_data
import cv2
import numpy as np

w, h = 1080, 1080 
while(1):
    input, output = generate_edge_data(w, h)
    cv2.imshow('input', input)
    cv2.imshow('output', output)

    key = cv2.waitKey(10)
    if key != -1 and key != 255:
        break
cv2.destroyAllWindows()