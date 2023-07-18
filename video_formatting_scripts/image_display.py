import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
image = cv2.imread('Trial.jpg')

#print(image.shape)
area = np.array([[201,831],[285,965],[1691,481],[1545,413]],np.int32)
color = (255, 0, 0)
thickness = 2

image = cv2.polylines(image, [area],
                      True, color, thickness)
# Displaying the image
while(1):
     
    cv2.imshow('image', image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
         
cv2.destroyAllWindows()
