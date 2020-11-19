import cv2
from skimage.feature import hog
import numpy as np

img_path = ("1002902/1184384.jpg")
img = cv2.imread(img_path)
img = cv2.resize(img, (192, 128))
hog_fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),transform_sqrt=True, visualize=True,feature_vector=True)
cv2.normalize(hog_img,hog_img,0,255,cv2.NORM_MINMAX)
hog_img = hog_img.astype(np.uint8)
cv2.imshow("hog", hog_img)
cv2.waitKey(0)
cv2.imwrite("hog.jpg", hog_img)