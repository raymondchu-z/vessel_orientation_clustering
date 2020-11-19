import cv2
img_path = "G:/research/repo/Feature_extraction-and-classification-for-Text-Document-Analysis-master/1002902/829912.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (192, 128))
cv2.imwrite("829912.jpg", img)