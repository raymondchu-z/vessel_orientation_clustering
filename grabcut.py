import cv2
import numpy as np
import os
import fnmatch


for dirpath, dirs, files in os.walk("G:/research/repo/Feature_extraction-and-classification-for-Text-Document-Analysis-master/03"):
    for filename in fnmatch.filter(files, '*.jpg'):
        #print (filename)
        img_path = dirpath+"/"+filename
        img_path = "1002902/829912.jpg"
        img = cv2.imread(img_path)
        #掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割的时候，也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函数；在处理结束之后，mask中会保存结果
        mask = np.zeros(img.shape[:2],np.uint8)

        #背景模型，如果为None，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
        bgdModel = np.zeros((1,65),np.float64)
        #fgdModel——前景模型，如果为None，函数内部会自动创建一个fgdModel；fgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
        fgdModel = np.zeros((1,65),np.float64)

        #用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
        
        rect = [0,0,383,255]  #格式为（x,y,w,h），分别为左上角坐标和宽度，高度
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)#函数返回值为mask,bgdModel,fgdModel
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')#0和2做背景
        img = img*mask2[:,:,np.newaxis]#使用蒙板来获取前景区域
        cv2.imshow('p',img)
        cv2.imwrite("grabcut.jpg", img)
        cv2.waitKey(0)