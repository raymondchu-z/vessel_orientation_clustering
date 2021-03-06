"""
HOG_Kmeans.py
把5个文件的流程合在一份代码，减少了中间文件。
实现读入目标文件夹内所有图片，然后提取HOG特征，并将HOG特征作为特征向量使用kmeans算法聚类。  
最后要实现每个文件夹下保存一个文件夹同名的x.csv文件，
然后还要选取数量最多的五个类，类内随机选取一张作为query
"""
import cv2
from skimage.feature import hog
import numpy as np
import pandas as pd
import os 
import fnmatch
from sklearn.cluster import KMeans
import time
from sklearn.externals import joblib
from joblib import Parallel, delayed

start_time = time.time()
debug_flag = 1
N_CLUSTERS = 5
dircount = 0
dataset_path = ("G:/research/repo/Feature_extraction-and-classification-for-Text-Document-Analysis-master/")
for dirpath, dirs, rootfiles in os.walk(dataset_path):
    for dir in dirs:
        if(dir != "03"):
            continue
        dircount+=1
        if os.path.exists("model.pkl"):#删除原来的
            os.remove("model.pkl")
            os.remove("file_label.csv")
            os.remove("file_seleted_label.csv")
        files = os.listdir( os.path.join(dataset_path,dir ))#图片列表
        # print(files[0][-3:])
        if not files[0][-3:]=='jpg':#如果删了前面三个文件，第一个仍不是图片，则跳过
            continue
        # print (dir)
        start_time_hog = time.time()#计算HOG耗时
        file_label_dict={}#重新清空
        file_hog_dict={}
        
        if(len(files) < N_CLUSTERS):#不够类别数目则取第一张
            img_path = os.path.join(dataset_path, dir, files[0])
            file_seleted_label_df = pd.DataFrame({'filename':files[0],'label':0,'path':img_path},pd.Index(range(1)))
            print(file_seleted_label_df)
            file_seleted_label_df.to_csv(dir+"/file_seleted_label.csv",index=False)
            continue
        for filename in fnmatch.filter(files, '*.jpg'):
            img_path = os.path.join(dataset_path, dir, filename)
            img = cv2.imread(img_path)
            # Convert to grayscale and apply Gaussian filtering
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #转灰度图
            # im_th = cv2.adaptiveThreshold(img_gray,255,1,1,11,2) # 自适应阈值，待测试
            # img_gray_cropped = img_gray[64:192, 96:288]# 裁剪坐标为[y0:y1, x0:x1]
            img = cv2.resize(img, (192, 128))
            # cv2.imshow('img_gray',img_gray)
            # cv2.waitKey(0)
            # img = np.power(img/float(np.max(img)), 1.5) #伽马矫正
            hog_fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),transform_sqrt=True, visualize=True,feature_vector=True)
            # print (len(hog_fd))  
            cv2.imshow("hog", hog_img)
            cv2.waitKey(0)          
            file_label_dict.setdefault('filename',[]).append(filename)
            file_label_dict.setdefault("path",[]).append(img_path)
            file_hog_dict.setdefault('filename',[]).append(filename)
            file_hog_dict.setdefault("path",[]).append(img_path)
            hog_features = np.array(hog_fd, 'float32')
            file_hog_dict.setdefault('hog',[]).append(hog_features)
        end_time_hog = time.time()#计算HOG耗时
        file_hog_df = pd.DataFrame(file_hog_dict)
        #####################################
        start_time_cluster = time.time()#计算cluster耗时
        X = file_hog_df['hog'].values
        X = np.vstack( X )
        model=KMeans(n_clusters=N_CLUSTERS, max_iter=300)#指定分类数量，即簇的数量
        label=model.fit_predict(X)#用训练器数据X拟合分类器模型并对训练器数据X进行预测
        time_hog = end_time_hog - start_time_hog
        time_cluster = time.time() - start_time_cluster
        outprint = "IMO:{:<10},file nember:{:<5},time_hog:{:<8.3f},time_cluster:{:<8.3f},time:{:<8.3f},precent:{:<8.2%}"
        outprint = outprint.format(dir, len(files), time_hog, time_cluster, time_hog+time_cluster, dircount/len(dirs))
        print(outprint)
        # print(dir+"  file nember:"+str(len(dir))+'    time_hog: %.2f    time_cluster: %.2f      time: %.2f' % (time_hog, time_cluster, time_hog+time_cluster)) #文件名，文件数，hog时间，cluster时间，总时间，百分比
        ###################################################################
        # '''把聚类的样本打标签'''
        # print("\nlabelPred=",model.labels_)
        # '''显示聚类的质心'''
        # print("\ncentroids=",model.cluster_centers_)
        # '''这个也可以看成损失，就是样本距其最近样本的平方总和'''
        # print("\ninertia=",model.inertia_)
        # #这下面是库里包装的方法
        # '''这个是返回每个样本与聚类质心的距离'''
        # print("model.transform",model.transform(X))#	Transform X to a cluster-distance space
        # '''这个我觉得和损失一样，评价聚类好坏'''
        # print("model.score",model.score(X))
        ##############################################################
        joblib.dump(model,  dir + '/model.pkl')#这里要改成文件夹名。
        file_label_dict["label"] = label
        file_label_df = pd.DataFrame(file_label_dict)
        file_label_df.to_csv(dir+"/file_label.csv",index=False)
        seleted_label = file_label_df['label'].value_counts().index.tolist()[:5]
        ##########################################测试
        file_label_sorted_df = file_label_df.sort_values(by = "label")
        file_label_sorted_df.to_csv(dir+"_file_label_sorted.csv",index=False)
        ##########################################
        # print(seleted_label)
        file_seleted_label_df = pd.DataFrame()
        for l in seleted_label:
            file_seleted_label_df = file_seleted_label_df.append(file_label_df[file_label_df['label']==l].iloc[0],ignore_index=True)#从file_label_df找到label降序排序前五的一条
        file_seleted_label_df['label'] = file_seleted_label_df['label'].astype(int)#因为空的df会默认用float
        # print(file_seleted_label_df)
        file_seleted_label_df.to_csv(dir+"/file_seleted_label.csv",index=False)

        


extime = time.time() - start_time
print('time: %.2f' % (extime)) 