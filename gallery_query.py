import pandas as pd
import os 
import numpy as np

def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False
# 读取整个csv文件
query_df = pd.read_csv("1002902/file_seleted_label.csv")
query_list = query_df['filename'].value_counts().index.tolist()
files_list = os.listdir( "1002902" )
files_list = list(filter(file_filter, files_list))

gallery_list = list(set(files_list)^set(query_list))
print(query_list)
print(gallery_list)
