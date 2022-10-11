# import os
# import os.path
# import numpy as np
# import random
# import h5py # (5)pip install h5py
# import torch
# import cv2 # (6)pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
# import glob,re
# import torch.utils.data as udata
# from utils import *
# from tqdm import *

# import threading
# from threading import Lock,Thread
# import time,os


# def Im2Patch(img, win, stride=1):
#     k = 0
#     endc = img.shape[0] # endc 3
#     endw = img.shape[1] # endw 512
#     endh = img.shape[2] # endh 384
#     patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride] # [:,0:413:100,0:285:100]
#     TotalPatNum = patch.shape[1] * patch.shape[2] # 5 * 3
#     Y = np.zeros([endc, win * win, TotalPatNum], np.float32) # Y为(3,10000,15)

#     for i in range(win): #　win 100
#         for j in range(win):
#             patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
#             Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
#             k = k + 1
#     return Y.reshape([endc, win, win, TotalPatNum]) # Y为(3,10000,15)
# def prepare_ITS_Data(patch,data_path = '/data1/hjh/ProjectData/Defogging/ITS/ITS/train', patch_size=192, stride=132, datasetname = 'ITS'):
#     # train
#     print('process training data')

#     # save_target_path = lmdb.open('train_input_{}'.format(datasetname),map_size=int(1e9))
#     # save_input_path = lmdb.open('test_input_{}'.format(datasetname),map_size=int(1e9))
    
#     # target_h5f = h5py.File(save_target_path, 'w')
#     # input_h5f = h5py.File(save_input_path, 'w')
    
#     set_target = dict()
#     train_num = 0

#     for input_file in tqdm(os.listdir(input_path)[patch *10000: patch* 10000 + 10000],desc=f'发送:{patch}'):
#         target_file = input_file.split('_')[0] + '.png'
#         target = cv2.imread(os.path.join(target_path,target_file))
#         bt, gt, rt = cv2.split(target)
#         target = cv2.merge([rt, gt, bt])

#         for j in range(3): # 增加数据的多样性
#             input_img = cv2.imread(os.path.join(input_path,input_file))
#             bi, gi, ri = cv2.split(input_img)
#             if j == 0: # 融合 初步无雨 RGB
#                 #==========堆叠Input为指定格式
#                 train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
#                 bi, gi, ri = cv2.split(input_img)
#                 train_ZeroData = np.vstack((train_ZeroData,ri.flatten()))
#                 train_ZeroData = np.vstack((train_ZeroData,gi.flatten()))
#                 train_ZeroData = np.vstack((train_ZeroData,bi.flatten()))
#                 #==========BLS预处理数据================= # 2022.09.03
#                 BLS_ri=BLS_Test(IMG=train_ZeroData[1:,:],width=input_img.shape[0],height=input_img.shape[1]) # 对有雨图像进行预处理，获得 R 通道背景
#                 #==========融合================ # 2022.09+.03
#                 input_img = cv2.merge([BLS_ri, gi, bi])
#                 target_img = target      
#             if j == 1: # 融合 初步无雨 IGB
#                 #==========堆叠Input为指定格式
#                 train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
#                 input_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
#                 hi, si, Ii = cv2.split(input_img_HSI)
#                 train_ZeroData = np.vstack((train_ZeroData,hi.flatten()))
#                 train_ZeroData = np.vstack((train_ZeroData,si.flatten()))
#                 train_ZeroData = np.vstack((train_ZeroData,Ii.flatten()))
#                 #==========BLS预处理数据================= # 2022.09.03
#                 BLS_Ii=BLS_Test(IMG=train_ZeroData[1:,:],width=input_img.shape[0],height=input_img.shape[1]) # 对有雨图像进行预处理，获得 R 通道背景
#                 #==========融合================ # 2022.09+.03
#                 input_img = cv2.merge([BLS_Ii, gi, bi])
#                 target_img = target                
#             if j == 2: # 融合 原图有雨 RGB
#                 input_img = cv2.merge([ri, gi, bi])
#                 target_img = target    
#                 target_img = cv2.flip(target_img, 1)
#                 input_img = cv2.flip(input_img, 1)

#             target_img = np.float32(normalize(target_img))
#             target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

#             input_img = np.float32(normalize(input_img))
#             input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
#             for n in range(target_patches.shape[3]):
#                 if target_file in set_target:
#                     target_h5f.create_dataset(str(train_num), data=set_target[target_file])
#                     # save_target. 
#                 else:
#                     set_target[target_file] = str(train_num)
#                     target_data = target_patches[:, :, :, n].copy()
#                     target_h5f.create_dataset(str(train_num), data=target_data)
#                 input_data = input_patches[:, :, :, n].copy()
#                 input_h5f.create_dataset(str(train_num), data=input_data)

#                 train_num += 1
#         print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))


#     print('training set, # samples %d\n' % train_num)
    
# datasetname = 'ITS'
# data_path = '/data1/hjh/ProjectData/Defogging/ITS/ITS/train' 
# input_path = os.path.join(data_path,'ITS_haze')
# target_path = os.path.join(data_path,'ITS_clear')
# save_target_path = os.path.join(data_path, 'train_target_{}test.h5'.format(datasetname))
# save_input_path = os.path.join(data_path, 'train_input_{}test.h5'.format(datasetname))
# target_h5f = h5py.File(save_target_path, 'w')
# input_h5f = h5py.File(save_input_path, 'w')

# t1 = threading.Thread(target=prepare_ITS_Data,args=(1,))     # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
# t2 = threading.Thread(target=prepare_ITS_Data,args=(2,))
# t1.start()
# t2.start()


from multicpu import multi_cpu
import multicpu
import time 
def process_job(job):
    time.sleep(1)
    return job
# # print(multicpu.__code__)
# print(multicpu.__file__)

jobs = [i for i in range(20)]
result = multi_cpu(process_job, jobs,10, 1)

