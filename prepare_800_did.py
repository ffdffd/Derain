import os
import os.path
import cv2 # (6)pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
import os.path
import threading
# import random
import numpy as np
import lmdb
np.random.seed(0)
import h5py # (5)pip install h5py
import torch
import cv2 # (6)pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import glob,re
import torch.utils.data as udata
from utils import *
from tqdm import *
from patchify import patchify
import lmdb,pickle
import torchvision.utils as utils


def prepare_data_Rain800(data_path, patch_size, stride):
    # train
    print('process training data')


    save_target_path = os.path.join(data_path, 'train_target_did_L.h5')
    save_input_path = os.path.join(data_path, 'train_input_did_L.h5')
    
    target_h5f = h5py.File(save_target_path, 'w') # 一个h5py文件是 “dataset” 和 “group” 二合一的容器 ； 主文件夹以 ‘/’ 开始，这又像Linux的树形结构。知道这些我们就可以开始向 h5py 文件读取或者写入了
    input_h5f = h5py.File(save_input_path, 'w')

    if '800' in data_path:
        ls1 = os.listdir(data_path+'/training')
        ls1 = [data_path+'/training/'+i for i in ls1]
    else:
        ls1 = [data_path + '/Rain_Heavy/train2018new/' + i for i in os.listdir(data_path+'/Rain_Light/train2018new')]
    
    train_num = 0
    for key in tqdm(ls1):  
        img = cv2.imread(key)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = np.float32(normalize(img)).transpose(2,0,1)
        target = img[:,:,0:img.shape[2]//2]
        input = img[:,:,img.shape[2]//2:]
        
        target_patches= patchify(target,(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        input_patches= patchify(input,(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        for n in range(target_patches.shape[0]): # target_patches.shape[3] = 15
           target_data = target_patches[n,:, :, :].copy()
           target_h5f.create_dataset(str(train_num), data=target_data)
           input_data = input_patches[n,:, :, :].copy()
           input_h5f.create_dataset(str(train_num), data=input_data)
           train_num += 1
    target_h5f.close()
    input_h5f.close()
    
class Dataset_DID_800(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_DID_800, self).__init__()
        if 'DID' in data_path:
            ls1 = [data_path + '/Rain_Heavy/train2018new/' + i for i in os.listdir(data_path+'/Rain_Heavy/train2018new')]
            # ls1 += [data_path + '/Rain_Medium/train2018new/' + i for i in os.listdir(data_path+'/Rain_Medium/train2018new')]
            # ls1 += [data_path + '/Rain_Light/train2018new/' + i for i in os.listdir(data_path+'/Rain_Light/train2018new')]
            
        else:
            ls1 = os.listdir(data_path)
            ls1 = [data_path+'/'+i for i in ls1]
        self.keys = ls1
    def __len__(self):
        return 3000
    def __getitem__(self, index):
        key = self.keys[index]
        img = cv2.imread(key)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = np.float32(normalize(img)).transpose(2,0,1)
        target = img[:,:,0:img.shape[2]//2]
        input = img[:,:,img.shape[2]//2:]
        target,input =np.expand_dims(target, 0), np.expand_dims(input, 0)

        high = target.shape[2]//32
        wight = target.shape[3]//32
        target = target[:,:, 0:high*32, 0:wight*32]
        
        high = input.shape[2]//32
        wight = input.shape[3]//32
        input = input[:,:, 0:high*32, 0:wight*32]
        return torch.Tensor(input).squeeze(0), torch.Tensor(target).squeeze(0)    


class Dataset_DID_800(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_DID_800, self).__init__()
        ls1 = os.listdir(data_path+'/train')
        ls1 = [data_path+'/'+i for i in ls1]
        self.keys = ls1
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        key = self.keys[index]
        img = cv2.imread(key)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = np.float32(normalize(img)).transpose(2,0,1)
        target = img[:,:,0:img.shape[2]//2]
        input = img[:,:,img.shape[2]//2:]
        target,input =np.expand_dims(target, 0), np.expand_dims(input, 0)

        high = target.shape[2]//32
        wight = target.shape[3]//32
        target = target[:,:, 0:high*32, 0:wight*32]
        
        high = input.shape[2]//32
        wight = input.shape[3]//32
        input = input[:,:, 0:high*32, 0:wight*32]
        return torch.Tensor(input).squeeze(0), torch.Tensor(target).squeeze(0)
    
# 800
# prepare_data_Rain800('/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain800/Rain800',192,192)

prepare_data_Rain800('/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-training',192,192)

