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
# import pexpect
#　pexpect.spawn('sudo [cmd]').sendline("[YAMTF1107]") # 解决调试时，权限不足的问题 BUG 应该用 sudo chmod 777 /home/jack/Data/RainData 解决

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0] # endc 3
    endw = img.shape[1] # endw 512
    endh = img.shape[2] # endh 384
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride] # [:,0:413:100,0:285:100]   (3,2,3)
    TotalPatNum = patch.shape[1] * patch.shape[2] # 5 * 3
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) # Y为(3,10000,15)

    for i in range(win): #　win 100
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) # Y为(3,10000,15)

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
        return 1500
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
    
    
class Dataset_800_h5(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain800'):
        super(Dataset_800_h5, self).__init__()
        self.data_path = data_path
        target_path = os.path.join(self.data_path, 'train_target_800.h5')
        input_path = os.path.join(self.data_path, 'train_input_800.h5')
        self.target_h5f = h5py.File(target_path, 'r')
        self.input_h5f = h5py.File(input_path, 'r')
        self.keys = list(self.target_h5f.keys())
        print(len(self.keys))

    def __len__(self):
        return(len(self.keys))
    def __getitem__(self, index):
        key = self.keys[index]
        target = np.array(self.target_h5f[key])
        input = np.array(self.input_h5f[key])
        return torch.Tensor(input), torch.Tensor(target)
    
class Dataset_did_h5(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-training'):
        super(Dataset_did_h5, self).__init__()
        self.data_path = data_path
        target_path = os.path.join(self.data_path, 'train_target_did_L.h5')
        input_path = os.path.join(self.data_path, 'train_input_did_L.h5')
        self.target_h5f = h5py.File(target_path, 'r')
        self.input_h5f = h5py.File(input_path, 'r')
        self.keys = list(self.target_h5f.keys())
        print(len(self.keys))

    def __len__(self):
        return(len(self.keys))
    def __getitem__(self, index):
        key = self.keys[index]
        target = np.array(self.target_h5f[key])
        input = np.array(self.input_h5f[key])
        return torch.Tensor(target),torch.Tensor(input)
    
class Dataset(udata.Dataset):
    # def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/ITS/train'):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData'):
        super(Dataset, self).__init__()
        self.data_path = data_path
        target_path = os.path.join(self.data_path, 'train_target_OTS.h5')
        input_path = os.path.join(self.data_path, 'train_input_OTS.h5')
        env1 = lmdb.open(target_path, map_size=int(1e11))
        env2 = lmdb.open(input_path, map_size=int(1e11))
        self.txn1 = env1.begin()
        self.txn2 = env2.begin()
    def __len__(self):
        return 107226
    def __getitem__(self, index):
        target = pickle.loads(self.txn1.get(str(index).encode()))# 将数据转化为矩阵
        input =  pickle.loads(self.txn2.get(str(index).encode()))
        return torch.from_numpy(input), torch.from_numpy(target)
    

class Dataset_200H_patch(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_200H_patch, self).__init__()
        self.data_path = data_path
        self.img_names =  os.listdir(data_path)
        self.num = 6    # 把一张一张大图分为num份
    def __len__(self):
        return len(self.img_names)*self.num

    def __getitem__(self, index):
        # target
        # GT path with target data
            # !!!!!!RGB
        y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index//self.num]))
        gt_path = os.path.join(self.data_path, self.img_names[index//self.num]).replace("hazy",'GT')   # #  I-HAZE O-HAZE
        gt = cv2.imread(gt_path)
        
        b, g, r = cv2.split(y_origin)
        y_origin = cv2.merge([r, g, b])
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        
        gt= patchify(gt.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        y= patchify(y_origin.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        # gt,y = progress(gt),progress(y)
        l = gt.shape[0]//(self.num-1)
        return y[index%self.num*l : (index%self.num+1)*l ,:,:,:], gt[index%self.num*l : (index%self.num+1)*l ,:,:,:]

class Dataset_200H_patch(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain14000/Rain12600'):
        super(Dataset_200H_patch, self).__init__()
        self.data_path = data_path
        self.img_names =  os.listdir(data_path)
        self.num = 6    # 把一张一张大图分为num份
    def __len__(self):
        return len(self.img_names)*self.num

    def __getitem__(self, index):
        # target
        # GT path with target data
            # !!!!!!RGB
        y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index//self.num]))
        gt_path = os.path.join(self.data_path, self.img_names[index//self.num]).replace("hazy",'GT')   # #  I-HAZE O-HAZE
        gt = cv2.imread(gt_path)
        
        b, g, r = cv2.split(y_origin)
        y_origin = cv2.merge([r, g, b])
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        
        gt= patchify(gt.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        y= patchify(y_origin.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        # gt,y = progress(gt),progress(y)
        l = gt.shape[0]//(self.num-1)
        return y[index%self.num*l : (index%self.num+1)*l ,:,:,:], gt[index%self.num*l : (index%self.num+1)*l ,:,:,:]
    

class Dataset_Rain200L(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200L/Rain200L/train'):
        super(Dataset_Rain200L, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
    def __len__(self):
        return len(os.listdir(self.input_names))
    def __getitem__(self, index):
        target_names = self.target_names + f'norain-{index+1}.png'
        input_names =  self.input_names + f'norain-{index+1}x2.png'
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt
    
class Dataset_Rain100H(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200L/Rain200L/train'):
        super(Dataset_Rain200L, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
    def __len__(self):
        return len(os.listdir(self.input_names))
    def __getitem__(self, index):
        target_names = self.target_names + f'norain-{index+1}.png'
        input_names =  self.input_names + f'norain-{index+1}x2.png'
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt
class Dataset_14000(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/home/huangjh/Data/ProjectData/RainData/Rain14000/Rain1400/train_input.h5'):
        super(Dataset_14000, self).__init__()
        self.data_path = data_path
        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')
        self.target_h5f = h5py.File(target_path, 'r')
        self.input_h5f = h5py.File(input_path, 'r')
        self.keys = list(self.target_h5f.keys())
        print(len(self.keys))

    def __len__(self):
        return(len(self.keys))
    def __getitem__(self, index):
        key = self.keys[index]
        target = np.array(self.target_h5f[key])
        input = np.array(self.input_h5f[key])
        return torch.Tensor(input), torch.Tensor(target)
    
def progress(y_origin):
    #  rbg 255 add_channel 32
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    
    y = normalize(np.float32(y)).transpose(2, 0, 1)
    high = y.shape[1]//32
    wight = y.shape[2]//32
    y = y[:, 0:high*32, 0:wight*32]
    # y = torch.Tensor(y)
    return y
class Dataset_X2(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/X2Data'):
        super(Dataset_X2, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        # self.input_names = [data_path+'/rain/'+i for i in input_names]
    def __len__(self):
        return 1800
    def __getitem__(self, index):
        y_origin = cv2.imread(self.input_names+f'norain-{index+1}x2.png')
        gt = cv2.imread(self.target_names+f'norain-{index+1}.png')
        y,gt = progress(y_origin),progress(gt)
        return y, gt
class Dataset_OTS(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_OTS, self).__init__()
        self.data_path ='/home/huangjiehui/Project/DerainNet/JackData/OTS'
        target_path = os.path.join(self.data_path, 'train_target_OTS.h5')
        input_path = os.path.join(self.data_path, 'train_input_OTS.h5')
        env1 = lmdb.open(target_path, map_size=int(1e11))
        env2 = lmdb.open(input_path, map_size=int(1e11))
        self.txn1 = env1.begin()
        self.txn2 = env2.begin()
    def __len__(self):
        return 69543
    def __getitem__(self, index):
        target = pickle.loads(self.txn1.get(str(index).encode()))# 将数据转化为矩阵
        input =  pickle.loads(self.txn2.get(str(index).encode()))
        return torch.from_numpy(input), torch.from_numpy(target)


class Dataset_DID(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-training'):
        super(Dataset_DID, self).__init__()
        ls1 = os.listdir(data_path+'/Rain_Heavy/train2018new/')
        ls1 = [data_path+'/Rain_Heavy/train2018new/'+i for i in ls1]
        ls2 = os.listdir(data_path+'/Rain_Heavy/train2018new/')
        ls1+=[data_path+'/Rain_Heavy/train2018new/'+i for i in ls2]
        ls3 = os.listdir(data_path+'/Rain_Heavy/train2018new/')
        ls1 += [data_path+'/Rain_Heavy/train2018new/'+i for i in ls3]
        self.keys = ls1
        print(len)
        
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        key = self.keys[index]
        img = cv2.imread(key)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = np.float32(normalize(img)).transpose(2,0,1)
        return torch.Tensor(img[:,:,0:512]), torch.Tensor(img[:,:,512:1024])
    

class Dataset_X2patch(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/X2Data'):
        super(Dataset_X2patch, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain_H/'
        self.input_names =  data_path+'/rain_H/'
        self.num = 3    # 把一张一张大图分为num份
    def __len__(self):
        return len(os.listdir(self.input_names))*self.num
    def __getitem__(self, index):
        # target
        # GT path with target data
            # !!!!!!RGB
        y_origin = cv2.imread(self.input_names+f'norain-{index//self.num+1}x2.png')
        gt = cv2.imread(self.target_names+f'norain-{index//self.num+1}.png')
        b, g, r = cv2.split(y_origin)
        y_origin = cv2.merge([r, g, b])
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        
        gt= patchify(gt.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        y= patchify(y_origin.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
        # gt,y = progress(gt),progress(y)
        index = index%self.num
        if gt.shape[0]% (self.num-1) == 0:
            l = gt.shape[0]//(self.num)
            return y[index*l : (index+1)*l ,:,:,:], gt[index.num*l : (index+1)*l ,:,:,:]
        else:
            l = gt.shape[0]//(self.num-1)
            if index == (self.num-1):
                index = 0
            return y[index*l : (index+1)*l ,:,:,:], gt[index*l : (index+1)*l ,:,:,:]

    

class Dataset_RainX2_H(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/X2Data'):
        super(Dataset_RainX2_H, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain_H/'
        self.input_names =  data_path+'/rain_H/'
    def __len__(self):
        return len(os.listdir(self.input_names))
    def __getitem__(self, index):
        target_names = self.target_names + f'norain-{index+1}.png'
        input_names =  self.input_names + f'norain-{index+1}x2.png'
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt

class Dataset_Rain200H(udata.Dataset):
    def __init__(self, data_path='/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200H/Rain200H/train'):
        super(Dataset_Rain200H, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
    def __len__(self):
        return len(os.listdir(self.input_names))
    def __getitem__(self, index):
        target_names = self.target_names + f'norain-{index+1}.png'
        input_names =  self.input_names + f'norain-{index+1}x2.png'
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt

if __name__ == "__main__":
    pass
    # set_target = dict()
    # prepare_ITS_Data(0)
    # prepare_OTS_Data(0)
    # result = []

    # result = multi_cpu(prepare_ITS_Data, [0,2,4,6,8], 20, 1)
    # x = Dataset()
    # print(x.__getitem__(2))
    x = Dataset_X2patch()
    x.__getitem__(2)