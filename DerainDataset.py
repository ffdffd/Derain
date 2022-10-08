import os
import os.path
import numpy as np
import random
import h5py # (5)pip install h5py
import torch
import cv2 # (6)pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import glob
import torch.utils.data as udata
from utils import *
# import pexpect
#　pexpect.spawn('sudo [cmd]').sendline("[YAMTF1107]") # 解决调试时，权限不足的问题 BUG 应该用 sudo chmod 777 /home/jack/Data/RainData 解决

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0] # endc 3
    endw = img.shape[1] # endw 512
    endh = img.shape[2] # endh 384
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride] # [:,0:413:100,0:285:100]
    TotalPatNum = patch.shape[1] * patch.shape[2] # 5 * 3
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) # Y为(3,10000,15)

    for i in range(win): #　win 100
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) # Y为(3,10000,15)


def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    # save_target_path = os.path.join('datasets/train/Rain12600/train_target.h5') #  BUG  斜杆符号不一致：datasets/train/Rain12600\train_input.h5  或者 data_path目录后边少了”/”，加上”/”，程序即可执行正确。
    save_input_path = os.path.join(data_path, 'train_input.h5')
    # save_input_path = os.path.join('datasets/train/Rain12600/train_input.h5')
    
    target_h5f = h5py.File(save_target_path, 'w') # 一个h5py文件是 “dataset” 和 “group” 二合一的容器 ； 主文件夹以 ‘/’ 开始，这又像Linux的树形结构。知道这些我们就可以开始向 h5py 文件读取或者写入了
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(90): # 源码设置900，为了便于测试，我们设置90
        target_file = "%d.jpg" % (i + 1) # 输出图片 1.jpg
        target = cv2.imread(os.path.join(target_path,target_file)) # shape为(512,384,3)
        b, g, r = cv2.split(target) # 将R G B三个通道的数据分别提为单通道，shape均为(512,384,3)
        target = cv2.merge([r, g, b]) # 将单通道合并回三通道

        for j in range(14): # 相当于epoch为14, 每次喂14张图片进去训练
            input_file = "%d_%d.jpg" % (i+1, j+1) # 输入图片 1_1.jpg
            input_img = cv2.imread(os.path.join(input_path,input_file)) # 保持同样大小(512,384,3)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img)) # 对像素值进行归一化，便于神经网络处理！
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride) # patch_size为100;stride为100; .transpose(2,0,1)操作对象是矩阵，让他转置，暂未理解透，先跳过，3-28
            # (512, 384, 3)-> (3, 512, 384) 把宽和高放后面，通道数放前面 https://blog.csdn.net/ll666634/article/details/79048638?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_antiscanv2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
            # target_img 处理后为 target_patches (3, 100, 100, 15) input_img 处理后为 input_patches (3, 100, 100, 15)
            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            # print("target file: %s # samples: %d" % (input_file, target_patches.shape[3])) # target file: 1_1.jpg # samples: 15 ****Jack 添加注释 4.6 省的运行时间长

            for n in range(target_patches.shape[3]): # target_patches.shape[3] = 15
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    # print('training set, # samples %d\n' % train_num) Jack 添加注释 4.6 省的运行时间长


def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy/')
    target_path = os.path.join(data_path, '')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')
    
    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%03d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%03d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2): # 多复制了一次图片，让数据显得更多
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1) # 将图像进行一次水平翻转
                input_img = cv2.flip(input_img, 1) 

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_X2Data(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    input_path = os.path.join(data_path, 'rain/') # \\ 等同于 / Jack 4.7 Great!
    target_path = os.path.join(data_path, 'norain/')
    for i in range(1900):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])

        for j in range(3): # 增加数据的多样性
            input_file = "norain-%dx2.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            bi, gi, ri = cv2.split(input_img)
            if j == 0: # 融合 初步无雨 RGB
                #==========堆叠Input为指定格式
                train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
                bi, gi, ri = cv2.split(input_img)
                train_ZeroData = np.vstack((train_ZeroData,ri.flatten()))
                train_ZeroData = np.vstack((train_ZeroData,gi.flatten()))
                train_ZeroData = np.vstack((train_ZeroData,bi.flatten()))
                #==========BLS预处理数据================= # 2022.09.03
                BLS_ri=BLS_Test(IMG=train_ZeroData[1:,:],width=input_img.shape[0],height=input_img.shape[1]) # 对有雨图像进行预处理，获得 R 通道背景
                #==========融合================ # 2022.09+.03
                input_img = cv2.merge([BLS_ri, gi, bi])
                target_img = target      
            if j == 1: # 融合 初步无雨 IGB
                #==========堆叠Input为指定格式
                train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
                input_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSI)
                hi, si, Ii = cv2.split(input_img_HSI)
                train_ZeroData = np.vstack((train_ZeroData,hi.flatten()))
                train_ZeroData = np.vstack((train_ZeroData,si.flatten()))
                train_ZeroData = np.vstack((train_ZeroData,Ii.flatten()))
                #==========BLS预处理数据================= # 2022.09.03
                BLS_Ii=BLS_Test(IMG=train_ZeroData[1:,:],width=input_img.shape[0],height=input_img.shape[1]) # 对有雨图像进行预处理，获得 R 通道背景
                #==========融合================ # 2022.09+.03
                input_img = cv2.merge([BLS_Ii, gi, bi])
                target_img = target                
            if j == 2: # 融合 原图有雨 RGB
                input_img = cv2.merge([ri, gi, bi])
                target_img = target    
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_X2Data_2(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    input_path = os.path.join(data_path, 'rain/') # \\ 等同于 / Jack 4.7 Great!
    target_path = os.path.join(data_path, 'norain/')
    for i in range(20): # 便于测试的时候用 20
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])

        for j in range(3): # 增加数据的多样性
            input_file = "norain-%dx2.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            bi, gi, ri = cv2.split(input_img)
            if j == 0: #==========融合================ # 2022.09.03
                input_img = cv2.merge([ri, gi, bi])
                target_img = target      
            if j == 1:
                input_img = cv2.merge([ri, gi, bi])
                target_img = target                
            if j == 2:
                input_img = cv2.merge([ri, gi, bi])
                target_img = target    
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key]) # 将数据转化为矩阵
        input = np.array(input_h5f[key])

        target_h5f.close() 
        input_h5f.close()
        # for zip(input, target):
        #     for x in range(size(input))
        return torch.Tensor(input), torch.Tensor(target)


