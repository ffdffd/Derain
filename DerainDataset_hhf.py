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
    for i in range(900): # 源码设置900，为了便于测试，我们设置90
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


def prepare_data_outdoorData(data_path, patch_size, stride,datasetname):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    input_path = os.path.join(data_path, 'hazy/') # \\ 等同于 / Jack 4.7 Great!
    target_path = os.path.join(data_path, 'gt/')
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


def prepare_data_X2Data_2(data_path, patch_size, stride,datasetname):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target_{}.h5'.format(datasetname))
    save_input_path = os.path.join(data_path, 'train_input_{}.h5'.format(datasetname))

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    input_path = os.path.join(data_path, 'hazy/') # \\ 等同于 / Jack 4.7 Great!
    target_path = os.path.join(data_path, 'gt/')
    for i in range(1800): # 便于测试的时候用 20
        target_file = "%04d.png" % (i + 1)
        if not os.path.exists(os.path.join(target_path,target_file)):
            print(os.path.join(target_path,target_file))
            continue
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])
        for j in range(3): # 增加数据的多样性
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


def prepare_Ihaze_Data(data_path, patch_size, stride,datasetname):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target_{}.h5'.format(datasetname))
    save_input_path = os.path.join(data_path, 'train_input_{}.h5'.format(datasetname))

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    input_path = os.path.join(data_path, 'hazy/') # \\ 等同于 / Jack 4.7 Great!
    target_path = os.path.join(data_path, 'GT/')
    for i in range(100):

        if datasetname == 'I-HAZE':
                target_file = "%02d_indoor_GT.jpg" % (i + 1)
        else :
                target_file = "%02d_outdoor_GT.jpg" % (i + 1)
        if not os.path.exists(os.path.join(target_path,target_file)):
            print(os.path.join(target_path,target_file))
            continue
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])

        for j in range(3): # 增加数据的多样性
            if datasetname == 'I-HAZE':
                input_file = "%02d_indoor_hazy.jpg" % (i + 1)
            else :
                input_file = "%02d_outdoor_hazy.jpg" % (i + 1)
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
                input_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
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
    
    
def prepare_ITS_Data(patch, 
                    patch_size=192, 
                    stride=132
                    ):
    # train
    print('process training data')
    datasetname = 'ITS'
    data_path = '/data1/hjh/ProjectData/Defogging/ITS/ITS/train'
    data = '/home/huangjiehui/Project/DerainNet/JackData/ITS/train'
    input_path = os.path.join(data_path,'ITS_haze')
    target_path = os.path.join(data_path,'ITS_clear')
    # save_target_path = os.path.join(data_path, 'train_target_{}.h5'.format(datasetname))
    # save_input_path = os.path.join(data_path, 'train_input_{}.h5'.format(datasetname))
    
    save_target_path = lmdb.open(os.path.join(data, 'train_target_{}.h5'.format(datasetname)),map_size=int(1e11))
    save_input_path = lmdb.open(os.path.join(data, 'train_input_{}.h5'.format(datasetname)),map_size=int(1e11))
    txn_target = save_target_path.begin(write=True)
    txn_input = save_input_path.begin(write=True)


    # save_target_path = os.path.join(data_path, 'train_target_{}.h5'.format(datasetname))
    # # save_input_path = os.path.join(data_path, 'train_input_{}.h5'.format(datasetname))
    # target_h5f = h5py.File(save_target_path, 'w')
    # input_h5f = h5py.File(save_input_path, 'w')
    train_num = 0
    ls = os.listdir(input_path)
    np.random.shuffle(ls)
    for input_file in tqdm(ls[0:10000],desc=f'正在保存:{patch}'):
        target_file = input_file.split('_')[0] + '.png'
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])

        for j in range(3): # 增加数据的多样性
            input_img = cv2.imread(os.path.join(input_path,input_file))
            bi, gi, ri = cv2.split(input_img)
            if j == 0: # 融合 初步无雨 RGB
                input_img = cv2.merge([ri, gi, bi])
                target_img = target    
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)
                
            if j == 1: # 融合 初步无雨 IGB
                #==========堆叠Input为指定格式
                train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
                input_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
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

        target_img = np.float32(normalize(target_img))
        # target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)
        target_patches= patchify(target_img.transpose(2, 0, 1),(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        input_img = np.float32(normalize(input_img))
        # input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
        input_patches= patchify(input_img.transpose(2, 0, 1),(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        for n in range(target_patches.shape[0]):
            # if target_file in set_target:
            #     # target_h5f.create_dataset(str(train_num), data=set_target[target_file])
            #     txn_target.put(str(train_num).encode(),set_target[target_file].encode())
            #     # save_target. 
            # else:
            target_data = target_patches[n,:, :, :].copy()
            txn_target.put(str(train_num).encode(), pickle.dumps(target_data))
            input_data = input_patches[n,:, :, :].copy()
            txn_input.put(str(train_num).encode(), pickle.dumps(input_data))
            train_num += 1
        # set_target[target_file] = str(train_num)
        print("target file: %s # samples: %d train_num:%d" % (input_file, target_patches.shape[0],train_num))
        txn_input.commit()
        txn_target.commit()
        txn_target = save_target_path.begin(write=True)
        txn_input = save_input_path.begin(write=True)

    save_target_path.close()
    save_input_path.close()
            # save_target_path = lmdb.open('train_target_{}'.format(datasetname),map_size=int(1e9))
            # save_input_path = lmdb.open('train_input_{}'.format(datasetname),map_size=int(1e9))
            # txn_target = save_target_path.begin(write=True)
            # txn_input = save_input_path.begin(write=True)

        


    print('training set, # samples %d\n' % train_num)
    
def prepare_OTS_Data(patch, 
                    patch_size=192, 
                    stride=132
                    ):
    # train
    print('process training data')
    # save_target_path = lmdb.open('train_input_{}'.format(datasetname),map_size=int(1e9))
    # save_input_path = lmdb.open('test_input_{}'.format(datasetname),map_size=int(1e9))
    datasetname = 'OTS'
    data_path = '/home/huangjiehui/Project/DerainNet/JackData/OTS'
    input_path = os.path.join(data_path,'haze')
    target_path = os.path.join(data_path,'clear_images')
    
    save_target_path = lmdb.open(os.path.join(data_path, 'train_target_{}.h5'.format(datasetname)),map_size=int(1e12))
    save_input_path = lmdb.open(os.path.join(data_path, 'train_input_{}.h5'.format(datasetname)),map_size=int(1e12))
    txn_target = save_target_path.begin(write=True)
    txn_input = save_input_path.begin(write=True)

    train_num = 0
    ls = os.listdir(input_path)
    # random.shuffle(ls)
    print(len(ls))
    for input_file in tqdm(ls[0:10000],desc=f'正在保存:{patch}'):
        target_file = input_file.split('_')[0] + '.jpg'
        target = cv2.imread(os.path.join(target_path,target_file))
        bt, gt, rt = cv2.split(target)
        target = cv2.merge([rt, gt, bt])

        for j in range(3): # 增加数据的多样性
            input_img = cv2.imread(os.path.join(input_path,input_file))
            bi, gi, ri = cv2.split(input_img)
            if j == 0: # 融合 初步无雨 RGB
                input_img = cv2.merge([ri, gi, bi])
                target_img = target    
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)
            if j == 1: # 融合 初步无雨 IGB
                #==========堆叠Input为指定格式
                train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
                input_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
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

        target_img = np.float32(normalize(target_img))
        if target_img.transpose(2, 0, 1).shape[1] < patch_size:
            print('wrong')
            continue
        target_patches= patchify(target_img.transpose(2, 0, 1),(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        input_img = np.float32(normalize(input_img))
        input_patches= patchify(input_img.transpose(2, 0, 1),(3,patch_size,patch_size),step=stride).reshape(-1,3,patch_size,patch_size)
        for n in range(target_patches.shape[0]):
            target_data = target_patches[n,:, :, :].copy()
            txn_target.put(str(train_num).encode(), pickle.dumps(target_data))

            input_data = input_patches[n,:, :, :].copy()
            txn_input.put(str(train_num).encode(), pickle.dumps(input_data))

            train_num += 1
        # set_target[target_file] = train_num-1
        txn_input.commit()
        txn_target.commit()
        txn_target = save_target_path.begin(write=True)
        txn_input = save_input_path.begin(write=True)
        print("target file: %s # samples: %d, train_num%d:" % (input_file, target_patches.shape[0],train_num))
    # target_h5f.close()
    # input_h5f.close()

    print('training set, # samples %d\n' % train_num)
class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()
        self.data_path = data_path
        target_path = os.path.join(self.data_path, 'train_target_ITS.h5')
        input_path = os.path.join(self.data_path, 'train_input_ITS.h5')
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