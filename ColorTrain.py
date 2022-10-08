from utils import *
import h5py
import cv2

data_path="/home/huangjh/Data/ProjectData/RainData/X2Data/Train/Light/"
tailor_size=96
stride=100
image_num=1600
BLS_option="Train"

print('process train data, wait ... ')

# 1. h5文件
save_input_path = os.path.join(data_path, 'train_input.h5')
save_target_path = os.path.join(data_path, 'target_input.h5')
input_h5f = h5py.File(save_input_path, 'w')  # 手动创建的h5文件，不要r，要w
target_h5f = h5py.File(save_target_path, 'w') 
# 2. 输入输出数据
input_path = os.path.join(data_path, 'rain/') # \\ 等同于 / Jack 4.7 Great!
target_path = os.path.join(data_path, 'norain/')
train_num = 0 # 这个数据代表了一张图像切割得到的patch总数

# ======================================================== RGB 变换处理 =================================
# 2.5 BLS预处理
import numpy as np
for image in range(image_num):
    input_file = "norain-%dx2.png" % (image+ 1)  
    target_file = "norain-%d.png" % (image + 1)     
    #==========Input处理===================
    input_img = cv2.imread(os.path.join(input_path,input_file)) # 读入第i张图片X
    if train_num==0:
        train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
        test_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten()
        train_all=train_ZeroData # train 包含了输入的有雨图像的展平集合
        test_all=test_ZeroData # test 包含了输入的无雨图像的3通道展平集合  
    bi, gi, ri = cv2.split(input_img) # 2022.08.18 参考PreNet
    train_all=np.vstack((train_all,ri.flatten())) # 对有雨图像3通道的 展平不断进行堆叠
    train_all=np.vstack((train_all,gi.flatten()))
    train_all=np.vstack((train_all,bi.flatten()))
    #==========Target处理===================
    target_img = cv2.imread(os.path.join(target_path,target_file)) # 读入第i张图片Y
    bt, gt, rt = cv2.split(target_img) # 2022.08.18 参考PreNet
    test_all=np.vstack((test_all,rt.flatten())) # 对无雨图像的R通道展平不断进行堆叠
    train_num = train_num+1
    print("The train_num%d" % train_num)
#==========BLS训练================
# np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/train_all.npy",train_all)
# np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/test_all.npy",test_all)
train_all = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/train_all.npy") # train_all shape 
train_all = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/test_all.npy")
WeightTop_Best,WeightEnhan_Best,WFSparse_Best,meanOfEachWindow_Best,distOfMaxAndMin_Best=BLS_Train(train_all[1:,:],test_all[1:,:]) # 训练过程 BLS_option 为train
#==========保存训练结果===========其他文件中已做预处理
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WeightTop_Best.npy",WeightTop_Best)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WeightEnhan_Best.npy",WeightEnhan_Best)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WFSparse_Best.npy",WFSparse_Best)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/meanOfEachWindow_Best.npy",meanOfEachWindow_Best)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/distOfMaxAndMin_Best.npy",distOfMaxAndMin_Best)

# ======================================================== HSI 变换处理 =================================
# 2.5 BLS预处理
import numpy as np
for image in range(image_num):
    input_file = "norain-%dx2.png" % (image+ 1)  
    target_file = "norain-%d.png" % (image + 1)     
    #==========Input处理===================
    input_img = cv2.imread(os.path.join(input_path,input_file)) # 读入第i张图片X
    if train_num==0:
        train_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten() 
        test_ZeroData = np.zeros((input_img.shape[0],input_img.shape[1])).flatten()
        train_all=train_ZeroData # train 包含了输入的有雨图像的展平集合
        test_all=test_ZeroData # test 包含了输入的无雨图像的3通道展平集合  
    bi, gi, ri = cv2.split(input_img) # 2022.08.18 参考PreNet
    train_all=np.vstack((train_all,ri.flatten())) # 对有雨图像3通道的 展平不断进行堆叠
    train_all=np.vstack((train_all,gi.flatten()))
    train_all=np.vstack((train_all,bi.flatten()))
    #==========Target处理===================
    target_img = cv2.imread(os.path.join(target_path,target_file)) # 读入第i张图片Y
    target_img_HSI = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
    ht, st, it = cv2.split(target_img_HSI) # 2022.08.18 参考PreNet
    test_all=np.vstack((test_all,it.flatten())) # 对无雨图像的I通道展平不断进行堆叠
    train_num = train_num+1
    print("The train_num%d" % train_num)
#==========BLS训练================
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/train_all_HSI.npy",train_all)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/test_all_HSI.npy",test_all)
train_all_HSI = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/train_all_HSI.npy")
test_all_HSI = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/test_all_HSI.npy")
WeightTop_Best_HSI,WeightEnhan_Best_HSI,WFSparse_Best_HSI,meanOfEachWindow_Best_HSI,distOfMaxAndMin_Best_HSI=BLS_Train(train_all_HSI[1:,:],test_all_HSI[1:,:]) # 训练过程 BLS_option 为train
#==========保存训练结果=========== 其他文件中已做预处理
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WeightTop_Best.npy",WeightTop_Best_HSI)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WeightEnhan_Best.npy",WeightEnhan_Best_HSI)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/WFSparse_Best.npy",WFSparse_Best_HSI)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/meanOfEachWindow_Best.npy",meanOfEachWindow_Best_HSI)
np.save("/home/huangjh/Project/DeRain/JackData/TrainLog/distOfMaxAndMin_Best.npy",distOfMaxAndMin_Best_HSI)


