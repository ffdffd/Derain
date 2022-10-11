import math
import torch
import re
import torch.nn as nn
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr BUG 改为如下
from skimage.metrics import peak_signal_noise_ratio as compare_psnr # (7)pip install scikit-image
import  os
import glob 
from torch.autograd import Variable

# ================================================JackAdd 2022.09.23==============================================
#=================== 用来评估测试性能的函数！
def psnr(pred, target):
    #print(pred.shape)
    #print(target.shape)
    mse = torch.mean( (pred - target) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def eval(model, epoch, eval_loader, criterion):
    model.eval()
    val_PSNR = 0
    loss = 0
    num_of_val_image = 0

    for j, (true_input, true_target) in enumerate(eval_loader):
        
        # To device
        # A is for input image, B is for target image
        true_input = true_input.cuda()
        true_target = true_target.cuda()

        # Forward propagation
        with torch.no_grad():
            fake_target = model(true_input)

        # Accumulate num of image and val_PSNR
        num_of_val_image += 1
        val_PSNR += psnr(fake_target[0], true_target)
        loss += criterion(fake_target[0], true_target)
    
    val_PSNR = val_PSNR/num_of_val_image
    loss = loss/num_of_val_image
    print('[epoch %d]: test_loss is %.4f, and test_PSNR is %.4f' % ((epoch+1), loss, val_PSNR))
    
    return loss, val_PSNR

# ================================================JackAdd 2022.09.23==============================================
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

#========================================Jack 循环控制模块加入 ============
def rain_judge(derain,rain,factor): # 当rain_streak_mean值为0说明图像除雨很好，无需再循环除雨，一般低于9即不错
    rain_streak = abs(derain-rain)
    rain_streak_mean = torch.mean(rain_streak.flatten(), dim=0, keepdim=False)
    if rain_streak_mean<factor: # data.shape (18, 3, 100, 100)
        result = "Clear"
    else:
        # print("========================Attention!========================") 
        result = "NotClear"
    return result
#============================================================ 以下函数专为BLS添加 ==============================================
def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
'''
参数压缩
'''
def shrinkage(a,b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
'''
参数稀疏化
'''
def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T,A)   
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m,n],dtype = 'double')
    ok = np.zeros([m,n],dtype = 'double')
    uk = np.zeros([m,n],dtype = 'double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1,A.T),b)
    for i in range(itrs):
        tempc = ok - uk
        ck =  L2 + np.dot(L1,tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
NumFea = 6
NumWin = 5
NumEnhan = 41
s = 0.8
C = 2**-30
import numpy as np
from sklearn import preprocessing
from numpy import random
import time
#=================== BLS模型进行前置处理   Jack 2022.09.03
def BLS_Train(train,target):
    # '''
    WeightTop_Best = WeightEnhan_Best=WFSparse_Best=meanOfEachWindow_Best=distOfMaxAndMin_Best= []
    test_ERR_Best = 999
    #===============train划分出x
    split_idx = int(len(train[1,:,:]) * 0.7) # train为R G B三色图，target为I通道(亮度)
    train_x = train.iloc[:split_idx]
    test_x = train.iloc[split_idx:]
    #===============target划分出y
    split_idy = int(len(target[1,:,:]) * 0.7)
    train_y = target.iloc[:split_idy] # 和 train_x 保持了一致的长度
    test_y = target.iloc[split_idy:]
    #==========================================借鉴BLS=============================
    for iter in range(len(train[1])):
        u = 0
        WF = list()
        for i in range(NumWin):
            random.seed(i+u)
            WeightFea=2*random.randn(train_x.shape[1]+1,NumFea)-1;
            WF.append(WeightFea)

        # random.seed(100)
        WeightEnhan=2*random.randn(NumWin*NumFea+1,NumEnhan)-1;
        time_start = time.time()
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])]);
        y = np.zeros([train_x.shape[0],NumWin*NumFea])
        WFSparse = list()
        distOfMaxAndMin = np.zeros(NumWin)
        meanOfEachWindow = np.zeros(NumWin)
        for i in range(NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)        
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse  = sparse_bls(A1,H1).T
            WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)
            meanOfEachWindow[i] = T1.mean()
            distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - meanOfEachWindow[i])/distOfMaxAndMin[i] 
            y[:,NumFea*i:NumFea*(i+1)] = T1

        H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
        T2 = H2.dot(WeightEnhan)
        T2 = tansig(T2);
        T3 = np.hstack([y,T2])
        WeightTop = pinv(T3,C).dot(train_y)

        Training_time = time.time()- time_start
        print('Training has been finished!');
        print('The Total Training Time is : ', round(Training_time,6), ' seconds' )
        NetoutTrain = T3.dot(WeightTop)

        RMSE = np.sqrt((NetoutTrain-train_y).T*(NetoutTrain-train_y)/train_y.shape[0])
        MAPE = sum(abs(NetoutTrain-train_y))/train_y.mean()/train_y.shape[0]
        train_ERR = RMSE
        train_MAPE = MAPE
        print('Training RMSE is : ',RMSE);
        print('Training MAPE is : ', MAPE)
        time_start = time.time()
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
        yy1=np.zeros([test_x.shape[0],NumWin*NumFea])
        for i in range(NumWin):
            WeightFeaSparse = WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1  = (TT1 - meanOfEachWindow[i])/distOfMaxAndMin[i]   
            yy1[:,NumFea*i:NumFea*(i+1)] = TT1

        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
        TT2 = tansig(HH2.dot( WeightEnhan));
        TT3 = np.hstack([yy1,TT2])
        NetoutTest = TT3.dot(WeightTop)
        RMSE = np.sqrt((NetoutTest-test_y).T*(NetoutTest-test_y)/test_y.shape[0]);
        MAPE = sum(abs(NetoutTest-test_y))/test_y.mean()/test_y.shape[0]
        test_ERR = RMSE
        test_MAPE = MAPE
        # Calculate the testing accuracy
        Testing_time = time.time() - time_start
        print('Testing has been finished!');
        print('The Total Testing Time is : ', round(Testing_time,6), ' seconds' );
        print('Testing RMSE is : ', RMSE) 
        print('Testing MAPE is : ', MAPE)
        if test_ERR<test_ERR_Best: # WeightTop_Best=WeightEnhan_Best=WFSparse_Best=meanOfEachWindow_Best=distOfMaxAndMin_Best
            test_ERR_Best=test_ERR # 将当前的 test_ERR_Best 更新
            WeightTop_Best=WeightTop # 将当前的 WeightTop_Best 更新
            WeightEnhan_Best=WeightEnhan # 将当前的 WeightEnhan_Best 更新
            WFSparse_Best=WFSparse # 将当前的 WFSparse_Best 更新
            meanOfEachWindow_Best=meanOfEachWindow # 将当前的 meanOfEachWindow_Best 更新
            distOfMaxAndMin_Best=distOfMaxAndMin # 将当前的 distOfMaxAndMin_Best 更新  
        iter=iter+1
        print("Iteration %d done!" % iter)        
    return WeightTop_Best,WeightEnhan_Best,WFSparse_Best,meanOfEachWindow_Best,distOfMaxAndMin_Best
    # '''

def BLS_Test(IMG,width,height):
    #===============train划分出x
    WeightTop_Best = np.load("/home/huangjiehui/Project/DerainNet/JackCode/Derain/para/WeightTop_Best.npy")
    WeightEnhan_Best = np.load("/home/huangjiehui/Project/DerainNet/JackCode/Derain/para/WeightEnhan_Best.npy")
    WFSparse_Best = np.load("/home/huangjiehui/Project/DerainNet/JackCode/Derain/para/WFSparse_Best.npy")
    meanOfEachWindow_Best = np.load("/home/huangjiehui/Project/DerainNet/JackCode/Derain/para/meanOfEachWindow_Best.npy")
    distOfMaxAndMin_Best = np.load("/home/huangjiehui/Project/DerainNet/JackCode/Derain/para/distOfMaxAndMin_Best.npy")
    # input = IMG.flatten() # 对输入的图像数据进行展平
    test_x = IMG.T # 整个train(输入的有雨图像)作为test_x，用来获得整个图像的映射
    #==========================================借鉴BLS=============================
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],NumWin*NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse_Best[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1  = (TT1 - meanOfEachWindow_Best[i])/distOfMaxAndMin_Best[i]   
        yy1[:,NumFea*i:NumFea*(i+1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
    TT2 = tansig(HH2.dot( WeightEnhan_Best));
    TT3 = np.hstack([yy1,TT2])
    NetoutTest = TT3.dot(WeightTop_Best) # WeightTop是非常关键的参数，即BLS网络训练过程学习到的特征权重
    NetoutTest = NetoutTest.reshape(width,height) # 对数据进行重整，变回原大小
    NetoutTest = np.array(NetoutTest,dtype='uint8')
    return NetoutTest 

def BLS_Test_HSI(IMG,width,height):
    #===============train划分出x
    WeightTop_Best = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/backup/WeightTop_Best_HSI.npy")
    WeightEnhan_Best = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/backup/WeightEnhan_Best_HSI.npy")
    WFSparse_Best = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/backup/WFSparse_Best_HSI.npy")
    meanOfEachWindow_Best = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/backup/meanOfEachWindow_Best_HSI.npy")
    distOfMaxAndMin_Best = np.load("/home/huangjh/Project/DeRain/JackData/TrainLog/backup/distOfMaxAndMin_Best_HSI.npy")
    # input = IMG.flatten() # 对输入的图像数据进行展平
    test_x = IMG.T # 整个train(输入的有雨图像)作为test_x，用来获得整个图像的映射
    #==========================================借鉴BLS=============================
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],NumWin*NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse_Best[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1  = (TT1 - meanOfEachWindow_Best[i])/distOfMaxAndMin_Best[i]   
        yy1[:,NumFea*i:NumFea*(i+1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
    TT2 = tansig(HH2.dot( WeightEnhan_Best));
    TT3 = np.hstack([yy1,TT2])
    NetoutTest = TT3.dot(WeightTop_Best) # WeightTop是非常关键的参数，即BLS网络训练过程学习到的特征权重
    NetoutTest = NetoutTest.reshape(width,height) # 对数据进行重整，变回原大小
    NetoutTest = np.array(NetoutTest,dtype='uint8')
    return NetoutTest 

