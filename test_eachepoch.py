from unittest.mock import NonCallableMagicMock
import cv2
import os
import argparse
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time
import tqdm
from patchify import patchify
import torchvision.utils as utils
parser = argparse.ArgumentParser(description="PReNet_Test1")
# parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
# parser.add_argument("--logdir", type=str, default="logs/Rain100H/PReNet6/", help='path to model and log files')  # 此处仅考虑rain100H的PReNet6和rainy
parser.add_argument("--logdir", type=str, default="/home/huangjh/Project/DeRain/JackCode/PreNet_Demo_copy/logs/PreNet_test/", help='path to model and log files')  # 使用自己训练得到的模型
# parser.add_argument("--save_path", type=str, default="results/results_PReNet/", help='path to save results')
parser.add_argument("--data_path", type=str, default="/home/huangjh/Data/ProjectData/RainData/TraditionalData/test/Rain100L/rainy/", help='path to training data')
# parser.add_argument("--data_path", type=str, default="/home/huangjh/Data/ProjectData/RainData/TraditionalData/test/test12/rainy/", help='path to training data')
# parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet", help='path to save results')
parser.add_argument("--save_path", type=str, default="/home/huangjh/Project/DeRain/JackCode/PreNet_Demo_copy/logs/Result/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()
import torch.utils.data as udata


from SSIM import SSIM
def progress(y_origin):
    #  rbg 255 add_channel 32
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    
    high = y.shape[2]//32
    wight = y.shape[3]//32
    y = y[:,:, 0:high*32, 0:wight*32]
    y = torch.Tensor(y)
    return y

#  I-HAZE
I_HAZE_data_path = '/home/huangjiehui/Project/DerainNet/JackData/I-HAZE/hazy'

#  O-HAZE
O_HAZE_data_path = '/data1/hjh/ProjectData/Defogging/O-HAZE/hazy'

#  SOTS-Ondoor
SOTS_Ondoor_data_path = '/data1/hjh/ProjectData/Defogging/SOTS/Outdoor/hazy'
SOTS_O_log_path = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/Train_OTS2'

#  SOTS-Indoor
SOTS_Indoor_data_path = '/data1/hjh/ProjectData/Defogging/SOTS/Indoor/hazy'

Rain_DID_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-test'




class Dataset_Rain200(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_Rain200, self).__init__()
        self.data_path = data_path
        if ('200' in self.data_path):
            self.target_names =  data_path+'/norain/'
            self.input_names =  data_path+'/rain/'
        elif  ('100' in self.data_path):
            self.target_names =  data_path + '/'
            self.input_names =  data_path+'/rainy/'
    def __len__(self):
        return len(self.input_names)
    def __getitem__(self, index):
        if ('200' in self.data_path):
            target_names = self.target_names + f'norain-{index+1}.png'
            input_names =  self.input_names + f'norain-{index+1}x2.png'
        elif  ('100' in self.data_path):
            target_names = self.target_names + 'norain-%03d.png'%(index+1)
            input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt
#  ITS
# ITS_data_path = '/home/huangjiehui/Project/DerainNet/JackData/ITS/train'
# log_path = '/home/huangjiehui/Project/DerainNet/Logs'



class Dataset(udata.Dataset):
    def __init__(self, data_path=SOTS_Ondoor_data_path):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.img_names =  os.listdir(data_path)
        self.num = 6    # 把一张一张大图分为num份
    def __len__(self):
        if ('O-HAZE' in self.data_path or 'I-HAZE'in self.data_path) :
            return len(self.img_names)*self.num
        return len(self.img_names)
    def __getitem__(self, index):
        # target
        # GT path with target data
        if 'O-HAZE' in self.data_path or 'I-HAZE'in self.data_path :
            # !!!!!!RGB
            y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index//self.num]))
            gt_path = os.path.join(self.data_path, self.img_names[index//self.num]).replace("hazy",'GT')   # #  I-HAZE O-HAZE
            gt = cv2.imread(gt_path)
            gt= patchify(gt.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
            y= patchify(y_origin.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
            # gt,y = progress(gt),progress(y)
            l = gt.shape[0]//(self.num-1)
            return y[index%self.num*l : (index%self.num+1)*l ,:,:,:], gt[index%self.num*l : (index%self.num+1)*l ,:,:,:]
        
        elif ('SOTS/O' in self.data_path or 'SOTS/I'in self.data_path) :
            y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index]))
            gt_path = os.path.join(self.data_path.replace('hazy','nohazy'), self.img_names[index].split('_')[0]+'.png') #  SOTS-Ondoor SOTS-Indoor 
            # gt = progress(cv2.imread(gt_path))
            y = progress(y_origin)
            if 'SOTS/Indoor' in self.data_path:
                gt = progress(cv2.imread(gt_path))[:,:,10:458,10:618]  # #  SOTS-Indoor
            else:
                gt = progress(cv2.imread(gt_path))
            return y, gt

class Dataset_DID_800(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_DID_800, self).__init__()
        self.data_path = data_path
        ls1 = os.listdir(data_path)
        ls1 = [data_path+'/'+i for i in ls1]
        self.keys = ls1
    def __len__(self):
        if 'DID' in self.data_path:
            return 1000
        else:
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
        return torch.Tensor(input), torch.Tensor(target)


from torch.utils.data import DataLoader
def test(model,model_path,datapath,name_index):
    model = model.cuda()
    i = 1
    while os.path.exists(os.path.join(model_path, f'net_epoch{i}.pth')):
        model.load_state_dict(torch.load(os.path.join(model_path, f'net_epoch{i}.pth')))
        print('load_model from ' + os.path.join(model_path, f'net_epoch{i}.pth'))
        model.eval()
        psnr_test ,pixel_metric,count,psnr_max,ssim_max = 0,0,0,0,0
        dataset = datapath
        with torch.no_grad(): 
            loader_train = DataLoader(dataset=dataset, num_workers=16, batch_size=1, shuffle=False)
            if opt.use_GPU:
                torch.cuda.synchronize()
                
            for out, gt in tqdm.tqdm(loader_train):
                out_o = out.cuda().type(torch.float32)
                gt = torch.squeeze(gt.cuda(),0).type(torch.float32)
                if (gt.shape[0] == 0 ):
                    continue
                # torch.FloatTensor()
                # try:
                # out, _ = model(torch.squeeze(gt,0))
                out, _ = model(torch.squeeze(out_o,0))
                
                out = torch.clamp(out, 0., 1.)
                criterion = SSIM()
                loss = criterion(out, gt) * out.shape[0]
                pixel_metric += loss
                psnr_cur = batch_PSNR(out,  gt, 1.) * out.shape[0]
                psnr_test += psnr_cur
                if psnr_cur >= psnr_max:
                    psnr_max = psnr_cur
                if loss >= ssim_max:
                    ssim_max = loss
                count += out.shape[0]
                print("[Test SSIM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================" % (pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))
                if 1: # 输出图像
                    # x = utils.make_grid(torch.cat((out,gt,torch.squeeze(out_o,0))), nrow=8, normalize=False, scale_each=True)
                    x = utils.make_grid(torch.cat((out,gt,torch.squeeze(out_o,0))))
                    x = np.uint8(255 * x.cpu().numpy().squeeze())
                    r, g, b = cv2.split(x.transpose(1, 2, 0))
                    cv2.imwrite(f'/home/huangjiehui/Project/DerainNet/JackCode/Derain/Result/{count}.jpg',cv2.merge([b ,g, r]))
            # except:
            #     pass
            # brea
            psnst_average = psnr_test / count
            pixel_metric_average = pixel_metric / count
            Note=open('/home/huangjiehui/Project/DerainNet/JackCode/Derain/Result_each_e'+'/log.txt','a')
            Note.write("[epoch %d] , pixel_metric: %.4f, PSNR: %.4f" %
                (i, pixel_metric_average.item(), psnst_average.item()))
            Note.write('\n')
            i+=1

        
    return psnst_average,psnr_max,pixel_metric_average,ssim_max

if __name__ == "__main__":
    
    # 模型地址
    I_HAZE_log_path = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/Train_ITS_2'
    O_HAZE_log_path = '/home/huangjiehui/Project/DerainNet/Train_OTS2'
    SOTS_I_log_path = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/Train_ITS_2'
    DID_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-test'
    data_path_800 = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain800/Rain800/rain800_test'
   
    path_200L = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/200L'
    path_X2_H_patch  = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/X2_H_patch'
    X2path = "/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/X2"
    path_X2_H = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/X2_H'
    path_JN_14000 = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/JN_14000'
    path_20H = "/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/X2_H"
    path_800 = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/_800'
    path_did = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/DID'
    path_did_patch = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/DID_patch_192_L'
    path_14000_patch = '/home/huangjiehui/Project/DerainNet/JackCode/Derain/Logs/AM2_14000'
    
    #  数据地址
    Rain_200H_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200H/Rain200H/test'
    Rain_200L_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200L/Rain200L/test'
    Rain_100H_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain100H'
    Rain_100L_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain100L'
    # model = AMCC2_o(use_GPU=True)
    model = AMCC2(use_GPU=True)
    
    res = []
    count = 0
    net = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain'
    
    para = X2path
    res += test(model,para,Dataset_Rain200(Rain_100L_data_path),count)
    # res += test(model,para,Dataset_DID_800(DID_data_path),count)
    # # quit()
    # count+=1
    # print(res)
    # res += test(model,para,Dataset_Rain200(Rain_200H_data_path),count)
    # count+=1
    # print(res)
    # res += test(model,para,Dataset_DID_800(data_path_800),count)
    # count+=1
    # print(res)
    # res += test(model,para,Dataset_Rain200(Rain_100L_data_path),count)
    # count+=1
    # print(res)
    # res += test(model,para,Dataset_Rain200(Rain_100H_data_path),count)
    # count+=1
    # print(res)
    # res += test(model,para,Dataset_DID_800(DID_data_path),count)
    print(res)
    
    
    
    
    
        

