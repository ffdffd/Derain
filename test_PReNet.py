import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time

parser = argparse.ArgumentParser(description="PReNet_Test")
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

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = AMCC2(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            y_origin = cv2.imread(img_path)
            
            if len(y_origin[:,0,0])%2 == 0: # 第二层KPN特性, 奇数大小的图像需要调整一下
                y = y_origin 
            else:
                y = y_origin[:-1,:-1,:] # ([1, 3, 321, 481]) 去除最后一个
            
            # y = y_origin
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                out, _ = model(y)
                
                #======================================查看测试损失 Start============================
                # input image
                gt_name = "no" + img_name
                gt_path = os.path.join(opt.data_path.replace("rainy/",""),gt_name )

                gt = cv2.imread(gt_path)
                b, g, r = cv2.split(gt)
                gt = cv2.merge([r, g, b])
                #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
                gt = normalize(np.float32(gt))
                gt = np.expand_dims(gt.transpose(2, 0, 1), 0)
                gt = Variable(torch.Tensor(gt))
                from SSIM import SSIM
                criterion = SSIM()
                if len(gt[0,0,0,:])%2==0:
                    pixel_metric = criterion(out, gt)
                    psnr_test = batch_PSNR(out,  gt, 1.)
                else:
                    pixel_metric = criterion(out, gt[:,:,:-1,:-1]) # KPN特性
                    psnr_test = batch_PSNR(out,  gt[:,:,:-1,:-1], 1.)
                print("[Test SSIM is] %f, [Test PSNR is] %d ==================" % (pixel_metric, psnr_test))
                #======================================查看测试损失 End============================
                
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

