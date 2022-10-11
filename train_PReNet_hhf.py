import os
import argparse # (1)conda create -n JackPRENET=3.6
import numpy as np # (2)pip install numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch  #　注意3.28官网不给10.2的源码下载了，哈哈哈vision tPorchau，还好俺们有清华源，将cu113 改为 cu102 即可，因为我电脑是10.2的cuda，装113会使得无法使用gpu！
# pip3 install torch torchdio --extra-index-url https://download.pytorch.org/whl/cu102　-i https://pypi.tuna.tsinghua.edu.cn/simple BUG emmm,好像无法写入,有个小BUG, 那就自己换conda命令吧
# (3)conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 成功~ (通道没加清华源就 末尾加 -c pytorch 用官网的)
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter # (4)pip install tensorboardX
from DerainDataset_hhf import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *

parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=48, help="Training batch size") # batch_size 设置为 18，游戏本顶不住，这里改为2
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs") # 为了快速得到结果，减少epoch 100 变为 5
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="/home/huangjiehui/Project/DerainNet/Logs", help='path to save models and log files')  # 3.28 BUG 防止add path时，斜杠不一致
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
# parser.add_argument("--data_path",type=str, default="/home/huangjh/Data/ProjectData/RainData/TraditionalData/train/Rain12600/",help='path to training data')  # 3.28 BUG 防止add path时，斜杠不一致
parser.add_argument("--data_path",type=str, default="/home/huangjh/Data/ProjectData/RainData/X2Data/Train/Light/",help='path to training data')  # 4.25 训练100H目录下的雨条纹图像
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args(args=[])

# import sys
# Logsave_path = sys.argv[1]


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    # model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    model = AMCC2(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    # print_network(model)
    
    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda() # BUG CUDA driver initialization failed, you might not have a CUDA gpu.
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)   

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0 
    # Note=open('/home/huangjh/Project/PDAnalysis/Result/'+Logsave_path,'a')
    # Note.truncate(0) # 初始化txt
    sum = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"]) # learning rate 0.001000
        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            # sum += batch_PSNR(input_train,target_train,1.0)
            # print(sum/(i+1))
            # continue
            model.train()  # nn.Module.train的继承
            model.zero_grad()
            optimizer.zero_grad()
            
            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            
            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train) # target_train.size()---torch.Size([2, 3, 100, 100]); out_train.size()---torch.Size([2, 3, 100, 100])
            loss = -pixel_metric                            

            loss.backward()
            optimizer.step() # BUG 2022.07.22 将step放在每个epoch训练完以后
            
            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.) # 将输入input张量每个元素的夹紧到区间 [min,max][0,1]，并返回结果到一个新张量
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            #====记录数据
            # Note=open('/home/huangjh/Project/PDAnalysis/Result/'+Logsave_path,'a')
            # Note.write("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  # (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 1 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                ## epoch training end
                if epoch % opt.save_freq == 0:
                    torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                    model.eval()
                    out_train, _ = model(input_train)
                    out_train = torch.clamp(out_train, 0., 1.)
                    im_target = utils.make_grid(target_train.data, nrow=8, normalize=False, scale_each=True)
                    im_input = utils.make_grid(input_train.data, nrow=8, normalize=False, scale_each=True)
                    im_derain = utils.make_grid(out_train.data, nrow=8, normalize=False, scale_each=True)
                    writer.add_image('clean image', im_target, epoch+1)
                    writer.add_image('rainy image', im_input, epoch+1)
                    writer.add_image('deraining image', im_derain, epoch+1)
                    # save model
                    torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
            step += 1


if __name__ == "__main__":
    main()

