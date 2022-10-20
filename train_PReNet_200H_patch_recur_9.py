import os
import argparse # (1)conda create -n JackPRENET=3.6
import numpy as np # (2)pip install numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=5, help="Training batch size") # batch_size 设置为 18，游戏本顶不住，这里改为2
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs") # 为了快速得到结果，减少epoch 100 变为 5
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="/home/huangjiehui/Project/DerainNet/Logs/200H_patch_re9", help='path to save models and log files')  # 3.28 BUG 防止add path时，斜杠不一致
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--Logsave_path",type=str, default="log1.txt",help='path to training data')  # 3.28 BUG 防止add path时，斜杠不一致
# parser.add_argument("--data_path",type=str, default="/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain14000/Rain12600",help='path to training data')  # 4.25 训练100H目录下的雨条纹图像
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=9, help='number of recursive stages')
opt = parser.parse_args(args=[])

import torch  #　注意3.28官网不给10.2的源码下载了，哈哈哈vision tPorchau，还好俺们有清华源，将cu113 改为 cu102 即可，因为我电脑是10.2的cuda，装113会使得无法使用gpu！
# pip3 install torch torchdio --extra-index-url https://download.pytorch.org/whl/cu102　-i https://pypi.tuna.tsinghua.edu.cn/simple BUG emmm,好像无法写入,有个小BUG, 那就自己换conda命令吧
# (3)conda install pytorch torch vision torchaudio cudatoolkit=10.2 -c pytorch 成功~ (通道没加清华源就 末尾加 -c pytorch 用官网的)
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
from tqdm import tqdm
import torch.distributed as dist
  
device = torch.device('cuda')

# import sys


if  torch.cuda.device_count()>1:
    # Logsave_path = sys.argv[1]
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    ## Initialization
    torch.distributed.init_process_group(backend="nccl" )
    torch.cuda.set_device(rank)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
def collate_fn(batch):
       return torch.cat([torch.tensor(i[0]) for i in batch],0), torch.cat([torch.tensor(i[1]) for i in batch],0)
def main():

    print('Loading dataset ...\n')
    
    dataset_train = Dataset_200H_patch()
    model = AMCC2(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()
    
    # loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batch_size, shuffle=True)
    if  torch.cuda.device_count()>1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = (train_sampler is None), sampler=train_sampler, pin_memory=True,collate_fn = collate_fn)
    else:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle = True,  pin_memory=True,collate_fn = collate_fn)
        
    print("# of training samples: %d\n" % int(len(dataset_train)))
    criterion = SSIM()

    # Move to GPU
    # if opt.use_gpu:
    # model = model.cuda() # BUG CUDA driver initialization failed, you might not have a CUDA gpu.
    # criterion.cuda()
    if  torch.cuda.device_count()>1:
        device_ids = [0, 1]
        # model = torch.nn.DataParallel(model,device_ids = device_ids)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,find_unused_parameters=True).cuda()
    # record training
    writer = SummaryWriter(opt.save_path)   

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        if  torch.cuda.device_count()>1:
            model.module.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
        else:
            model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
            
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates
    # start training
    step = 0 
    # Note=open('/home/huangjh/Project/PDAnalysis/Result/'+Logsave_path,'a')
    # Note.truncate(0) # 初始化tx
    sum = 0
    
    for epoch in range(initial_epoch, opt.epochs):
        if  torch.cuda.device_count()>1:
            train_sampler.set_epoch(epoch)   
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"]) # learning rate 0.001000
        ## epoch training start
        for i, (input_train, target_train) in enumerate(tqdm(loader_train)):
            # sum += batch_PSNR(input_train,target_train,1.0)
            # print(sum/(i+1))
            # continue
            input_train = input_train.reshape(-1,3,96,96).type(torch.float32)
            target_train = target_train.reshape(-1,3,96,96).type(torch.float32)
            model.train()  # nn.Module.train的继承
            model.zero_grad()
            optimizer.zero_grad()
            
            count, psnr_all= 0,0
            # input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            # with autocast():
            out_train, times = model(input_train)
            pixel_metric = criterion(target_train, out_train) # target_train.size()---torch.Size([2, 3, 100, 100]); out_train.size()---torch.Size([2, 3, 100, 100])
            loss = -pixel_metric  
            # loss = -batch_PSNR(out_train, target_train, 1.) 
            if 0:
                x = utils.make_grid(torch.cat((target_train.detach(),input_train.detach())), nrow=8, normalize=False, scale_each=True)
                x = np.uint8(255 * x.cpu().numpy().squeeze())
                r, g, b = cv2.split(x.transpose(1, 2, 0))
                cv2.imwrite('/home/huangjiehui/Project/DerainNet/JackCode/Derain/1.jpg',cv2.merge([b ,g, r]))
                print("1111111")
            if 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else :
                loss.backward() 
                if  torch.cuda.device_count()>1:
                    dist.barrier()
                optimizer.step() # BUG 2022.07.22 将step放在每个epoch训练完以后
            
            # training curve
            # model.eval()
            # out_train, _ = model(input_train)
            # out_train = torch.clamp(out_train, 0., 1.) # 将输入input张量每个元素的夹紧到区间 [min,max][0,1]，并返回结果到一个新张量
            psnr_all += batch_PSNR(out_train, target_train, 1.)* out_train.shape[0]
            count += out_train.shape[0]
            step += 1
            if step % 1 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_all/count, step)
                Note=open(opt.save_path+'/log.txt','a')
                Note.write("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
              (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_all/count))
                Note.write('\n')
        
        #====记录数据
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f, times:%d" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_all/count,times))
            

        if  torch.cuda.device_count()>1:
            if epoch % opt.save_freq == 0 and rank == 0:
                torch.save(model.module.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                torch.save(model.module.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        else:
            if epoch % opt.save_freq == 0:
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
                     
                        
            # from test_PReNet import test
            # psnr_test_average,pixel_metric_average = test(model)
            # # 可视化
            # # out_train, _ = model(input_train)
            # # out_train = torch.clamp(out_train, 0., 1.)
            im_target = utils.make_grid(target_train.data, nrow=8, normalize=False, scale_each=True)
            # # im_input = utils.make_grid(input_train.data, nrow=8, normalize=False, scale_each=True)
            # # im_derain = utils.make_grid(out_train.data, nrow=8, normalize=False, scale_each=True)
            # # writer.add_image('clean image', im_target, epoch+1)
            # # writer.add_image('rainy image', im_input, epoch+1)
            # # writer.add_image('deraining image', im_derain, epoch+1)

            # print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
            #   (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric_average, psnr_test_average))
            # # save model

if __name__ == "__main__":
    main()

