class AMCC2_o(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True, factor=10e-3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(AMCC2_o, self).__init__()
        #============================================ Layer 2 ===================================
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False) # ([18, 3, 96, 96]) -> ([18, 64, 96, 96])
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积 # 基于空间注意力的上采样
        self.conv6_spatial = Basic(512+512, 512, channel_att=channel_att, spatial_att=TRUE)
        self.conv7_spatial = Basic(256+512, 256, channel_att=channel_att, spatial_att=TRUE)
        self.conv8_spatial = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=TRUE)
        # 6~8层要先上采样再卷积 # 基于通道注意力的上采样
        self.conv6_channel = Basic(512+512, 512, channel_att=TRUE, spatial_att=spatial_att)
        self.conv7_channel = Basic(256+512, 256, channel_att=TRUE, spatial_att=spatial_att)
        self.conv8_channel = Basic(256+128, out_channel, channel_att=TRUE, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.conv_final = nn.Conv2d(in_channels=75, out_channels=3, kernel_size=3, stride=1, padding=1) # 抛弃 KernelConv 将 75 channel 变为 12 channel
        #============================================ Layer 3 ===================================
        self.factor = factor # 定义循环控制变量
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(12, 32, 3, 1, 1), # pre1 3channels===pre2 3channels====x 3channels====input 3channels   3*4=12 channels
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        #============================================ Layer 2 ===================================
        conv1 = self.conv1(input) # ([18, 3, 96, 96]) -> ([18, 64, 96, 96])
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=3, stride=2,padding=1)) # ([18, 64, 96, 96]) -> ([18, 64, 48, 48]) -> ([18, 128, 48, 48]) kernel_size=2 变为 3 同时加入 padding=1
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=3, stride=2,padding=1)) # ([18, 128, 48, 48]) -> ([18, 64, 24, 24]) -> ([18, 256, 24, 24])
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=3, stride=2,padding=1)) # ([18, 256, 24, 24]) -> ([18, 64, 12, 12]) -> ([18, 512, 12, 12])
        conv5 = self.conv5(F.avg_pool2d(conv4,  kernel_size=3, stride=2,padding=1)) # ([18, 512, 12, 12]) -> ([18, 64, 6, 6]) -> ([18, 512, 6, 6])
        # 开始上采样  同时要进行skip connection
        for i in range(2):
            if i==0: # 基于空间注意力的上采样
                conv6 = self.conv6_spatial(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1)) # self.upMode为bilinear ([18, 512, 12, 12])
                conv7 = self.conv7_spatial(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1)) # ([18, 256, 24, 24])
                #print(conv7.size())
                conv8 = self.conv8_spatial(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1)) # ([18, 75, 48, 48])
                # return channel K*K*N
                core1 = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode)) # 恢复到原来的图像大小 ([18, 75, 96, 96])
                pred1 = self.conv_final(core1) 
            elif i==1:  # 基于通道注意力的上采样
                conv6 = self.conv6_channel(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1)) # self.upMode为bilinear
                conv7 = self.conv7_channel(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
                #print(conv7.size())
                conv8 = self.conv8_channel(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
                # return channel K*K*N
                core2 = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode)) # 恢复到原来的图像大小
                pred2 = self.conv_final(core2) 
        #============================================ Layer 3 ===================================
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        input1 = torch.cat((input, pred1), 1) # 将input和core1进行拼接 ([18, 3, 96, 96]) 
        input2 = torch.cat((input1, pred2), 1) # 将input1和core2进行拼接
        for i in range(self.iteration):  #　以下逐级扩张
            if i==0: # 第一次进行扩张
                x = torch.cat((input2, x), 1)
                x = self.conv0(x)
                resx = x
                x = F.relu(self.res_conv1(x) + resx)
                resx = x
                x = F.relu(self.res_conv2(x) + resx)
                resx = x
                x = F.relu(self.res_conv3(x) + resx)
                resx = x
                x = F.relu(self.res_conv4(x) + resx)
                resx = x
                x = F.relu(self.res_conv5(x) + resx)
                x = self.conv(x)
                x = x + input # 残差相加，不影响channels数量
                x_list.append(x)   
                derain=x     
            elif rain_judge(derain,rain=input, factor=self.factor)=="NotClear": # 不同于PreNet的是，AMCC针对扩张次数进行了限定
                x = torch.cat((input2, x), 1)
                x = self.conv0(x)
                resx = x
                x = F.relu(self.res_conv1(x) + resx)
                resx = x
                x = F.relu(self.res_conv2(x) + resx)
                resx = x
                x = F.relu(self.res_conv3(x) + resx)
                resx = x
                x = F.relu(self.res_conv4(x) + resx)
                resx = x
                x = F.relu(self.res_conv5(x) + resx)
                x = self.conv(x)
                x = x + input
                x_list.append(x)
                derain=x

        return x, x_list