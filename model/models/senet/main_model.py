#coding=utf-8
from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from senet import senet154


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream3 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, mid_channels, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(in_channels, mid_channels*2, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y
'''

空间上的attention 模块
'''
class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True) # 由hwc 变为 hw1
        # 3-by-3 conv
        h = x.size(2)
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (h,h), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        return x  
        ## 返回的是h*w*1 的 soft map

'''
通道上的attention 模块
'''
class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]) #输出是1*1*c
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        # 返回的是 1*1*c 的map

'''
空间和通道上的attention 融合
就是空间和通道上的attention做一个矩阵乘法
'''
class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """
    
    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y
        # 返回的是 hwc

'''
输出的是STN 需要的theta
'''
class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""
    
    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4*2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        # 初始化 参数
        # if x_t = 0  the performance is very low
        self.fc.bias.data.copy_(torch.tensor([0.3, -0.3, 0.3, 0.3, -0.3, 0.3, -0.3, -0.3], dtype=torch.float))

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta
         #  返回的是 2T  T为区域数量。 因为尺度会固定。 所以只要学位移的值


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""
    
    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta



class MODEL(nn.Module):
    '''
    the mian model of cvper2020
    '''
    def __init__(self, num_classes, senet154_weight, nchannels=[256,512,1024,2048], multi_scale = False ,learn_region=True, use_gpu=True):
        super(MODEL,self).__init__()
        self.learn_region=learn_region
        self.use_gpu = use_gpu
        self.conv = ConvBlock(3, 32, 3, s=2, p=1)
        self.senet154_weight = senet154_weight
        self.multi_scale = multi_scale
        self.num_classes = num_classes

        # construct SEnet154 
        senet154_ = senet154(num_classes=1000, pretrained=None)
        senet154_.load_state_dict(torch.load(self.senet154_weight))


        self.extract_feature = senet154_.layer0
        #global backbone
        self.global_layer1 = senet154_.layer1
        self.global_layer2 = senet154_.layer2
        self.global_layer3 = senet154_.layer3
        self.global_layer4 = senet154_.layer4


        self.classifier_global =nn.Sequential(
                                nn.Linear(2048*2, 2048), # 将4个区域 融合成一个 需要加上batchnorma1d, 和 relu
                                nn.BatchNorm1d(2048),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(2048, num_classes),
                                )
        if self.multi_scale:
            self.global_fc = nn.Sequential(
                                nn.Linear(2048+512+1024, 2048), # 将4个区域 融合成一个 需要加上batchnorma1d, 和 relu
                                nn.BatchNorm1d(2048),
                                nn.ReLU(),
                                )

            self.global_out = nn.Linear(2048,num_classes)  # global 分类
        else:

            self.global_out = nn.Linear(2048,num_classes)  # global 分类
        self.ha2  = HarmAttn(nchannels[1])
        self.ha3  = HarmAttn(nchannels[2])
        self.ha4  = HarmAttn(nchannels[3])
        

        self.dropout = nn.Dropout(0.2)  #  分类层之前使用dropout

        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(nchannels[1], nchannels[1])
            self.local_conv2 = InceptionB(nchannels[2], nchannels[2])
            self.local_conv3 = InceptionB(nchannels[3], nchannels[3])    
            self.local_fc = nn.Sequential(
                                nn.Linear(2048+512+1024, 2048), # 将4个区域 融合成一个 需要加上batchnorma1d, 和 relu
                                nn.BatchNorm1d(2048),
                                nn.ReLU(),
                                )
            self.classifier_local = nn.Linear(2048,num_classes)




    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        #  the s_w and s_h is fixed. 
        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float))
     

    def stn(self, x, theta):
        """Performs spatial transform
        
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,:,-1] = theta_i
        if self.use_gpu: theta = theta.cuda()
        return theta

    def forward(self, x):
        batch_size = x.size()[0]  # obtain the batch size
        x = self.extract_feature(x)  # output shape is 128 * 56 *56  senet154第0层layer 提取特征

        #  =================layer 1 ===============

        # global branch
        x1 = self.global_layer1(x)  # the output shape is 256*56*56
        
        #============layer 2================
        #global branch

        x2 = self.global_layer2(x1)  # x2 is 512*28*28
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
 
        if self.multi_scale:
            #  attention global layer1  avg pooling 
            x2_avg = F.adaptive_avg_pool2d(x2_out, (1, 1)).view(x2_out.size(0), -1)  #512 向量

        # local branch
        if self.learn_region:
            x2_local_list = []

            for region_idx in range(4):
                x2_theta_i = x2_theta[:,region_idx,:]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x2, x2_theta_i)  #256*56*26
                x2_trans_i = F.upsample(x2_trans_i, (56, 56), mode='bilinear', align_corners=True)
                x2_local_i = x2_trans_i 
                x2_local_i = self.local_conv1(x2_local_i) # 512*28*28
                x2_local_list.append(x2_local_i)


        #============layer 3================
        #global branch

        x3 = self.global_layer3(x2_out)  # x3 is 1024*14*14
        # print('layer3 output')
        # print(x3.size())
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        
        
        if self.multi_scale:
                #  attention global layer1  avg pooling 
            x3_avg = F.adaptive_avg_pool2d(x3_out, (1, 1)).view(x3_out.size(0), -1)  #1024 向量
        
        
        # local branch
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:,region_idx,:]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x3, x3_theta_i) #512*28*28
                x3_trans_i = F.upsample(x3_trans_i, (28, 28), mode='bilinear', align_corners=True)
                x3_local_i = x3_trans_i 
                x3_local_i = self.local_conv2(x3_local_i) # 1024*14*14
                x3_local_list.append(x3_local_i)

        #============layer 4================
        #global branch
        x4 = self.global_layer4(x3_out)  # 2048*7*7
        x4_attn, x4_theta = self.ha4(x4)
        x4_out = x4 * x4_attn
        

        # local branch
        if self.learn_region:
            x4_local_list = []
            for region_idx in range(4):
                x4_theta_i = x4_theta[:,region_idx,:]
                x4_theta_i = self.transform_theta(x4_theta_i, region_idx)
                x4_trans_i = self.stn(x4, x4_theta_i) #1024*14*14
                x4_trans_i = F.upsample(x4_trans_i, (14,14), mode='bilinear', align_corners=True)
                x4_local_i = x4_trans_i 
                x4_local_i = self.local_conv3(x4_local_i) # 2048*7*7
                x4_local_list.append(x4_local_i)
        # ============== Feature generation ==============
        # global branch
        x4_avg = F.avg_pool2d(x4_out, x4_out.size()[2:]).view(x4_out.size(0),  -1) #全局pooling 2048 之前已经relu过了
        
        if self.multi_scale:
            multi_scale_feature = torch.cat([x2_avg, x3_avg, x4_avg],1)
            global_fc = self.global_fc(multi_scale_feature)
            global_out = self.global_out(self.dropout(global_fc))

        else:
            global_out = self.global_out(x4_avg)  # 2048 -> num_classes

        if self.learn_region:
            x_local_list = []

            local_512 = torch.randn(batch_size, 4, 512).cuda()
            local_1024 = torch.randn(batch_size, 4, 1024).cuda()
            local_2048 = torch.randn(batch_size, 4, 2048).cuda()

            for region_idx in range(4):

                x2_local_i = x2_local_list[region_idx]
                x2_local_i = F.avg_pool2d(x2_local_i, x2_local_i.size()[2:]).view(x2_local_i.size(0), -1) #每个local 都全局pooling
                local_512[:,region_idx] = x2_local_i

                x3_local_i = x3_local_list[region_idx]
                x3_local_i = F.avg_pool2d(x3_local_i, x3_local_i.size()[2:]).view(x3_local_i.size(0), -1) #每个local 都全局pooling
                local_1024[:,region_idx] = x3_local_i


                x4_local_i = x4_local_list[region_idx]
                x4_local_i = F.avg_pool2d(x4_local_i, x4_local_i.size()[2:]).view(x4_local_i.size(0), -1) #每个local 都全局pooling
                local_2048[:,region_idx] = x4_local_i

            local_512_maxpooing = local_512.max(1)[0]
            local_1024_maxpooing = local_1024.max(1)[0]
            local_2048_maxpooing = local_2048.max(1)[0]
            local_concate = torch.cat([local_512_maxpooing, local_1024_maxpooing, local_2048_maxpooing], 1)
            local_fc = self.local_fc(local_concate)
            local_out = self.classifier_local(self.dropout(local_fc))

        if self.multi_scale:
            out = torch.cat([global_fc,local_fc],1)
        else:        
            out = torch.cat([x4_avg, local_512_maxpooing, local_1024_maxpooing, local_2048_maxpooing], 1) # global  和  local 一起做拼接 2048*2

        out = self.classifier_global(out)
        
        if self.learn_region:
            return out, global_out,local_out
        else:
            return global_out