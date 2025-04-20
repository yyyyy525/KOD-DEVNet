import torch
import torch.nn as nn
import torch.nn.functional as F
from DVNet.lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#####
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#####------------------------------------
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SAM(nn.Module):
    def __init__(self, ch_in=32, reduction=16):
        super(SAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_h, x_l):
        #print('x_h shape, x_l shape,',x_h.shape, x_l.shape)
        b, c, _, _ = x_h.size()
        #print('self.avg_pool(x_h)',self.avg_pool(x_h).shape)
        y_h = self.avg_pool(x_h).view(b, c) # squeeze操作
        #print('***this is Y-h shape',y_h.shape)
        h_weight=self.fc_wight(y_h)
        #print('h_weight',h_weight.shape,h_weight) ##(batch,1)
        y_h = self.fc(y_h).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('y_h.expand_as(x_h)',y_h.expand_as(x_h).shape)
        x_fusion_h=x_h * y_h.expand_as(x_h)
        x_fusion_h=torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
##################----------------------------------
        b, c, _, _ = x_l.size()
        y_l = self.avg_pool(x_l).view(b, c) # squeeze操作
        l_weight = self.fc_wight(y_l)
        #print('l_weight',l_weight.shape,l_weight)
        y_l = self.fc(y_l).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('***this is y_l shape', y_l.shape)
        x_fusion_l=x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
#################-------------------------------
        #print('x_fusion_h shape, x_fusion_l shape,h_weight shape',x_fusion_h.shape,x_fusion_l.shape,h_weight.shape)
        x_fusion=x_fusion_h+x_fusion_l

        return x_fusion # 注意力作用每一个通道上

#####
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs

        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x

class Hitnet(nn.Module):
    def __init__(self, channel=32,n_feat=32,scale_unetfeats=32,kernel_size=3,reduction=4,bias=False,act=nn.PReLU()):
        super(Hitnet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        ## cod
        self.Translayer2_0_c = BasicConv2d(64, channel, 1)
        self.Translayer2_1_c = BasicConv2d(128, channel, 1)
        self.Translayer3_1_c = BasicConv2d(320, channel, 1)
        self.Translayer4_1_c = BasicConv2d(512, channel, 1)
        ## uni
        self.Translayer2_0_u = BasicConv2d(64, channel, 1)
        self.Translayer2_1_u = BasicConv2d(128, channel, 1)
        self.Translayer3_1_u = BasicConv2d(320, channel, 1)
        self.Translayer4_1_u = BasicConv2d(512, channel, 1)
        ## sod
        self.Translayer2_0_s = BasicConv2d(64, channel, 1)
        self.Translayer2_1_s = BasicConv2d(128, channel, 1)
        self.Translayer3_1_s = BasicConv2d(320, channel, 1)
        self.Translayer4_1_s = BasicConv2d(512, channel, 1)
        ## 多分类分支
        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM_c = SAM()
        self.SAM_u = SAM()
        self.SAM_s = SAM()
        self.SAM = SAM()

        self.out_SAM_c = nn.Conv2d(channel, 1, 1)
        self.out_SAM_u = nn.Conv2d(channel, 1, 1)
        self.out_SAM_s = nn.Conv2d(channel, 1, 1)
        self.out_CFM_c = nn.Conv2d(channel, 1, 1)
        self.out_CFM_u = nn.Conv2d(channel, 1, 1)
        self.out_CFM_s = nn.Conv2d(channel, 1, 1)
        self.out_SAM = nn.Conv2d(channel, 2, 1)
        self.out_CFM = nn.Conv2d(channel, 2, 1)

        ###############################
        self.decoder_level1_c = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1_c = nn.Sequential(*self.decoder_level1_c)

        self.decoder_level1_u = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1_u = nn.Sequential(*self.decoder_level1_u)

        self.decoder_level1_s = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1_s = nn.Sequential(*self.decoder_level1_s)

        self.decoder_level1 = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        ###############################

        self.upsample_4_c = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_4_u = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_4_s = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.compress_out_c = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)
        self.compress_out_u = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)
        self.compress_out_s = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)
        self.compress_out = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)

        ###############################

        self.decoder_level4_c = [CAB(n_feat,  kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level4_c = nn.Sequential(*self.decoder_level4_c)

        self.decoder_level4_u = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level4_u = nn.Sequential(*self.decoder_level4_u)

        self.decoder_level4_s = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level4_s = nn.Sequential(*self.decoder_level4_s)

        self.decoder_level4 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        ###############################

        self.upsample_c = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_u = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_s = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder_level3_c = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3_c = nn.Sequential(*self.decoder_level3_c)
        self.decoder_level3_u = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3_u = nn.Sequential(*self.decoder_level3_u)
        self.decoder_level3_s = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3_s = nn.Sequential(*self.decoder_level3_s)
        self.decoder_level3 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        ###############################

        self.compress_out2_c = BasicConv2d(2 * channel, channel, kernel_size=1)
        self.compress_out2_u = BasicConv2d(2 * channel, channel, kernel_size=1)
        self.compress_out2_s = BasicConv2d(2 * channel, channel, kernel_size=1)
        self.compress_out2 = BasicConv2d(2 * channel, channel, kernel_size=1)

        ###############################

        self.decoder_level2_c = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2_c = nn.Sequential(*self.decoder_level2_c)
        self.decoder_level2_u = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2_u = nn.Sequential(*self.decoder_level2_u)
        self.decoder_level2_s = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2_s = nn.Sequential(*self.decoder_level2_s)
        self.decoder_level2 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)

        ###############################

        self.conv4_c = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv4_u = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv4_s = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        ###############################

        self.down05_c = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down05_u = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down05_s = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

#############-------------------------------------------------
        ### CIM
        cim_feature_c = self.decoder_level1_c(x1)
        cim_feature_u = self.decoder_level1_u(x1)
        cim_feature_s = self.decoder_level1_s(x1)
        cim_feature = self.decoder_level1(x1)

        ####
        # CFM
        # cod
        x2_t = self.Translayer2_1_c(x2)
        x3_t = self.Translayer3_1_c(x3) 
        x4_t = self.Translayer4_1_c(x4)
        ## uni
        y2_t = self.Translayer2_1_u(x2)  
        y3_t = self.Translayer3_1_u(x3) 
        y4_t = self.Translayer4_1_u(x4)
        ## sod
        z2_t = self.Translayer2_1_s(x2) 
        z3_t = self.Translayer3_1_s(x3)  
        z4_t = self.Translayer4_1_s(x4)
        # 多分类语义分割分支
        n2_t = self.Translayer2_1(x2)
        n3_t = self.Translayer3_1(x3) 
        n4_t = self.Translayer4_1(x4)

####stage 1--------------------------------------------------
        stage_c_loss = list()
        stage_u_loss = list()
        stage_s_loss = list()
        stage_loss=list()
        cfm_feature_c = None
        cfm_feature_u = None
        cfm_feature_s = None
        cfm_feature=None
        ###### x cod ; y uni ; z sod
        for iter in range(4):
            if cfm_feature_c == None and cfm_feature_u == None and cfm_feature_s == None:
                x4_t = x4_t
                y4_t = y4_t
                z4_t = z4_t
                n4_t = n4_t
            else:
                x4_t = torch.cat((self.upsample_4_c(x4_t), cfm_feature_c), 1)
                x4_t = self.compress_out_c(x4_t)
                y4_t = torch.cat((self.upsample_4_u(y4_t), cfm_feature_u), 1)
                y4_t = self.compress_out_u(y4_t)
                z4_t = torch.cat((self.upsample_4_s(z4_t), cfm_feature_s), 1)
                z4_t = self.compress_out_s(z4_t)
                n4_t = torch.cat((self.upsample_4(n4_t), cfm_feature), 1)
                n4_t = self.compress_out(n4_t)

            x4_t_feed = self.decoder_level4_c(x4_t)  ######channel=32, width and height
            y4_t_feed = self.decoder_level4_u(y4_t)  ######channel=32, width and height
            z4_t_feed = self.decoder_level4_s(z4_t)  ######channel=32, width and height
            n4_t_feed = self.decoder_level4(n4_t)  ######channel=32, width and height

            # ############################################

            x3_t_feed = torch.cat((x3_t, self.upsample_c(x4_t_feed)), 1)
            x3_t_feed = self.decoder_level3_c(x3_t_feed)
            y3_t_feed = torch.cat((y3_t, self.upsample_u(y4_t_feed)), 1)
            y3_t_feed = self.decoder_level3_u(y3_t_feed)
            z3_t_feed = torch.cat((z3_t, self.upsample_s(z4_t_feed)), 1)
            z3_t_feed = self.decoder_level3_s(z3_t_feed)
            n3_t_feed = torch.cat((n3_t, self.upsample(n4_t_feed)), 1)
            n3_t_feed = self.decoder_level3(n3_t_feed)
            if iter > 0:
                x2_t = torch.cat((x2_t, cfm_feature_c), 1)
                x2_t = self.compress_out2_c(x2_t)
                y2_t = torch.cat((y2_t, cfm_feature_u), 1)
                y2_t = self.compress_out2_u(y2_t)
                z2_t = torch.cat((z2_t, cfm_feature_s), 1)
                z2_t = self.compress_out2_s(z2_t)
                n2_t = torch.cat((n2_t, cfm_feature), 1)
                n2_t = self.compress_out2(n2_t)

            x2_t_feed = torch.cat((x2_t, self.upsample_c(x3_t_feed)), 1)
            x2_t_feed = self.decoder_level2_c(x2_t_feed)
            y2_t_feed = torch.cat((y2_t, self.upsample_u(y3_t_feed)), 1)
            y2_t_feed = self.decoder_level2_u(y2_t_feed)
            z2_t_feed = torch.cat((z2_t, self.upsample_s(z3_t_feed)), 1)
            z2_t_feed = self.decoder_level2_s(z2_t_feed)
            n2_t_feed = torch.cat((n2_t, self.upsample(n3_t_feed)), 1)
            n2_t_feed = self.decoder_level2(n2_t_feed)

            cfm_feature_c = self.conv4_c(x2_t_feed)
            cfm_feature_u = self.conv4_u(y2_t_feed)
            cfm_feature_s = self.conv4_s(z2_t_feed)
            cfm_feature = self.conv4(n2_t_feed)

            prediction1_c = self.out_CFM_c(cfm_feature_c)
            prediction1_u = self.out_CFM_u(cfm_feature_u)
            prediction1_s = self.out_CFM_s(cfm_feature_s)
            prediction2 = self.out_CFM(cfm_feature)

            prediction1c_8 = F.interpolate(prediction1_c, scale_factor=8, mode='bilinear')
            prediction1u_8 = F.interpolate(prediction1_u, scale_factor=8, mode='bilinear')
            prediction1s_8 = F.interpolate(prediction1_s, scale_factor=8, mode='bilinear')
            prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')

            stage_c_loss.append(prediction1c_8)
            stage_u_loss.append(prediction1u_8)
            stage_s_loss.append(prediction1s_8)
            stage_loss.append(prediction2_8)
###-----------------------
        # SAM
        T2_c = self.Translayer2_0_c(cim_feature_c)
        T2_c = self.down05_c(T2_c)
        sam_feature_c = self.SAM_c(cfm_feature_c, T2_c)

        T2_u = self.Translayer2_0_u(cim_feature_u)
        T2_u = self.down05_u(T2_u)
        sam_feature_u = self.SAM_u(cfm_feature_u, T2_u)

        T2_s = self.Translayer2_0_s(cim_feature_s)
        T2_s = self.down05_s(T2_s)
        sam_feature_s = self.SAM_s(cfm_feature_s, T2_s)

        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2_c = self.out_SAM_c(sam_feature_c)
        prediction2_u = self.out_SAM_u(sam_feature_u)
        prediction2_s = self.out_SAM_s(sam_feature_s)
        prediction22 = self.out_SAM(sam_feature)

        prediction2c_8 = F.interpolate(prediction2_c, scale_factor=8, mode='bilinear')
        prediction2u_8 = F.interpolate(prediction2_u, scale_factor=8, mode='bilinear')
        prediction2s_8 = F.interpolate(prediction2_s, scale_factor=8, mode='bilinear')
        prediction22_8 = F.interpolate(prediction22, scale_factor=8, mode='bilinear')
        return stage_c_loss, stage_u_loss, stage_s_loss, prediction2c_8, prediction2u_8, prediction2s_8, stage_loss, prediction22_8


if __name__ == '__main__':
    model = Hitnet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2, prediction3, prediction4, prediction5, prediction6 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
