import torch
import torch.nn as nn
from resnet import *
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, strides, pads, 
        dilation = 1, inplace = True, has_bias = False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = ksize, stride = strides,
                                padding = pads, dilation = dilation, bias = has_bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(AttentionRefinement, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        ax = self.channel_attention(x)
        x = x * ax

        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 4):
        super(FeatureFusion, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bias = False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, out_planes // reduction, 1, 1, 0, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_planes // reduction, out_planes, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim = 1)
        x = self.conv_3x3(x)
        x_se = self.channel_attention(x)
        output = x * x_se + x
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DoubleConv, self).__init__()
        self.conv_3x3_1 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bias = False)
        self.conv_3x3_2 = ConvBnRelu(out_planes, out_planes, 3, 1, 1, has_bias = False)

    def forward(self, x):
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return x

class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3, has_bias = False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1, has_bias = False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1, has_bias = False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0, has_bias = False)#1x1 delete?

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output

class BiSeNet(nn.Module):
    def __init__(self, num_class, train = True):
        super(BiSeNet, self).__init__()
        self.training = train
        self.spatial_path = SpatialPath(3, 128)
        self.context_path = resnet18()
        conv_channel = 128
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0, has_bias = False)
        )
        self.arm_32 = AttentionRefinement(512, 512)
        self.arm_16 = AttentionRefinement(256, 256)
        self.refine_merge32 = DoubleConv(512 + 128, 512)
        self.refine_merge16 = DoubleConv(512 + 256, 256)
        self.ffm = FeatureFusion(conv_channel * 3, conv_channel * 3, reduction = 4)
        self.conv_3x3_1 = ConvBnRelu(512, 256, 3, 1, 1, has_bias = False)
        self.conv_1x1_1 = ConvBnRelu(256, num_class, 1, 1, 0, has_bias = False)
        self.conv_3x3_2 = ConvBnRelu(256, 128, 3, 1, 1, has_bias = False)
        self.conv_1x1_2 = ConvBnRelu(128, num_class, 1, 1, 0, has_bias = False)
        self.conv_1x1_3 = ConvBnRelu(conv_channel * 3, num_class, 1, 1, 0, has_bias = False)

    def forward(self, data):
        spatial_out = self.spatial_path(data)
        context_block1, context_block2, context_block3, context_block4 = self.context_path(data)

        global_context = self.global_pool(context_block4)
        global_context = F.interpolate(global_context, size = context_block4.size()[2:],
                                       mode = 'bilinear', align_corners = True)
        arm32 = self.arm_32(context_block4)
        arm16 = self.arm_16(context_block3)
        merge32 = torch.cat((arm32, global_context), dim = 1)#512+128->640
        refine32 = self.refine_merge32(merge32)#512+128->512
        refine32 = F.interpolate(refine32, size = context_block3.size()[2:],
                                      mode = 'bilinear', align_corners = True)
        merge16 = torch.cat((refine32, arm16), dim = 1)
        context_out = self.refine_merge16(merge16)#512+256->256
        context_scale = F.interpolate(context_out, scale_factor = 2,
                                      mode = 'bilinear', align_corners = True)
        ffm = self.ffm(spatial_out, context_scale)
        ffm = self.conv_1x1_3(ffm)
        #main_out = F.interpolate(ffm, scale_factor = 8, mode = 'bilinear', align_corners = True)
        main_out = ffm
        #print(main_out.size())
        if self.training:
            refine32 = self.conv_3x3_1(refine32)
            refine32 = self.conv_1x1_1(refine32)
            refine32scale = F.interpolate(refine32, scale_factor = 2,
                                      mode = 'bilinear', align_corners = True)#scale_factor 16->2
            context_out = self.conv_3x3_2(context_out)
            context_out = self.conv_1x1_2(context_out)
            refine16scale = F.interpolate(context_out, scale_factor = 2,
                                      mode = 'bilinear', align_corners = True)#scale_factor 16->2

            return main_out, refine16scale, refine32scale

        return main_out

def creat_model(training = True, num_classes = 19):
    model = BiSeNet(num_classes, training)
    #init
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            #nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.normal_()

    return model
