from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
import torch
from torch import nn
from functools import partial

# unet原型





class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.dv = DoubleConv2(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        x = self.maxpool(x)
        x1 = self.dv(x)
        x =self.conv1(x)
        x = torch.cat((x, x1),dim=1)
        x = self.conv2(x)

        return x


class DoubleConv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels

        super(DoubleConv2, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        diff_y = x1.size()[2] - g1.size()[2]
        diff_x = x1.size()[3] - g1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        g1 = F.pad(g1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        self.att=Attention_block(in_channels//2,in_channels//2,in_channels//4)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = DySample(in_channels//2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 =self.conv1(x1)

        x1 = self.up(x1)


        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dC = DoubleConv(in_size, out_size)
        self.dsv = nn.Sequential(
            # nn.Conv2d(in_channels=in_size,out_channels=out_size,kernel_size=1,padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear',align_corners=True), )

    def forward(self, input):

        input = self.dC(input)

        return self.dsv(input)






class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )





class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0),
            nn.BatchNorm2d(1))
    def forward(self,x):

        x=self.conv(x)

        x=torch.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0),
            nn.BatchNorm2d(int(out_channels/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu=nn.ReLU()
        self.activation = nn.ReLU()

    def forward(self,x):

        x=nn.AvgPool2d(x.size()[2:])(x)
        x1 = nn.MaxPool2d(x.size()[2:])(x)

        x=x+x1

        x=self.relu(x)

        x=self.conv1(x)
        x=self.activation(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x

class SCSE_Block(nn.Module):
    def __init__(self, out_channels):
        super(SCSE_Block, self).__init__()
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x,x1):
        g1 = self.spatial_gate(x)

        g2 = self.channel_gate(x)


        x = g1 * x1 + g2 * x1+x1
        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)



class cat1(nn.Module):
    def __init__(self, out_channels):
        super(cat1, self).__init__()
        half = out_channels//2
        self.conv1 = nn.Conv2d(out_channels, half, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels*2, half, kernel_size=1)

        self.up = DySample(out_channels*2)


    def forward(self,x1,x2):

        x1 = self.conv1(x1)

        x2 = self.up(x2)

        x2 =self.conv2(x2)

        # input = [x1,x2]
        # x=self.cat(input)
        x=torch.cat([x1,x2],dim=1)
        return x

class cat2(nn.Module):
    def __init__(self, out_channels):
        super(cat2, self).__init__()
        out3 = out_channels//3
        self.conv1 = nn.Conv2d(out_channels, out3, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels*4, out3, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels-2*out3, kernel_size=1)

        self.up1  = DySample(out_channels*4,scale=4)
        self.up2 = DySample(out_channels*2,scale=2)

    def forward(self,x1,x2,x3):

        x1 = self.conv1(x1)
        x2 = self.up1(x2)

        x2 =self.conv2(x2)
        x3 = self.up2(x3)
        x3 =self.conv3(x3)

        x=torch.cat([x1,x2,x3],dim=1)

        return x

class cat3(nn.Module):
    def __init__(self, out_channels):
        super(cat3, self).__init__()
        out4 = out_channels//4
        self.conv1 = nn.Conv2d(out_channels, out4, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels*8, out4, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels*4, out4, kernel_size=1)
        self.conv4 = nn.Conv2d(out_channels*2, out_channels-3*out4, kernel_size=1)

        self.up1  = DySample(out_channels*8,scale=8)
        self.up2 = DySample(out_channels*4,scale=4)
        self.up3 = DySample(out_channels*2,scale=2)

    def forward(self,x1,x2,x3,x4):

        x1 = self.conv1(x1)
        x2 = self.up1(x2)

        x2 =self.conv2(x2)
        x3 = self.up2(x3)
        x3 =self.conv3(x3)
        x4 = self.up3(x4)
        x4 =self.conv4(x4)

        x=torch.cat([x1,x2,x3,x4],dim=1)

        return x


class cat4(nn.Module):
    def __init__(self, out_channels):
        super(cat4, self).__init__()
        out5 = out_channels//5
        self.conv1 = nn.Conv2d(out_channels, out5, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels*16, out5, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels*8, out5, kernel_size=1)
        self.conv4 = nn.Conv2d(out_channels*4, out5, kernel_size=1)
        self.conv5 = nn.Conv2d(out_channels*2, out_channels-4*out5, kernel_size=1)

        self.up1  = DySample(out_channels*16,scale=16)
        self.up2 = DySample(out_channels*8,scale=8)
        self.up3 = DySample(out_channels*4,scale=4)
        self.up4 = DySample(out_channels*2,scale=2)

    def forward(self,x1,x2,x3,x4,x5):

        x1 = self.conv1(x1)
        x2 = self.up1(x2)

        x2 =self.conv2(x2)
        x3 = self.up2(x3)
        x3 =self.conv3(x3)
        x4 = self.up3(x4)
        x4 =self.conv4(x4)
        x5=self.up4(x5)
        x5 =self.conv5(x5)

        x=torch.cat([x1,x2,x3,x4,x5],dim=1)

        return x

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.dsv1 = UnetDsv3(in_size=256, out_size=128, scale_factor=128)
        self.dsv2 = UnetDsv3(in_size=128, out_size=64, scale_factor=256)
        self.dsv3 = UnetDsv3(in_size=64, out_size=32, scale_factor=512)

        # self.corss_attention_end1 = corss_attention_end(in_channels_1=32, in_channels_2=32, mid_channels=32)
        self.corss_attention_end1 = SCSE_Block(128)
        self.corss_attention_end2 = SCSE_Block(64)
        self.corss_attention_end3 = SCSE_Block(32)

        self.endup1 = UnetDsv3(in_size=64, out_size=32, scale_factor=512)
        self.endup2 = UnetDsv3(in_size=128, out_size=32, scale_factor=512)
        self.endup3 = UnetDsv3(in_size=256, out_size=32, scale_factor=512)

        self.cat1 = cat1(256)
        self.cat2 = cat2(128)
        self.cat3 = cat3(64)
        self.cat4 = cat4(32)




    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.cat1(x4,x5)
        x_1 = self.up1(x5, x4)
        x3 = self.cat2(x3,x5,x_1)
        x_2 = self.up2(x_1, x3)
        x2 = self.cat3(x2,x5,x_1,x_2)
        x_3 = self.up3(x_2, x2)
        x1 = self.cat4(x1,x5,x_1,x_2,x_3)
        x_4 = self.up4(x_3, x1)

        x_11 = self.dsv1(x_1)
        x_21 = self.corss_attention_end1(x_11,x_2)
        x_22 = self.dsv2(x_21)
        x_32 = self.corss_attention_end2(x_22,x_3)
        x_33 = self.dsv3(x_32)
        x_43 = self.corss_attention_end3(x_33,x_4)



        x_32 =self.endup1(x_32)
        x_21 =self.endup2(x_21)
        x_1 =self.endup3(x_1)

        logits1 = self.out_conv(x_43)
        logits2 = self.out_conv(x_32)
        logits3 = self.out_conv(x_21)
        logits4 = self.out_conv(x_1)


        return {"out": logits1, "out1": logits2, "out2": logits3, "out3": logits4}
        # return {"out": logits1}

