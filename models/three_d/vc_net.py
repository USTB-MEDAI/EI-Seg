"""
Author: Yifan Wang
Date created: 11/09/2020
PyTorch implementation of VC-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MipArg(nn.Module):
    def __init__(self, cube_size=(64, 64, 64), channel_2d=32):
        super(MipArg, self).__init__()
        self.cube_size = cube_size
        self.channel_2d = channel_2d
        self.s = self.cube_size[2] // 16 * 5

    def forward(self, x):
        # 使用reshape替代view，并确保张量维度正确
        x = x.reshape(-1, self.channel_2d, 3, self.cube_size[0], 2, self.cube_size[1])
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(-1, self.channel_2d, 6, self.cube_size[0], self.cube_size[1])
        # Stack 20 times for 20-sliced MIP
        x_slice_list = [x for _ in range(self.s)]
        x = torch.stack(x_slice_list, dim=-1) 
        # ? [batch_size, channel_2d, 6, self.cube_size[0], self.cube_size[1], self.s]
        # ? [bs, 32, 6, 64, 64, 20]
        return x


class Unproject(nn.Module):
    def __init__(self, cube_size=(64, 64, 64)):
        super(Unproject, self).__init__()
        self.cube_size = cube_size

    def forward(self, x):
        # Split along the 4th dimension
        a = torch.unbind(x, dim=-4) # ? x : [bs, 32, 6, 64, 64, 20]; 

        # Pad each slice
        a0 = F.pad(a[0], (0, 44, 0, 0, 0, 0, 0, 0))
        a1 = F.pad(a[1], (8, 36, 0, 0, 0, 0, 0, 0))
        a2 = F.pad(a[2], (16, 28, 0, 0, 0, 0, 0, 0))
        a3 = F.pad(a[3], (24, 20, 0, 0, 0, 0, 0, 0))
        a4 = F.pad(a[4], (32, 12, 0, 0, 0, 0, 0, 0))
        a5 = F.pad(a[5], (44, 0, 0, 0, 0, 0, 0, 0)) # ? [bs, 32, 6, 64, 64, 64]

        # Stack and get maximum
        a_concat = torch.stack([a0, a1, a2, a3, a4, a5], dim=-1)
        a_max = torch.max(a_concat, dim=-1)[0] # ? (bs, 32, 64, 64, 64)

        return a_max


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=False):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, depth=4, batch_norm=False):
        super(UNet3D, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.down_path.append(ConvBlock3D(in_ch, out_ch, batch_norm=batch_norm))
            in_ch = out_ch

        # Upsampling path
        for i in range(depth - 1):
            in_ch = base_filters * (2 ** (depth - 1 - i))
            out_ch = base_filters * (2 ** (depth - 2 - i))
            self.up_path.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                    ConvBlock3D(out_ch * 2, out_ch, batch_norm=batch_norm),
                )
            )

    def forward(self, x):
        # Downsampling
        features = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                features.append(x)
                x = F.max_pool3d(x, 2)

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up[0](x)  # 先进行转置卷积
            x = torch.cat([x, features[-(i + 1)]], dim=1)  # 拼接特征
            x = up[1](x)  # 再进行卷积块处理

        return x


class ConvBlock2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dropout=0.0, batch_norm=False
    ):
        super(ConvBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UNet2D(nn.Module):
    def __init__(
        self, in_channels=1, base_filters=32, depth=4, dropout=0.0, batch_norm=False
    ):
        super(UNet2D, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.down_path.append(
                ConvBlock2D(in_ch, out_ch, dropout=dropout, batch_norm=batch_norm)
            )
            in_ch = out_ch

        # Upsampling path
        for i in range(depth - 1):
            in_ch = base_filters * (2 ** (depth - 1 - i))
            out_ch = base_filters * (2 ** (depth - 2 - i))
            self.up_path.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    ConvBlock2D(
                        out_ch * 2, out_ch, dropout=dropout, batch_norm=batch_norm
                    ),
                )
            )

    def forward(self, x):
        # Downsampling
        features = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                features.append(x)
                x = F.max_pool2d(x, 2)

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up[0](x)  # 先进行转置卷积
            x = torch.cat([x, features[-(i + 1)]], dim=1)  # 拼接特征
            x = up[1](x)  # 再进行卷积块处理

        return x

class Projection(nn.Module):
    # ? the original size in the paper is 128,128,16 , where the 16 is the vertical size
    # ? ensure that the vertical size is divisible by 16
    def __init__(self, size_3D=[64, 64, 64]): 
        super(Projection, self).__init__()
        self.m = 6
        self.Vertical_size = size_3D[2]
        self.s = self.Vertical_size // 16 * 5
        self.t = math.ceil((self.Vertical_size - self.s) / self.m)
        
        
        
    # ? the shape of x is [batch_size, 1, 64, 64, 64]
    def forward(self, x):
        x = x.squeeze(1)
        
        x_1 = x[:, :, :, 0: self.s].max(dim=3)[0]
        x_2 = x[:, :, :, self.t: self.s + self.t].max(dim=3)[0]
        x_3 = x[:, :, :, self.t * 2: self.s + self.t * 2].max(dim=3)[0]
        x_4 = x[:, :, :, self.t * 3: self.s + self.t * 3].max(dim=3)[0]
        x_5 = x[:, :, :, self.t * 4: self.s + self.t * 4].max(dim=3)[0]
        x_6 = x[:, :, :, self.Vertical_size - self.s: self.Vertical_size].max(dim=3)[0]
        
        arg_1 = (x[:, :, :, 0: self.s] == x[:, :, :, 0: self.s].max(dim=3, keepdim=True)[0])
        arg_2 = (x[:, :, :, self.t: self.s + self.t] == x[:, :, :, self.t: self.s + self.t].max(dim=3, keepdim=True)[0])
        arg_3 = (x[:, :, :, self.t * 2: self.s + self.t * 2] == x[:, :, :, self.t * 2: self.s + self.t * 2].max(dim=3, keepdim=True)[0])
        arg_4 = (x[:, :, :, self.t * 3: self.s + self.t * 3] == x[:, :, :, self.t * 3: self.s + self.t * 3].max(dim=3, keepdim=True)[0])
        arg_5 = (x[:, :, :, self.t * 4: self.s + self.t * 4] == x[:, :, :, self.t * 4: self.s + self.t * 4].max(dim=3, keepdim=True)[0])
        arg_6 = (x[:, :, :, self.Vertical_size - self.s: self.Vertical_size] == x[:, :, :, self.Vertical_size - self.s: self.Vertical_size].max(dim=3, keepdim=True)[0])
        
        x_line_1 = torch.cat([x_1, x_2, x_3], dim=2)
        x_line_2 = torch.cat([x_4, x_5, x_6], dim=2)
        
        x_res = torch.cat([x_line_1, x_line_2], dim=1) # ? [batch_size, 2 * 64, 3 * 64]
        
        x_res = x_res.unsqueeze(1) # ? [batch_size, 1, 2 * 64, 3 * 64]
        
        arg_res = torch.stack([arg_1, arg_2, arg_3, arg_4, arg_5, arg_6], dim=-1)
        # print("arg_res.shape: ", arg_res.shape)
        arg_res = arg_res.permute(0, 4, 1, 2, 3)# ? [batch_size, 6, 64, 64]

        
        return x_res, arg_res


class VCNet(nn.Module):
    def __init__(
        self,
        in_channels=1, # ? the channel of the 3D input must be 1
        cube_size=(64, 64, 64),
        dropout_2d=0.0,
        batch_norm=False,
        arg_2d=0.2,
    ):
        super(VCNet, self).__init__()
        self.cube_size = cube_size
        self.arg_2d = arg_2d
        
        self.projection = Projection(cube_size)

        # 3D UNet
        self.unet3d = UNet3D(in_channels=in_channels, batch_norm=batch_norm)

        # 2D UNet
        self.unet2d = UNet2D(
            in_channels=1, dropout=dropout_2d, batch_norm=batch_norm
        )

        # MIP and Unprojection layers
        self.mip_arg = MipArg(self.cube_size)
        self.unproject = Unproject(self.cube_size)

        # Final fusion layers
        self.fusion_conv1 = nn.Conv3d(96, 32, kernel_size=1)
        self.fusion_conv2 = nn.Conv3d(32, 2, kernel_size=1)
        self.final_conv2d = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_3d):
        x_2d, arg_2d = self.projection(x_3d)
        
        # 3D path
        fea_3d = self.unet3d(x_3d)

        # 2D path
        fea_2d = self.unet2d(x_2d)
        final_2d = torch.sigmoid(self.final_conv2d(fea_2d))

        # MIP and back-projection
        final_reshape = self.mip_arg(fea_2d)
        back3d = final_reshape * arg_2d
        fea_2d_3d = self.unproject(back3d)

        # Final fusion
        fea_fuse = torch.cat([fea_3d, fea_2d_3d], dim=1)
        fea_fuse = F.relu(self.fusion_conv1(fea_fuse))
        final_3d = torch.sigmoid(self.fusion_conv2(fea_fuse))

        return final_3d, final_2d, x_2d


def get_vc_net(
    cube_size=(1, 128, 128, 16),
    patch_size=(384, 256),
    num_channels_2d=1,
    dropout_2d=0.0,
    batch_norm=False,
):
    """
    创建VC-Net模型

    参数:
        cube_size: 3D输入的大小 (channels, height, width, depth)
        patch_size: 2D输入的大小 (height, width)
        num_channels_2d: 2D输入的通道数
        dropout_2d: 2D UNet中的dropout率
        batch_norm: 是否使用批归一化
    """
    model = VCNet(
        cube_size=cube_size,
        patch_size=patch_size,
        num_channels_2d=num_channels_2d,
        dropout_2d=dropout_2d,
        batch_norm=batch_norm,
    )
    return model


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(1, 1, 64, 64, 64)
    # p = Projection()
    # x_2d, arg_2d = p(x)
    # print("x_2d.shape: ", x_2d.shape)
    # print("arg_2d.shape: ", arg_2d.shape)
    
    # model = UNet2D()
    # x_2d = model(x_2d)
    # print("x_2d.shape: ", x_2d.shape)
    
    # mip = MipArg()
    # b = mip(x_2d)
    # print("b.shape: ", b.shape)
    
    # b = b * arg_2d
    # print("b.shape: ", b.shape)
    
    # unproject = Unproject()
    # b = unproject(b)
    # print("b.shape: ", b.shape)
    
    model = VCNet()
    output1 , output2, x_2d = model(x)
    print("output1.shape: ", output1.shape)
    print("output2.shape: ", output2.shape)
    print("x_2d.shape: ", x_2d.shape)
