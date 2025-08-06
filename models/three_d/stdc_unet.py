from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, padding=None):
        super(ConvX, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        if padding == None:
            padding = kernel // 2
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class stdc_module(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.stride = stride
        # ? can't understand
        # TODO
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv3d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False,
                ),
                nn.BatchNorm3d(out_planes // 2),
            )
            self.skip = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ConvX(
            in_planes, out_planes // 2, kernel=1, stride=1
        )  # * 1*1 convx
        self.block2 = ConvX(out_planes // 2, out_planes // 4, kernel=3, stride=1)
        self.block3 = ConvX(out_planes // 4, out_planes // 8, kernel=1, stride=1)
        self.block4 = ConvX(out_planes // 8, out_planes // 8, kernel=3, stride=1)
        self.block_list = nn.ModuleList(
            [self.block1, self.block2, self.block3, self.block4]
        )

    def forward(self, x):
        out_list = []
        out1 = self.block_list[0](x)
        for i, block in enumerate(self.block_list[1:]):
            if i == 0:
                if self.stride == 2:
                    out = block(self.avd_layer(out1))
                else:
                    out = block(out1)
            else:
                out = block(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class STDC_Net(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(STDC_Net, self).__init__()
        self.convx1 = ConvX(in_channels, init_features, stride=1)
        self.convx2 = ConvX(init_features, init_features * 2, stride=2)

        # self.conv_last = ConvX(init_features * 32, max(1024, init_features * 32), kernel=1, stride=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(
            max(1024, init_features * 32), max(1024, init_features * 32), bias=False
        )
        self.bn = nn.BatchNorm3d(max(1024, init_features * 32))
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(max(1024, init_features * 32), out_channels, bias=False)

        self.stage3 = nn.Sequential(
            stdc_module(
                in_planes=init_features * 2, out_planes=init_features * 4, stride=2
            ),
            stdc_module(
                in_planes=init_features * 4, out_planes=init_features * 4, stride=1
            ),
        )
        self.stage4 = nn.Sequential(
            stdc_module(
                in_planes=init_features * 4, out_planes=init_features * 8, stride=2
            ),
            stdc_module(
                in_planes=init_features * 8, out_planes=init_features * 8, stride=1
            ),
        )
        self.stage5 = nn.Sequential(
            stdc_module(
                in_planes=init_features * 8, out_planes=init_features * 16, stride=2
            ),
            stdc_module(
                in_planes=init_features * 16, out_planes=init_features * 16, stride=1
            ),
        )
        self.neck = UnetBlock(init_features * 8, init_features * 16)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.convx1(x)
        out2 = self.convx2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)  # [1,256,4,4,4] should be [1,256,8,8,8]
        out5 = self.stage5(out4)  # [1,512,4,4,4]
        # out5 = self.stage5(out4)
        # out5 = self.conv_last(out5
        # out = self.conv_last(out)
        # out = self.global_pool(out).flatten(1)
        # # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # out = self.relu(out)
        # out = self.drop(out)
        # out = self.linear(out)
        out_list = [out1, out2, out3, out4, out5]

        # return out1, out2, out3, out4, out5
        return out_list
        # out=self.bn(out)


class UnetUP(nn.Module):
    def __init__(self, base, out_channels) -> None:
        super().__init__()

        self.upconv4 = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=2, stride=2)
        self.decoder4 = UnetBlock((base * 8) * 2, base * 8)

        self.upconv3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.decoder3 = UnetBlock((base * 4) * 2, base * 4)

        self.upconv2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.decoder2 = UnetBlock((base * 2) * 2, base * 2)

        self.upconv1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.decoder1 = UnetBlock(base * 2, base)

        self.conv = nn.Conv3d(
            in_channels=base, out_channels=out_channels, kernel_size=1
        )

    def forward(self, out_list):
        out1, out2, out3, out4, out5 = out_list
        dec4 = self.upconv4(out5)
        dec4 = torch.cat((dec4, out4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, out3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, out2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, out1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs


class UnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_planes, out_planes, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_planes, out_planes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class StdcUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32) -> None:
        super().__init__()
        self.encoder = STDC_Net(in_channels=in_channels, init_features=init_features)
        self.decoder = UnetUP(base=init_features, out_channels=out_channels)

    def forward(self, x):
        out_list = self.encoder(x)
        out = self.decoder(out_list)
        return out


if __name__ == "__main__":
    import os

    from torchsummary import summary

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(1, 1, 64, 64, 64).to(device)

    net = StdcUnet()
    net.cuda()
    net.eval()
    # out = net(x)

    # onnx_path = './saved_model.onnx'
    # out, out16, out32 = net(in_ten)
    # torch.onnx.export(net, in_ten, onnx_path)
    # netron.start(onnx_path)
    # torch.save(net.state_dict(), 'STDCNet813.pth')
    summary(net, (1, 64, 64, 64))
