import torch
import torch.nn as nn

# from torchsummary import summary


class ConvX(nn.Module):

    def __init__(self, in_planes, out_planes, kernel=3, stride=1, padding=None, dim='3d', use_LN=False):
        super(ConvX, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        if padding == None:
            padding = kernel // 2
        if dim == '2d':
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_planes)
        elif dim == '3d':
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            # if use_LN:
            #     self.bn=nn.LayerNorm(out_planes)
            # else:
            self.bn = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class UnetBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dim='2d') -> None:
        super().__init__()
        if dim == '2d':
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(out_planes)
            self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm3d(out_planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class stdc_module(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, dim='3d', use_LN=False):
        super().__init__()
        self.stride = stride
        if stride == 2:
            if dim == '2d':
                self.avd_layer = nn.Sequential(
                    nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2, bias=False),
                    nn.BatchNorm2d(out_planes // 2),
                )
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if dim == '3d':
                self.avd_layer = nn.Sequential(
                    nn.Conv3d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2, bias=False),
                    nn.BatchNorm3d(out_planes // 2),
                )
                self.skip = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ConvX(in_planes, out_planes // 2, kernel=1, stride=1, dim=dim, use_LN=use_LN)  # * 1*1 convx
        self.block2 = ConvX(out_planes // 2, out_planes // 4, kernel=3, stride=1, dim=dim, use_LN=use_LN)
        self.block3 = ConvX(out_planes // 4, out_planes // 8, kernel=1, stride=1, dim=dim, use_LN=use_LN)
        self.block4 = ConvX(out_planes // 8, out_planes // 8, kernel=3, stride=1, dim=dim, use_LN=use_LN)
        self.block_list = nn.ModuleList([self.block1, self.block2, self.block3, self.block4])

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


class StdcEncoder(nn.Module):

    def __init__(self, in_channels=1, init_features=32, seperate=False, no_stdc=False):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        super(StdcEncoder, self).__init__()
        self.seperate = seperate

        self.convx1 = ConvX(in_channels, init_features, stride=1, dim='3d', use_LN=True)
        self.convx2 = ConvX(init_features, init_features * 2, stride=2, dim='3d', use_LN=True)

        # self.conv_last = ConvX(init_features * 32, max(1024, init_features * 32), kernel=1, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(max(1024, init_features * 32), max(1024, init_features * 32), bias=False)
        # self.bn = nn.BatchNorm2d(max(1024, init_features * 32))
        # self.relu = nn.ReLU(inplace=True)
        # self.drop = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(max(1024, init_features * 32), out_channels, bias=False)

        if no_stdc:
            self.stage3 = nn.Sequential(
                ConvX(in_planes=init_features * 2, out_planes=init_features * 4, stride=2, dim='3d', use_LN=True),
                ConvX(in_planes=init_features * 4, out_planes=init_features * 4, stride=1, dim='3d', use_LN=True),
            )
            self.stage4 = nn.Sequential(
                ConvX(in_planes=init_features * 4, out_planes=init_features * 8, stride=2, dim='3d', use_LN=True),
                ConvX(in_planes=init_features * 8, out_planes=init_features * 8, stride=1, dim='3d', use_LN=True),
            )
            self.stage5_2d = nn.Sequential(
                ConvX(in_planes=init_features * 8, out_planes=init_features * 16, stride=2, dim='2d'),
                ConvX(in_planes=init_features * 16, out_planes=init_features * 16, stride=1, dim='2d'),
            )
            self.stage5_3d = nn.Sequential(
                ConvX(in_planes=init_features * 8, out_planes=init_features * 16, stride=2, dim='3d'),
                ConvX(in_planes=init_features * 16, out_planes=init_features * 16, stride=1, dim='3d'),
            )
        else:
            self.stage3 = nn.Sequential(
                stdc_module(in_planes=init_features * 2, out_planes=init_features * 4, stride=2, dim='3d', use_LN=True),
                stdc_module(in_planes=init_features * 4, out_planes=init_features * 4, stride=1, dim='3d', use_LN=True),
            )
            self.stage4 = nn.Sequential(
                stdc_module(in_planes=init_features * 4, out_planes=init_features * 8, stride=2, dim='3d', use_LN=True),
                stdc_module(in_planes=init_features * 8, out_planes=init_features * 8, stride=1, dim='3d', use_LN=True),
            )
            self.stage5_2d = nn.Sequential(
                stdc_module(in_planes=init_features * 8, out_planes=init_features * 16, stride=2, dim='2d'),
                stdc_module(in_planes=init_features * 16, out_planes=init_features * 16, stride=1, dim='2d'),
            )
            self.stage5_3d = nn.Sequential(
                stdc_module(in_planes=init_features * 8, out_planes=init_features * 16, stride=2, dim='3d'),
                stdc_module(in_planes=init_features * 16, out_planes=init_features * 16, stride=1, dim='3d'),
            )
        self.neck = UnetBlock(init_features * 8, init_features * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # * 3d
        out1 = self.convx1(x)
        out2 = self.convx2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)  # [1,256,4,4,4] should be [1,256,8,8,8]
        out5_3d = self.stage5_3d(out4)
        out_3d_list = [out1, out2, out3, out4, out5_3d]
        # * 3d project to 2d
        if not self.seperate:
            xy1_plane, xz1_plane, yz1_plane, s1_mask = self.projection(out1)
            xy2_plane, xz2_plane, yz2_plane, s2_mask = self.projection(out2)
            xy3_plane, xz3_plane, yz3_plane, s3_mask = self.projection(out3)
            xy4_plane, xz4_plane, yz4_plane, s4_mask = self.projection(out4)
            xy5_out,   xz5_out,   yz5_out,   s5_mask = self.projection(out5_3d)
            
            dia1_plane, dia1_index = self.projection_diagonal(out1)
            dia2_plane, dia2_index = self.projection_diagonal(out2)
            dia3_plane, dia3_index = self.projection_diagonal(out3)
            dia4_plane, dia4_index = self.projection_diagonal(out4)
            dia5_out = self.stage5_2d(dia4_plane)

            xy_planes = [xy1_plane, xy2_plane, xy3_plane, xy4_plane, xy5_out]
            xz_planes = [xz1_plane, xz2_plane, xz3_plane, xz4_plane, xz5_out]
            yz_planes = [yz1_plane, yz2_plane, yz3_plane, yz4_plane, yz5_out]
            dia_planes = [dia1_plane, dia2_plane, dia3_plane, dia4_plane, dia5_out]
            
            # * column stack for decouple
            path1_stack = torch.stack((xy1_plane, xz1_plane, yz1_plane), dim=1)  # * after : [bs,3,32,64,64]
            path2_stack = torch.stack((xy2_plane, xz2_plane, yz2_plane), dim=1)
            path3_stack = torch.stack((xy3_plane, xz3_plane, yz3_plane), dim=1)
            path4_stack = torch.stack((xy4_plane, xz4_plane, yz4_plane), dim=1)
            stack_list = [path1_stack, path2_stack, path3_stack, path4_stack]
            mask_list = [s4_mask, s3_mask, s2_mask, s1_mask]
            dia_mask_list = [dia4_index, dia3_index, dia2_index, dia1_index]
            mask = s1_mask
            if self.training:
                return xy_planes, xz_planes, yz_planes, mask_list, out_3d_list, stack_list, dia_planes, dia_mask_list
            else:
                return xy_planes, xz_planes, yz_planes, mask, out_3d_list, stack_list, dia_planes, dia_mask_list
        else:
            return out_3d_list


    def projection(self, inputs):
        xy_plane = torch.max(inputs, -1).values  # * after projection shape:[1,32,64,64]
        xz_plane = torch.max(inputs, -2).values  # * after projection shape:[1,32,64,64]
        yz_plane = torch.max(inputs, -3).values  # * after projection shape:[1,32,64,64]

        xy_mask = (inputs == inputs.max(dim=-1, keepdim=True)[0])
        xz_mask = (inputs == inputs.max(dim=-2, keepdim=True)[0])  # * shape [1,32,64,64,64]
        yz_mask = (inputs == inputs.max(dim=-3, keepdim=True)[0])

        planes = [xy_plane, xz_plane, yz_plane]
        masks = [xy_mask, xz_mask, yz_mask]
        return xy_plane, xz_plane, yz_plane, masks
    
    def projection_diagonal(self, inputs):
        Bs, C, D, H, W = inputs.shape
        
        assert H == W, "H and W must be equal"
        
        device = inputs.device
        
        # 创建网格索引
        i_grid, j_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 计算每个位置对应的对角线索引 (i + j) / 2
        diagonal_sum = i_grid + j_grid  # [H, W]
        
        # 只保留偶数和值（因为我们要的是 i + j = 2*h 形式）
        valid_diagonals = diagonal_sum % 2 == 0
        diagonal_indices = diagonal_sum // 2  # 对角线索引 h
        
        # 过滤掉超出范围的对角线
        valid_range = diagonal_indices < H
        valid_mask = valid_diagonals & valid_range
        
        # 获取有效的坐标
        valid_i, valid_j = torch.where(valid_mask)
        valid_h = diagonal_indices[valid_i, valid_j]
        
        # 提取所有有效位置的值 [Bs, C, D, num_valid_positions]
        all_values = inputs[:, :, :, valid_i, valid_j]
        
        # 初始化结果
        res = torch.full((Bs, C, D, H), float('-inf'), device=device)
        index = torch.zeros_like(inputs)
        
        # 使用循环方式来找到每个对角线的最大值
        for h in range(H):
            mask = (valid_h == h)
            if mask.any():
                h_values = all_values[:, :, :, mask]
                h_max_values, h_max_indices = torch.max(h_values, dim=-1)
                
                res[:, :, :, h] = h_max_values
                
                # 找到最大值对应的原始坐标
                masked_positions = mask.nonzero(as_tuple=False)
                if masked_positions.numel() > 0:
                    # 确保squeeze不会意外删除batch维度
                    if masked_positions.dim() > 1:
                        masked_positions = masked_positions.squeeze(-1)
                    
                    max_pos_indices = masked_positions[h_max_indices]
                    
                    max_i = valid_i[max_pos_indices]
                    max_j = valid_j[max_pos_indices]
                    
                    # 设置索引 - 确保维度匹配
                    batch_idx = torch.arange(Bs, device=device)[:, None, None]
                    channel_idx = torch.arange(C, device=device)[None, :, None]
                    depth_idx = torch.arange(D, device=device)[None, None, :]
                    
                    index[batch_idx, channel_idx, depth_idx, max_i, max_j] = 1
        
        return res, index
            
                
                    
            


class UnetDecoder(nn.Module):

    def __init__(self, base, out_channels, dim='2d') -> None:
        super().__init__()

        if dim == '2d':
            self.upconv4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
            self.upconv3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
            self.upconv1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)

            self.decoder4 = UnetBlock((base * 8) * 2, base * 8)
            self.decoder3 = UnetBlock((base * 4) * 2, base * 4)
            self.decoder2 = UnetBlock((base * 2) * 2, base * 2)
            self.decoder1 = UnetBlock(base * 2, base)
            self.conv = nn.Conv2d(in_channels=base, out_channels=out_channels, kernel_size=1)
        elif dim == '3d':
            self.upconv4 = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=2, stride=2)
            self.upconv3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
            self.upconv2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
            self.upconv1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)

            self.decoder4 = UnetBlock((base * 8) * 2, base * 8, dim='3d')
            self.decoder3 = UnetBlock((base * 4) * 2, base * 4, dim='3d')
            self.decoder2 = UnetBlock((base * 2) * 2, base * 2, dim='3d')
            self.decoder1 = UnetBlock(base * 2, base, dim='3d')
            self.conv = nn.Conv3d(in_channels=base, out_channels=out_channels, kernel_size=1)

    def forward(self, plane_list):
        plane1, plane2, plane3, plane4, plane5 = plane_list
        dec4 = self.upconv4(plane5)
        dec4 = torch.cat((dec4, plane4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, plane3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, plane2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, plane1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        m_outputs = [dec4, dec3, dec2, dec1]
        return outputs, m_outputs
    
if __name__ == "__main__":
    model = StdcEncoder()
    a = torch.randn(2, 32, 64, 64, 64)
    res, index = model.projection_diagonal(a)
    print(res.shape)
    print(index.shape)
