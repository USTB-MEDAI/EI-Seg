# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .stdc_helper import StdcEncoder, UnetDecoder
import torch.nn as nn
from einops import repeat
import math
import torch


class ProjectionSegNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, tsne_mode=False, tsne_layer='dec4', plain_num=3) -> None:
        super().__init__()
        self.img_size = 64
        self.init_features = init_features
        self.tsne_mode = tsne_mode
        self.tsne_layer = tsne_layer
        self.plain_num = plain_num
        self.conv1 = nn.Conv3d(in_channels, init_features, kernel_size=3, stride=1, padding=1)  # *  increase chanels, keep feature map size
        self.encoder = StdcEncoder(in_channels, init_features)

        self.path1 = UnetDecoder(init_features, init_features, dim='2d')
        self.path2 = UnetDecoder(init_features, init_features, dim='2d')
        self.path3 = UnetDecoder(init_features, init_features, dim='2d')
        
        if self.plain_num == 4:
            self.path4 = UnetDecoder(init_features, init_features, dim='2d')
        # self.path4 = UnetDecoder(init_features, init_features, dim='2d')
        
        self.seg_path = UnetDecoder(init_features, out_channels, dim='3d')
        self.conv_head = nn.Conv3d(in_channels=init_features, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.rec_head = nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, inputs):
        # * 3d 2d Path
        xy_planes, xz_planes, yz_planes, masks, out_3d_list, stack_list, dia_planes, dia_mask_list = self.encoder(inputs)

        # * calculate spatial loss (decouple)
        path1_stack, path2_stack, path3_stack, path4_stack = stack_list
        # t_loss = self.decouple_loss(path4_stack)
        _, _, _, dia4_plane, _ = dia_planes

        # * recover 2d -> 3d
        xy_out, xy_decs = self.path1(xy_planes)
        xz_out, xz_decs = self.path2(xz_planes)
        yz_out, yz_decs = self.path3(yz_planes)  # * shape [1,32,64,64]
        if self.plain_num == 4:
            dia_out, dia_decs = self.path4(dia_planes)
            dia_expand = repeat(dia_out, 'b c x y -> b c x y z', z=self.img_size)
            _ ,_ ,_ ,dia1_mask = dia_mask_list
            dia_rec = self.unproject(dia_expand, dia1_mask)
        
        xy_expand = repeat(xy_out, 'b c x y -> b c x y z', z=self.img_size)
        xz_expand = repeat(xz_out, 'b c x z -> b c x y z', y=self.img_size)
        yz_expand = repeat(yz_out, 'b c y z -> b c x y z', x=self.img_size)
        

        if self.training:
            _, _, _, s1_masks = masks
        else:
            s1_masks = masks
        xy_mask, xz_mask, yz_mask = s1_masks
        
        
        
        # print(xy_expand.shape, xy_mask.shape)

        xy_rec = xy_expand * xy_mask
        xz_rec = xz_expand * xz_mask
        yz_rec = yz_expand * yz_mask
        
        
        
        if self.plain_num == 1:
            path4_stack = path4_stack[:, 0, :, :]
            out = xy_rec
        elif self.plain_num == 2:
            path4_stack = path4_stack[:, 0:2, :, :]
            out = (xy_rec + xz_rec) / 2
        elif self.plain_num == 3:
            path4_stack = path4_stack[:, 0:3, :, :]
            out = (xy_rec + xz_rec + yz_rec) / 3
        elif self.plain_num == 4:
            path4_stack = torch.cat([path4_stack, dia4_plane], dim=1)
            out = (xy_rec + xz_rec + yz_rec + dia_rec) / 4
        else:
            raise ValueError(f"plain_num must be in [1,2,3,4], but got {self.plain_num}")
        
        out = self.conv_head(out)

        # * get 2d middle decoder to 3d
        if self.training:
            t_loss = self.decouple_loss(path4_stack)
            dec4_list, dec3_list, dec2_list, dec1_list = [], [], [], []
            for i in [xy_decs, xz_decs, yz_decs]:
                dec4_list.append(i[0])
                dec3_list.append(i[1])
                dec2_list.append(i[2])
                dec1_list.append(i[3])
            decs = [dec4_list, dec3_list, dec2_list, dec1_list]
            m_out = []
            for index, (m, dec) in enumerate(zip(
                    masks, decs)):  # * masks [s4_masks,...] s4_masks contain [xy_mask,...]  decs [dec4_list,...] dec4_list: [xy_dec4,xz_dec4...]
                xy_mask, xz_mask, yz_mask = m
                xy_dec, xz_dec, yz_dec = dec

                t_xy_expand = repeat(xy_dec, 'b c x y -> b c x y z', z=4 * int(math.pow(2, index + 1)))
                t_xz_expand = repeat(xz_dec, 'b c x z -> b c x y z', y=4 * int(math.pow(2, index + 1)))
                t_yz_expand = repeat(yz_dec, 'b c y z -> b c x y z', x=4 * int(math.pow(2, index + 1)))
                t_xy_rec = t_xy_expand * xy_mask
                t_xz_rec = t_xz_expand * xz_mask
                t_yz_rec = t_yz_expand * yz_mask
                if self.plain_num == 1:
                    t_out = t_xy_rec
                elif self.plain_num == 2:
                    t_out = (t_xy_rec + t_xz_rec) / 2
                elif self.plain_num == 3:
                    t_out = (t_xy_rec + t_xz_rec + t_yz_rec) / 3
                m_out.append(t_out)  # * sequence: [4,3,2,1]

            # * seg path for co-train
            out_3d, out_decs = self.seg_path(out_3d_list)
            # out_3d=self.rec_head(out_3d)
            return out, out_3d, m_out, out_decs, t_loss
        else:
            out_3d, out_decs = self.seg_path(out_3d_list)
            if self.tsne_mode:
                # if self.tsne_layer == 'dec4':
                #     return out,  path4_stack
                # elif self.tsne_layer == 'dec3':
                #     return out,  path3_stack
                # elif self.tsne_layer == 'dec2':
                #     return out,  path2_stack
                # elif self.tsne_layer == 'dec1':
                #     return out,  path1_stack
                if self.tsne_layer == 'before_mask':
                    return xy_out, xz_out, yz_out
                elif self.tsne_layer == 'recover':
                    return xy_rec,xz_rec,yz_rec
                elif self.tsne_layer == 'enc4':
                    # path4_stack = path4_stack.squeeze()
                    xy_enc4 = path4_stack[:, 0, :, :]
                    xz_enc4 = path4_stack[:, 1, :, :]
                    yz_enc4 = path4_stack[:, 2, :, :]

                    return xy_enc4, xz_enc4, yz_enc4

            else:
                return out, out_3d
        
    def unproject(self, dia_expand, dia1_mask):
        """
        将对角线投影的2D特征重新投影回3D空间
        
        Args:
            dia_expand: 扩展后的对角线特征 [Bs, C, H, W, D]
            dia1_mask: 对角线投影时记录的索引掩码 [Bs, C, D, H, W]
            
        Returns:
            dia_rec: 重新投影后的3D特征 [Bs, C, D, H, W]
        """
        # 获取输入张量的维度大小：批次大小、通道数、高度、宽度、深度
        Bs, C, H, W, D = dia_expand.shape
        # 获取张量所在的设备（CPU或GPU）
        device = dia_expand.device
        
        # 初始化输出张量：创建一个全零的3D张量，用于存储重新投影的结果
        dia_rec = torch.zeros(Bs, C, D, H, W, device=device)
        
        # 创建网格索引，用于计算对角线映射
        # i_grid: 行索引网格 [H, W]，每行都是[0,1,2,...,H-1]
        # j_grid: 列索引网格 [H, W]，每列都是[0,1,2,...,W-1]
        i_grid, j_grid = torch.meshgrid(
            torch.arange(H, device=device),  # 创建行索引数组 [0, 1, 2, ..., H-1]
            torch.arange(W, device=device),  # 创建列索引数组 [0, 1, 2, ..., W-1]
            indexing='ij'  # 使用ij索引方式，确保i_grid是行索引，j_grid是列索引
        )
        
        # 计算每个位置对应的对角线索引
        # diagonal_sum: 每个位置的i+j值，用于确定对角线
        diagonal_sum = i_grid + j_grid
        # valid_diagonals: 只保留i+j为偶数的位置（因为对角线定义为i+j=2*h）
        valid_diagonals = diagonal_sum % 2 == 0
        # diagonal_indices: 对角线索引h = (i+j)/2
        diagonal_indices = diagonal_sum // 2
        # valid_range: 确保对角线索引不超过H的范围
        valid_range = diagonal_indices < H
        # valid_mask: 综合所有条件，确定哪些位置是有效的对角线点
        valid_mask = valid_diagonals & valid_range
        
        # 矢量化操作：一次性找到所有有效的对角线位置
        # valid_i, valid_j: 所有有效位置的坐标
        valid_i, valid_j = torch.where(valid_mask)
        # valid_h: 所有有效位置对应的对角线索引
        valid_h = diagonal_indices[valid_i, valid_j]
        
        # 批量提取特征和应用掩码
        # 检查是否有有效的对角线位置
        if len(valid_i) > 0:
            # 为每个有效位置提取特征和应用掩码
            # 遍历所有有效的对角线位置
            for idx in range(len(valid_i)):
                # 获取当前位置的坐标：将tensor转换为python数值
                i, j = valid_i[idx].item(), valid_j[idx].item()
                # 获取对应的对角线索引
                h_idx = valid_h[idx].item()
                
                # 从dia_expand中提取对应对角线的特征
                # dia_expand[:, :, h_idx, j, :] 表示：所有批次、所有通道、第h_idx个对角线、第j列、所有深度
                diagonal_features = dia_expand[:, :, h_idx, j, :]  # 结果形状：[Bs, C, D]
                
                # 使用mask来确定哪些位置应该有值
                # dia1_mask[:, :, :, i, j] 表示：所有批次、所有通道、所有深度、第i行、第j列的掩码
                mask_slice = dia1_mask[:, :, :, i, j]  # 结果形状：[Bs, C, D]
                
                # 将特征值分配到对应的3D位置
                # 只有在mask为1的位置才会有值，其他位置保持0
                dia_rec[:, :, :, i, j] = diagonal_features * mask_slice
        
        # 返回重新投影后的3D特征
        return dia_rec

    def decouple_loss(self, feature_map):
        # * input [bs,3,c,h,w] 3: xy,xz,yz path feature map
        bs, path_num, c_num, _, _ = feature_map.shape
        contrast_loss = 0.0
        i1, i2 = 0, 0
        for i in range(path_num):
            pos_vec = feature_map[:, i, ...]  # * shape:[bs,c,h,w]
            for j in range(4, c_num // 2 - 1, 16):
                i1 += 1
                data_list = []
                data_list.append(pos_vec[:, j, ...])
                data_list.append(pos_vec[:, j + c_num // 4, ...])
                # * 放了两个元素,每个元素是取出来的一个channel,shape[bs,8]
                neg_list = [feature_map[:, k, ...][:, j - 4:j + 5, ...] for k in range(path_num) if k != i]
                # * 取出来 9 个相邻的通道
                data_list.extend(neg_list[idx][:, m, ...] for idx in range(path_num - 1) for m in range(neg_list[0].shape[1]))
                contrast_loss += self.one_modality_loss(data_list, t=0.07)
        return contrast_loss / (i1 + i2)

    def one_modality_loss(self, data_list, t=0.07):
        pos_score = self.score(data_list[0], data_list[1], t)
        all_score = 0.0
        for i in range(1, len(data_list)):
            all_score += self.score(data_list[0], data_list[i], t)
        contrast = -torch.log(pos_score / all_score + 1e-5).mean()

        return contrast

    def score(self, fm_1, fm_2, t):
        '''
        t: temperature
        '''
        # if torch.norm(fm_1, dim=1).mean().item() <= 0.001 or torch.norm(fm_2, dim=1).mean().item() <= 0.001:
        #     print(torch.norm(fm_1, dim=1).mean().item(), torch.norm(fm_2, dim=1).mean().item())

        return torch.exp((fm_1 * fm_2).sum(1) / (t * (torch.norm(fm_1, dim=1) * torch.norm(fm_2, dim=1)) + 1e-5))

class SeperateSegNet(nn.Module):

    def __init__(self, in_channels=1,out_channels=1, init_features=32):
        super().__init__()
        self.init_features = init_features
        self.encoder = StdcEncoder(in_channels, init_features, seperate=True)
        self.seg_path = UnetDecoder(init_features, out_channels, dim='3d')

    def forward(self, x):
        out_3d_list = self.encoder(x)
        out,out_decs = self.seg_path(out_3d_list)
        return out, out_decs


if __name__ == "__main__":
    # import os
    # from torchsummary import summary

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(4, 1, 64, 64, 64).to(device)

    # net = ProjectionSegNet()
    # net.cuda()
    # out, p_out, m_out, out_decs, t_loss = net(x)
    # print(t_loss)
    # mask=torch.where(dec4>0)
    # print(out.shape)

    # # onnx_path = './saved_model.onnx'
    # # out, out16, out32 = net(in_ten)
    # # torch.onnx.export(net, in_ten, onnx_path)
    # # netron.start(onnx_path)
    # # torch.save(net.state_dict(), 'STDCNet813.pth')
    # summary(net, (1, 64, 64, 64))
    # import torch
    # a = torch.randn(2, 3, 3)
    # #* help generate a [2,3,3] torch tensor
    # b = a * True
    # print(b)
    model = ProjectionSegNet()
    model.train()
    x = torch.randn(4, 1, 64, 64, 64)
    out, out_3d, m_out, out_decs, t_loss = model(x)
    print(out.shape)
    print(out_3d.shape)
    # print(m_out.shape)
    # print(out_decs.shape)
    # print(t_loss)