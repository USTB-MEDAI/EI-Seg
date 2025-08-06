def select_network(config):
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet
        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet
        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net
        model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "re_net":
        from models.three_d.RE_net import RE_Net
        model = RE_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == 'csrnet':
        from models.three_d.csrnet import CSRNet
        model = CSRNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == 'vtnet':
        from models.three_d.vtnet import VTUNet
        model = VTUNet(num_classes=config.out_classes, input_dim=config.in_classes, zero_head=config.zero_head, embed_dim=config.embed_dim, win_size=config.win_size)
    elif config.network == 'unetr':
        from models.three_d.unetr import UNETR
        model = UNETR(input_dim=config.in_classes, output_dim=config.out_classes)
    elif config.network == "ei-seg" or (config.network == "ei-seg-ablation" and (config.Ablation_setting == 'w/o_decp_loss' or config.Ablation_setting == 'w/o_cycle_loss')):
        from models.three_d.stdc2d_v2 import ProjectionSegNet
        model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32, plain_num=config.plain_num if 'plain_num' in config else 3)
    elif config.network == "ei-seg-ablation" and config.Ablation_setting == 'w/o_stdc':
        from models.three_d.stdc2d_v2_no_stdc import ProjectionSegNet
        model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "ei-seg-ablation" and config.Ablation_setting == '3d_only_w/o_stdc':
        from models.three_d.stdc2d_v2_no_stdc import SeperateSegNet
        model = SeperateSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "ei-seg-ablation" and config.Ablation_setting == '3d_only':
        from models.three_d.stdc2d_v2 import SeperateSegNet
        model = SeperateSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == 'ei-seg-ablation' and config.Ablation_setting == 'no_shared':
        from models.three_d.abalation_no_shared import ProjectionSegNet
        model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == 'ei-seg-ablation' and config.Ablation_setting == '2d_only':
        from models.three_d.ablation_2d_only import ProjectionSegNet
        model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == 'vc_net':
        from models.three_d.vc_net import get_vc_net
        model = get_vc_net()
        
    elif config.network == 'vnet':
        from models.three_d.vnet3d import VNet
        model = VNet(in_channels=config.in_classes, classes=config.out_classes)
    return model