import os
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import torch.distributed as dist
import torchio as tio
from torchio.transforms import (
    ZNormalization, )
from tqdm import tqdm
from torchvision import utils
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from process_input import process_x, process_gt
import numpy as np
from logger import create_logger
from utils import yaml_read
from utils.conf_base import Default_Conf
from rich.progress import Progress, TextColumn, MofNCompleteColumn, BarColumn, TimeRemainingColumn
import hydra
from accelerate import Accelerator
import logging
from rich.logging import RichHandler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # ! solve warning


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--conf_path', type=str, help='conf_path')
    parser.add_argument('--gpus', type=str, help='use which gpu')
    parser.add_argument('--epochs', type=int, help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, help='batch-size')
    parser.add_argument('--network', type=str, help='decide which network to use')
    parser.add_argument("--init_lr", type=float, help="learning rate")
    parser.add_argument("--load_mode", type=int, help="decide how to load model")
    parser.add_argument('-k', "--ckpt", type=str, help="path to the checkpoints to resume training")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument('--aug', action='store_true', help='data augmentation')
    parser.add_argument('--save_arch', type=str, help="save arch")
    parser.add_argument('--file_name', type=str, default=os.path.basename(__file__).split('.')[0], help='file name')

    parser.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


def predict(model, conf, logger):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = conf.cudnn_enabled
    torch.backends.cudnn.benchmark = conf.cudnn_benchmark
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * load model
    assert type(conf.ckpt) == str, "You must specify the checkpoint path"
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.rank], output_device=conf.rank)
    logger.info(f"load model from:{conf.ckpt}")
    ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    # model.cuda()
    model.eval()

    # * load datasetBs
    from dataloader import Dataset
    dataset = Dataset(conf).subjects  # ! notice in predict.py should use Dataset(conf).subjects
    znorm = ZNormalization()
    file_tqdm = progress.add_task("[red]Predicting file", total=len(dataset))

    dice_ls, cldice_ls, precision_ls, recall_ls, hs95_ls = [], [], [], [], []
    dice_ls_3d, cldice_ls_3d, precision_ls_3d, recall_ls_3d, hs95_ls_3d = [], [], [], [], []
    # b0_gt_ls, b1_gt_ls, b2_gt_ls, b0_pred_ls, b1_pred_ls, b2_pred_ls = [], [], [], [], [], []
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    # start progess
    progress.start()
    for i, item in enumerate(dataset):
        progress.update(file_tqdm, completed=i + 1)
        item = znorm(item)
        grid_sampler = tio.inference.GridSampler(item, patch_size=(conf.patch_size), patch_overlap=(4, 4, 36))
        affine = item['source']['affine']
        spacing = item.spacing
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=True)

        if conf.batch_size != 1:
            logger.info('in this version, batch_size must be set to 1. automatically change batch_size to 1')
            conf.batch_size = 1

        # patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=conf.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        #    sampler=dist_sampler)
        patch_loader = tio.SubjectsLoader(
            dataset=grid_sampler,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        if i == 0:
            batch_tqdm = progress.add_task("[blue]file batch", total=len(patch_loader))
        else:
            progress.reset(batch_tqdm, total=len(patch_loader))

        pred_aggregator = tio.inference.GridAggregator(grid_sampler)
        pred3d_aggregator = tio.inference.GridAggregator(grid_sampler)
        gt_aggregator = tio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for j, batch in enumerate(patch_loader):
                progress.update(batch_tqdm, completed=j + 1)
                locations = batch[tio.LOCATION]

                x = batch['source'][tio.DATA]
                gt = batch['gt'][tio.DATA]

                x = x.type(torch.FloatTensor).cuda()
                gt = gt.type(torch.FloatTensor).cuda()

                # if conf.network == "stdc" and conf.see_3d:
                #     pred, pred_3d = model(x)
                #     mask_3d = torch.sigmoid(pred_3d.clone())
                #     mask_3d[mask_3d > 0.5] = 1
                #     mask_3d[mask_3d <= 0.5] = 0
                #     pred3d_aggregator.add_batch(mask_3d, locations)

                # else:
                #     pred= model(x)
                if conf.network == 'ei-seg-ablation' and conf.Ablation_setting == '3d_only':
                        pred, out_decs = model(x)
                    
                elif conf.network == 'ei-seg-ablation' and conf.Ablation_setting == '2d_only':
                        pred = model(x)
                    
                elif conf.network == 'ei-seg' or conf.network == 'ei-seg-ablation':
                        pred, pred_3d = model(x)
                        # # * unpack decs
                        # m_dec4, m_dec3, m_dec2, m_dec1 = m_outputs
                        # seg_dec4, seg_dec3, seg_dec2, seg_dec1 = seg_m_outputs
                        # if conf.upper_0:
                        #     seg_dec4 = seg_dec4 * (m_dec4 > 0)
                        #     seg_dec3 = seg_dec3 * (m_dec3 > 0)
                        #     seg_dec2 = seg_dec2 * (m_dec2 > 0)
                        #     seg_dec1 = seg_dec1 * (m_dec1 > 0)
                else:
                        pred = model(x)

                # mask = torch.sigmoid(pred.clone())
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0
                mask = pred.clone().argmax(dim=1, keepdim=True)
                mask_3d = pred_3d.clone().argmax(dim=1, keepdim=True)

                pred_aggregator.add_batch(mask, locations)
                gt_aggregator.add_batch(gt, locations)
                pred3d_aggregator.add_batch(mask_3d, locations)
                progress.refresh()
            
            pred_t = pred_aggregator.get_output_tensor()
            gt_t = gt_aggregator.get_output_tensor()
            if conf.network == "ei-seg" and conf.see_3d:
                pred3d_t = pred3d_aggregator.get_output_tensor()
                dice_3d, cl_dice_3d, precision_3d, recall_3d, hs95_3d = metric(gt_t, pred3d_t, spacing)
                dice_ls_3d.append(dice_3d)
                cldice_ls_3d.append(cl_dice_3d)
                precision_ls_3d.append(precision_3d)
                recall_ls_3d.append(recall_3d)
                hs95_ls_3d.append(hs95_3d)

            # * save pred mhd file
            save_mhd(conf, pred_t, affine, i)

            # * calculate metrics
            dice, cl_dice, precision, recall, hs95, = metric(gt_t, pred_t, spacing)
            dice_ls.append(dice)
            cldice_ls.append(cl_dice)
            precision_ls.append(precision)
            recall_ls.append(recall)
            hs95_ls.append(hs95)
            # b0_gt, b1_gt, b2_gt = b_gt
            # b0_pred, b1_pred, b2_pred = b_pred
            # b0_gt_ls.append(b0_gt)
            # b1_gt_ls.append(b1_gt)
            # b2_gt_ls.append(b2_gt)
            # b0_pred_ls.append(b0_pred)
            # b1_pred_ls.append(b1_pred)
            # b2_pred_ls.append(b2_pred)
            if conf.network == "ei-seg" and conf.see_3d:
                logger.info(f'\ndice:{dice}'
                            f'\ncl_dice:{cl_dice}'
                            f'\nprecision:{precision}'
                            f'\nrecall:{recall}'
                            f'\nhs95:{hs95}'
                            f'\n------ 3d metrics ------'
                            f'\ndice_3d:{dice_3d}'
                            f'\ncl_dice_3d:{cl_dice_3d}'
                            f'\nprecision:{precision_3d}'
                            f'\nrecall:{recall_3d}'
                            f'\nhs95:{hs95_3d}')
            else:
                logger.info(f'\ndice:{dice}'
                            f'\ncl_dice:{cl_dice}'
                            f'\nprecision:{precision}'
                            f'\nrecall:{recall}'
                            f'\nhs95:{hs95}')

    save_csv(conf,dice_ls, cldice_ls, precision_ls, recall_ls, hs95_ls)
    if conf.network == "ei-seg" and conf.see_3d:
        save_csv(conf,dice_ls_3d, cldice_ls_3d, precision_ls_3d, recall_ls_3d, hs95_ls_3d, is_3d=True)
        
    dice_mean = np.mean(dice_ls)
    cl_dice_mean = np.mean(cldice_ls)
    precision_mean = np.mean(precision_ls)
    recall_mean = np.mean(recall_ls)
    hs95_mean = np.mean(hs95_ls)
    # b0_gt_mean = np.mean(b0_gt_ls)
    # b1_gt_mean = np.mean(b1_gt_ls)
    # b2_gt_mean = np.mean(b2_gt_ls)
    # b0_pred_mean = np.mean(b0_pred_ls)
    # b1_pred_mean = np.mean(b1_pred_ls)
    # b2_pred_mean = np.mean(b2_pred_ls)
    dice_mean_3d = np.mean(dice_ls_3d)
    cl_dice_mean_3d = np.mean(cldice_ls_3d)
    precision_mean_3d = np.mean(precision_ls_3d)
    recall_mean_3d = np.mean(recall_ls_3d)
    hs95_mean_3d = np.mean(hs95_ls_3d)
    # print('-' * 40)
    if conf.network == "ei-seg" and conf.see_3d:
        logger.info(
            f'\ndice_mean:{dice_mean}'
            f'\ncl_dice_mean:{cl_dice_mean}'
            f'\nprecision_mean:{precision_mean}'
            f'\nrecall_mean:{recall_mean}'
            f'\nhs95_mean:{hs95_mean}'
            f'\n------ 3d metrics ------'
            f'\ndice_3d_mean:{dice_mean_3d}'
            f'\ncl_dice_3d_mean:{cl_dice_mean_3d}'
            f'\nprecision:_mean{precision_mean_3d}'
            f'\nrecall_mean:{recall_mean_3d}'
            f'\nhs95_mean:{hs95_mean_3d}'
        )


def save_csv(conf,dice_ls, cldice_ls, precision_ls, recall_ls, hs95_ls, is_3d=False):
    import pandas as pd
    data = {'dice': dice_ls, 'cl_dice': cldice_ls, 'precision': precision_ls, 'recall_ls': recall_ls, 'hs95': hs95_ls}
    df = pd.DataFrame(data)
    mean_data = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean(), df.iloc[:, 2].mean(), df.iloc[:, 3].mean(), df.iloc[:, 4].mean()]
    std_data = [df.iloc[:, 0].std(), df.iloc[:, 1].std(), df.iloc[:, 2].std(), df.iloc[:, 3].std(), df.iloc[:, 4].std()]
    df.loc[len(df)] = mean_data
    df.loc[len(df)] = std_data
    if is_3d:
        save_path = os.path.join(conf.hydra_path, 'metrics_3d.csv')
    else:
        save_path = os.path.join(conf.hydra_path, 'metrics.csv')
    df.to_csv(save_path, index=False)


def save_mhd(conf,pred, affine, index):
    save_base = os.path.join(conf.hydra_path, 'pred_file')
    os.makedirs(save_base, exist_ok=True)
    pred_data = tio.ScalarImage(tensor=pred, affine=affine)
    pred_data.save(os.path.join(save_base, f'pred-{index:04d}.mhd'))

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(conf):
    conf = conf["config"]
    if conf.predict.cycle_val:
        raise NotImplementedError("Cycle val is not implemented yet, Please use predict_cycleval.py to predict the model")
    os.makedirs(conf.hydra_path, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu

    # #* distributed training
    # dist.init_process_group(backend='nccl')
    # conf.rank = dist.get_rank()
    # torch.cuda.set_device(conf.rank)
    # device = torch.device('cuda', conf.rank)
    if type(conf.patch_size) == str:
        assert (len(conf.patch_size.split(",")) <= 3), f'patch size can only be one str or three str but got {len(conf.patch_size.split(","))}'
        if len(conf.patch_size.split(",")) == 3:
            conf.patch_size = tuple(map(int, conf.patch_size.split(",")))
        else:
            conf.patch_size = int(conf.patch_size)

    # # * model selection
    # if conf.network == "res_unet":
    #     from models.three_d.residual_unet3d import UNet
    #     model = UNet(in_channels=conf.in_classes, n_classes=conf.out_classes, base_n_filter=32)
    # elif conf.network == "unet":
    #     from models.three_d.unet3d import UNet3D  # * 3d unet
    #     model = UNet3D(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == "er_net":
    #     from models.three_d.ER_net import ER_Net
    #     model = ER_Net(classes=conf.out_classes, channels=conf.in_classes)
    # elif conf.network == "re_net":
    #     from models.three_d.RE_net import RE_Net
    #     model = RE_Net(classes=conf.out_classes, channels=conf.in_classes)
    # elif conf.network == 'csrnet':
    #     from models.three_d.csrnet import CSRNet
    #     model = CSRNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == 'vtnet':
    #     from models.three_d.vtnet import VTUNet
    #     model = VTUNet(num_classes=conf.out_classes, input_dim=conf.in_classes, zero_head=conf.zero_head, embed_dim=conf.embed_dim, win_size=conf.win_size)
    # elif conf.network == 'unetr':
    #     from models.three_d.unetr import UNETR
    #     model = UNETR(img_shape=conf.patch_size, input_dim=conf.in_classes, output_dim=conf.out_classes,
    #                   embed_dim=conf.embed_dim, patch_size=conf.unetr_patch_size, num_heads=conf.num_heads, dropout=conf.dropout)
    # elif conf.network == "stdc":
    #     from models.three_d.stdc_unet import StdcUnet
    #     model = StdcUnet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == "stdc2d":
    #     from models.three_d.stdc2d import ProjectionNet
    #     model = ProjectionNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == "stdc2d-seg":
    #     from models.three_d.stdc2d_v2 import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == "stdc3d-only":
    #     from models.three_d.stdc2d_v2 import SeperateSegNet
    #     model = SeperateSegNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == 'vnet':
    #     from models.three_d.vnet3d import VNet
    #     model = VNet(in_channels=conf.in_classes, classes=conf.out_classes)
    # elif conf.network == 'abalation_no_shared':
    #     from models.three_d.abalation_no_shared import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    # elif conf.network == 'ablation_2d':
    #     from models.three_d.ablation_2d_only import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=conf.in_classes, out_channels=conf.out_classes, init_features=32)
    
    
    from utils.select_network import select_network
    model = select_network(conf)
    # * create logger
    logger = get_logger(conf)
    info = '\nParameter Settings:\n'
    for k, v in conf.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    predict(model, conf, logger)
    logger.info(f'tensorboard file saved in:{conf.hydra_path}')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    # parser = parse_training_args(parser)
    # args, _ = parser.parse_known_args()
    # args = parser.parse_args()
    # args_dict = vars(args)

    # conf_path = args.conf_path
    # conf = Default_Conf()
    # conf.update(yaml_read(conf_path))
    # conf.update_from_args(args_dict)
    main()
