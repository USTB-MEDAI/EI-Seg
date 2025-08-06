import argparse
import os
import time
from glob import glob
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import imageio
import torch
import torch.distributed as dist
from timm.utils import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from logger import create_logger
from process_input import process_gt, process_x
from utils import yaml_read
from utils.conf_base import Default_Conf
from utils.metric import metric
import hydra
import logging
from rich.logging import RichHandler
from accelerate import Accelerator
import torchio as tio
from rich.progress import Progress, TextColumn, MofNCompleteColumn, BarColumn, TimeRemainingColumn
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning

BLUE = "\033[94m"
YELLOW = "\033[93m"
END = "\033[0m"


def weights_init_normal(init_type):

    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02
        # init_type = conf.init_type

        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


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

def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--conf_path", type=str, help="conf_path")
    parser.add_argument("--gpus", type=str, help="use which gpu")
    parser.add_argument("--epochs", type=int, help="Number of total epochs to run")
    parser.add_argument("--batch_size", type=int, help="batch-size")
    parser.add_argument("--network", type=str, help="decide which network to use")
    parser.add_argument("--init_lr", type=float, help="learning rate")
    parser.add_argument("--load_mode", type=int, help="decide how to load model")
    parser.add_argument("-k", "--ckpt", type=str, help="path to the checkpoints to resume training")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument("--aug", action="store_true", help="data augmentation")
    parser.add_argument("--save_arch", type=str, help="save arch")
    parser.add_argument(
        "--file_name",
        type=str,
        default=os.path.basename(__file__).split(".")[0],
        help="file name",
    )

    parser.add_argument("--cudnn-enabled", default=True, help="Enable cudnn")
    parser.add_argument("--cudnn-benchmark", default=True, help="Run cudnn benchmark")

    return parser


def train(model, config, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # * init averageMeter
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    recall_meter = AverageMeter()
    spe_meter = AverageMeter()  # * specificity meter  (avoid too long variable name)
    max_memory = AverageMeter()

    # * set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    # * set loss function
    from loss_function import Binary_Loss, DiceLoss, cross_entropy_3D

    criterion = Binary_Loss().cuda()
    seg_criterion = DiceLoss().cuda()
    dec4_criterion = torch.nn.MSELoss().cuda()
    dec3_criterion = torch.nn.MSELoss().cuda()
    dec2_criterion = torch.nn.MSELoss().cuda()
    dec1_criterion = torch.nn.MSELoss().cuda()

    # * set scheduler strategy
    if config.use_scheduler:
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # * load model
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[config.rank],
    #     output_device=config.rank,
    #     find_unused_parameters=True,
    # )
    if config.load_mode == 1:  # * load weights from checkpoint
        logger.info(f"load model from: {os.path.join(config.ckpt, config.latest_checkpoint_file)}")
        ckpt = torch.load(
            os.path.join(config.ckpt, config.latest_checkpoint_file),
            map_location=lambda storage, loc: storage,
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
        # elapsed_epochs = 0
    else:
        elapsed_epochs = 0

    model.train()

    # * tensorboard writer
    writer = SummaryWriter(config.hydra_path)

    # * load datasetBs
    from dataloader import Dataset

    train_dataset = Dataset(config)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset.queue_dataset, shuffle=True)
    # #! in distributed training, the 'shuffle' must be false!
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset.queue_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True,
    #     sampler=train_sampler,
    # )
    train_loader = tio.SubjectsLoader(
        train_dataset.queue_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    
    accelerator = Accelerator()
    # * accelerate prepare
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)
    
    epochs = config.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )
    
    epoch_tqdm = progress.add_task(description="[red]epoch progress", total=epochs)
    batch_tqdm = progress.add_task(description="[blue]batch progress", total=len(train_loader))

    progress.start()
    for epoch in range(1, epochs + 1):
        progress.update(epoch_tqdm, completed=epoch)
        epoch += elapsed_epochs

        num_iters = 0

        load_meter = AverageMeter()
        train_time = AverageMeter()
        load_start = time.time()  # * initialize

        # train_loader.sampler.set_epoch(epoch)  # ! must set epoch for DistributedSampler!
        for i, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                progress.update(batch_tqdm, completed=i + 1)
                train_start = time.time()
                load_time = time.time() - load_start
                optimizer.zero_grad()

                # x = process_x(config, batch)  # * from batch extract x:[bs,4 or 1,h,w,d]
                # gt = process_gt(config, batch)  # * from batch extract gt:[bs,4 or 1,h,w,d]
                x = batch['source'][tio.DATA]
                gt = batch['gt'][tio.DATA]
                
                gt_back = torch.zeros_like(gt)
                gt_back[gt == 0] = 1
                gt = torch.cat([gt_back, gt], dim=1)  # * [bs,2,h,w,d]

                x = x.type(torch.FloatTensor).cuda()
                gt = gt.type(torch.FloatTensor).cuda()

                # if config.network == 'stdc2d-seg' or config.network == 'abalation_no_shared':
                #     pred, seg_pred, m_outputs, seg_m_outputs, decouple_loss = model(x)
                #     # * unpack decs
                #     m_dec4, m_dec3, m_dec2, m_dec1 = m_outputs
                #     seg_dec4, seg_dec3, seg_dec2, seg_dec1 = seg_m_outputs
                #     if config.upper_0:
                #         seg_dec4 = seg_dec4 * (m_dec4 > 0)
                #         seg_dec3 = seg_dec3 * (m_dec3 > 0)
                #         seg_dec2 = seg_dec2 * (m_dec2 > 0)
                #         seg_dec1 = seg_dec1 * (m_dec1 > 0)
                # else:
                #     pred = model(x)
                
                if config.network == 'ei-seg-ablation' and config.Ablation_setting == '3d_only':
                    pred, out_decs = model(x)
                
                elif config.network == 'ei-seg-ablation' and config.Ablation_setting == '2d_only':
                    pred = model(x)
                
                elif config.network == 'ei-seg' or config.network == 'ei-seg-ablation':
                    pred, seg_pred, m_outputs, seg_m_outputs, decouple_loss = model(x)
                    # * unpack decs
                    m_dec4, m_dec3, m_dec2, m_dec1 = m_outputs
                    seg_dec4, seg_dec3, seg_dec2, seg_dec1 = seg_m_outputs
                    if config.upper_0:
                        seg_dec4 = seg_dec4 * (m_dec4 > 0)
                        seg_dec3 = seg_dec3 * (m_dec3 > 0)
                        seg_dec2 = seg_dec2 * (m_dec2 > 0)
                        seg_dec1 = seg_dec1 * (m_dec1 > 0)
                else:
                    pred = model(x)
                    
                mask = pred.argmax(dim=1, keepdim=True)  # * [bs,1,h,w,d]

                # *  pred -> mask (0 or 1)
                # mask = torch.sigmoid(pred.clone())  # TODO should use softmax, because it returns two probability (sum = 1)
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0
                if config.network == 'ei-seg-ablation' and config.Ablation_setting == '3d_only':
                    loss = criterion(pred, gt)
                
                elif config.network == 'ei-seg-ablation' and config.Ablation_setting == '2d_only':
                    loss = criterion(pred, gt)

                elif config.network == 'ei-seg' or config.network == 'ei-seg-ablation':
                    seg_2dloss = criterion(pred, gt) + seg_criterion(pred, gt)
                    seg_3dloss = criterion(seg_pred, gt) + seg_criterion(seg_pred, gt)
                    dec4_loss = dec4_criterion(m_dec4, seg_dec4)
                    dec3_loss = dec3_criterion(m_dec3, seg_dec3)  # 循环loss
                    dec2_loss = dec2_criterion(m_dec2, seg_dec2)
                    dec1_loss = dec1_criterion(m_dec1, seg_dec1)
                    
                    cycle_loss = dec4_loss + dec3_loss + dec2_loss + dec1_loss
                    loss = seg_2dloss + seg_3dloss
                    loss += config.decp_loss_lamda * decouple_loss + config.cycle_loss_lamda * cycle_loss
                    # if config.network == 'ei-seg-ablation' and config.Ablation_setting == 'w/o_decp_loss':
                    #     loss = seg_2dloss + seg_3dloss + dec4_loss + dec3_loss + dec2_loss + dec1_loss
                    # if config.network == 'ei-seg-ablation' and config.Ablation_setting == 'w/o_cycle_loss':
                    #     loss = seg_2dloss + seg_3dloss + decouple_loss
                else:
                    # loss = criterion(pred, gt) + seg_criterion(pred, gt)
                    loss = criterion(pred, gt)
                accelerator.backward(loss)
                progress.refresh()

            optimizer.step()

            num_iters += 1
            iteration += 1

            # * calculate metrics
            # TODO use reduce to sum up all rank's calculation results
            gt = gt.argmax(dim=1, keepdim=True)
            dice= metric(gt.cpu(), mask.cpu(), 0)
            # dice = dist.all_reduce(dice, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # recall = dist.all_reduce(recall, op=dist.ReduceOp.SUM) / dist.get_world_size()
            # specificity = dist.all_reduce(specificity, op=dist.ReduceOp.SUM) / dist.get_world_size()

            writer.add_scalar("Training/Loss", loss.item(), iteration)
            # writer.add_scalar('Training/recall', recall, iteration)
            # writer.add_scalar('Training/specificity', specificity, iteration)
            writer.add_scalar("Training/dice", dice, iteration)

            # print('lr:' + str(scheduler._last_lr[0]))

            temp_file_base = os.path.join(config.hydra_path, "train_temp")
            os.makedirs(temp_file_base, exist_ok=True)
            # if (i % 20 == 0):
            #     with torch.no_grad():
            #         #! if dataset is brats ,it will automatically save flair modality as nii.gz
            #         if (conf.dataset == 'brats'):
            #             affine = batch['flair']['affine'][0].numpy()
            #             flair_source = tio.ScalarImage(tensor=x[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_source.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-source" + conf.save_arch))
            #             flair_gt = tio.ScalarImage(tensor=gt[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_gt.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-gt" + conf.save_arch))
            #             flair_pred = tio.ScalarImage(tensor=pred[:, 0, :, :, :].cpu().detach().numpy(), affine=affine)
            #             flair_pred.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-pred" + conf.save_arch))
            #         else:
            #             affine = batch['source']['affine'][0].numpy()
            #             source = tio.ScalarImage(tensor=x[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             source.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-source" + conf.save_arch))
            #             gt_data = tio.ScalarImage(tensor=gt[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             gt_data.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-gt" + conf.save_arch))
            #             pred_data = tio.ScalarImage(tensor=pred[0, :, :, :, :].cpu().detach().numpy(), affine=affine)
            #             pred_data.save(os.path.join(temp_file_base, f"epoch-{epoch:04d}-batch-{i:02d}-pred" + conf.save_arch))
            # * record metris
            loss_meter.update(loss.item(), x.size(0))
            dice_meter.update(dice, x.size(0))
            # recall_meter.update(recall, x.size(0))
            # spe_meter.update(specificity, x.size(0))
            train_time.update(time.time() - train_start)
            load_meter.update(load_time)
            # logger.info('batch used time: {:.3f} s\n'.format(batch_time.val))
            logger.info(f"\nEpoch: {epoch} Batch: {i}, data load time: {load_meter.val:.3f}s , train time: {train_time.val:.3f}s\n"
                        f"Loss: {loss_meter.val}\n"
                        f"Dice: {dice_meter.val}\n")
            # f'{BLUE}Recall:{END} {recall_meter.val}\n'
            # f'{BLUE}Specificity:{END} {spe_meter.val}\n')

            load_start = time.time()

        if config.use_scheduler:
            scheduler.step()
            logger.info(f"Learning rate: {scheduler.get_last_lr()[0]}")

        # * one epoch logger
        logger.info(f"\nEpoch {epoch} used time: {load_meter.sum+train_time.sum:.3f} s\n"
                    f"Loss Avg: {loss_meter.avg}\n"
                    f"Dice Avg: {dice_meter.avg}\n")
        # f'{BLUE}Recall Avg:{END} {recall_meter.avg}\n'
        # f'{BLUE}Specificity Avg:{END} {spe_meter.avg}\n')

        # Store latest checkpoint in each epoch
        scheduler_dict = scheduler.state_dict() if config.use_scheduler else None
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler_dict,
                "epoch": epoch,
            },
            os.path.join(config.hydra_path, config.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % config.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                },
                os.path.join(config.hydra_path, f"checkpoint_{epoch:04d}.pt"),
            )
    writer.close()
    
@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    # parser = argparse.ArgumentParser(description="PyTorch Medical Segmentation Training")
    # parser = parse_training_args(parser)
    # args, _ = parser.parse_known_args()
    # args = parser.parse_args()
    # args_dict = vars(args)

    # config_path = args.config_path
    # config = Default_config()
    # config.update(yaml_read(config_path))
    # config.update_from_args(args_dict)
    # print(config)
    config=config["config"]
    
    if config.cycle_val:
        raise NotImplementedError("Cycle validation is not implemented in this code; Please use train_cycleval.py")

    if type(config.patch_size) == str:
        assert (len(config.patch_size.split(",")) <= 3), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)
    # if type(config.img_shape) == str:
    #     assert (len(config.img_shape.split(",")) <= 3), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
    #     if len(config.img_shape.split(",")) == 3:
    #         config.img_shape = tuple(map(int, config.img_shape.split(",")))
    #     else:
    #         config.img_shape = int(config.img_shape)

    os.makedirs(config.hydra_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # # #* distributed training
    # dist.init_process_group(backend="nccl")
    # config.rank = dist.get_rank()
    # torch.cuda.set_device(config.rank)
    # device = torch.device("cuda", config.rank)

    # # * model selection
    # if config.network == "res_unet":
    #     from models.three_d.residual_unet3d import UNet
    #     model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    # elif config.network == "unet":
    #     from models.three_d.unet3d import UNet3D  # * 3d unet
    #     model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == "er_net":
    #     from models.three_d.ER_net import ER_Net
    #     model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    # elif config.network == "re_net":
    #     from models.three_d.RE_net import RE_Net
    #     model = RE_Net(classes=config.out_classes, channels=config.in_classes)
    # elif config.network == 'csrnet':
    #     from models.three_d.csrnet import CSRNet
    #     model = CSRNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == 'vtnet':
    #     from models.three_d.vtnet import VTUNet
    #     model = VTUNet(num_classes=config.out_classes, input_dim=config.in_classes, zero_head=config.zero_head, embed_dim=config.embed_dim, win_size=config.win_size)
    # elif config.network == 'unetr':
    #     from models.three_d.unetr import UNETR
    #     model = UNETR(img_shape=config.img_shape, input_dim=config.in_classes, output_dim=config.out_classes,
    #                   embed_dim=config.embed_dim, patch_size=config.unetr_patch_size, num_heads=config.num_heads, dropout=config.dropout)
    # # elif config.network == "stdc":
    # #     from models.three_d.stdc_unet import StdcUnet
    # #     model = StdcUnet(in_channels=config.out_classes, out_channels=config.in_classes, init_features=32)
    # # elif config.network == "stdc2d":
    # #     from models.three_d.stdc2d import ProjectionNet
    # #     model = ProjectionNet(in_channels=config.out_classes, out_channels=config.in_classes, init_features=32)
    # elif config.network == "ei-seg" or (config.network == "ei-seg-ablation" and (config.Ablation_setting == 'w/o_decp_loss' or config.Ablation_setting == 'w/o_cycle_loss')):
    #     from models.three_d.stdc2d_v2 import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == "ei-seg-ablation" and config.Ablation_setting == '3d-only':
    #     from models.three_d.stdc2d_v2 import SeperateSegNet
    #     model = SeperateSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == 'ei-seg-ablation' and config.Ablation_setting == 'no_shared':
    #     from models.three_d.abalation_no_shared import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == 'ei-seg-ablation' and config.Ablation_setting == '2d_only':
    #     from models.three_d.ablation_2d_only import ProjectionSegNet
    #     model = ProjectionSegNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    # elif config.network == 'vc_net':
    #     from models.three_d.vc_net import get_vc_net
    #     model = get_vc_net()
        
    # elif config.network == 'vnet':
    #     from models.three_d.vnet3d import VNet
    #     model = VNet(in_channels=config.in_classes, classes=config.out_classes)
    
    from utils.select_network import select_network
    model = select_network(config)

    model.apply(weights_init_normal(config.init_type))
    # model = model.to(device)

    # * create logger
    logger = get_logger(config)
    
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    train(model, config, logger)
    logger.info(f"tensorboard file saved in:{config.hydra_path}")
    
if __name__ == "__main__":
    main()
    
