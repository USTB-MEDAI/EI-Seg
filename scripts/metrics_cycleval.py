import torchio as tio
import SimpleITK as sitk
from pathlib import Path
import argparse
import torch
import numpy as np
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
import pandas as pd
from logger import create_logger, metrics_logger
from tqdm import tqdm
from hparam import hparams as hp
from cl_dice import clDice_metric, clDice
from monai.metrics import compute_hausdorff_distance
import glob
from metrics import metric_evaluation

BLUE = '\033[94m'
END = '\033[0m'

# def make_subject(image_paths, label_paths, pred_suffixes, gt_suffixes):
#     images_dir = Path(image_paths)
#     labels_dir = Path(label_paths)

#     image_paths = sorted(images_dir.glob(pred_suffixes))
#     label_paths = sorted(labels_dir.glob(gt_suffixes))
#     subjects = []
#     for (image_path, label_path) in zip(image_paths, label_paths):
#         print(image_path, label_path)
#         subject = tio.Subject(
#             pred=tio.ScalarImage(image_path),
#             gt=tio.LabelMap(label_path),
#         )
#         subjects.append(subject)
#     return subjects


def save_csv(data, path):
    df = pd.DataFrame(data)
    # 添加列名
    df.columns = ["hs95", "dice", "cl_dice", "score"]
    means = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean(), df.iloc[:, 2].mean(), df.iloc[:, 3].mean()]
    stds = [df.iloc[:, 0].std(), df.iloc[:, 1].std(), df.iloc[:, 2].std(), df.iloc[:, 3].std()]
    df.loc[len(df)] = means
    df.loc[len(df)] = stds
    for i,(m,s) in enumerate(zip(means, stds)):
        print(f"{df.columns[i]}: {m:.4f} ± {s:.4f}")
    # 保存为csv文件
    df.to_csv(path, encoding='utf-8')


def metric_cycleval(predict_dir, labels_dir, logger, pred_suffixes="*.mhd", gt_suffixes="*.nii.gz"):
    fold_dir = sorted(glob.glob(os.path.join(predict_dir, '*')))
    all_data = []
    for i, predict_dir in enumerate(fold_dir):

        Data, mean_dice = metric_evaluation(predict_dir, labels_dir, None, '*.mhd', '*.nii.gz', i)
        
        all_data.append(Data)
        
    return all_data


# def soft_cldice(y_true, y_pred, iters=2, alpha=0.5, smooth=1.):

#     skel_pred = soft_skel(y_pred, iters)
#     skel_true = soft_skel(y_true, iters)
#     tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) +
#              smooth) / (torch.sum(skel_pred[:, 1:, ...]) + smooth)
#     tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) +
#              smooth) / (torch.sum(skel_true[:, 1:, ...]) + smooth)
#     cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
#     return cl_dice


if __name__ == '__main__':
    
    pred_dir = '/disk/cc/STDC-Backbone-new/logs/csrnet/predict_cycleval-2025-06-30/12-14-01/pred_file'
    gt_dir = '/home/cc/tof_data_nii/test/label1_nii/'
    metric_save_path = pred_dir
    # output_dir = './logs/test_unet'
    # hp = hp()
    # parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    # parser = parse_training_args(parser)
    # args, _ = parser.parse_known_args()
    # args = parser.parse_args()
    # os.makedirs(args.output_dir, exist_ok=True)

    # logger = metrics_logger(output_dir=output_dir, name='Evaluate Metrics')

    

    # skel_path = os.path.join(args.output_dir, "skel")
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metric_save_path, exist_ok=True)
    # os.makedirs(skel_path, exist_ok=True)

    # logger.info(f'\nPrediction results: {pred_dir}\n'
    #             f'Ground truth: {gt_dir}\n'
    #             f'Metric saved in: {metric_save_path}\n')
    print(f'\nPrediction results: {pred_dir}\n'
                f'Ground truth: {gt_dir}\n'
                f'Metric saved in: {metric_save_path}\n')

    # metric and save csv
    all_data = metric_cycleval(pred_dir, gt_dir, None, '*.mhd', '*.nii.gz')
    # csv_path = os.path.join(metric_save_path, 'metric_evaluation.csv')
    for i, Data in enumerate(all_data):
        csv_path = os.path.join(metric_save_path, f'metric_evaluation_{i}.csv')
        save_csv(Data, csv_path)
    # logger.info(f'\n{BLUE}Mean dice{END}: {mean_dice}\n'
    #             f'{BLUE}Metric saved in{END}: {csv_path}\n')

    # save hparam to json
    # with open(os.path.join(metric_dir, 'hparam.json'), 'w') as f:
    #     json.dump(hp.__dict__, f, indent=4)
    # print("hparam.json saveed at", os.path.join(metric_dir, 'hparam.json'))
