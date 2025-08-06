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

BLUE = '\033[94m'
END = '\033[0m'

def make_subject(image_paths, label_paths, pred_suffixes, gt_suffixes):
    images_dir = Path(image_paths)
    labels_dir = Path(label_paths)

    image_paths = sorted(images_dir.glob(pred_suffixes))
    label_paths = sorted(labels_dir.glob(gt_suffixes))
    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects


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
    
    


def metric_evaluation(predict_dir, labels_dir, logger, pred_suffixes="*.mhd", gt_suffixes="*.mhd", fold_num=None):

    subjects = make_subject(predict_dir, labels_dir, pred_suffixes, gt_suffixes)

    training_set = tio.SubjectsDataset(subjects)

    dice_arr = []
    Data = []

    for i, subj in tqdm(enumerate(training_set._subjects), total=len(training_set._subjects),
                        desc="Generating metrics"if fold_num is None else f"Fold{fold_num}: Generating metrics"):
        data = []
        gt = subj['gt'][tio.DATA].squeeze()
        spacing = subj.spacing

        # subj = toc(subj)
        pred = subj['pred'][tio.DATA].squeeze()  # .permute(0,1,3,2)

        # preds.append(pred)
        # gts.append(gt)

        preds = pred.numpy()
        gts = gt.numpy()

        pred = preds.astype(int)  # float data does not support bit_and and bit_or
        gdth = gts.astype(int)  # float data does not support bit_and and bit_or

        # ========= hs 95 =========
        hs_pred = pred[None, None, :]
        hs_gt = gdth[None, None, :]
        hs95 = compute_hausdorff_distance(hs_pred, hs_gt, percentile=95, spacing=spacing)
        hs95 = hs95.detach().numpy().squeeze()

        fp_array = copy.deepcopy(pred)  # keep pred unchanged
        fn_array = copy.deepcopy(gdth)
        gdth_sum = np.sum(gdth)
        pred_sum = np.sum(pred)

        intersection = gdth & pred  # only both 1 will be 1
        union = gdth | pred
        intersection_sum = np.count_nonzero(intersection)  # sum of nonzero elements
        union_sum = np.count_nonzero(union)  #

        tp_array = intersection

        tmp = pred - gdth
        fp_array[tmp < 1] = 0  # fp : false positive

        tmp2 = gdth - pred
        fn_array[tmp2 < 1] = 0

        tn_array = np.ones(gdth.shape) - union

        tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

        smooth = 0.001
        # precision = tp / (pred_sum + smooth)
        # recall = tp / (gdth_sum + smooth)
        cl_dice, skel_pred, skel_gt = clDice_metric(preds, gts)

        #! save skel
        # img_skel_pred=sitk.GetImageFromArray(skel_pred)
        # img_skel_gt=sitk.GetImageFromArray(skel_gt)

        # img_skel_pred.SetSpacing([1.0, 1.0, 1.0])
        # img_skel_gt.SetSpacing([1.0, 1.0, 1.0])

        # img_skel_pred.SetOrigin([0.0, 0.0, 0.0])
        # img_skel_gt.SetOrigin([0.0, 0.0, 0.0])

        # sitk.WriteImage(img_skel_pred,os.path.join(skel_path,f"skel_pred_{i}.mhd"))
        # sitk.WriteImage(img_skel_gt,os.path.join(skel_path,f"skel_gt_{i}.mhd"))

        # img_skel_pred.SetPixelID(sitk.sitkFloat64)
        # img_skel_gt.SetPixelID(sitk.sitkFloat64)

        # img_skel_pred.SetRegions([skel_pred.shape[2], skel_pred.shape[1], skel_pred.shape[0]])
        # img_skel_gt.SetRegions([skel_gt.shape[2], skel_gt.shape[1], skel_gt.shape[0]])

        # false_positive_rate = fp / (fp + tn + smooth)
        # false_negative_rate = fn / (fn + tp + smooth)

        jaccard = intersection_sum / (union_sum + smooth)
        dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

        # logger.info(  # f'\nprecision: {precision}\n'
        #     # f'recall: {recall}\n'
        #     # f'false_positive_rate: {false_positive_rate}\n'
        #     # f'false_negative_rate: {false_negative_rate}\n'
        #     f'\nhs95: {hs95}\n'
        #     f'cl_dice: {cl_dice}\n'
        #     f'dice: {dice}\n')
        print(f'\nhs95: {hs95}\n'
            f'cl_dice: {cl_dice}\n'
            f'dice: {dice}\n')
        dice_arr.append(dice)
        # data.append(precision)
        # data.append(recall)
        # data.append(false_positive_rate)
        # data.append(false_negative_rate)
        data.append(hs95)
        data.append(dice)
        data.append(cl_dice)
        data.append((hs95+dice+cl_dice)/3)
        Data.append(data)
    return Data, np.mean(dice_arr)


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
    
    pred_dir = '/disk/cc/STDC-Backbone-new/prediction/norm_aug/IXI/csrnet/pred_file'
    gt_dir = '/home/cc/tof_data_nii/test/label1_nii/'
    metric_save_path = './csv_temp'
    output_dir = './logs/test_unet'
    # hp = hp()
    # parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    # parser = parse_training_args(parser)
    # args, _ = parser.parse_known_args()
    # args = parser.parse_args()
    # os.makedirs(args.output_dir, exist_ok=True)

    # logger = metrics_logger(output_dir=output_dir, name='Evaluate Metrics')

    

    # skel_path = os.path.join(args.output_dir, "skel")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metric_save_path, exist_ok=True)
    # os.makedirs(skel_path, exist_ok=True)

    # logger.info(f'\nPrediction results: {pred_dir}\n'
    #             f'Ground truth: {gt_dir}\n'
    #             f'Metric saved in: {metric_save_path}\n')
    print(f'\nPrediction results: {pred_dir}\n'
                f'Ground truth: {gt_dir}\n'
                f'Metric saved in: {metric_save_path}\n')

    # metric and save csv
    Data, mean_dice = metric_evaluation(pred_dir, gt_dir, None, '*.mhd', '*.nii.gz')
    csv_path = os.path.join(metric_save_path, 'metric_evaluation.csv')
    save_csv(Data, csv_path)
    # logger.info(f'\n{BLUE}Mean dice{END}: {mean_dice}\n'
    #             f'{BLUE}Metric saved in{END}: {csv_path}\n')

    # save hparam to json
    # with open(os.path.join(metric_dir, 'hparam.json'), 'w') as f:
    #     json.dump(hp.__dict__, f, indent=4)
    # print("hparam.json saveed at", os.path.join(metric_dir, 'hparam.json'))
