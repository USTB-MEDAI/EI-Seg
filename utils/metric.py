import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from monai.metrics import compute_hausdorff_distance
from cl_dice import clDice_metric
from utils.betti_numbers import BinaryBettiCalculator


def all_metric(gt, wt_pred, et_pred, tc_pred):
    wt_dice, wt_recall, wt_specificity, wt_hs95 = metric(gt[0], wt_pred)
    et_dice, et_recall, et_specificity, et_hs95 = metric(gt[1], et_pred)
    tc_dice, tc_recall, tc_specificity, tc_hs95 = metric(gt[2], tc_pred)
    return [wt_dice, wt_recall, wt_specificity, wt_hs95], [et_dice, et_recall, et_specificity, et_hs95], [tc_dice, tc_recall, tc_specificity, tc_hs95]


def metric(gt, pred, *args):
    # * input shape: (batch, channel, height, width)

    spacing = args[0]

    # gt = gt.squeeze()  # (240,240)
    # pred = pred.squeeze()  # (240,240)

    preds = pred.detach().numpy()
    gts = gt.detach().numpy()

    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    if spacing:
        hs_pred = pred[None, :, :, :, :]
        hs_gd = gdth[None, :, :, :, :]
        hs95 = compute_hausdorff_distance(hs_pred, hs_gd, percentile=95, spacing=spacing).numpy()[0][0]
        
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)
    specificity = tn / (tn + fp + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)
    try:
        cl_dice, _, _ = clDice_metric(preds, gts)
    except:
        cl_dice = 0

    # hs95 = hausdorff_95(gdth, pred, (1, 1))
    # hs95 = hausdorff_95(preds, gts, (1, 1, 1))
    # if spacing:
    #     # * compute betti numbers
    #     betti_calculator = BinaryBettiCalculator()
    #     betti_0, betti_1, betti_2 = betti_calculator.compute_all_betti(preds.squeeze())
    #     b_pred = [betti_0, betti_1, betti_2]
        
    #     b0, b1, b2 = betti_calculator.compute_all_betti(gdth.squeeze())
    #     b_gt = [b0, b1, b2]
    
    
    if spacing: 
        return dice, cl_dice, precision, recall, hs95
    else:
        return dice

#
# def hausdorff_95(gt_array, pred_array, spacing):
#     '''
#     :params gt_array: ground true mask
#     :params pred_array: the result of segmentation
#     :params num_class: label number
#     :params spacing: spacing of the image
#     '''
#     Hausdorff_95_score = []
#     gt_array = gt_array.astype(bool)
#     pred_array = pred_array.astype(bool)
# #
#     # compute Hausdorff_95 score
#     surface_distances = surface_distance.compute_surface_distances(pred_array, gt_array, spacing)
# #     hs95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
#     return hs95

if __name__ == "__main__":
    preds = torch.randn(1, 240, 240, 240)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    
    print(preds.sum())
    gts = torch.randn(1, 1, 240, 240, 240)
    spacing = (1.0, 1.0, 1.0)
    betti_calculator = BinaryBettiCalculator()
    betti_0, betti_1, betti_2 = betti_calculator.compute_all_betti(preds.squeeze())
    print(betti_0, betti_1, betti_2)
