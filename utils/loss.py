# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06

import numpy as np
import h5py
import os
import time
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from utils.emd import earth_mover_distance
from utils.chamfer_distance import ChamferDistance

def get_metrics(keep, target):
    with torch.no_grad():
        TP = (keep * target).nonzero().shape[0]
        FN = (~keep * target).nonzero().shape[0]
        FP = (keep * ~target).nonzero().shape[0]
        TN = (~keep * ~target).nonzero().shape[0]

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

def get_emd_loss(original_pc, reconstruct_pc):
    with torch.no_grad():
        emd_loss = earth_mover_distance(original_pc, reconstruct_pc, transpose=False)
    return emd_loss

def get_cd_loss(original_pc, reconstruct_pc):
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(original_pc, reconstruct_pc)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss

def BCEWithLogitsLoss(feat, target, weight=1.):
    """
    binary cross entropy loss with sgmoid
    """

    pos_coords_ids = target.nonzero()[:,0]
    neg_coords_ids = (~target).nonzero()[:0]

    pos_feats = torch.clamp(torch.sigmoid(feat[pos_coords_ids]), 1e-7, 1.0 - 1e-7)
    neg_feats = torch.clamp(torch.sigmoid(feat[neg_coords_ids]), 1e-7, 1.0 - 1e-7)

    loss = weight * torch.mean(-torch.log(pos_feats)) + torch.mean(-torch.log(1.0 - neg_feats))
 
    return loss
