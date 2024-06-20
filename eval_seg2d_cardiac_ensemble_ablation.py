#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the estimating process for the evaluation of 2d segmentation on cardiac

'''

import os
from os import listdir
from os.path import isfile, join
import argparse
import itertools
import moval
import nibabel as nib
import numpy as np
from moval.solvers.utils import ComputMetric
from eval_seg3d_brainlesion_ensemble_ablation import test_cls

parser = argparse.ArgumentParser(description='Cardiac 2D Segmentation Performance Evaluation of ensemble models')
parser.add_argument('--dataset', default='', type=str, help='saving checkpoint name, Cardiac')
parser.add_argument('--predpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/cardiac/cardiacval/results', type=str, help='pred path of the test cases')
parser.add_argument('--gtpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Cardiac/1', type=str, help='gt path of the test cases')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--savingpath', default='./results_cardiac_syn.txt', type=str, help='txt file to save the evaluation results')
parser.add_argument('--portion', default=19, type=int, help='number of validation data used for estimating')

args = parser.parse_args()

def main():

    # test data
    predpath = args.predpath
    predfiles = [f for f in listdir(predpath) if isfile(join(predpath, f))]

    logits = []
    gt = []
    for predfile in predfiles:
        if predfile.split('_')[-2][-1] == '0':
            # grep the caseID
            caseID = f"{predfile.split('_')[-3]}_{predfile.split('_')[-2][:-4]}"
            #
            GT_file = f"{args.gtpath}/{caseID}/seg.nii.gz"
            #
            logit_cls0_file = f"{predpath}/pred_{caseID}cls0_prob.nii.gz"
            logit_cls1_file = f"{predpath}/pred_{caseID}cls1_prob.nii.gz"
            logit_cls2_file = f"{predpath}/pred_{caseID}cls2_prob.nii.gz"
            logit_cls3_file = f"{predpath}/pred_{caseID}cls3_prob.nii.gz"
            #
            logit_cls0_read = nib.load(logit_cls0_file)
            logit_cls1_read = nib.load(logit_cls1_file)
            logit_cls2_read = nib.load(logit_cls2_file)
            logit_cls3_read = nib.load(logit_cls3_file)
            #
            logit_cls0      = logit_cls0_read.get_fdata()   # ``(H, W, D)``
            logit_cls1      = logit_cls1_read.get_fdata()
            logit_cls2      = logit_cls2_read.get_fdata()
            logit_cls3      = logit_cls3_read.get_fdata()
            #
            GT_read         = nib.load(GT_file)
            GTimg           = GT_read.get_fdata()           # ``(H, W, D)``
            #
            logit_cls = np.stack((logit_cls0, logit_cls1, logit_cls2, logit_cls3))  # ``(d, H, W, D)``
            # only including the slices that contains labels
            for dslice in range(GTimg.shape[2]):
                if np.sum(GTimg[:, :, dslice]) > 0:
                    logits.append(logit_cls[:, :, :, dslice])
                    gt.append(GTimg[:, :, dslice])

    # logits is a list of length ``n``,  each element has ``(d, H, W)``.
    # gt is a list of length ``n``,  each element has ``(H, W)``.
    # H and W could differ for different cases.

    mode = "segmentation"
    metric = args.metric
    estim_algorithm = "moval-ensemble-seg2d-" + metric

    seednums = [13, 35, 57, 79, 93]
    for seednum in seednums:

        results_files = args.savingpath[:-4] + "seed" + str(seednum) + ".txt"
        # clean previous results
        if os.path.isfile(results_files):
            os.remove(results_files)

        err_test, metric_estim, metric_real = test_cls(
            estim_algorithm = estim_algorithm,
            mode = mode,
            metric = args.metric,
            logits_test = logits,
            gt_test = gt,
            dataset = args.dataset,
            portion = args.portion,
            seednum = seednum
        )

        test_condition = f"estim_algorithm = {estim_algorithm}, mode = {mode}, metric = {args.metric}"

        with open(results_files, 'a') as f:
            f.write(test_condition)
            f.write('\n')
            f.write("test err: ")
            f.write(str(err_test))
            f.write('\n')
            f.write("estimated metric: ")
            f.write(str(metric_estim))
            f.write('\n')
            f.write("real metric: ")
            f.write(str(metric_real))
            f.write('\n')
            f.write('\n')

if __name__ == "__main__":
    
    main()
