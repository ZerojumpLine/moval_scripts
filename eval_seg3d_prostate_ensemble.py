#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the estimating process for the evaluation of 3d segmentation on prostate

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
from eval_seg3d_brainlesion_ensemble import test_cls

parser = argparse.ArgumentParser(description='Prostate 3D Segmentation Performance Evaluation of ensemble models')
parser.add_argument('--dataset', default='', type=str, help='saving checkpoint name, Prostate')
parser.add_argument('--predpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/prostate/prostateval/results', type=str, help='pred path of the test cases')
parser.add_argument('--gtpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Prostate/BMC', type=str, help='gt path of the test cases')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--savingpath', default='./results_prostate_syn.txt', type=str, help='txt file to save the evaluation results')

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
            caseID = f"{predfile.split('_')[-2][:-4]}"
            #
            GT_file = f"{args.gtpath}/{caseID}/seg.nii.gz"
            #
            logit_cls0_file = f"{predpath}/pred_{caseID}cls0_prob.nii.gz"
            logit_cls1_file = f"{predpath}/pred_{caseID}cls1_prob.nii.gz"
            #
            logit_cls0_read = nib.load(logit_cls0_file)
            logit_cls1_read = nib.load(logit_cls1_file)
            #
            logit_cls0      = logit_cls0_read.get_fdata()   # ``(H, W, D)``
            logit_cls1      = logit_cls1_read.get_fdata()
            #
            GT_read         = nib.load(GT_file)
            GTimg           = GT_read.get_fdata()           # ``(H, W, D)``
            #
            logit_cls = np.stack((logit_cls0, logit_cls1))  # ``(d, H, W, D)``
            logits.append(logit_cls)
            gt.append(GTimg)

    # logits is a list of length ``n``,  each element has ``(d, H, W, D)``.
    # gt is a list of length ``n``,  each element has ``(H, W, D)``.
    # H, W and D could differ for different cases.

    mode = "segmentation"
    metric = args.metric
    estim_algorithm = "moval-ensemble-seg3d-" + metric

    results_files = args.savingpath
    # clean previous results
    if os.path.isfile(results_files):
        os.remove(results_files)


    err_test, metric_estim, metric_real = test_cls(
        estim_algorithm = estim_algorithm,
        mode = mode,
        metric = args.metric,
        logits_test = logits,
        gt_test = gt,
        dataset = args.dataset
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
