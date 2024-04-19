#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the fitting process for the evaluation of 3c segmentation tasks

'''

import os
from os import listdir
from os.path import isfile, join
import argparse
import itertools
import moval
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser(description='Estimating MOVAL Parameters for 3d segmentation Performance Evaluation')
parser.add_argument('--dataset', default='', type=str, help='saving checkpoint name, Cardiac | Prostate | Brainlesionlas | Brainlesionci1 | Brainlesionci2 | Brainlesionrl1 | Brainlesionrl2')
parser.add_argument('--predpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/prostate/prostateval/results', type=str, help='pred path of the validation cases')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--gtpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Prostate/BMC', type=str, help='gt path of the validation cases')
parser.add_argument('--partion', default=0, type=int, help='run in batches')

args = parser.parse_args()

def main():

    # validation data
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

    moval_options = list(itertools.product(moval.models.get_estim_options(),
                               ["segmentation"],
                               moval.models.get_conf_options(),
                               [False, True]))
    
    # ac-model does not need class-speicfic variants
    for moval_option in moval_options:
        if moval_option[0] == 'ac-model' and moval_option[-1] == True:
            moval_options.remove(moval_option)

    if args.partion == 0:
        k_conds = range(len(moval_options))
    else:
        k_conds = range(args.partion-1, args.partion)

    for k_cond in k_conds:
        
        mode = moval_options[k_cond][1]
        confidence_scores = moval_options[k_cond][2]
        estim_algorithm = moval_options[k_cond][0]
        class_specific = moval_options[k_cond][3]

        moval_model = moval.MOVAL(
                mode = mode,
                metric = args.metric,
                confidence_scores = confidence_scores,
                estim_algorithm = estim_algorithm,
                class_specific = class_specific
                )

        #
        moval_model.fit(logits, gt)

        ckpt_savname = f"./{args.dataset}_{mode}_{args.metric}_{confidence_scores}_{estim_algorithm}_{class_specific}.pkl"

        moval_model.save(ckpt_savname)



if __name__ == "__main__":
    
    main()
