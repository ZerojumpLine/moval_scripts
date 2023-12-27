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

parser = argparse.ArgumentParser(description='Cardiac 2D Segmentation Performance Evaluation')
parser.add_argument('--predpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/cardiac/cardiacval/results', type=str, help='pred path of the test cases')
parser.add_argument('--gtpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Cardiac/1', type=str, help='gt path of the test cases')
parser.add_argument('--savingpath', default='./results_cardiac_syn.txt', type=str, help='txt file to save the evaluation results')

args = parser.parse_args()

def test_cls(estim_algorithm, mode, confidence_scores, class_specific, logits_test, gt_test):
    """Test MOVAL with different conditions for 2d segmentation tasks

    Args:
        mode (str): The given task to estimate model performance.
        confidence_scores (str):
            The method to calculate the confidence scores. We provide a list of confidence score calculation methods which
            can be displayed by running :py:func:`moval.models.get_conf_options`.
        estim_algorithm (str):
            The algorithm to estimate model performance. We also provide a list of estimation algorithm which can be displayed by
            running :py:func:`moval.models.get_estim_options`.
        class_specific (bool):
            If ``True``, the calculation will match class-wise confidence to class-wise accuracy.
        logits_test:  The network testing output (logits) of a list of n' ``(d, H', W', (D'))`` for segmentation.
        gt_test: The cooresponding testing annotation of a list of n' ``(H', W', (D'))`` for segmentation.

    Returns:
        err_test (float): testing error.
        dsc_estims: the estimated dsc of D'-1 classes.
        dsc_reals: the real dsc of D'-1 classes.

    """

    ckpt_savname = f"./Cardiac_{mode}2d_{confidence_scores}_{estim_algorithm}_{class_specific}.pkl"

    moval_model = moval.MOVAL.load(ckpt_savname)

    # save the test err in the result files.
    # the gt_guide for test data is optional
    gt_guide_test = []
    for n_case in range(len(logits_test)):
        gt_case_test     = gt_test[n_case]
        gt_exist_test = []
        for k_cls in range(logits_test[0].shape[0]):
            gt_exist_test.append(np.sum(gt_case_test == k_cls) > 0)
        gt_guide_test.append(gt_exist_test)

    estim_dsc_test = moval_model.estimate(logits_test, gt_guide = gt_guide_test)

    DSC_list_test = []
    for n_case in range(len(logits_test)):
        pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H', W', (D'))``
        gt_case     = gt_test[n_case] # ``(H', W', (D'))``

        dsc_case = np.zeros(logits_test[n_case].shape[0])
        for kcls in range(1, logits_test[n_case].shape[0]):
            if np.sum(gt_case == kcls) == 0:
                dsc_case[kcls] = -1
            else:
                dsc_case[kcls] = ComputMetric(pred_case == kcls, gt_case == kcls)
        DSC_list_test.append(dsc_case)

    # only aggregate the ones which are not -1
    DSC_list_test = np.array(DSC_list_test) # ``(n, d)``
    m_DSC_test = []
    m_DSC_test.append(0.)
    for kcls in range(1, logits_test[n_case].shape[0]):
        m_DSC_test.append(DSC_list_test[:, kcls][DSC_list_test[:,kcls] >= 0].mean())

    m_DSC_test = np.array(m_DSC_test)
    err_test = np.abs( m_DSC_test[1:] - estim_dsc_test )

    return err_test, estim_dsc_test, m_DSC_test[1:]

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

    moval_options = list(itertools.product(moval.models.get_estim_options(),
                               ["segmentation"],
                               moval.models.get_conf_options(),
                               [False, True]))
    
    # ac-model does not need class-speicfic variants
    for moval_option in moval_options:
        if moval_option[0] == 'ac-model' and moval_option[-1] == True:
            moval_options.remove(moval_option)

    results_files = args.savingpath
    # clean previous results
    if os.path.isfile(results_files):
        os.remove(results_files)

    for k_cond in range(len(moval_options)):

        err_test, dsc_estim, dsc_real = test_cls(
            estim_algorithm = moval_options[k_cond][0],
            mode = moval_options[k_cond][1],
            confidence_scores = moval_options[k_cond][2],
            class_specific = moval_options[k_cond][3],
            logits_tests = logits,
            gt_tests = gt
        )

        test_condition = f"estim_algorithm = {moval_options[k_cond][0]}, mode = {moval_options[k_cond][1]}, confidence_scores = {moval_options[k_cond][2]}, class_specific = {moval_options[k_cond][3]}"

        with open(results_files, 'a') as f:
            f.write(test_condition)
            f.write('\n')
            f.write("test err: ")
            f.write(str(err_test))
            f.write('\n')
            f.write("estimated dsc: ")
            f.write(str(dsc_estim))
            f.write('\n')
            f.write("real dsc: ")
            f.write(str(dsc_real))
            f.write('\n')
            f.write('\n')

if __name__ == "__main__":
    
    main()
