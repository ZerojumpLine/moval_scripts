#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the estimating process for the evaluation of 3d segmentation on brain lesion

'''

import os
from os import listdir
from os.path import isfile, join
import argparse
import itertools
import moval
import nibabel as nib
import numpy as np
from moval.models.utils import cal_softmax
from moval.solvers.utils import ComputMetric, ComputAUC

parser = argparse.ArgumentParser(description='Brainlesion 3D Segmentation Performance Evaluation of ensemble models')
parser.add_argument('--dataset', default='Brainlesionlas', type=str, help='saving checkpoint name, Brainlesionlas | Brainlesionci1 | Brainlesionci2 | Brainlesionrl1 | Brainlesionrl2')
parser.add_argument('--predpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas/atlasval/results', type=str, help='pred path of the test cases')
parser.add_argument('--gtpath', default='/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Brain_lesion/Siemens Trio', type=str, help='gt path of the test cases')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--savingpath', default='./results_brainlesion_syn.txt', type=str, help='txt file to save the evaluation results')

args = parser.parse_args()

def test_cls(estim_algorithm, mode, metric, logits_test, gt_test, dataset):
    """Test MOVAL with different conditions for 2d/3d segmentation tasks

    Args:
        mode (str): The given task to estimate model performance.
        metrc (str): The metric to be estimated.
        estim_algorithm (str):
            The algorithm to estimate model performance. We also provide a list of estimation algorithm which can be displayed by
            running :py:func:`moval.models.get_estim_options`.
        logits_test:  The network testing output (logits) of a list of n' ``(d, H', W', (D'))`` for segmentation.
        gt_test: The cooresponding testing annotation of a list of n' ``(H', W', (D'))`` for segmentation.

    Returns:
        err_test (float): testing error.
        metric_estims: the estimated accuracy or the estimated metric of D'-1 classes.
        metric_reals: the real accuracy or the estimated metric of D'-1 classes.

    """

    ckpt_savname = f"./{dataset}_{mode}_{metric}_{estim_algorithm}.pkl"

    moval_model = moval.MOVAL.load(ckpt_savname)

    # save the test err in the result files.
    # the gt_guide for test data is optional, but kind of important for 2d evaluation.
    # because for 2d cases, there could be many sliced without GT.
    gt_guide_test = []
    for n_case in range(len(logits_test)):
        gt_case_test     = gt_test[n_case]
        gt_exist_test = []
        for k_cls in range(logits_test[0].shape[0]):
            gt_exist_test.append(np.sum(gt_case_test == k_cls) > 0)
        gt_guide_test.append(gt_exist_test)

    if metric == "auc":
        # to accelrate the inference of auc, crop a bit.
        logits_test_crop, gt_guide_test_crop = moval_model.crop(logits = logits_test, gt = gt_guide_test, approximate_boundary = 30)
        estim_metric_test = moval_model.estimate(logits_test_crop, gt_guide = gt_guide_test_crop)
    else:
        estim_metric_test = moval_model.estimate(logits_test, gt_guide = gt_guide_test)

    if metric == "accuracy":
        pred_all_flatten = []
        gt_all_flatten = []
        for n_case in range(len(logits_test)):

            pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
            gt_case     = gt_test[n_case] # ``(H, W, (D))``
            
            pred_all_flatten.append(pred_case.flatten())
            gt_all_flatten.append(gt_case.flatten())
        
        pred_all_flatten = np.concatenate(pred_all_flatten)
        gt_all_flatten = np.concatenate(gt_all_flatten)

        acc = np.sum(gt_all_flatten == pred_all_flatten) / len(gt_all_flatten)
        m_metric_test = acc
    elif metric == "sensitivity":
        sensitivity = []
        for n_case in range(len(logits_test)):

            pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
            gt_case     = gt_test[n_case] # ``(H, W, (D))``

            sensitivity_case = np.zeros(logits_test[n_case].shape[0])
            for kcls in range(1, logits_test[n_case].shape[0]):
                if np.sum(gt_case == kcls) == 0:
                    sensitivity_case[kcls] = -1
                else:
                    _, sensitivity_case[kcls], _ = ComputMetric(gt_case == kcls, pred_case == kcls)
            sensitivity.append(sensitivity_case)
        
        # only aggregate the ones which are not -1
        sensitivity = np.array(sensitivity) # ``(n, d)``
        sensitivity_mean = []
        for kcls in range(logits_test[n_case].shape[0]):
            # I am not sure, if the real sensitivity is 0, I think the network cannot learn anything
            # but any calculate this case in the estim, may we can improve here.
            sensitivity_mean.append(sensitivity[:, kcls][sensitivity[:,kcls] >= 0].mean())
        m_metric_test = np.array(sensitivity_mean)
    elif metric == "precision":

        precision = []
        for n_case in range(len(logits_test)):

            pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
            gt_case     = gt_test[n_case] # ``(H, W, (D))``

            precision_case = np.zeros(logits_test[n_case].shape[0])
            for kcls in range(1, logits_test[n_case].shape[0]):
                if np.sum(gt_case == kcls) == 0:
                    precision_case[kcls] = -1
                else:
                    _, _, precision_case[kcls] = ComputMetric(gt_case == kcls, pred_case == kcls)
            precision.append(precision_case)
        
        # only aggregate the ones which are not -1
        precision = np.array(precision) # ``(n, d)``
        precision_mean = []
        for kcls in range(logits_test[n_case].shape[0]):
            # I am not sure, if the real precision is 0, I think the network cannot learn anything
            # but any calculate this case in the estim, may we can improve here.
            precision_mean.append(precision[:, kcls][precision[:,kcls] >= 0].mean())
        m_metric_test = np.array(precision_mean)
    elif metric == "f1score":

        dsc = []
        for n_case in range(len(logits_test)):

            pred_case   = np.argmax(logits_test[n_case], axis = 0) # ``(H, W, (D))``
            gt_case     = gt_test[n_case] # ``(H, W, (D))``

            dsc_case = np.zeros(logits_test[n_case].shape[0])
            for kcls in range(1, logits_test[n_case].shape[0]):
                if np.sum(gt_case == kcls) == 0:
                    dsc_case[kcls] = -1
                else:
                    dsc_case[kcls], _, _ = ComputMetric(gt_case == kcls, pred_case == kcls)
            dsc.append(dsc_case)
        
        # only aggregate the ones which are not -1
        dsc = np.array(dsc) # ``(n, d)``
        dsc_mean = []
        for kcls in range(logits_test[n_case].shape[0]):
            # I am not sure, if the real dsc is 0, I think the network cannot learn anything
            # but any calculate this case in the estim, may we can improve here.
            dsc_mean.append(dsc[:, kcls][dsc[:,kcls] >= 0].mean())
        m_metric_test = np.array(dsc_mean)
    elif metric == "auc":
                
        auc = []
        for n_case in range(len(logits_test)):
            
            inp_case = logits_test[n_case] # ``(d, H, W, (D))``
            # from ``(d, H, W, (D))`` to ``(n, d)``
            d, *rest_of_dimensions = inp_case.shape
            flatten_dim = np.prod(rest_of_dimensions)
            inp_case = inp_case.reshape((d, flatten_dim))
            inp_case = inp_case.T # ``(n, d)``
            probability = cal_softmax(inp_case) # ``(n, d)``                    
            gt_case     = gt_test[n_case].flatten() # ``(n, )``

            auc_case = ComputAUC(gt_case, probability) # ``(d, )``
            auc.append(auc_case)
        
        # only aggregate the ones which are not -1
        auc = np.array(auc) # ``(n, d)``
        auc_mean = []
        for kcls in range(logits_test[n_case].shape[0]):
            # I am not sure, if the real dsc is 0, I think the network cannot learn anything
            # but any calculate this case in the estim, may we can improve here.
            auc_mean.append(auc[:, kcls][auc[:,kcls] >= 0].mean())
        m_metric_test = np.array(auc_mean)


    err_test = np.mean( np.abs( m_metric_test - estim_metric_test ) )

    return err_test, estim_metric_test, m_metric_test

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
    estim_algorithm = "moval-ensemble-seg-" + metric

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
