#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the estimating process for the evaluation of classification on cifar10

'''

import os
import argparse
import itertools
import moval
import pandas as pd
import numpy as np

from moval.solvers.utils import ComputMetric, ComputAUC
from moval.models.utils import cal_softmax

parser = argparse.ArgumentParser(description='CIFAR10-LT Performance Evaluation of ensemble models')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='saving checkpoint name, CIFAR10 | CIFAR10ci1 | CIFAR10ci2 | CIFAR10rl1 | CIFAR10rl2')
parser.add_argument('--testpath', default='', type=str, help='csv path of the test prediction conditions')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--savingpath', default='./results_CIFAR10_syn.txt', type=str, help='txt file to save the evaluation results')

args = parser.parse_args()

def test_cls(estim_algorithm, mode, metric, logits_tests, gt_tests, dataset):
    """Test MOVAL with different conditions for classification tasks

    Args:
        mode (str): The given task to estimate model performance.
        metrc (str): The metric to be estimated.
        estim_algorithm (str):
            The algorithm to estimate model performance. We also provide a list of estimation algorithm which can be displayed by
            running :py:func:`moval.models.get_estim_options`.
        logits_tests:  A list of m test conditions ``(n', d)``.
        gt_test: The cooresponding annotation of a list of m ``(n', )``.

    Returns:
        err_test: A list of m test err.
        acc_estims: A list of m estimated accuracy.
        acc_reals: A list of m real accuracy.

    """

    ckpt_savname = f"./{dataset}_{mode}_{metric}_{estim_algorithm}.pkl"

    moval_model = moval.MOVAL.load(ckpt_savname)

    # save the test err in the result files.

    err_tests = []
    metric_estims = []
    metric_reals = []
    for k_test in range(len(logits_tests)):

        _logits_test = logits_tests[k_test]
        _gt_test = gt_tests[k_test]

        estim_metric_val = moval_model.estimate(_logits_test)
        metric_estims.append(estim_metric_val)
        pred_test = np.argmax(_logits_test, axis = 1)

        if metric == "accuracy":
            real_metric_val = np.sum(_gt_test == pred_test) / len(_gt_test)
        elif metric == "sensitivity":
            sensitivities = []
            for kcls in range(_logits_test.shape[1]):
                pos_cls = np.where(_gt_test == kcls)[0]
                # if there do not exist any samplies for class kcls
                if len(pos_cls) == 0:
                    sensitivity_cls = -1
                else:
                    _, sensitivity_cls, _ = ComputMetric(_gt_test == kcls, pred_test == kcls)
                sensitivities.append(sensitivity_cls)
            sensitivities = np.array(sensitivities)
            sensitivity_mean = sensitivities[sensitivities >= 0].mean()
            sensitivities[sensitivities < 0] = sensitivity_mean

            real_metric_val = sensitivities
        elif metric == "precision":
            precisions = []
            for kcls in range(_logits_test.shape[1]):
                pos_cls = np.where(_gt_test == kcls)[0]
                # if there do not exist any samplies for class kcls
                if len(pos_cls) == 0:
                    precision_cls = -1
                else:
                    _, _, precision_cls = ComputMetric(_gt_test == kcls, pred_test == kcls)
                precisions.append(precision_cls)
            precisions = np.array(precisions)
            precision_mean = precisions[precisions >= 0].mean()
            precisions[precisions < 0] = precision_mean
            real_metric_val = precisions
        elif metric == "f1score":
            f1scores = []
            for kcls in range(_logits_test.shape[1]):
                pos_cls = np.where(_gt_test == kcls)[0]
                # if there do not exist any samplies for class kcls
                if len(pos_cls) == 0:
                    f1score_cls = -1
                else:
                    f1score_cls, _, _ = ComputMetric(_gt_test == kcls, pred_test == kcls)
                f1scores.append(f1score_cls)
            f1scores = np.array(f1scores)
            f1score_mean = f1scores[f1scores >= 0].mean()
            f1scores[f1scores < 0] = f1score_mean
            real_metric_val = f1scores
        else:
            aucs = ComputAUC(_gt_test, cal_softmax(_logits_test))
            auc_mean = aucs[aucs > 0].mean()
            aucs[aucs == 0] = auc_mean
            real_metric_val = aucs

        err_test = np.mean( np.abs( real_metric_val - estim_metric_val ) )
        #
        err_tests.append(err_test)
        metric_reals.append(real_metric_val)


    return err_tests, metric_estims, metric_reals

def main():

    # test data
    num_classes = 10
    cnn_pred_test = pd.read_csv(args.testpath)
    targets_all_test = np.array(cnn_pred_test[['target_' + str(i) for i in range(0, num_classes)]])
    logits_test = np.array(cnn_pred_test[['logit_' + str(i) for i in range(0, num_classes)]])
    gt_test = np.argmax(targets_all_test, axis = 1)

    #

    testc_indx_1 = list(range(0, 7000))
    testc_indx_2 = list(range(7000, 14000))
    testc_indx_3 = list(range(14000, 21000))
    testc_indx_4 = list(range(21000, 28000))
    testc_indx_5 = list(range(28000, 35000))

    testc_indxs = [testc_indx_1, testc_indx_2, testc_indx_3, testc_indx_4, testc_indx_5]

    #
    logits_tests = []
    gt_tests = []
    #
    for testc_indx in testc_indxs:
        #
        logits_tests.append(logits_test[testc_indx, :])
        gt_tests.append(gt_test[testc_indx])

    #

    mode = "classification"
    metric = args.metric
    estim_algorithm = "moval-ensemble-cls-" + metric

    err_test, metric_estim, metric_real = test_cls(
        estim_algorithm = estim_algorithm,
        mode = mode,
        metric = metric,
        logits_tests = logits_tests,
        gt_tests = gt_tests,
        dataset = args.dataset
    )

    test_condition = f"estim_algorithm = {estim_algorithm}, mode = {mode}, metric = {metric}"

    results_files = args.savingpath
    # clean previous results
    if os.path.isfile(results_files):
        os.remove(results_files)

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
