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


parser = argparse.ArgumentParser(description='CIFAR10-LT Performance Evaluation')
parser.add_argument('--dataset', default='', type=str, help='saving checkpoint name, CIFAR10 | CIFAR10ci1 | CIFAR10ci2 | CIFAR10rl1 | CIFAR10rl2')
parser.add_argument('--testpath', default='', type=str, help='csv path of the test prediction conditions')
parser.add_argument('--savingpath', default='./results_CIFAR10_syn.txt', type=str, help='txt file to save the evaluation results')

args = parser.parse_args()

def test_cls(estim_algorithm, mode, confidence_scores, class_specific, logits_tests, gt_tests, dataset):
    """Test MOVAL with different conditions for classification tasks

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
        logits_tests:  A list of m test conditions ``(n', d)``.
        gt_test: The cooresponding annotation of a list of m ``(n', )``.

    Returns:
        err_test: A list of m test err.
        acc_estims: A list of m estimated accuracy.
        acc_reals: A list of m real accuracy.

    """

    ckpt_savname = f"./{dataset}_{mode}_{confidence_scores}_{estim_algorithm}_{class_specific}.pkl"

    moval_model = moval.MOVAL.load(ckpt_savname)

    # save the test err in the result files.

    err_tests = []
    acc_estims = []
    acc_reals = []
    for k_test in range(len(logits_tests)):

        _logits_test = logits_tests[k_test]
        _gt_test = gt_tests[k_test]

        estim_acc_test = moval_model.estimate(_logits_test)
        pred_test = np.argmax(_logits_test, axis = 1)
        err_test = np.abs( np.sum(_gt_test == pred_test) / len(_gt_test) - estim_acc_test )
        #
        err_tests.append(err_test)
        acc_estims.append(estim_acc_test)
        acc_reals.append(np.sum(_gt_test == pred_test) / len(_gt_test))

    return err_tests, acc_estims, acc_reals

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

    moval_options = list(itertools.product(moval.models.get_estim_options(),
                               ["classification"],
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

        err_test, acc_estim, acc_real = test_cls(
            estim_algorithm = moval_options[k_cond][0],
            mode = moval_options[k_cond][1],
            confidence_scores = moval_options[k_cond][2],
            class_specific = moval_options[k_cond][3],
            logits_tests = logits_tests,
            gt_tests = gt_tests,
            dataset = args.dataset
        )

        test_condition = f"estim_algorithm = {moval_options[k_cond][0]}, 
                            mode = {moval_options[k_cond][1]}, 
                            confidence_scores = {moval_options[k_cond][2]}, 
                            class_specific = {moval_options[k_cond][3]}"

        with open(results_files, 'a') as f:
            f.write(test_condition)
            f.write('\n')
            f.write("test err: ")
            f.write(str(err_test))
            f.write('\n')
            f.write("estimated acc: ")
            f.write(str(acc_estim))
            f.write('\n')
            f.write("real acc: ")
            f.write(str(acc_real))
            f.write('\n')
            f.write('\n')

if __name__ == "__main__":
    
    main()
