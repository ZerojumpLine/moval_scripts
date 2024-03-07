#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the estimating process for the evaluation of classification on skin lesion

'''

import os
import argparse
import itertools
import moval
import pandas as pd
import numpy as np
from eval_cls_cifar10 import test_cls

parser = argparse.ArgumentParser(description='HAM10000 Performance Evaluation')
parser.add_argument('--dataset', default='', type=str, help='saving checkpoint name, HAM')
parser.add_argument('--testpath', default='', type=str, help='csv path of the test prediction conditions')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--savingpath', default='./results_HAM_syn.txt', type=str, help='txt file to save the evaluation results')

args = parser.parse_args()

def main():

    # test data
    num_classes = 7
    cnn_pred_test = pd.read_csv(args.testpath)
    targets_all_test = np.array(cnn_pred_test[['target_' + str(i) for i in range(0, num_classes)]])
    logits_test = np.array(cnn_pred_test[['logit_' + str(i) for i in range(0, num_classes)]])
    gt_test = np.argmax(targets_all_test, axis = 1)

    #

    logits_tests = []
    gt_tests = []
    #

    logits_tests.append(logits_test)
    gt_tests.append(gt_test)

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

        err_test, metric_estim, metric_real = test_cls(
            estim_algorithm = moval_options[k_cond][0],
            mode = moval_options[k_cond][1],
            metric = args.metric,
            confidence_scores = moval_options[k_cond][2],
            class_specific = moval_options[k_cond][3],
            logits_tests = logits_tests,
            gt_tests = gt_tests,
            dataset = args.dataset
        )

        test_condition = f"estim_algorithm = {moval_options[k_cond][0]}, mode = {moval_options[k_cond][1]}, metric = {args.metric}, confidence_scores = {moval_options[k_cond][2]}, class_specific = {moval_options[k_cond][3]}"

        with open(results_files, 'a') as f:
            f.write(test_condition)
            f.write('\n')
            f.write("test err: ")
            f.write(str(err_test))
            f.write('\n')
            f.write("estimated acc: ")
            f.write(str(metric_estim))
            f.write('\n')
            f.write("real acc: ")
            f.write(str(metric_real))
            f.write('\n')
            f.write('\n')

if __name__ == "__main__":
    
    main()
