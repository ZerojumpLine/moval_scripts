#!/usr/bin/env python
# Copyright (c) 2023, Zeju Li
# All rights reserved.

'''Runing the fitting process for the evaluation of classification tasks

'''

import os
import argparse
import itertools
import moval
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='Estimating MOVAL ensemble Parameters for classification Performance Evaluation')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='saving checkpoint name, CIFAR10 | CIFAR10ci1 | CIFAR10ci2 | CIFAR10rl1 | CIFAR10rl2 | CIFAR100 | HAM')
parser.add_argument('--numcls', default=10, type=int, help='number of class for model fitting')
parser.add_argument('--batch', default=8, type=int, help='group size for the optimization process')
parser.add_argument('--metric', default='accuracy', type=str, help='type of estimation metrics, accuracy | sensitivity | precision | f1score | auc')
parser.add_argument('--valpath', default='', type=str, help='csv path of the validation prediction conditions')

args = parser.parse_args()

def main():

    # validation data
    num_classes = args.numcls
    cnn_pred = pd.read_csv(args.valpath)
    targets_all = np.array(cnn_pred[['target_' + str(i) for i in range(0, num_classes)]])
    logits_val = np.array(cnn_pred[['logit_' + str(i) for i in range(0, num_classes)]])
    gt_val = np.argmax(targets_all, axis = 1)
    # logits is of shape ``(n, d)``
    # gt is of shape ``(n, )``

    mode = "classification"
    metric = args.metric
    estim_algorithm = "moval-ensemble-cls-" + metric

    moval_model = moval.MOVAL(
        mode = mode,
        metric = metric,
        estim_algorithm = estim_algorithm
        )

    #
    moval_model.fit(logits_val, gt_val, args.batch)

    ckpt_savname = f"./{args.dataset}_{mode}_{args.metric}_{estim_algorithm}_batch.pkl"

    moval_model.save(ckpt_savname)



if __name__ == "__main__":
    
    main()
