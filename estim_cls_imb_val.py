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

parser = argparse.ArgumentParser(description='Estimating MOVAL Parameters for classification Performance Evaluation')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='saving checkpoint name, CIFAR10 | CIFAR10ci1 | CIFAR10ci2 | CIFAR10rl1 | CIFAR10rl2 | CIFAR100 | HAM')
parser.add_argument('--numcls', default=10, type=int, help='number of class for model fitting')
parser.add_argument('--batch', default=1, type=int, help='group size for the optimization process')
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

    # resample it to be long-tailed..
    # [300, ..., 30]
    np.random.seed(0)
    max_samples = 300
    min_samples = 300
    n_classes = 10
    decay_factor = np.log(max_samples / min_samples) / (n_classes - 1)
    samples = [int(max_samples * np.exp(-decay_factor * i)) for i in range(n_classes)]
    #
    logits_val_ = []
    gt_val_ = []
    for n_cls in range(10):
        logits_val_cls = logits_val[gt_val == n_cls]
        gt_val_cls = gt_val[gt_val == n_cls]
        #
        selected_indices = np.random.choice(len(gt_val_cls), size=np.min((samples[n_cls], len(gt_val_cls))), replace=False)
        logits_val_sel = logits_val_cls[selected_indices]
        gt_val_sel = gt_val_cls[selected_indices]
        #
        logits_val_.append(logits_val_sel)
        gt_val_.append(gt_val_sel)
    
    logits_val = np.concatenate(logits_val_)
    gt_val = np.concatenate(gt_val_)

    moval_options = list(itertools.product(moval.models.get_estim_options(),
                               ["classification"],
                               moval.models.get_conf_options(),
                               [False, True]))
    
    # ac-model does not need class-speicfic variants
    for moval_option in moval_options:
        if moval_option[0] == 'ac-model' and moval_option[-1] == True:
            moval_options.remove(moval_option)

    for k_cond in range(len(moval_options)):
        
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
        moval_model.fit(logits_val, gt_val, args.batch)

        ckpt_savname = f"./{args.dataset}_{mode}_{args.metric}_{confidence_scores}_{estim_algorithm}_{class_specific}_imb_val.pkl"

        moval_model.save(ckpt_savname)



if __name__ == "__main__":
    
    main()
