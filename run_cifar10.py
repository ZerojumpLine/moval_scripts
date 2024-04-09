#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on cifar10 classification

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_cls.py --dataset {dataset} --numcls {numcls} --metric {metric} --valpath {valpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_cls_cifar10.py --dataset {dataset} --testpath {testpath} --metric {metric} --savingpath {savingpath} \n'

## baseline training
resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/baseline'

## advanced training
resultspath_c1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/ci1'
resultspath_c2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/ci2'
resultspath_r1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/rl1'
resultspath_r2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/rl2'


'''Running inference

'''
training_conds = [resultspath, resultspath_c1, resultspath_c2, resultspath_r1, resultspath_r2]

## test on synthesized conditions
## for each one return x5 results.

test_syn_conds = ['gaussian_blur', 'motion_blur', 'snow', 'contrast', 'jpeg_compression', 'shot_noise', 'saturate',
                  'impulse_noise', 'pixelate', 'speckle_noise', 'frost', 'defocus_blur', 'brightness',
                  'gaussian_noise', 'zoom_blur', 'fog', 'spatter', 'glass_blur', 'elastic_transform']
metrics = ["accuracy", "sensitivity", "precision", "f1score", "auc"]
numcls = 10

# estimate 36 conditions
with open('./cifar10_estim.txt', 'w') as fpr:
    metric = metrics[0]
    for training_cond in training_conds:
        ext = training_cond.split('/')[-1]
        if ext == 'baseline':
            dataset = 'CIFAR10'
        else:
            dataset = f"CIFAR10{ext}"
        valpath = f"{training_cond}/predictions_val.csv"
        fpr.write(cmd_estim.format(dataset=dataset, numcls=numcls, metric=metric, valpath=f'"{valpath}"'))
    
    training_cond = training_conds[0]
    for metric in metrics[1:]:
        ext = training_cond.split('/')[-1]
        if ext == 'baseline':
            dataset = 'CIFAR10'
        else:
            dataset = f"CIFAR10{ext}"
        valpath = f"{training_cond}/predictions_val.csv"
        fpr.write(cmd_estim.format(dataset=dataset, numcls=numcls, metric=metric, valpath=f'"{valpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./cifar10_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./cifar10_eval.txt', 'w+') as fpr:
    metric = metrics[0]
    for training_cond in training_conds:
        ext = training_cond.split('/')[-1]
        if ext == 'baseline':
            dataset = 'CIFAR10'
        else:
            dataset = f"CIFAR10{ext}"

        for test_syn_cond in test_syn_conds:

            testpath = f"{training_cond}/predictions_val_{test_syn_cond}.csv"
            savingpath = f"./results_{dataset}_{metric}_{test_syn_cond}.txt"
            fpr.write(cmd_eval.format(dataset=dataset, testpath=f'"{testpath}"', metric=metric, savingpath=f'"{savingpath}"'))

    training_cond = training_conds[0]
    for metric in metrics[1:]:
        ext = training_cond.split('/')[-1]
        if ext == 'baseline':
            dataset = 'CIFAR10'
        else:
            dataset = f"CIFAR10{ext}"

        for test_syn_cond in test_syn_conds:

            testpath = f"{training_cond}/predictions_val_{test_syn_cond}.csv"
            savingpath = f"./results_{dataset}_{metric}_{test_syn_cond}.txt"
            fpr.write(cmd_eval.format(dataset=dataset, testpath=f'"{testpath}"', metric=metric, savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./cifar10_eval.txt')