#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on cifar10 classification

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_cls_ensemble_ablation_batch.py --dataset {dataset} --numcls {numcls} --metric {metric} --valpath {valpath} --portion {portion}\n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_cls_cifar10_ensemble_ablation_batch.py --dataset {dataset} --testpath {testpath} --metric {metric} --savingpath {savingpath} --portion {portion}\n'

## baseline training
resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/baseline'

## advanced training
resultspath_c1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/ci1'
resultspath_c2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/ci2'
resultspath_r1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/rl1'
resultspath_r2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar10results/rl2'


'''Running inference

'''
training_conds = [resultspath]

## test on synthesized conditions
## for each one return x5 results.

test_syn_conds = ['gaussian_blur', 'motion_blur', 'snow', 'contrast', 'jpeg_compression', 'shot_noise', 'saturate',
                  'impulse_noise', 'pixelate', 'speckle_noise', 'frost', 'defocus_blur', 'brightness',
                  'gaussian_noise', 'zoom_blur', 'fog', 'spatter', 'glass_blur', 'elastic_transform']
metrics = ["accuracy", "sensitivity", "precision", "f1score", "auc"]
portions = [100, 50, 20, 10, 5, 3]
numcls = 10

# estimate 36 conditions
with open('./cifar10_estim_ensemble_ablation_batch.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:
            for portion in portions:
                ext = training_cond.split('/')[-1]
                if ext == 'baseline':
                    dataset = 'CIFAR10'
                else:
                    dataset = f"CIFAR10{ext}"
                valpath = f"{training_cond}/predictions_val.csv"
                fpr.write(cmd_estim.format(dataset=dataset, numcls=numcls, metric=metric, valpath=f'"{valpath}"', portion=portion))

print(f'fsl_sub -q long -R 128 -l logs -t ./cifar10_estim_ensemble_ablation_batch.txt')

# evaluate 36 condtions for all test conditions
with open('./cifar10_eval_ensemble_ablation_batch.txt', 'w+') as fpr:
    for metric in metrics:
        for training_cond in training_conds:
            for portion in portions:
                ext = training_cond.split('/')[-1]
                if ext == 'baseline':
                    dataset = 'CIFAR10'
                else:
                    dataset = f"CIFAR10{ext}"

                for test_syn_cond in test_syn_conds:

                    testpath = f"{training_cond}/predictions_val_{test_syn_cond}.csv"
                    savingpath = f"./results_{dataset}_{metric}_{test_syn_cond}_ensemble_{portion}_batch.txt"
                    fpr.write(cmd_eval.format(dataset=dataset, testpath=f'"{testpath}"', metric=metric, savingpath=f'"{savingpath}"', portion=portion))

print(f'fsl_sub -q long -R 128 -l logs -t ./cifar10_eval_ensemble_ablation_batch.txt')