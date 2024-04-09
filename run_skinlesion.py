#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on skin lesion classification

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_cls.py --dataset {dataset} --numcls {numcls} --metric {metric} --valpath {valpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_cls_skinlesion.py --dataset {dataset} --testpath {testpath} --metric {metric} --savingpath {savingpath} \n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/skinresults'

'''Running inference

'''
training_conds = [resultspath]
## test on synthesized and natural conditions

## submit x5 conditions.

test_syn_conds = ['gaussian_blur', 'motion_blur', 'snow', 'contrast', 'jpeg_compression', 'shot_noise', 'saturate',
                  'impulse_noise', 'pixelate', 'speckle_noise', 'frost', 'defocus_blur', 'brightness',
                  'gaussian_noise', 'zoom_blur', 'fog', 'spatter', 'glass_blur', 'elastic_transform']
metrics = ["accuracy", "sensitivity", "precision", "f1score", "auc"]

test_nat_conds = ['BCN', 'D7P', 'MSK', 'OTH', 'PH2', 'UDA', 'VIE']

dataset = 'HAM'
numcls = 7

# estimate 36 conditions
with open('./skinlesion_estim.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:

            valpath = f"{training_cond}/predictions_val.csv"
            fpr.write(cmd_estim.format(dataset=dataset, numcls=numcls, metric=metric, valpath=f'"{valpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./skinlesion_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./skinlesion_eval.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:

            for test_syn_cond in test_syn_conds:

                for knum in range(1, 6):
                    testpath = f"{training_cond}/predictions_test_{test_syn_cond}_{knum}.csv"
                    savingpath = f"./results_{dataset}_{metric}_{test_syn_cond}_{knum}.txt"
                    fpr.write(cmd_eval.format(dataset=dataset, testpath=f'"{testpath}"', metric=metric, savingpath=f'"{savingpath}"'))
            
            for testnat_cond in test_nat_conds:

                testpath = f"{training_cond}/predictions_test_{testnat_cond}.csv"
                savingpath = f"./results_{dataset}_{metric}_{testnat_cond}.txt"
                fpr.write(cmd_eval.format(dataset=dataset, testpath=f'"{testpath}"', metric=metric, savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./skinlesion_eval.txt')