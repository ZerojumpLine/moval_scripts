#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on cifar100 classification

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_cls.py --dataset {dataset} --numcls {numcls} --valpath {valpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_cls_cifar100.py --testpath {testpath} --savingpath {savingpath} \n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Skin-Lesion-Classification/cifar100results/baseline'

'''Running inference

'''
training_conds = [resultspath]
## test on synthesized conditions

test_syn_conds = ['gaussian_blur', 'motion_blur', 'snow', 'contrast', 'jpeg_compression', 'shot_noise', 'saturate',
                  'impulse_noise', 'pixelate', 'speckle_noise', 'frost', 'defocus_blur', 'brightness',
                  'gaussian_noise', 'zoom_blur', 'fog', 'spatter', 'glass_blur', 'elastic_transform']


dataset = 'CIFAR100'
numcls = 100


# estimate 36 conditions
with open('./cifar100_estim.txt', 'w') as fpr:
    for training_cond in training_conds:
        valpath = f"{training_cond}/predictions_val.csv"
        fpr.write(cmd_estim.format(dataset=dataset, numcls=numcls, valpath=valpath))

print(f'fsl_sub -q short -R 128 -l logs -t ./cifar100_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./cifar100_eval.txt', 'w') as fpr:
    for training_cond in training_conds:

        for test_syn_cond in test_syn_conds:

            testpath = f"{training_cond}/predictions_val_{test_syn_cond}.csv"
            savingpath = f"./results_{dataset}_{test_syn_cond}.txt"
            fpr.write(cmd_eval.format(testpath=testpath, savingpath=savingpath))

print(f'fsl_sub -q short -R 128 -l logs -t ./cifar100_eval.txt')