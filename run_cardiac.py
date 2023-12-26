#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on cardiac mri segmentation

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_seg2d.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_seg2d_cardiac.py --predpath {predpath} --gtpath {gtpath} --savingpath {savingpath} \n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/cardiac'

'''Running inference

'''
training_conds = [resultspath]
## test on synthesized and natural conditions

test_syn_conds = list(range(1, 84))

test_nat_conds = ['2', '3', '4', '5']

datapath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Cardiac'
dataset = 'Cardiac'

# estimate 36 conditions
with open('./cardiac_estim.txt', 'w') as fpr:
    for training_cond in training_conds:
        predpath = f"{training_cond}/cardiacval/results"
        gtpath = f"{datapath}/1"
        fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./cardiac_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./cardiac_eval.txt', 'w') as fpr:
    for training_cond in training_conds:

        for test_syn_cond in test_syn_conds:

            predpath = f"{training_cond}/cardiactestsyn_{test_syn_cond}/results"
            gtpath = f"{datapath}/1"
            savingpath = f"./results_{dataset}_syn_{test_syn_cond}.txt"
            fpr.write(cmd_eval.format(predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', savingpath=f'"{savingpath}"'))
        
        for test_nat_cond in test_nat_conds:

            predpath = f"{training_cond}/cardiactest_{test_nat_cond}/results"
            gtpath = f"{datapath}/{test_nat_cond}"
            savingpath = f"./results_{dataset}_nat_{test_nat_cond}.txt"
            fpr.write(cmd_eval.format(predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./cardiac_eval.txt')