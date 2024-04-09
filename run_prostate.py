#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on prostate mri segmentation

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_seg3d.py --dataset {dataset} --predpath {predpath} --metric {metric} --gtpath {gtpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_seg3d_prostate.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} --metric {metric} --savingpath {savingpath} \n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/prostate'

'''Running inference

'''
training_conds = [resultspath]
## test on synthesized and natural conditions

test_syn_conds = list(range(1, 84))

test_nat_conds = ['BIDMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
metrics = ["f1score", "accuracy", "sensitivity", "precision", "auc"]

datapath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Prostate'

dataset = 'Prostate'

# estimate 36 conditions
with open('./prostate_estim.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:
            predpath = f"{training_cond}/prostateval/results"
            gtpath = f"{datapath}/BMC"
            fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', metric=metric, gtpath=f'"{gtpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./prostate_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./prostate_eval.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:

            for test_syn_cond in test_syn_conds:

                predpath = f"{training_cond}/prostatetestsyn_{test_syn_cond}/results"
                gtpath = f"{datapath}/BMC"
                savingpath = f"./results_{dataset}_{metric}_syn_{test_syn_cond}.txt"
                fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))
            
            for test_nat_cond in test_nat_conds:

                predpath = f"{training_cond}/prostatetest_{test_nat_cond}/results"
                gtpath = f"{datapath}/{test_nat_cond}"
                savingpath = f"./results_{dataset}_{metric}_nat_{test_nat_cond}.txt"
                fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./prostate_eval.txt')