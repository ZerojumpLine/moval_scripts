#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on cardiac mri segmentation

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_seg2d_ensemble_ablation.py --dataset {dataset} --predpath {predpath} --metric {metric} --gtpath {gtpath} --portion {portion}\n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_seg2d_cardiac_ensemble_ablation.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} --metric {metric} --savingpath {savingpath} --portion {portion}\n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/cardiac'

'''Running inference

'''
training_conds = [resultspath]
## test on synthesized and natural conditions

test_syn_conds = list(range(1, 84))

test_nat_conds = ['2', '3', '4', '5']
metrics = ["f1score"]
portions = [12, 6, 3, 2, 1]

datapath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Cardiac'
dataset = 'Cardiac'

# estimate 36 conditions
with open('./cardiac_estim_ensemble_ablation.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:
            for portion in portions:
                predpath = f"{training_cond}/cardiacval/results"
                gtpath = f"{datapath}/1"
                fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', metric=metric, gtpath=f'"{gtpath}"', portion=portion))

print(f'fsl_sub -q long -R 128 -l logs -t ./cardiac_estim_ensemble_ablation.txt')

# evaluate 36 condtions for all test conditions
with open('./cardiac_eval_ensemble_ablation.txt', 'w') as fpr:
    for metric in metrics:
        for training_cond in training_conds:
            for portion in portions:

                for test_syn_cond in test_syn_conds:

                    predpath = f"{training_cond}/cardiactestsyn_{test_syn_cond}/results"
                    gtpath = f"{datapath}/1"
                    savingpath = f"./results_{dataset}_{metric}_syn_{test_syn_cond}_ensemble.txt"
                    fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"', portion=portion))
                
                for test_nat_cond in test_nat_conds:

                    predpath = f"{training_cond}/cardiactest_{test_nat_cond}/results"
                    gtpath = f"{datapath}/{test_nat_cond}"
                    savingpath = f"./results_{dataset}_{metric}_nat_{test_nat_cond}_ensemble.txt"
                    fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"', portion=portion))

print(f'fsl_sub -q long -R 128 -l logs -t ./cardiac_eval_ensemble_ablation.txt')