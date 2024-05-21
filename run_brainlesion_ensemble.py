#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on brain lesion mri segmentation

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_seg3d_ensemble.py --dataset {dataset} --predpath {predpath} --metric {metric} --gtpath {gtpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_seg3d_brainlesion_ensemble.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} --metric {metric} --savingpath {savingpath} \n'

## baseline training

resultspath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas'

## advanced training

resultspath_c1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas_ci1'
resultspath_c2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas_ci2'
resultspath_r1 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas_rl1'
resultspath_r2 = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/output/atlas_rl2'

'''Running inference

'''
training_conds = [resultspath, resultspath_c1, resultspath_c2, resultspath_r1, resultspath_r2]
## test on synthesized and natural conditions

test_syn_conds = list(range(1, 84))

test_nat_conds = ['GE 750 Discovery', 'GE Signa Excite', 'GE Signa HD-X', 'Philips', 'Philips Achieva', 
                  'Siemens Allegra', 'Siemens Magnetom Skyra', 'Siemens Prisma', 'Siemens Skyra', 
                  'Siemens Sonata', 'Siemens TrioTim', 'Siemens Verio', 'Siemens Vision']
metrics = ["f1score", "accuracy", "sensitivity", "precision", "auc"]

datapath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Brain_lesion'

# estimate 36 conditions
with open('./brainlesion_estim_dsc_ensemble.txt', 'w') as fpr:
    metric = metrics[0]
    for training_cond in training_conds:
        
        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        predpath = f"{training_cond}/atlasval/results"
        gtpath = f"{datapath}/Siemens Trio"

        fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', metric=metric, gtpath=f'"{gtpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./brainlesion_estim_dsc_ensemble.txt')

# estimate 36 conditions for 4 other metrics
with open('./brainlesion_estim_4metrics_ensemble.txt', 'w') as fpr:
    training_cond = training_conds[0]
    for metric in metrics[1:]:
        
        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        predpath = f"{training_cond}/atlasval/results"
        gtpath = f"{datapath}/Siemens Trio"

        fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', metric=metric, gtpath=f'"{gtpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./brainlesion_estim_4metrics_ensemble.txt')

# evaluate 36 condtions for all test conditions
with open('./brainlesion_eval_dsc_ensemble.txt', 'w') as fpr:
    metric = metrics[0]
    for training_cond in training_conds:

        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        for test_syn_cond in test_syn_conds:

            predpath = f"{training_cond}/atlastestcondition_{test_syn_cond}/results"
            gtpath = f"{datapath}/Siemens Trio"
            savingpath = f"./results_{dataset}_{metric}_syn_{test_syn_cond}_ensemble.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))
        
        for test_nat_cond in test_nat_conds:

            predpath = f"{training_cond}/atlastest_{test_nat_cond}/results"
            gtpath = f"{datapath}/{test_nat_cond}"
            savingpath = f"./results_{dataset}_{metric}_nat_{test_nat_cond}_ensemble.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_dsc_ensemble.txt')

# evaluate 36 condtions for 4 other metrics
with open('./brainlesion_eval_4metrics_ensemble.txt', 'w') as fpr:
    training_cond = training_conds[0]
    for metric in metrics[1:]:

        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        for test_syn_cond in test_syn_conds:

            predpath = f"{training_cond}/atlastestcondition_{test_syn_cond}/results"
            gtpath = f"{datapath}/Siemens Trio"
            savingpath = f"./results_{dataset}_{metric}_syn_{test_syn_cond}_ensemble.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))
        
        for test_nat_cond in test_nat_conds:

            predpath = f"{training_cond}/atlastest_{test_nat_cond}/results"
            gtpath = f"{datapath}/{test_nat_cond}"
            savingpath = f"./results_{dataset}_{metric}_nat_{test_nat_cond}_ensemble.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', metric=metric, savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_4metrics_ensemble.txt')


