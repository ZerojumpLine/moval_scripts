#!/usr/bin/env python
'''This script will generate the cmd for summerizing the evaluation results on brain lesion mri segmentation

'''
import os

bin = '/well/win-fmrib-analysis/users/gqu790/conda/skylake/envs/moval/bin/python'
cmd_estim = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/estim_seg3d.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} \n'
cmd_eval = bin + ' /well/win-fmrib-analysis/users/gqu790/moval/moval_scripts/eval_seg3d_brainlesion.py --dataset {dataset} --predpath {predpath} --gtpath {gtpath} --savingpath {savingpath} \n'

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


datapath = '/well/win-fmrib-analysis/users/gqu790/moval/Robust-Medical-Segmentation/data/Dataset_Brain_lesion'

# estimate 36 conditions
with open('./brainlesion_estim.txt', 'w') as fpr:
    for training_cond in training_conds:
        
        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        predpath = f"{training_cond}/atlasval/results"
        gtpath = f"{datapath}/Siemens Trio"
        fpr.write(cmd_estim.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./brainlesion_estim.txt')

# evaluate 36 condtions for all test conditions
with open('./brainlesion_eval.txt', 'w') as fpr:
    for training_cond in training_conds:

        ext = training_cond.split('/')[-1][-3:]
        dataset = f"Brainlesion{ext}"

        for test_syn_cond in test_syn_conds:

            predpath = f"{training_cond}/atlastestcondition_{test_syn_cond}/results"
            gtpath = f"{datapath}/Siemens Trio"
            savingpath = f"./results_{dataset}_syn_{test_syn_cond}.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', savingpath=f'"{savingpath}"'))
        
        for test_nat_cond in test_nat_conds:

            predpath = f"{training_cond}/atlastest_{test_nat_cond}/results"
            gtpath = f"{datapath}/{test_nat_cond}"
            savingpath = f"./results_{dataset}_nat_{test_nat_cond}.txt"
            fpr.write(cmd_eval.format(dataset=dataset, predpath=f'"{predpath}"', gtpath=f'"{gtpath}"', savingpath=f'"{savingpath}"'))

print(f'fsl_sub -q short -R 128 -l logs -t ./brainlesion_eval.txt')