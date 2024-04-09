## To run the all the experiments on BMRC, with command line.

First, estimate the parameters with:
```
fsl_sub -q long -R 128 -l logs -t ./cifar10_estim.txt
fsl_sub -q long -R 128 -l logs -t ./cifar100_estim.txt
fsl_sub -q long -R 128 -l logs -t ./skinlesion_estim.txt

fsl_sub -q long -R 128 -l logs -t ./brainlesion_estim_dsc.txt
fsl_sub -q long -R 128 -l logs -t ./brainlesion_estim_4metrics.txt
fsl_sub -q long -R 128 -l logs -t ./cardiac_estim.txt
fsl_sub -q long -R 128 -l logs -t ./prostate_estim.txt
```

Then, run the evaluation:
```
fsl_sub -q long -R 128 -l logs -t ./cifar10_eval.txt
fsl_sub -q long -R 128 -l logs -t ./cifar100_eval.txt
fsl_sub -q long -R 128 -l logs -t ./skinlesion_eval.txt

fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_dsc.txt
fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_4metrics.txt
fsl_sub -q long -R 128 -l logs -t ./cardiac_eval.txt
fsl_sub -q long -R 128 -l logs -t ./prostate_eval.txt
```