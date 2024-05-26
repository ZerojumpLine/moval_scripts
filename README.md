## To run the all the experiments on BMRC, with command line.

### Most experiments

First, estimate the parameters with:
```
fsl_sub -q short -R 128 -l logs -t ./cifar10_estim.txt
fsl_sub -q short -R 128 -l logs -t ./cifar100_estim.txt
fsl_sub -q short -R 128 -l logs -t ./skinlesion_estim.txt

fsl_sub -q short -R 128 -l logs -t ./brainlesion_estim_dsc.txt
fsl_sub -q short -R 128 -l logs -t ./brainlesion_estim_4metrics.txt
fsl_sub -q short -R 128 -l logs -t ./cardiac_estim.txt
fsl_sub -q short -R 128 -l logs -t ./prostate_estim.txt
```

Then, run the evaluation:
```
fsl_sub -q short -R 128 -l logs -t ./cifar10_eval.txt
fsl_sub -q short -R 128 -l logs -t ./cifar100_eval.txt
fsl_sub -q short -R 128 -l logs -t ./skinlesion_eval.txt

fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_dsc.txt
fsl_sub -q long -R 128 -l logs -t ./brainlesion_eval_4metrics.txt
fsl_sub -q short -R 128 -l logs -t ./cardiac_eval.txt
fsl_sub -q short -R 128 -l logs -t ./prostate_eval.txt
```

### Ensemble experiments

First, estimate the parameters with:
```
fsl_sub -q short -R 128 -l logs -t ./cifar10_estim_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./cifar100_estim_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./skinlesion_estim_ensemble.txt

fsl_sub -q short -R 128 -l logs -t ./brainlesion_estim_dsc_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./brainlesion_estim_4metrics_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./cardiac_estim_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./prostate_estim_ensemble.txt
```

Then, run the evaluation:
```
fsl_sub -q short -R 128 -l logs -t ./cifar10_eval_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./cifar100_eval_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./skinlesion_eval_ensemble.txt

fsl_sub -q short -R 200 -l logs -t ./brainlesion_eval_dsc_ensemble.txt
fsl_sub -q short -R 200 -l logs -t ./brainlesion_eval_4metrics_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./cardiac_eval_ensemble.txt
fsl_sub -q short -R 128 -l logs -t ./prostate_eval_ensemble.txt
```