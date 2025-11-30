# MOVAL Scripts

This repository contains scripts and results using MOVAL (Model Validation) framework.

## Project Structure

```
moval_scripts/
├── README.md                # This file
├── scripts/                 # Main execution scripts
│   ├── brainlesion/         # Brain lesion segmentation experiments
│   ├── cardiac/             # Cardiac segmentation experiments
│   ├── cifar10/             # CIFAR10 classification experiments
│   ├── cifar100/            # CIFAR100 classification experiments
│   ├── prostate/            # Prostate segmentation experiments
│   └── skinlesion/          # Skin lesion classification experiments
├── src/                     # Source code modules
│   ├── estimation/          # Estimation algorithms
│   │   ├── Classification/  # Classification estimation methods
│   │   └── Segmentation/    # Segmentation estimation methods
│   ├── evaluation/          # Evaluation metrics and analysis
│   │   ├── Classification/  # Classification evaluation
│   │   └── Segmentation/    # Segmentation evaluation
│   └── utils/               # Utility functions and helpers
├── results/                 # Experimental results organized by dataset
│   ├── brainlesion/         # Brain lesion segmentation results
│   ├── cardiac/             # Cardiac segmentation results
│   ├── cifar/               # CIFAR classification results
│   ├── prostate/            # Prostate segmentation results
│   └── skinlesion/          # Skin lesion classification results
├── models/                  # Trained model files (.pkl)
├── data/                    # Data files and models
│   └── *.csv                # CSV data files
├── notebooks/               # Jupyter notebooks for analysis
├── figures/           # PDF reports and documentation
└── logs/                    # Log files from experiments
```

## Datasets

### 1. Brain Lesion Segmentation
- **Task**: Medical image segmentation
- **Experiments**: Ensemble methods, different confidence measures
- **Files**: `brainlesion_*.txt` in `results/brainlesion/`

### 2. Cardiac Segmentation
- **Task**: Cardiac MRI segmentation
- **Experiments**: Natural and synthetic data variations
- **Files**: `cardiac_*.txt` in `results/cardiac/`

### 3. CIFAR Classification
- **Task**: Image classification
- **Experiments**: Brightness variations, ensemble methods
- **Files**: `cifar*.txt` in `results/cifar/`

### 4. Prostate Segmentation
- **Task**: Prostate MRI segmentation
- **Experiments**: Different confidence measures
- **Files**: `prostate_*.txt` in `results/prostate/`

### 5. Skin Lesion Classification
- **Task**: Skin lesion classification (HAM dataset)
- **Experiments**: Brightness and contrast variations
- **Files**: `HAM_*.txt` in `results/skinlesion/`

## API Documentation

### Estimation Module (`src/estimation/`)

#### Classification Estimation
- **`estim_cls.py`**: Basic classification estimation
- **`estim_cls_batch.py`**: Batch processing for classification
- **`estim_cls_ensemble.py`**: Ensemble methods for classification
- **`estim_cls_ensemble_batch.py`**: Batch ensemble classification
- **`estim_cls_ensemble_ablation.py`**: Ablation studies for ensemble classification
- **`estim_cls_ensemble_ablation_batch.py`**: Batch ablation studies
- **`estim_cls_imb_val.py`**: Imbalanced validation for classification

#### Segmentation Estimation
- **`estim_seg2d.py`**: 2D segmentation estimation
- **`estim_seg2d_ensemble.py`**: Ensemble 2D segmentation
- **`estim_seg2d_ensemble_ablation.py`**: Ablation studies for 2D segmentation
- **`estim_seg3d.py`**: 3D segmentation estimation
- **`estim_seg3d_ensemble.py`**: Ensemble 3D segmentation
- **`estim_seg3d_ensemble_ablation.py`**: Ablation studies for 3D segmentation

### Evaluation Module (`src/evaluation/`)

#### Classification Evaluation
- **`eval_cls_cifar10.py`**: CIFAR-10 classification evaluation
- **`eval_cls_cifar10_batch.py`**: Batch CIFAR-10 evaluation
- **`eval_cls_cifar10_ensemble.py`**: Ensemble CIFAR-10 evaluation
- **`eval_cls_cifar10_ensemble_batch.py`**: Batch ensemble CIFAR-10 evaluation
- **`eval_cls_cifar10_ensemble_ablation.py`**: Ablation studies for CIFAR-10
- **`eval_cls_cifar10_ensemble_ablation_batch.py`**: Batch ablation studies
- **`eval_cls_cifar10_imb_val.py`**: Imbalanced validation evaluation
- **`eval_cls_cifar100.py`**: CIFAR-100 classification evaluation
- **`eval_cls_cifar100_ensemble.py`**: Ensemble CIFAR-100 evaluation
- **`eval_cls_skinlesion.py`**: Skin lesion classification evaluation
- **`eval_cls_skinlesion_ensemble.py`**: Ensemble skin lesion evaluation

#### Segmentation Evaluation
- **`eval_seg2d_cardiac.py`**: 2D cardiac segmentation evaluation
- **`eval_seg2d_cardiac_ensemble.py`**: Ensemble 2D cardiac evaluation
- **`eval_seg2d_cardiac_ensemble_ablation.py`**: Ablation studies for 2D cardiac
- **`eval_seg3d_brainlesion.py`**: 3D brain lesion segmentation evaluation
- **`eval_seg3d_brainlesion_ensemble.py`**: Ensemble 3D brain lesion evaluation
- **`eval_seg3d_brainlesion_ensemble_ablation.py`**: Ablation studies for 3D brain lesion
- **`eval_seg3d_prostate.py`**: 3D prostate segmentation evaluation
- **`eval_seg3d_prostate_ensemble.py`**: Ensemble 3D prostate evaluation

### Utility Module (`src/utils/`)
- Common helper functions and utilities
- Data processing functions
- Visualization utilities

## Usage

### 1. Run Experiments
Use the scripts in `scripts/` directory:
```bash
# CIFAR classification
python scripts/cifar10/run_cifar10.py
```

### 2. Run Specific Estimations
```bash
# Classification estimation
python src/estimation/estim_cls.py

# Segmentation estimation
python src/estimation/estim_seg2d.py

# Ensemble methods
python src/estimation/estim_cls_ensemble.py
```

### 3. Evaluate Results
```bash
# Classification evaluation
python src/evaluation/eval_cls_cifar10.py

# Segmentation evaluation
python src/evaluation/eval_seg2d_cardiac.py
```

### 4. Analyze Results
- Check organized results in `results/` directory
- Access trained models in `data/models/`
- Use Jupyter notebooks in `notebooks/` for analysis
