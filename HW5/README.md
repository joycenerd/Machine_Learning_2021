# Gaussian Process & SVM

Implementation of Gaussian Process and SVM (both used kernel tricks)

## Set up the environment

You should have Anaconda or Miniconda installed with Python>=3.7

```
conda env create -f environment.yml
conda activate SVM
```

## Reproducing the experiments

```
python gaussian_process.py
```
<img src="./results/initial_gaussian_process.jpg">
<img src="./results/optimize_gaussian_process.jpg">

```
python SVM.py --part 1
```
<img src="./results/compare_kernels.png">

```
python SVM.py --part 2 --kernel-type rbf
```
<img src="./results/rbf_confusion_matrix.jpg">
<img src="./results/rbf_best_param_and_acc.PNG">

```
python SVM.py --part 3
```

<img src="./results/linear_rbf_kernel.PNG">
