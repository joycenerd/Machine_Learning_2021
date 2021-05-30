# Kernel K-means and Spectral Clustering

Zhi-Yi Chin

## Set-up

You should have Anaconda or Miniconda with Python3 already installed.

```
conda env create -f environment.yml
conda activate cluster
```

## Demo

All the results that our program generates are already in `./results`, in case you want to replicate the experiments we provide steps below.

```
python kernel_K_means.py [-h] [--clusters CLUSTERS] [--gamma-s GAMMA_S]
                         [--gamma-c GAMMA_C] [--iterations ITERATIONS]
                         [--init-mode INIT_MODE] [--kernel-type KERNEL_TYPE]
                         [--kappa KAPPA] [--c C]
```

ex:

```
python kernel_K_means.py --clusters 2 --gamma-s 2.5 --gamma-c 2.5 --iterations 50 --init-mode k-means++ --kernel-type rbf
```

```
python spectral_clustering.py [-h] [--kernel-type KERNEL_TYPE]
                              [--gamma-s GAMMA_S] [--gamma-c GAMMA_C]
                              [--sigma SIGMA] [--cut CUT] [--K K]
                              [--init-mode INIT_MODE]
                              [--iterations ITERATIONS]
```

ex:

```
python spectral_clustering.py --kernel-type Laplace_rbf --sigma 0.1 --cut normalized --K 2 --init-mode k-means++ --iterations 50
```


