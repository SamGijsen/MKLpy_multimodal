This fork of MKLpy (see below) features the following edits:
- facilitates multimodal use
- enables use of AverageMKL for regression

Multimodal modeling (using AverageMKL or EasyMKL) can be done as follows, where `N` is a list of length M in case of M modalities and `M[i]` denotes the amount of features belonging to modality `i`. 

```py
from MKLpy.utils.cv_wrappers import EasyMKLWrapper, AverageMKLWrapper, KernelTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Specify model and hyperparameters
kernel_types=["rbf", "rbf"]

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ('kernel_transform', KernelTransformer(
       N=[20, 10], norm_data=False, kernel_types=kernel_types)),
    ('AvgMKL', AverageMKLWrapper(learner=SVC(probability=True), classification=True)) 
])
grid = {"AvgMKL__learner__C": [0.01, 1., 100.],
        "kernel_transform__gamma": [0.001, 0.01, 0.1]}

# Specify stratification
# X_train, Y_train are training data and labels respectively
cv = StratifiedKFold(3, shuffle=True, random_state=1)      
cv_splits = cv.split(X_train, Y_train)

# Cross-validation + Grid-search
gs = GridSearchCV(estimator=pipeline, param_grid=grid, n_jobs=1,
                  cv=cv_splits, return_train_score=True, refit=True, verbose=True)

gs = gs.fit(X_train, Y_train)
```

and for regression:

```py
from sklearn.svm import SVR

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ('kernel_transform', KernelTransformer(
       N=[20, 10], norm_data=False, kernel_types=kernel_types)),
    ('AvgMKL', AverageMKLWrapper(learner=SVR(), classification=False)) 
])

grid = {"AvgMKL__classification": [False], # !
        "AvgMKL__learner__C": [0.01, 1., 100.],
        "kernel_transform__gamma": [0.001, 0.01, 0.1]}
```


MKLpy
=====

[![Documentation Status](https://readthedocs.org/projects/mklpy/badge/?version=latest)](https://mklpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/IvanoLauriola/MKLpy.svg?branch=master)](https://travis-ci.com/IvanoLauriola/MKLpy)
[![Coverage Status](https://coveralls.io/repos/github/IvanoLauriola/MKLpy/badge.svg?branch=master)](https://coveralls.io/github/IvanoLauriola/MKLpy?branch=master)
[![PyPI version](https://badge.fury.io/py/MKLpy.svg)](https://badge.fury.io/py/MKLpy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


**MKLpy** is a framework for Multiple Kernel Learning (MKL)  inspired by the [scikit-learn](http://scikit-learn.org/stable) project.

This package contains:
* the implementation of some MKL algorithms;
* tools to operate on kernels, such as normalization, centering, summation, average...;
* metrics, such as kernel_alignment, radius of Minimum Enclosing Ball, margin between classes, spectral ratio...;
* kernel functions, including boolean kernels (disjunctive, conjunctive, DNF, CNF) and string kernels (spectrum, fixed length and all subsequences).


The main MKL algorithms implemented in this library are

|Name       |Short description | Status | Source |
|-----------|------------------|--------|:------:|
| AverageMKL| Computes the simple average of base kernels         | Available | - |
| EasyMKL   | Fast and memory efficient margin-based combination  | Available |[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003653) |
| GRAM      | Radius/margin ratio optimization                    | Available |[[2]](https://www.researchgate.net/publication/318468451_Radius-Margin_Ratio_Optimization_for_Dot-Product_Boolean_Kernel_Learning)   |
| R-MKL     | Radius/margin ratio optimization                    | Available |[[3]](https://link.springer.com/content/pdf/10.1007/978-3-642-04180-8_39.pdf)  |
| MEMO      | Margin maximization and complexity minimization     | Available |[[4]](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-181.pdf) |
| PWMK      | Heuristic based on individual kernels performance   | Avaible   |[[5]](https://ieeexplore.ieee.org/abstract/document/4586335) |
| FHeuristic| Heuristic based on kernels alignment                | Available |[[6]](https://ieeexplore.ieee.org/abstract/document/4731239) |
| CKA       | Centered kernel alignment optimization in closed form| Available|[[7]](https://static.googleusercontent.com/media/research.google.com/it//pubs/archive/36468.pdf) |
| SimpleMKL | Alternate margin maximization                       | Work in progress |[[5]](http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf)|


The documentation of MKLpy is available on [readthedocs.io](https://mklpy.readthedocs.io/en/latest/)!



Installation
------------

**MKLpy** is also available on PyPI:
```sh
pip install MKLpy
```

**MKLpy** leverages multiple scientific libraries, that are [numpy](https://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/), and [CVXOPT](https://cvxopt.org/).


Examples
--------
The folder *examples* contains several scripts and snippets of codes to show the potentialities of **MKLpy**. The examples show how to train a classifier, how to process data, and how to use kernel functions.

Additionally, you may read our [tutorials](https://mklpy.readthedocs.io/en/latest/)


Work in progress
----------------
**MKLpy** is under development! We are working to integrate several features, including:
* additional MKL algorithms;
* more kernels for structured data;
* efficient optimization




Citing MKLpy
------------
If you use MKLpy for a scientific purpose, please **cite** the following preprint.

```
@article{lauriola2020mklpy,
  title={MKLpy: a python-based framework for Multiple Kernel Learning},
  author={Lauriola, Ivano and Aiolli, Fabio},
  journal={arXiv preprint arXiv:2007.09982},
  year={2020}
}
```