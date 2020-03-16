# Bess: An python Package for Best Subset Selection in Linear and Logistic Model


## Introduction

One of the main tasks of statistical modeling is to exploit the association between
a response variable and multiple predictors. Linear model (LM), as a simple parametric
regression model, is often used to capture linear dependence between response and
predictors. The anothe model--generalized linear model (GLM), can be considered as
the extensions of linear model, depending on the types of responses. Parameter estimation in these models
can be computationally intensive when the number of predictors is large. Meanwhile,
Occam’s razor is widely accepted as a heuristic rule for statistical modeling,
which balances goodness of ﬁt and model complexity. This rule leads to a relative 
small subset of important predictors. 

**bess** package provides solutions for best subset selection problem for sparse LM,
and GLM models.

We consider a primal-dual active set (PDAS) approach to exactly solve the best subset
selection problem for sparse LM, GLM and CoxPH models. The PDAS algorithm for linear 
least squares problems was ﬁrst introduced by [Ito and Kunisch (2013)](https://iopscience.iop.org/article/10.1088/0266-5611/30/1/015001)
and later discussed by [Jiao, Jin, and Lu (2015)](https://arxiv.org/abs/1403.0515) and [Huang, Jiao, Liu, and Lu (2017)](https://arxiv.org/abs/1701.05128). 
It utilizes an active set updating strategy and ﬁts the sub-models through use of
complementary primal and dual variables. We generalize the PDAS algorithm for 
general convex loss functions with the best subset constraint, and further 
extend it to support both sequential and golden section search strategies
for optimal k determination. 


## Install

Python Version
- python >= 3.5

Modules needed
- numpy 

The package has been publish in PyPI. You can easy install by:
```sh
$ pip install bess
```

## Example
```python console
>>> from bess.bess import *
>>> import numpy as np
>>> np.random.seed(123)   # fix seed to get the same result

### Data information
>>> train_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))     # train_x
>>> train_y = np.random.normal(0, 1, 10)                          # train_y
>>> test_X = np.random.normal(0, 1, 10 * 5).reshape((10, 5))      # test_x
>>> data_type = 1   #data_type: 1:regression 2:2-classification

### Model information.
>>> sequence = [1, 2, 3]
>>> model = SPDAS_LM(sequence=sequence)

### Fit model.
>>> model.fit(train_X, train_y, data_type)

### Predict.
>>> model.predict(test_X)
[0.8487986542970937, -0.3262548528320405, -1.2427679709665753, -1.1018898695690327, -1.1589700150336006, 0.9170341511542658, -0.7894714094101583, -0.40658125651728766, -0.5161022067202307, -0.17317826396384567]

```

## Reference

- Wen, C. , Zhang, A. , Quan, S. , & Wang, X. . (2017). [Bess: an r package for best subset selection in linear, logistic and coxph models](https://arxiv.org/pdf/1709.06254.pdf)


## Bug report

Connect to [@Jiang-Kangkang](https://github.com/Jiang-Kangkang), or send an email to Jiang Kangkang(jiangkk3@mail2.sysu.edu.cn)

