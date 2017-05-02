Rank-kernel-appr
===========================

## Citation
If you use the codes, please cite the following paper:
```
@article{Chen2017Ranking,
  title={Ranking Support Vector Machine with Kernel Approximation.},
  author={Chen, Kai and Li, Rongchun and Dou, Yong and Liang, Zhengfa and Lv, Qi},
  journal={Computational Intelligence & Neuroscience},
  volume={2017},
  year={2017},
}
```
    
## Overview
Rank-kernel-appr is a ranking SVM (RankSVM) algorithm with kernel approximation to solve the problem of lengthy training time of kernel RankSVM.

Kernel approximation methods
- The Nystrom method
- Improved Nystrom method
- Random Fourier features

Linear RankSVM
- RankSVM-primal
- Fenchel

## The Nystrom method vs random Fourier features

<img src="https://github.com/KaenChan/rank-kernel-appr/blob/master/test/ncomponets-map-all.png" height="500" width="600" >

## Demo

The datasets can be found in [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor).
Please download and copy the datasets to ./data folder.


Load data and transform the data to our structrue.

```
>> addpath(genpath('.'))
>> load_data_to_mat
```

Run rank-kernel-appr with the Nystrom methods on OHSUMED dataset.

```
>> test_ranksvm_run
Fold1 N=500 C=2^-6 g=2^-6 Time=1.0557 s EVAL=(0.4801 0.4806 0.3529)
Fold2 N=500 C=2^-6 g=2^-6 Time=1.3700 s EVAL=(0.4913 0.3457 0.4461)
Fold3 N=500 C=2^-6 g=2^-6 Time=1.1567 s EVAL=(0.4545 0.4474 0.4491)
Fold4 N=500 C=2^-6 g=2^-6 Time=1.1895 s EVAL=(0.4297 0.4592 0.5160)
Fold5 N=500 C=2^-6 g=2^-6 Time=1.0851 s EVAL=(0.4338 0.5157 0.4725)

test_ranksvm_option
dataset     = OHSUMED
X_train     = [9219 45]
Q_train     = 63
X_vali      = [3538 45]
Q_vali      = 21
X_test      = [3383 45]
Q_test      = 22
metric_type = MAP
k_ndcg      = 5
N           = [500;500;500;500;500]
C           = [-6;-6;-6;-6;-6]
gammas      = [-6;-6;-6;-6;-6]
appr_type   = nystroem
kernel      = rbf

====================================================================
      | N    | C     | gamma | Train Time | MAP                    |
--------------------------------------------------------------------
Fold1 | 500  | 2^-6  | 2^-6  |  1.0557 s  | (0.4801 0.4806 0.3529) |
Fold2 | 500  | 2^-6  | 2^-6  |  1.3700 s  | (0.4913 0.3457 0.4461) |
Fold3 | 500  | 2^-6  | 2^-6  |  1.1567 s  | (0.4545 0.4474 0.4491) |
Fold4 | 500  | 2^-6  | 2^-6  |  1.1895 s  | (0.4297 0.4592 0.5160) |
Fold5 | 500  | 2^-6  | 2^-6  |  1.0851 s  | (0.4338 0.5157 0.4725) |
--------------------------------------------------------------------
Mean  |                      |  1.1714 s  | (0.4579 0.4497 0.4473) |
--------------------------------------------------------------------
 
Performance on testing set
qid     NDCG@1  NDCG@2  NDCG@3  NDCG@4  NDCG@5  NDCG@6  NDCG@7  NDCG@8  NDCG@9  NDCG@10 MeanNDCG 
fold1   0.5000  0.4545  0.4175  0.4366  0.4250  0.4062  0.4022  0.3938  0.3947  0.3955  0.5012 
fold2   0.5397  0.5000  0.4992  0.5111  0.4891  0.4825  0.4712  0.4643  0.4618  0.4576  0.5377 
fold3   0.5397  0.4762  0.4701  0.4614  0.4578  0.4585  0.4490  0.4419  0.4363  0.4356  0.5393 
fold4   0.5873  0.5079  0.4737  0.4754  0.4652  0.4653  0.4668  0.4749  0.4623  0.4593  0.5433 
fold5   0.6984  0.6032  0.5765  0.5732  0.5530  0.5422  0.5223  0.5154  0.5138  0.5047  0.5675 
n_mean  0.5730  0.5084  0.4874  0.4915  0.4780  0.4709  0.4623  0.4581  0.4538  0.4505  0.5378 
 
qid     P@1     P@2     P@3     P@4     P@5     P@6     P@7     P@8     P@9     P@10    MAP 
fold1   0.5909  0.5455  0.4848  0.5000  0.4727  0.4318  0.4091  0.3807  0.3788  0.3773  0.3529 
fold2   0.6667  0.5952  0.5714  0.5714  0.5238  0.5079  0.4762  0.4524  0.4497  0.4476  0.4461 
fold3   0.6667  0.5714  0.5873  0.5714  0.5714  0.5714  0.5442  0.5357  0.5344  0.5333  0.4491 
fold4   0.7143  0.6667  0.6190  0.6190  0.6000  0.5873  0.5986  0.6012  0.5714  0.5667  0.5160 
fold5   0.7619  0.7143  0.6825  0.7143  0.6762  0.6587  0.6190  0.5952  0.5820  0.5667  0.4725 
m_mean  0.6801  0.6186  0.5890  0.5952  0.5688  0.5514  0.5294  0.5130  0.5033  0.4983  0.4473 
```

Run rank-kernel-appr with the Nystrom methods on MQ2007 dataset.

```
>> test_ranksvm_run
dataset     = MQ2007
X_train     = [42158 46]
Q_train     = 1017
X_vali      = [13813 46]
Q_vali      = 339
X_test      = [13652 46]
Q_test      = 336
metric_type = MAP
k_ndcg      = 5
N           = [2000;2000;2000;2000;2000]
C           = [4;4;4;4;4]
gammas      = [-8;-8;-8;-8;-8]
appr_type   = nystroem
kernel      = rbf

====================================================================
      | N    | C     | gamma | Train Time | MAP                    |
--------------------------------------------------------------------
Fold1 | 2000 | 2^4   | 2^-8  | 10.9219 s  | (0.4750 0.4817 0.4952) |
Fold2 | 2000 | 2^4   | 2^-8  | 10.3446 s  | (0.4815 0.4935 0.4619) |
Fold3 | 2000 | 2^4   | 2^-8  | 12.3584 s  | (0.4840 0.4618 0.4732) |
Fold4 | 2000 | 2^4   | 2^-8  | 10.9939 s  | (0.4878 0.4726 0.4416) |
Fold5 | 2000 | 2^4   | 2^-8  | 10.9360 s  | (0.4884 0.4503 0.4756) |
--------------------------------------------------------------------
Mean  |                      | 11.1110 s  | (0.4834 0.4720 0.4695) |
--------------------------------------------------------------------

Performance on testing set
qid     NDCG@1  NDCG@2  NDCG@3  NDCG@4  NDCG@5  NDCG@6  NDCG@7  NDCG@8  NDCG@9  NDCG@10 MeanNDCG
fold1   0.4435  0.4472  0.4476  0.4532  0.4593  0.4614  0.4652  0.4688  0.4736  0.4763  0.5293
fold2   0.4208  0.4063  0.4025  0.4034  0.4089  0.4147  0.4214  0.4275  0.4322  0.4384  0.4910
fold3   0.4425  0.4334  0.4266  0.4280  0.4330  0.4386  0.4412  0.4458  0.4495  0.4544  0.5132
fold4   0.3913  0.3557  0.3638  0.3722  0.3840  0.3870  0.3945  0.4024  0.4111  0.4168  0.4724
fold5   0.4228  0.4179  0.4287  0.4299  0.4325  0.4343  0.4382  0.4472  0.4547  0.4605  0.5121
n_mean  0.4242  0.4121  0.4138  0.4173  0.4235  0.4272  0.4321  0.4383  0.4442  0.4493  0.5036  test

qid     P@1     P@2     P@3     P@4     P@5     P@6     P@7     P@8     P@9     P@10    MAP
fold1   0.5030  0.4821  0.4643  0.4546  0.4470  0.4311  0.4230  0.4133  0.4074  0.4009  0.4952
fold2   0.4897  0.4513  0.4326  0.4226  0.4212  0.4140  0.4105  0.4052  0.4009  0.3965  0.4619
fold3   0.5074  0.4720  0.4405  0.4204  0.4142  0.4066  0.3991  0.3901  0.3825  0.3743  0.4732
fold4   0.4661  0.4100  0.4081  0.4004  0.4006  0.3869  0.3835  0.3798  0.3746  0.3681  0.4417
fold5   0.4779  0.4587  0.4513  0.4366  0.4289  0.4179  0.4054  0.4008  0.3956  0.3882  0.4756
m_mean  0.4888  0.4548  0.4394  0.4269  0.4224  0.4113  0.4043  0.3978  0.3922  0.3856  0.4695  test
```
