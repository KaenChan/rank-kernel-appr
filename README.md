Rank-kernel-appr
===========================

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


```
>> addpath(genpath('.'))
>> load_data_to_mat
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
