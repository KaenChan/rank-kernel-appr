Rank-kernel-appr
===========================

Rank-kernel-appr is a ranking SVM (RankSVM) algorithm with kernel approximation to solve the problem of lengthy training time of kernel RankSVM.

Supported kernel approximation methods
- The Nystrom method
- Improved Nystrom method
- Random Fourier features

Supported RankSVM
- RankSVM-primal
- Fenchel

## The Nystrom method vs random Fourier features

<img src="https://github.com/KaenChan/rank-kernel-appr/blob/master/test/ncomponets-map-all.png" height="300" width="400" >

## Demo

The datasets can be found in [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor).
Please download and copy the datasets to ./data folder.


```
>> addpath(genpath('.'))
>> load_data_to_mat
>> test_ranksvm_run
Fold1 N=500 C=2^-6 g=2^-6 Time=1.0023 s EVAL=(0.4802 0.4790 0.3539)
Fold2 N=500 C=2^-6 g=2^-6 Time=1.1820 s EVAL=(0.4908 0.3438 0.4448)
Fold3 N=500 C=2^-6 g=2^-6 Time=1.0530 s EVAL=(0.4546 0.4496 0.4543)
Fold4 N=500 C=2^-6 g=2^-6 Time=0.9119 s EVAL=(0.4283 0.4593 0.5140)
Fold5 N=500 C=2^-6 g=2^-6 Time=0.8332 s EVAL=(0.4326 0.5152 0.4714)

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
appr_type   = fourier
kernel      = rbf

====================================================================
      | N    | C     | gamma | Train Time | MAP                    |
--------------------------------------------------------------------
Fold1 | 500  | 2^-6  | 2^-6  |  1.0023 s  | (0.4802 0.4790 0.3539) |
Fold2 | 500  | 2^-6  | 2^-6  |  1.1820 s  | (0.4908 0.3438 0.4448) |
Fold3 | 500  | 2^-6  | 2^-6  |  1.0530 s  | (0.4546 0.4496 0.4543) |
Fold4 | 500  | 2^-6  | 2^-6  |  0.9119 s  | (0.4283 0.4593 0.5140) |
Fold5 | 500  | 2^-6  | 2^-6  |  0.8332 s  | (0.4326 0.5152 0.4714) |
--------------------------------------------------------------------
Mean  |                      |  0.9965 s  | (0.4573 0.4494 0.4477) |
--------------------------------------------------------------------
 
Performance on testing set
qid     NDCG@1  NDCG@2  NDCG@3  NDCG@4  NDCG@5  NDCG@6  NDCG@7  NDCG@8  NDCG@9  NDCG@10 MeanNDCG 
fold1   0.5000  0.4545  0.4284  0.4385  0.4211  0.4057  0.4105  0.3995  0.3950  0.3965  0.5022 
fold2   0.5397  0.5000  0.4878  0.5013  0.4861  0.4733  0.4654  0.4636  0.4589  0.4552  0.5353 
fold3   0.5556  0.5079  0.4964  0.4796  0.4647  0.4631  0.4470  0.4492  0.4481  0.4384  0.5463 
fold4   0.5873  0.4921  0.4768  0.4830  0.4619  0.4651  0.4691  0.4730  0.4689  0.4692  0.5388 
fold5   0.6508  0.5794  0.5698  0.5650  0.5497  0.5329  0.5136  0.5051  0.5030  0.4955  0.5640 
n_mean  0.5667  0.5068  0.4918  0.4935  0.4767  0.4680  0.4611  0.4581  0.4548  0.4510  0.5373 
 
qid     P@1     P@2     P@3     P@4     P@5     P@6     P@7     P@8     P@9     P@10    MAP 
fold1   0.5909  0.5455  0.5000  0.5000  0.4636  0.4242  0.4221  0.3920  0.3838  0.3773  0.3539 
fold2   0.6667  0.5952  0.5556  0.5595  0.5238  0.4921  0.4762  0.4643  0.4497  0.4429  0.4448 
fold3   0.7143  0.6190  0.6190  0.5952  0.5810  0.5794  0.5442  0.5476  0.5503  0.5238  0.4543 
fold4   0.7143  0.6667  0.6508  0.6429  0.5905  0.5952  0.6054  0.6012  0.5926  0.5905  0.5140 
fold5   0.7143  0.6905  0.6825  0.7024  0.6857  0.6349  0.5986  0.5893  0.5714  0.5619  0.4714 
m_mean  0.6801  0.6234  0.6016  0.6000  0.5689  0.5452  0.5293  0.5189  0.5096  0.4993  0.4477 
```
