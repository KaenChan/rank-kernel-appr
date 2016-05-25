
load letor3_ohsumed_fold1

metric_type.name = 'MAP';
% metric_type.name = 'NDCG';
metric_type.k_ndcg = 5;


option.r = 0.9;
option.err = 0.001;
option.max_iter = 1000;
option.verbose = 1;
option.metric_type = metric_type;

fenchel_rank_train(X_train, Y_train, Q_train, option)
