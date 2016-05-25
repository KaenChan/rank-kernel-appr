function model = fenchel_rank_train(X_train, Y_train, Q_train, option)

% option:
%  r : the constrain of ||w|| <= r
%  max_iter : the maximum number of iterations
%  err : the required accuray

t1 = tic;

% Q_train = Q_train(1);
% idx = Q_train{1};
% X_train = X_train(idx);
% Y_train = Y_train(idx);

qids = zeros(size(Y_train));
for i=1:length(Q_train)
	qids(Q_train{i}) = i;
end

qids = int32(qids);

X = [X_train ones(size(X_train,1),1)];

w = fenchel_rank(X, Y_train, qids, option.r, option.max_iter, option.err, option.verbose);
w'
pred=(X * w);                       %   the actual output of the training data

TrainMAP  = compute_map(pred, Y_train, Q_train);
TrainNDCG = compute_ndcg(pred, Y_train, Q_train, option.metric_type.k_ndcg);

model = option;
model.weights = w;
model.TrainTime = toc(t1);
model.TrainMAP = TrainMAP;
model.TrainNDCG = TrainNDCG;
