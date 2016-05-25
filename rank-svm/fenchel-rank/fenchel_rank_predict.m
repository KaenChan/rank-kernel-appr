function [pred, TestTime, TestEVAL] = fenchel_rank_predict(model, X_test, Y_test, Q_test)

% option:
%  r : the constrain of ||w|| <= r
%  max_iter : the maximum number of iterations
%  err : the required accuray

t1 = tic;

X = [X_test ones(size(X_train,1),1)];

pred=(X * model.w);

TestEVAL = compute_metric(pred, Y_test, Q_test, model.metric_type);

t2 = clock;
TestTime = etime(t2,t1);
