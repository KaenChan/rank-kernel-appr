function test_kernel_appr_multiclass()

datasets{1} = 'glass';      % n:214     p:9       c:6
datasets{2} = 'vehicle';    % n:846     p:18      c:4
datasets{3} = 'vowel';      % n:990     p:10      c:11
datasets{4} = 'segment';    % n:2310    p:19      c:7
datasets{5} = 'dna';        % n:2586    p:180     c:3
datasets{6} = 'usps';       % n:9298    p:256     c:10
datasets{7} = 'pendigits';  % n:10992   p:16      c:10
datasets{8} = 'protein';    % n:21516   p:357     c:3
datasets{9} = 'sector';     % n:9619    p:55197   c:105

dataset = datasets{6};

rng('default');

train_split = 0.8;
seed = 0;
[X_train, Y_train, X_test, Y_test, onehot_map] = load_multiclass_clf_data(dataset, 1);


option.n_components = 500;
option.c_rho = 10;
option.metric_type.name = 'acc';
option.learn_type = 'classifier';

option.appr_type = 'rbf';
% option.appr_type = 'nystroem';
% option.appr_type = 'improvednystroem';

% RBFSampler
option.rbf.gamma = 0.1;
% NystroemSampler
option.nystroem.kernel = 'rbf';
option.nystroem.gamma = 0.001;
option.nystroem.coef0 = 1; 
option.nystroem.degree = 3;

info = '';
info = [info sprintf('dataset      = %s\n', dataset)];
info = [info sprintf('trainsize    = %s\n', mat2str(size(X_train)))];
info = [info sprintf('testsize     = %s\n', mat2str(size(X_test)))];
info = [info sprintf('metric_type  = %s\n', option.metric_type.name)];
info = [info sprintf('n_components = %s\n', mat2str(option.n_components ))];
info = [info sprintf('appr_type    = %s\n', option.appr_type)];
info = [info sprintf('c_rho        = %d\n', option.c_rho)];
info = [info sprintf('\n')];

fprintf(info);


model = kernel_approximation_train(X_train, Y_train, option);

pred_train = kernel_approximation_predict(model, X_train, Y_train);
pred_test = kernel_approximation_predict(model, X_test, Y_test);

TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
model.EVAL = [TrainEVAL TestEVAL];

fprintf('N=%-8d | TrainTime=%.4f s | %s (%.4f %.4f) ||\n', ...
    model.N, model.TrainTime, option.metric_type.name, model.EVAL(1), model.EVAL(2));

fprintf('\n');

