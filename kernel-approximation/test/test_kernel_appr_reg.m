function test_kernel_appr_reg()

datasets{1} = 'housing';
datasets{2} = 'space_ga';
datasets{3} = 'abalone';
datasets{4} = 'kinematics';
datasets{5} = 'cadata';
datasets{6} = 'census_16h';
datasets{7} = 'mpg';
datasets{8} = 'bodyfat';
datasets{9} = 'triazines';
datasets{10} = 'mg';
datasets{11} = 'cpusmall';
datasets{12} = 'data_sinc_1000';
dataset = datasets{11};

rng('default');

train_split = 0.8;
seed = 0;
[X_train, Y_train, X_test, Y_test] = load_reg_data(dataset, train_split, seed);


option.n_components = 100;
option.c_rho = 10;
option.metric_type.name = 'rmse';
option.learn_type = 'regression';

option.appr_type = 'rbf';
option.appr_type = 'nystroem';
% option.appr_type = 'improvednystroem';

% RBFSampler
option.rbf.gamma = 0.2;
% NystroemSampler
option.nystroem.kernel = 'rbf';
option.nystroem.gamma = 0.1;
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

