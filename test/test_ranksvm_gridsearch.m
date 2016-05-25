function test_ranksvm_gridsearch2()
% clear;

% dataset = 'td2003';
% dataset = 'td2004';
% dataset = 'OHSUMED';
% dataset = 'MQ2007';
dataset = 'MQ2008';
% dataset = 'mslr10k';

[dataset_txt, dataset_mat] = get_dataset_name(dataset);

metric_type.name = 'NDCG';
metric_type.name = 'MAP';
metric_type.k_ndcg = 5;

% n_components  = [400 600 800];
% n_components  = [200:200:3000];
% n_components  = [1000:1000:8000];
% n_components  = [100 200 300 400 500];
% n_components  = 2000;
n_components  = 500;

gammas = -12:1:2;   % appr-fenchel-rank-svm

% primal
Cs = -12:2:6;
% gammas = -6;

% ker-appr-ranksvm
% Cs = -14:2:0;
% gammas = -10:2:0;

% Cs=-2; gammas=-8;

% Cs = 1; gammas = -8;  % appr-fenchel-rank-svm ohsumed  test-map:0.4341
% Cs = 4; gammas = -11; % appr-fenchel-rank-svm ohsumed  test-map:0.4341

load([dataset_mat num2str(1)]);

% RBFSampler
% NystroemSampler
appr_type = 'nystroem';
% appr_type = 'improvednystroem';
% appr_type = 'fourier';
kernel = 'rbf';
coef0 = 1; 
degree = 3;
seed = 0;

load([dataset_mat num2str(1)]);

info = 'ranksvm gridsearch2\n';
info = [info sprintf('dataset     = %s\n', dataset)];
info = [info sprintf('X_train     = %s\n', mat2str(size(X_train)))];
info = [info sprintf('Q_train     = %d\n', length(Q_train))];
info = [info sprintf('X_vali      = %s\n', mat2str(size(X_vali)))];
info = [info sprintf('Q_vali      = %d\n', length(Q_vali))];
info = [info sprintf('X_test      = %s\n', mat2str(size(X_test)))];
info = [info sprintf('Q_test      = %d\n', length(Q_test))];
info = [info sprintf('metric_type = %s\n', metric_type.name)];
info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
info = [info sprintf('N           = %s\n', mat2str(n_components))];
info = [info sprintf('C           = %s\n', mat2str(Cs))];
info = [info sprintf('gammas      = %s\n', mat2str(gammas))];
info = [info sprintf('appr_type   = %s\n', appr_type)];
info = [info sprintf('kernel      = %s\n', kernel)];
info = [info '\n'];

fprintf(info);

t1=tic;
t2=tic;
bestvalid = 0;
for N = [n_components]
for C = [Cs]
for g = [gammas]
        fprintf('[%.2f s (%.2f s)] N=%d C=2^%d g=2^%d ', toc(t1), toc(t2), N, C, g);
        t2=tic;
        for i=1:5, % Loop over the folds
            % Read the training and validation data
            load([dataset_mat num2str(i)]);
            option.n_components = N;
            option.metric_type = metric_type;
            option.verbose = 0;
            option.iter_max_Newton = 20;

            % model = PrimalRankSVM(2^C);
            model = KernelApprRankSVM(N, 2^C, appr_type, kernel, 2^g, coef0, degree, seed);
            % model = FenchelRankSVM(2^C, 0.01, 1000);
            % model = KernelApprFenchelRankSVM(N, 2^C,  0.01, 1000, appr_type, kernel, 2^g, coef0, degree, seed);

            model = fit(model, X_train, Y_train, Q_train, option);

            TrainMAP = model.TrainMAP;
            TrainNDCG = model.TrainNDCG;
            [pred, ValidTime] = predict(model, X_vali);
            ValidMAP = compute_map(pred, Y_vali, Q_vali);
            ValidNDCG = compute_ndcg(pred, Y_vali, Q_vali,metric_type.k_ndcg);

            if strcmp(metric_type.name, 'MAP')
                validScore(i) = ValidMAP;
            elseif strcmp(metric_type.name, 'NDCG')
                validScore(i) = ValidNDCG;
            end;

        end;
        validScore_avg = mean(validScore);
        if bestvalid < validScore_avg,
            bestvalid = validScore_avg;
            clear bestmodel;
            bestmodel.trained_model = model;
            bestmodel.N = N;
            bestmodel.C = C;
            bestmodel.g = g;
            bestmodel.Time = model.TrainTime;
            bestmodel.validScore = validScore_avg;
        end;
        fprintf('%.4f\t|\tBest model N=%d C=2^%d g=2^%d EVAL=%.4f\n', ...
            validScore_avg, bestmodel.N, bestmodel.C, bestmodel.g, bestmodel.validScore);
    end;
    end;
end;
bestmodels = [];
for i=1:5
    bestmodels = [bestmodels bestmodel];
end
save bestmodels bestmodels;
result = zeros(5, 6);

% fprintf(info);

test_ranksvm_bestmodel(dataset_txt, dataset_mat, bestmodels, 1);

