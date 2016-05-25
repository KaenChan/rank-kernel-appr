function test_ranksvm_bestmodel(dataset_txt, dataset_mat, bestmodels, retrain)
    % dataset_txt = 'E:\work\ml-datasets\letor3.0\OHSUMED\QueryLevelNorm';
    % dataset_mat = 'data/letor3_ohsumed_fold'
    for i=1:5, % Loop over the folds
        load([dataset_mat num2str(i)]);
        if retrain==0
            model = bestmodels(i).trained_model;
            TrainEVAL = bestmodels(i).EVAL(1);
            ValidEVAL = bestmodels(i).EVAL(2);
        else
            model = bestmodels(i).trained_model;
            n_components = bestmodels(i).N;
            C = bestmodels(i).C;
            g = bestmodels(i).g;
            seed = fix(mod(cputime,100));
            option.seed = seed;
            option.verbose = 0;
            option.metric_type = model.metric_type;
            
            if strcmp(class(model), 'PrimalRankSVM')
                model = PrimalRankSVM(2^C);
            elseif strcmp(class(model), 'KernelApprFenchelRankSVM')
                model = KernelApprFenchelRankSVM(n_components, 2^C,  0.01, 1000, model.appr_type, ...
                    model.kernel, 2^g, model.coef0, model.degree, seed);
            elseif strcmp(class(model), 'KernelApprRankSVM')
                appr_type = model.appr_type;
                kernel = model.kernel;
                coef0 = model.coef0; 
                degree = model.degree;
                clear model;
                model = KernelApprRankSVM(n_components, 2^C, appr_type, kernel, 2^g, coef0, degree, seed);
            end
            model = fit(model, X_train, Y_train, Q_train, option);
            pred_train = predict(model, X_train);
            pred_valid = predict(model, X_vali);
            TrainEVAL = compute_metric(pred_train, Y_train, Q_train, model.metric_type);
            ValidEVAL = compute_metric(pred_valid, Y_vali, Q_vali, model.metric_type);
        end
        pred = predict(model, X_test);
        TestEVAL = compute_metric(pred, Y_test, Q_test, model.metric_type);
        Times{i} = model.TrainTime;
        EVALs{i} = [TrainEVAL, ValidEVAL, TestEVAL];
        write_out(pred,i,'test',dataset_txt);
    end;

    result = zeros(5, 7);
    fprintf('====================================================================\n');
    fprintf('      | N    | C     | gamma | Train Time | %-6s                 |\n', ...
        model.metric_type.name);
    fprintf('--------------------------------------------------------------------\n');
    for i=1:5
        bestmodel = bestmodels(i);
        Time = Times{i};
        EVAL = EVALs{i};
        fprintf('Fold%d | %-4d | 2^%-3d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) |\n', ...
            i, bestmodel.N, bestmodel.C, bestmodel.g, Time, ...
            EVAL(1), EVAL(2), EVAL(3));
        result(i,1) = Time;
        result(i,2:4) = EVAL;
    end;
  
    avgmap = mean(result);
    fprintf('--------------------------------------------------------------------\n');
    fprintf('Mean  |                      | %7.4f s  | (%.4f %.4f %.4f) |\n', ...
        avgmap(1), avgmap(2), avgmap(3), avgmap(4));
    fprintf('--------------------------------------------------------------------\n');

    disp(' '); disp('Performance on testing set');
    system(['python src/evaluation/run_evaluation.py ' dataset_txt ' test']);
    % disp(' '); disp('Performance on validation set');
    % system(['python src/evaluation/run_evaluation.py ' dataset_txt ' vali']);
    % disp(' '); disp('Performance on training set');
    % system(['python src/evaluation/run_evaluation.py ' dataset_txt ' train']);
