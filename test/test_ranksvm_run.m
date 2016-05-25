function test_ranksvm_run()
    % dataset_txt = 'E:\work\ml-datasets\letor3.0\OHSUMED\QueryLevelNorm';
    % dataset_mat = 'data/letor3_ohsumed_fold'

    % dataset = 'td2003';
    % dataset = 'td2004';
    dataset = 'OHSUMED';
    % dataset = 'MQ2007';
    % dataset = 'MQ2008';
    % dataset = 'mslr10k';

    [dataset_txt, dataset_mat] = get_dataset_name(dataset);

    metric_type.name = 'NDCG';
    metric_type.name = 'MAP';
    metric_type.k_ndcg = 5;


    % RBFSampler
    % NystroemSampler
    appr_type = 'nystroem';
    % appr_type = 'improvednystroem';
    % appr_type = 'fourier';
    kernel = 'rbf';
    coef0 = 1; 
    degree = 3;
    seed = 0;

    if strcmp(dataset, 'td2004')
        if strcmp(appr_type, 'nystroem')
            Ns = ones(5,1) * 500;
            Cs = ones(5,1) * 0;
            gs = ones(5,1) * -8;
        elseif strcmp(appr_type, 'nystroem-diff')
            Ns = ones(5,1) * 500;
            Cs = ones(5,1) * 0;
            gs = ones(5,1) * -8;
        elseif strcmp(appr_type, 'improvednystroem')
            Ns = ones(5,1) * 500;
            Cs = ones(5,1) * 0;
            gs = ones(5,1) * -8;
        elseif strcmp(appr_type, 'fourier')
            Ns = ones(5,1) * 500;
            Cs = ones(5,1) * 0;
            gs = ones(5,1) * -8;
        end
    elseif strcmp(dataset, 'OHSUMED')
        if strcmp(appr_type, 'nystroem')
            Ns = ones(5,1) * 500;
            % Cs = ones(5,1) * -8;
            % gs = ones(5,1) * -4;
            Cs = ones(5,1) * -6;
            gs = ones(5,1) * -6;
        elseif strcmp(appr_type, 'improvednystroem')
            Ns = ones(5,1) * 500;
            Cs = ones(5,1) * -8;
            gs = ones(5,1) * -4;
        elseif strcmp(appr_type, 'fourier')
            Ns = ones(5,1) * 500;
            % Cs = ones(5,1) * -4;
            % gs = ones(5,1) * -10;
            Cs = ones(5,1) * -6;
            gs = ones(5,1) * -6;
        end
    elseif strcmp(dataset, 'MQ2007')
        if strcmp(appr_type, 'nystroem')
            Ns = ones(5,1) * 2000;
            Cs = ones(5,1) * 4;
            gs = ones(5,1) * -8;
        elseif strcmp(appr_type, 'improvednystroem')
            Ns = ones(5,1) * 2000;
            Cs = ones(5,1) * 4;
            gs = ones(5,1) * -8;
        elseif strcmp(appr_type, 'fourier')
            Ns = ones(5,1) * 2000;
            Cs = ones(5,1) * 4;
            gs = ones(5,1) * -8;
        end
    end

    % Cs = ones(5,1) * -12;  % primal-ranksvm ohsumed
    % Cs = ones(5,1) * -12;  % primal-ranksvm td2004
    % Cs = ones(5,1) * 6;  % primal-ranksvm mslr10k
    % gs = ones(5,1) * -5;

    load([dataset_mat num2str(1)]);

    info = '\ntest_ranksvm_option\n';
    info = [info sprintf('dataset     = %s\n', dataset)];
    info = [info sprintf('X_train     = %s\n', mat2str(size(X_train)))];
    info = [info sprintf('Q_train     = %d\n', length(Q_train))];
    info = [info sprintf('X_vali      = %s\n', mat2str(size(X_vali)))];
    info = [info sprintf('Q_vali      = %d\n', length(Q_vali))];
    info = [info sprintf('X_test      = %s\n', mat2str(size(X_test)))];
    info = [info sprintf('Q_test      = %d\n', length(Q_test))];
    info = [info sprintf('metric_type = %s\n', metric_type.name)];
    info = [info sprintf('k_ndcg      = %d\n', metric_type.k_ndcg)];
    info = [info sprintf('N           = %s\n', mat2str(Ns))];
    info = [info sprintf('C           = %s\n', mat2str(Cs))];
    info = [info sprintf('gammas      = %s\n', mat2str(gs))];
    info = [info sprintf('appr_type   = %s\n', appr_type)];
    info = [info sprintf('kernel      = %s\n', kernel)];
    info = [info '\n'];

    for i=1:5, % Loop over the folds
        load([dataset_mat num2str(i)]);
        C = Cs(i);
        g = gs(i);
        seed = fix(mod(cputime,100));
        option.seed = seed;
        option.verbose = 0;
        option.metric_type = metric_type;
        option.iter_max_Newton = 100;
        
        N = Ns(i); C = Cs(i); g = gs(i);
        fprintf('Fold%d N=%d C=2^%d g=2^%d ', i, N, C, g);
        
        model = KernelApprRankSVM(N, 2^C, appr_type, kernel, 2^g, coef0, degree, seed);
        % model = PrimalRankSVM(2^C);
        % model = FenchelRankSVM(2^C, 0.01, 1000);
        % model = KernelApprFenchelRankSVM(N, 2^C,  0.01, 1000, appr_type, kernel, 2^g, coef0, degree, seed);

        model = fit(model, X_train, Y_train, Q_train, option);
        pred_train = predict(model, X_train);
        pred_valid = predict(model, X_vali);
        TrainEVAL = compute_metric(pred_train, Y_train, Q_train, model.metric_type);
        ValidEVAL = compute_metric(pred_valid, Y_vali, Q_vali, model.metric_type);

        pred = predict(model, X_test);
        TestEVAL = compute_metric(pred, Y_test, Q_test, model.metric_type);
        fprintf('Time=%.4f s EVAL=(%.4f %.4f %.4f)\n', model.TrainTime, TrainEVAL, ValidEVAL, TestEVAL);
        Times{i} = model.TrainTime;
        EVALs{i} = [TrainEVAL, ValidEVAL, TestEVAL];
        write_out(pred,i,'test',dataset_txt);
    end;

    fprintf(info);

    result = zeros(5, 7);
    fprintf('====================================================================\n');
    fprintf('      | N    | C     | gamma | Train Time | %-6s                 |\n', ...
        model.metric_type.name);
    fprintf('--------------------------------------------------------------------\n');
    for i=1:5
        Time = Times{i};
        EVAL = EVALs{i};
        fprintf('Fold%d | %-4d | 2^%-3d | 2^%-3d | %7.4f s  | (%.4f %.4f %.4f) |\n', ...
            i, Ns(i), Cs(i), gs(i), Time, ...
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
    system(['python evaluation/run_evaluation.py ' dataset_txt ' test']);
    % disp(' '); disp('Performance on validation set');
    % system(['python src/evaluation/run_evaluation.py ' dataset_txt ' vali']);
    % disp(' '); disp('Performance on training set');
    % system(['python src/evaluation/run_evaluation.py ' dataset_txt ' train']);
