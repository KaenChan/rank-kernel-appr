function test_ranksvm_run_mq2007fold1

    % dataset = 'OHSUMED';
    dataset = 'MQ2007';
    [dataset_txt, dataset_mat] = get_dataset_name(dataset);

    metric_type.name = 'NDCG';
    % metric_type.name = 'MAP';
    metric_type.k_ndcg = 0;
        
    % RBFSampler
    % NystroemSampler
    % appr_type = 'nystroem';
    % appr_type = 'improvednystroem';
    % appr_type = 'fourier';
    kernel = 'rbf';
    coef0 = 1; 
    degree = 3;
    seed = 0;

    Ns = ones(1,10)*2000;
    Cs = -10:2:4;
    Cs = -4;
    gs = 1;
    gs = -10:2:0;
    
    Cs = -2;   gs = -5;
    
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

    fprintf(info);

    for N = Ns
    for C = Cs
    for g = gs
        i = 1;
        load([dataset_mat num2str(i)]);
        seed = fix(mod(cputime,100));
        option.seed = seed;
        option.verbose = 0;
        option.metric_type = metric_type;
        option.iter_max_Newton = 100;

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

        % system(['perl src/evaluation/run_evaluation.py ' dataset_txt ' test']);
        cmd = sprintf('perl src/evaluation/evaluate_pl/Eval-Score-4.0.pl %s/Fold%d/test.txt %s/rankelm/test.fold%d temp 0', ...
            dataset_txt, i, dataset_txt, i);
        system(cmd);
        system('cat temp');
    end
    end
    end
