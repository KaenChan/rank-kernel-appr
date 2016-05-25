classdef FenchelRankSVM

properties
    r;          %  r : the constrain of ||w|| <= r
    max_iter;   %  max_iter : the maximum number of iterations
    err;        %  err : the required accuray
    w;
    TrainTime;
    TrainMAP;
    TrainNDCG;
    metric_type;
end

methods

    function obj = FenchelRankSVM(r,err,max_iter)
        obj.r = r;
        obj.err = err;
        obj.max_iter = max_iter;
    end

    function obj = fit(obj, X, y, qids, option)
        %% input option
        obj.metric_type = option.metric_type;

        t1=tic;

        obj.w = fenchel_rank(X, y, qids, obj.r, obj.max_iter, obj.err, option.verbose);

        obj.TrainTime = toc(t1);

        % TrainingTime=toc;
        pred=(X * obj.w);

        obj.TrainMAP  = compute_map(pred, y, qids);
        obj.TrainNDCG = compute_ndcg(pred, y, qids, obj.metric_type.k_ndcg);
    end

    function [pred, TestTime] = predict(obj, X)
        t1=tic;
        pred=(X * obj.w);
        t2 = clock;
        TestTime = toc(t1);
    end

end
end
