function eva = compute_metric(Ypred,Yt,qids,metric_type)
    switch lower(metric_type.name)
        case {'map'}
            eva = compute_map(Ypred,Yt,qids);

        case {'r2score'}
            eva = compute_r2score(Ypred,Yt,qids);

        case {'mse'}
            eva = compute_mse(Ypred,Yt,qids);
            
        case {'mae'}
            eva = compute_mae(Ypred,Yt,qids);

        case {'ndcg'}
            eva = compute_ndcg(Ypred,Yt,qids,metric_type.k_ndcg);

        case {'auc'}
            eva = compute_auc(Ypred,Yt,metric_type.posclass);
    end