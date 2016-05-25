function ndcg_1q = compute_ndcgk_query(r, k)
    dcg = compute_dcg_query(r);
    ideal_dcg = compute_dcg_query(sort(r, 'descend'));
    if ideal_dcg > 0
        ndcg_temp = dcg ./ ideal_dcg;
    else
        ndcg_temp = zeros(1,length(r));
    end;
    if k>length(r)
        idx=r;
    else
        idx=k;
    end
    if k==0
        ndcg_1q = mean(ndcg_temp);
    else
        ndcg_1q = ndcg_temp(idx);
    end
