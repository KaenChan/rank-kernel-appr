classdef PrimalRankSVM

properties
    C;
    w;
    TrainTime;
    TrainMAP;
    TrainNDCG;
    metric_type;
end

methods

    function obj = PrimalRankSVM(C)
        obj.C = C;
    end

    function obj = fit(obj, X, y, qids, option)
        %% input option
        obj.metric_type = option.metric_type;

        t1=tic;

        A = generate_constraints(y, qids);

        opt.lin_cg=1;
        opt.verbose=option.verbose;
        if isfield(option,'iter_max_Newton')
           opt.iter_max_Newton = option.iter_max_Newton;
        end;  
        obj.w = ranksvm(X, A, obj.C*ones(size(A,1),1), zeros(size(X,2),1),opt);

        obj.TrainTime = toc(t1);

        % TrainingTime=toc;
        %%%%%%%%%%% Calculate the training accuracy
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

function A = generate_constraints(y, qids)
  nq = length(qids);
  
  I=zeros(1e7,1); J=I; V=I; nt = 0;
  
  ind = 0;
  for i=1:nq
    qs = qids{i};
    ind = ind(end)+[1:length(qs)]';
    y2 = y(qs);
    n = length(ind);
    [I1,I2] = find(repmat(y2,1,n)>repmat(y2',n,1));
    n = length(I1);
    I(2*nt+1:2*nt+2*n) = nt+[1:n 1:n]'; 
    J(2*nt+1:2*nt+2*n) = [ind(I1); ind(I2)];
    V(2*nt+1:2*nt+2*n) = [ones(n,1); -ones(n,1)];
    nt = nt+n;
  end;
  A = sparse(I(1:2*nt),J(1:2*nt),V(1:2*nt));    
  if size(A,2)~=size(y,1)
      A(size(A,1),size(y,1))=0;
  end
end
