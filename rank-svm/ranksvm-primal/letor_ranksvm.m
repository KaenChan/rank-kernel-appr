function letor_ranksvm(dataset)
% The function should be called with the path of the dataset from
% the Letor 3.0 distribution. For instance, if you are in
% Gov/QueryLevelNorm, type: letor_ranksvm('2003_hp_dataset')

  global X Xt;
  mkdir(dataset,['ranksvm']); % Where the prdictions and metrics will be stored
  
  for i=1:5, % Loop over the folds
    % Read the training and validation data
    dname = [dataset '/Fold' num2str(i) '/'];
    [X, Y ] = read_letor([dname '/train.txt']);
    [Xt,Yt] = read_letor([dname '/vali.txt']);

    use_kernel_appr = 1
    if use_kernel_appr
      n_components = 1000;

      option.appr_type = 'rbf';
      option.appr_type = 'nystroem';
      % option.appr_type = 'improvednystroem';
      gamma = 0.01;
      seed = 0;

      % RBFSampler
      option.rbf.gamma = gamma;
      % NystroemSampler
      option.nystroem.kernel = 'rbf';
      option.nystroem.gamma = gamma;
      option.nystroem.coef0 = 1; 
      option.nystroem.degree = 3;

      switch lower(option.appr_type)
          case 'rbf'
              kernelsampler = RBFSampler(X, n_components, option.nystroem.gamma, seed);
          case 'nystroem'
              kernelsampler = NystroemSampler(n_components, option.nystroem.kernel, ...
                  option.nystroem.gamma, option.nystroem.coef0, option.nystroem.degree, ...
                  seed);
              kernelsampler = fit(kernelsampler, X);
          case 'improvednystroem'
              kernelsampler = ImprovedNystroemSampler(n_components, option.nystroem.kernel, ...
                  option.nystroem.gamma, option.nystroem.coef0, option.nystroem.degree, ...
                  seed);
              kernelsampler = fit(kernelsampler, X);
          otherwise
              warning('error');
      end

      X = transform(kernelsampler, X);
      Xt = transform(kernelsampler, Xt);
    end

    % Generate the preference pairs; see ranksvm.m for the format of this matrix.
    A = generate_constraints(Y);
    clear w;
    for j=1:5 % Model selection
      opt.lin_cg=1;
      C = 10^(j-3)/size(A,1); % Dividing C by the number of pairs
      w(:,j) = ranksvm(X, A, C*ones(size(A,1),1),zeros(size(X,2),1),opt);
      map(j) = compute_map(Xt*w(:,j),Yt); % MAP value on the validation set
    end;
    fprintf('C = %f, MAP = %f\n',[10.^[-2:2]; map]) 
    [foo, j] = max(map); % Best MAP value
    w = w(:,j);
    % Print predictions and compute the metrics.
    write_out(X*w,i,'train',dataset)
    write_out(Xt*w,i,'vali',dataset)
    [Xt,Yt] = read_letor([dname '/test.txt']);
    if use_kernel_appr
      Xt = transform(kernelsampler, Xt);
    end
    write_out(Xt*w,i,'test',dataset)
  end;
  system(['python src/evaluation/run_evaluation.py ' dataset ' test']);
  
function [X,Y] = read_letor(filename)
  f = fopen(filename);
  X = zeros(2e5,0);
  qid = '';
  i = 0; q = 0;
  while 1
    l = fgetl(f);
    if ~ischar(l), break; end;
    i = i+1; 
    [lab,  foo1, foo2, ind] = sscanf(l,'%d qid:',1); l(1:ind-1)=[];
    [nqid, foo1, foo2, ind] = sscanf(l,'%s',1); l(1:ind-1)=[]; 
    if ~strcmp(nqid,qid)
      q = q+1;
      qid = nqid;
      Y{q} = lab;
    else 
      Y{q} = [Y{q}; lab];
    end;
    tmp = sscanf(l,'%d:%f'); 
    X(i,tmp(1:2:end)) = tmp(2:2:end);
  end;
  X = X(1:i,:);
  fclose(f);

function write_out(output,i,name,dataset)
  output = output + 1e-10*randn(length(output),1);  % Break ties at random
  fname = [dataset '/ranksvm/' name '.fold' num2str(i)];
  save(fname,'output','-ascii');
  % Either copy the evaluation script in the current directory or
  % change the line below with the correct path 
%   system(['perl Eval-Score-3.0.pl ' dataset '/Fold' num2str(i) '/' name ...
%           '.txt ' fname ' ' fname '.metric 0']);
  % system(['perl Eval-Score-4.0.pl ' dataset '/Fold' num2str(i) '/' name ...
  %         '.txt ' fname ' ' fname '.metric 0']);
  system(['python e:/work/ml-work/learning-to-rank/src-work/pgbrt/pgbrt/scripts/evaluate.py ' dataset '/Fold' num2str(i) '/' name '.txt ' fname]);
       

function A = generate_constraints(Y)
  nq = length(Y);
  
  I=zeros(1e7,1); J=I; V=I; nt = 0;
  
  ind = 0;
  for i=1:nq
    ind = ind(end)+[1:length(Y{i})]';
    Y2 = Y{i};
    n = length(ind);
    [I1,I2] = find(repmat(Y2,1,n)>repmat(Y2',n,1));
    n = length(I1);
    I(2*nt+1:2*nt+2*n) = nt+[1:n 1:n]'; 
    J(2*nt+1:2*nt+2*n) = [ind(I1); ind(I2)];
    V(2*nt+1:2*nt+2*n) = [ones(n,1); -ones(n,1)];
    nt = nt+n;
  end;
  A = sparse(I(1:2*nt),J(1:2*nt),V(1:2*nt));    

function map = compute_map(Y,Yt)
  ind = 0;
  for i=1:length(Yt)
    ind = ind(end)+[1:length(Yt{i})];
    [foo,ind2] = sort(-Y(ind));
    r = Yt{i}(ind2)>0;
    p = cumsum(r) ./ [1:length(r)]';
    if sum(r)> 0 
      map(i) = r'*p / sum(r);
    else
      map(i)=0;
    end;
  end;
  map=mean(map);

