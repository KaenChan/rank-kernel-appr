function [X,Y_out,qids] = read_letor(filename)
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
  
  Y_out = zeros(i,1);
  iter = 0;
  for i=1:length(Y)
      Y_q = Y{i};
      qid = zeros(length(Y_q),1);
      for j=1:length(Y_q)
          iter = iter + 1;
          Y_out(iter) = Y_q(j);
          qid(j) = iter;
      end
      qids{i} = qid;
  end

