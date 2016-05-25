function write_out(output,i,name,dataset)
    if exist([dataset '/rankelm/'],'dir')==0
       mkdir([dataset '/rankelm/']);
    end
%   output = output + 1e-10*randn(length(output),1);  % Break ties at random
    fname = [dataset '/rankelm/' name '.fold' num2str(i)];
%   save(fname,'output','-ascii');
    fidout=fopen(fname,'w');
    for k = 1:length(output)
    fprintf(fidout,'%f\n',output(k));
    end;
    fclose(fidout);
%   Either copy the evaluation script in the current directory or
%   change the line below with the correct path 
%   system(['perl Eval-Score-3.0.pl ' dataset '/Fold' num2str(i) '/' name ...
%           '.txt ' fname ' ' fname '.metric 0']);
%   system(['perl src/evaluation/evaluate_pl/Eval-Score-4.0.pl ' dataset '/Fold' num2str(i) '/' name ...
%           '.txt ' fname ' ' fname '.metric 0']);
%   ['python e:/work/ml-work/learning-to-rank/src-work/pgbrt/pgbrt/scripts/evaluate.py ' dataset '/Fold' num2str(i) '/' name '.txt ' fname ' > result.txt']
%   system(['python src/evaluation/evaluate_py/evaluate.py ' dataset '/Fold' num2str(i) '/' name '.txt ' fname ' > result.txt']);
%   f = fopen('result.txt');
%   l = fgetl(f);
%   result = sscanf(l,'rmse: %f, err: %f, ndcg: %f',3);
%   fclose(f);
%   err = result(2);
%   ngcd = result(3);