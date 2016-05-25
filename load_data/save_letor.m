function save_letor(filename, X, Y, qids)
  disp(['save_letor -> ' filename]);
  fid = fopen(filename, 'w');

  for i = 1:length(qids)
    q = qids{i};
    for j = 1:length(q)
      features = X(q(j),:);
      y = Y(q(j));
      s = sprintf('%d qid:%d ', y, i);
      for k = 1:length(features)
        if features(k)~=0
          s = [s sprintf('%d:%f ',k,features(k))];
        end
      end
      fprintf(fid,[s '\n']);
    end
  end
  fclose(fid);
