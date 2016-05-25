function x = query_level_normalize(x, qids, norm_type)
	for i = 1:length(qids)
		q = qids{i};
		sub_x = x(q);
		if strcmp(norm_type, 'zscore')
			m = mean(sub_x, 1);
			s = std(sub_x, 1);
		elseif strcmp(norm_type, 'minmax')
			xmin = min(sub_x, [], 1);
			xmax = max(sub_x, [], 1);
			m = xmin;
			s = xmax-xmin;
		end
		s(s==0) = 1;
	    sub_x=bsxfun(@minus,sub_x,m);
		sub_x=bsxfun(@rdivide,sub_x,s);
		x(q) = sub_x;
	end
end
