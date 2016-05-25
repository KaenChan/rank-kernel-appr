function make()
try
	% mex -g fenchel_rank.cpp fenchel_dual_svm.cpp
	mex fenchel_rank.cpp fenchel_dual_svm.cpp
catch err
	fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
	disp(err.message);
	fprintf('=> Please check README for detailed instructions.\n');
end
