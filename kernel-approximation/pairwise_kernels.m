classdef pairwise_kernels
    % Compute the kernel between arrays X and optional array Y.

    % This method takes either a vector array or a kernel matrix, and returns
    % a kernel matrix. If the input is a vector array, the kernels are
    % computed. If the input is a kernel matrix, it is returned instead.

    % This method provides a safe way to take a kernel matrix as input, while
    % preserving compatibility with many other algorithms that take a vector
    % array.

    % If Y is given (default is None), then the returned matrix is the pairwise
    % kernel between the arrays from both X and Y.

    % Valid values for metric are::
    %     ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

    % Parameters
    % ----------
    % X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
    %          [n_samples_a, n_features] otherwise
    %     Array of pairwise kernels between samples, or a feature array.

    % Y : array [n_samples_b, n_features]
    %     A second feature array only if X has shape [n_samples_a, n_features].

    % metric : string, or callable
    %     The metric to use when calculating kernel between instances in a
    %     feature array. If metric is a string, it must be one of the metrics
    %     in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    %     If metric is "precomputed", X is assumed to be a kernel matrix.
    %     Alternatively, if metric is a callable function, it is called on each
    %     pair of instances (rows) and the resulting value recorded. The callable
    %     should take two arrays from X as input and return a value indicating
    %     the distance between them.

    properties
        metric;
        gamma;
        coef0;
        degree;
    end

	methods
    function obj = pairwise_kernels(metric, gamma, coef0, degree);
    	obj.metric = metric;
        obj.gamma = gamma;
        obj.coef0 = coef0;
        obj.degree = degree;
    end

    function K = apply(obj, X, Y)
	    switch lower(obj.metric)
	        case 'linear'
	        	K = linear_kernel(X, Y);
	        case {'poly','polynomial'}
	        	K = polynomial_kernel(X, Y, obj.degree, obj.gamma, obj.coef0);
	        case 'rbf'
	        	K = rbf_kernel(X, Y, obj.gamma);
	        case 'cosine'
	        	K = cosine_similarity(X, Y);
	        case 'chi2'
	        	K = chi2_kernel(X, Y);
	        case 'sigmoid'
	        	K = sigmoid_kernel(X, Y);
	        case 'additive_chi2'
	        	K = additive_chi2_kernel(X, Y);
	        otherwise
	            warning('error');
        end
	end
end
end

% Kernels
function K = linear_kernel(X, Y)
    
    % Compute the linear kernel between X and Y.

    % Parameters
    % ----------
    % X : array of shape (n_samples_1, n_features)

    % Y : array of shape (n_samples_2, n_features)

    % Returns
    % -------
    % Gram matrix : array of shape (n_samples_1, n_samples_2)
    
    K = X * Y';
end


function K = polynomial_kernel(X, Y, degree, gamma, coef0)
    
    % Compute the polynomial kernel between X and Y::

    %     K(X, Y) = (gamma <X, Y> + coef0)^degree

    % Parameters
    % ----------
    % X : ndarray of shape (n_samples_1, n_features)

    % Y : ndarray of shape (n_samples_2, n_features)

    % coef0 : int, functionault 1

    % degree : int, functionault 3

    % Returns
    % -------
    % Gram matrix : array of shape (n_samples_1, n_samples_2)
    
    % gamma = 1.0 / X.shape[1]

    K = X * Y';
    K = (K * gamma + coef0) ^ degree;
end


function sigmoid_kernel(X, Y, gamma, coef0)
    
    % Compute the sigmoid kernel between X and Y::

    %     K(X, Y) = tanh(gamma <X, Y> + coef0)

    % Parameters
    % ----------
    % X : ndarray of shape (n_samples_1, n_features)

    % Y : ndarray of shape (n_samples_2, n_features)

    % coef0 : int, functionault 1

    % Returns
    % -------
    % Gram matrix: array of shape (n_samples_1, n_samples_2)
    
    % gamma = 1.0 / X.shape[1]

    K = X * Y';
    K = K * gamma + coef0;
    K = np.tanh(K, K);   % compute tanh in-place
end


function K = rbf_kernel(X, Y, gamma)
    
    % Compute the rbf (gaussian) kernel between X and Y::

    %     K(x, y) = exp(-gamma ||x-y||^2)

    % for each pair of rows x in X and y in Y.

    % Parameters
    % ----------
    % X : array of shape (n_samples_X, n_features)

    % Y : array of shape (n_samples_Y, n_features)

    % gamma : float

    % Returns
    % -------
    % kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    
    % gamma = 1.0 / X.shape[1]

    % K = euclidean_distances(X, Y, squared=True)
	K = distSquared(X, Y);
    K = exp(K * (-gamma));
end


function K = cosine_similarity(X, Y)
    % Compute cosine similarity between samples in X and Y.

    % Cosine similarity, or the cosine kernel, computes similarity as the
    % normalized dot product of X and Y:

    %     K(X, Y) = <X, Y> / (||X||*||Y||)

    % On L2-normalized data, this function is equivalent to linear_kernel.

    % Parameters
    % ----------
    % X : array_like, sparse matrix
    %     with shape (n_samples_X, n_features).

    % Y : array_like, sparse matrix (optional)
    %     with shape (n_samples_Y, n_features).

    % Returns
    % -------
    % kernel matrix : array
    %     An array with shape (n_samples_X, n_samples_Y).
    
    % to avoid recursive import

    X = X';
    mu = mean(X);
    sigma = std(X);
    X=bsxfun(@minus,X,mu);
	X_normalized=bsxfun(@rdivide,X,sigma);

    Y = Y';
    mu = mean(Y);
    sigma = std(Y);
    Y=bsxfun(@minus,Y,mu);
	Y_normalized=bsxfun(@rdivide,Y,sigma);

    K = X_normalized' * Y_normalized;
end


function K = additive_chi2_kernel(X, Y)
    % Computes the additive chi-squared kernel between observations in X and Y

    % The chi-squared kernel is computed between each pair of rows in X and Y.  X
    % and Y have to be non-negative. This kernel is most commonly applied to
    % histograms.

    % The chi-squared kernel is given by::

    %     k(x, y) = -Sum [(x - y)^2 / (x + y)]

    % It can be interpreted as a weighted difference per entry.

    % Notes
    % -----
    % As the negative of a distance, this kernel is only conditionally positive
    % functioninite.


    % Parameters
    % ----------
    % X : array-like of shape (n_samples_X, n_features)

    % Y : array of shape (n_samples_Y, n_features)

    % Returns
    % -------
    % kernel_matrix : array of shape (n_samples_X, n_samples_Y)

    % References
    % ----------
    % * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
    %   Local features and kernels for classification of texture and object
    %   categories: A comprehensive study
    %   International Journal of Computer Vision 2007
    %   http://eprints.pascal-network.org/archive/00002309/01/Zhang06-IJCV.pdf


    % See also
    % --------
    % chi2_kernel : The exponentiated version of the kernel, which is usually
    %     preferable.

    % sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
    %     to this kernel.
    
    warning('todo next');
    % k(x, y) = -Sum [(x - y)^2 / (x + y)]
end


function K = chi2_kernel(X, Y, gamma)
    % Computes the exponential chi-squared kernel X and Y.

    % The chi-squared kernel is computed between each pair of rows in X and Y.  X
    % and Y have to be non-negative. This kernel is most commonly applied to
    % histograms.

    % The chi-squared kernel is given by::

    %     k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

    % It can be interpreted as a weighted difference per entry.

    % Parameters
    % ----------
    % X : array-like of shape (n_samples_X, n_features)

    % Y : array of shape (n_samples_Y, n_features)

    % gamma : float, functionault=1.
    %     Scaling parameter of the chi2 kernel.

    % Returns
    % -------
    % kernel_matrix : array of shape (n_samples_X, n_samples_Y)

    % References
    % ----------
    % * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
    %   Local features and kernels for classification of texture and object
    %   categories: A comprehensive study
    %   International Journal of Computer Vision 2007
    %   http://eprints.pascal-network.org/archive/00002309/01/Zhang06-IJCV.pdf

    % See also
    % --------
    % additive_chi2_kernel : The additive version of this kernel

    % sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
    %     to the additive version of this kernel.
    
    K = exp( gamma * additive_chi2_kernel(X, Y) );
end

function D2 = distSquared(X, Y)
%
nx	= size(X,1);
ny	= size(Y,1);
%
D2 = (sum((X.^2), 2) * ones(1,ny)) + (ones(nx, 1) * sum((Y.^2),2)') - ...
     2*X*Y';
end
