classdef ImprovedNystroemSampler
    % Approximate a kernel map using a subset of the training data.

    % The Improved Nystrom method uses K-means clustering centers as the 
    % landmark points, which can greatly improve approximation quality of the 
    % kernel matrix. Efficiency of the kmeans step therefore should be taken 
    % into account. Considering that the kmeans function in matlab is very 
    % inefficient, [1] wrote a faster one (eff_kmeans.m); the number of K-means 
    % iterations should be a small number, e.g. 5.  

    % Constructs an approximate feature map for an arbitrary kernel
    % using a subset of the data as basis.

    % Parameters
    % ----------
    % kernel  string or callable, functionault="rbf"
    %     Kernel map to be approximated. A callable should accept two arguments
    %     and the keyword arguments passed to this object as kernel_params, and
    %     should return a floating point number.

    % n_components  int
    %     Number of features to construct.
    %     How many data points will be used to construct the mapping.

    % gamma  float, functionault
    %     Gamma parameter for the RBF, polynomial, exponential chi2 and
    %     sigmoid kernels. Interpretation of the functionault value is left to
    %     the kernel; see the documentation for sklearn.metrics.pairwise.
    %     Ignored by other kernels.

    % coef0  float, functionault=1
    %     Zero coefficient for polynomial and sigmoid kernels.
    %     Ignored by other kernels.

    % kernel_params  mapping of string to any, optional
    %     Additional parameters (keyword arguments) for kernel function passed
    %     as callable object.

    % random_state  {int, RandomState}, optional
    %     If int, random_state is the seed used by the random number generator;
    %     if RandomState instance, random_state is the random number generator.


    % Attributes
    % ----------
    % components_  array, shape (n_components, n_features)
    %     Subset of training points used to construct the feature map.

    % component_indices_  array, shape (n_components)
    %     Indices of ``components_`` in the training set.

    % normalization_  array, shape (n_components, n_components)
    %     Normalization matrix needed for embedding.
    %     Square root of the kernel matrix on ``components_``.


    % References
    % ----------
    % [1] Kai Zhang, Ivor W. Tsang, James T. Kwok. 
    %   "Improved Nystrom Low Rank Approximation and Error Analysis". 
    %   In the 25th International Conference on Machine Learning (ICML 2008), 
    %   Helsinki, Finland, June 2008.

    % [2] Kai Zhang and James T. Kwok, 
    %   "Clustered Nyström Method for Large Scale Manifold Learning and Dimension Reduction", 
    %   IEEE Transactions on Neural Networks, Volume: 21 , Issue: 10 Page(s): 1576 - 1587, 2010.

    properties
        kernel;
        gamma;
        coef0;
        degree;
        n_components;
        seed;
        components_;
        component_idxs_;
        normalization_;
        pairwise_kernels_inst;
    end

    methods
    function obj = ImprovedNystroemSampler(n_components, kernel, gamma, coef0, degree, seed)
        % kernel = 'rbf';
        % gamma = 1;
        % coef0 = 1; 
        % degree = 3;
        obj.kernel = kernel;
        obj.gamma = gamma;
        obj.coef0 = coef0;
        obj.degree = degree;
        obj.n_components = n_components;
        obj.seed = seed;
    end

    function obj = fit(obj, X)
        % Fit estimator to data.

        % Samples a subset of training points, computes kernel
        % on these and computes normalization matrix.

        % Parameters
        % ----------
        % X  array-like, shape=(n_samples, n_feature)
        %     Training data.
        
        n_samples = size(X, 1);

        % get basis vectors

        if obj.n_components > n_samples
            % XXX should we just bail?
            n_components_ = n_samples;
            warning(['n_components > n_samples. This is not possible.\n'
                     'n_components was set to n_samples, which results'
                     ' in inefficient evaluation of the full kernel.'])
        else
            n_components_ = obj.n_components;
        end
        n_components_ = min(n_samples, n_components_);

        % #iteration is restricted to 5
        [idxs, basis, n_components_] = eff_kmeans(X, n_components_, 5); 

        obj.pairwise_kernels_inst = pairwise_kernels(obj.kernel, obj.gamma, obj.coef0, obj.degree);
        basis_kernel = apply(obj.pairwise_kernels_inst, basis, basis);

        % sqrt of kernel matrix on basis vectors
        W = basis_kernel;
        [Ve, Va] = eig(W);
        va = diag(Va);
        pidx = find(va > 1e-6);
        inVa = sparse(diag(va(pidx).^(-0.5)));
        G = Ve(:,pidx) * inVa;
        obj.normalization_ = G;
        obj.components_ = basis;
        obj.component_idxs_ = idxs;
    end

    function projection = transform(obj, X)
        % Apply feature map to X.

        % Computes an approximate feature map using the kernel
        % between some training points and X.

        % Parameters
        % ----------
        % X  array-like, shape=(n_samples, n_features)
        %     Data to transform.

        % Returns
        % -------
        % X_transformed  array, shape=(n_samples, n_components)
        %     Transformed data.
        
        embedded = apply(obj.pairwise_kernels_inst, X, obj.components_);
        projection = embedded * obj.normalization_;
    end
    end
end


function [idx, center, m] = eff_kmeans(data, m, MaxIter);

[n, dim] = size(data);
dex = randperm(n);
center = data(dex(1:m),:);

for i = 1:MaxIter;
    nul = zeros(m,1);
    [xx, idx] = min(sqdist(center', data'));
    for j = 1:m;
        dex = find(idx == j);
        l = length(dex);
        cltr = data(dex,:);
        if l > 1;
            center(j,:) = mean(cltr);
        elseif l == 1;
            center(j,:) = cltr;
        else
            nul(j) = 1;
        end;
    end;
    dex = find(nul == 0);
    m = length(dex);
    center = center(dex,:);
end;

end

function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.*a,1); bb = sum(b.*b,1); ab = a'*b; 
d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

end