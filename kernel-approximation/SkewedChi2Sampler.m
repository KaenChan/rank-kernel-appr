classdef SkewedChi2Sampler
    % Approximates feature map of the "skewed chi-squared" kernel by Monte
    % Carlo approximation of its Fourier transform.

    % Parameters
    % ----------
    % skewedness : float
    %     "skewedness" parameter of the kernel. Needs to be cross-validated.

    % n_components : int
    %     number of Monte Carlo samples per original feature.
    %     Equals the dimensionality of the computed feature space.

    % random_state : {int, RandomState}, optional
    %     If int, random_state is the seed used by the random number generator;
    %     if RandomState instance, random_state is the random number generator.

    % References
    % ----------
    % See "Random Fourier Approximations for Skewed Multiplicative Histogram
    % Kernels" by Fuxin Li, Catalin Ionescu and Cristian Sminchisescu.
    
    properties
        skewedness;
        n_components;
        seed;
        random_weights;
        random_offset;
    end
    
    methods
        %Constructer
        function obj =  SkewedChi2Sampler(X, n_components, skewedness, seed)
            if nargin  ==  4
                rng('default');
                rng(seed);
                obj.seed = seed;
            end
            obj.skewedness = skewedness;
            obj.n_components = n_components ;
            n_features = size(X,2);

            % normally distribute
	        uniform = randn(n_features, n_components);
	        % transform by inverse CDF of sech
	        random_weights = (1. / pi * log(tan(pi / 2. * uniform)));
            % uniform distribute
	        random_offset = rand(0, 2 * pi, n_components)

            if nargin  ==  4
                rng('default');
            end
        end
        
        function projection = transform(obj, X)
	        X = log(X + obj.skewedness);
            projection = X * obj.random_weights;
            projection = projection + repmat(obj.random_offset, size(X,1), 1);
            projection = cos(projection);
            projection = projection * (sqrt(2) / sqrt(obj.n_components));
        end
      
    end
    
end

