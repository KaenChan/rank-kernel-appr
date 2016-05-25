classdef RBFSampler
    % Approximates feature map of an RBF kernel by Monte Carlo approximation
    % of its Fourier transform.

    % It implements a variant of Random Kitchen Sinks.[1]

    % Parameters
    % ----------
    % gamma : float
    %     Parameter of RBF kernel: exp(-gamma * x^2)

    % n_components : int
    %     Number of Monte Carlo samples per original feature.
    %     Equals the dimensionality of the computed feature space.

    % random_state : {int, RandomState}, optional
    %     If int, random_state is the seed used by the random number generator;
    %     if RandomState instance, random_state is the random number generator.

    % Notes
    % -----
    % See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    % Benjamin Recht.

    % [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    % minimization with randomization in learning" by A. Rahimi and
    % Benjamin Recht.
    % (http://www.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    
    properties
        gamma;
        n_components;
        seed;
        random_weights;
        random_offset;
    end
    
    methods
        %Constructer
        function obj = RBFSampler(n_components, n_features, gamma, seed)
            if nargin  ==  4
                rng('default');
                rng(seed);
                obj.seed = seed;
            end
            obj.gamma = gamma;
            obj.n_components = n_components ;
            
            % normally distribute
            obj.random_weights = (sqrt(2 * gamma) * randn(n_features, n_components));
            % uniform distribute
            obj.random_offset = rand(1,n_components) * 2 * pi;

            if nargin  ==  4
                rng('default');
            end
        end
        
        function projection = transform(obj, X)
            projection = X * obj.random_weights;
            projection = projection + repmat(obj.random_offset, size(X,1), 1);
            projection = cos(projection);
            projection  = projection * (sqrt(2) / sqrt(obj.n_components));
        end
      
        function [obj, w_scale] = appr_keep(obj, inds)
            new_n = length(inds);
            w_scale = sqrt(new_n) / sqrt(obj.n_components);
            obj.n_components = new_n;
            obj.random_weights = obj.random_weights(:,inds);
            obj.random_offset = obj.random_offset(inds);
        end
    end
    
end

