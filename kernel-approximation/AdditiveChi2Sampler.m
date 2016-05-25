classdef AdditiveChi2Sampler
    % Approximate feature map for additive chi2 kernel.

    % Uses sampling the fourier transform of the kernel characteristic
    % at regular intervals.

    % Since the kernel that is to be approximated is additive, the components of
    % the input vectors can be treated separately.  Each entry in the original
    % space is transformed into 2*sample_steps+1 features, where sample_steps is
    % a parameter of the method. Typical values of sample_steps include 1, 2 and
    % 3.

    % Optimal choices for the sampling interval for certain data ranges can be
    % computed (see the reference). The default values should be reasonable.

    % Parameters
    % ----------
    % sample_steps : int, optional
    %     Gives the number of (complex) sampling points.
    % sample_interval : float, optional
    %     Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    % Notes
    % -----
    % This estimator approximates a slightly different version of the additive
    % chi squared kernel then ``metric.additive_chi2`` computes.

    % References
    % ----------
    % See `"Efficient additive kernels via explicit feature maps"
    % <http://eprints.pascal-network.org/archive/00006964/01/vedaldi10.pdf>`_
    % Vedaldi, A. and Zisserman, A., Computer Vision and Pattern Recognition 2010
    
    properties
        skewedness;
        n_components;
        seed;
        random_weights;
        random_offset;
    end
    
    methods
        %Constructer
        function obj =  AdditiveChi2Sampler(X, n_components, gamma, seed)
            if nargin  ==  4
                rng('default');
                rng(seed);
                obj.seed = seed;
            end
            obj.gamma = gamma;
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
        
        function projection = rbfsample_apply(obj, X)
            projection = X * obj.random_weights;
            projection = projection + repmat(obj.random_offset, size(X,1), 1);
            projection = cos(projection);
            projection  = projection * (sqrt(2) / sqrt(obj.n_components));
        end
      
    end
    
end


