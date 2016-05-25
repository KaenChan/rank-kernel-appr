function model = kernel_approximation_train(X_train, Y_train, option)

%% input option
n_components = option.n_components;
c_rho        = 2^option.c_rho;
learn_type   = option.learn_type;

% learn_type = 'regression';
% learn_type = 'classifier';

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end
rng('default');
rng(seed);

%% Woad training dataset

T=Y_train;

n_data=size(X_train,1);
n_features=size(X_train,2);

T=double(T);

t1=tic;

%% Calculate weights & biases

switch lower(option.appr_type)
    case 'rbf'
        kernelsampler = RBFSampler(X_train, n_components, option.nystroem.gamma, seed);
    case 'nystroem'
        kernelsampler = NystroemSampler(n_components, option.nystroem.kernel, ...
            option.nystroem.gamma, option.nystroem.coef0, option.nystroem.degree, ...
            seed);
        kernelsampler = fit(kernelsampler, X_train);
    case 'improvednystroem'
        kernelsampler = ImprovedNystroemSampler(n_components, option.nystroem.kernel, ...
            option.nystroem.gamma, option.nystroem.coef0, option.nystroem.degree, ...
            seed);
        kernelsampler = fit(kernelsampler, X_train);
    otherwise
        warning('error');
end

H = transform(kernelsampler, X_train);
n_components = size(H,2);

%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=((H'*H+(eye(n)/c_rho))\(H'*T)); 

if size(H,1) > size(H,2)
    HH = H'*H;
    HT = H'*T;
    OutputWeight=((HH+(eye(size(H,2))/c_rho))\(HT)); 
else
    HH = H*H';
    OutputWeight=H'*((HH+(eye(size(H,1))/c_rho))\(T)); 
end

TrainTime = toc(t1);

% TrainTime=toc;
%%%%%%%%%%% Calculate the training accuracy
pred = (H * OutputWeight);

if strcmp(learn_type, 'classifier')
    %%%%%%%%%% Calculate training & testing classification accuracy
    missclassified=0;

    for i = 1 : size(X_train, 1)
        [x, label_index_expected]=max(pred(i,:));
        [x, label_index_actual]=max(Y_train(i,:));
        if label_index_actual ~= label_index_expected
            missclassified = missclassified + 1;
        end
    end
    TrainEVAL = 1-missclassified/n_data;
elseif strcmp(learn_type, 'regression')
    TrainEVAL = sqrt(mse(Y_train - pred));
end

model = option;
model.learn_type = learn_type;
model.kernelsampler = kernelsampler;
model.n_components = n_components;
model.N = n_components;
model.OutputWeight = OutputWeight;
model.c_rho = option.c_rho;
model.TrainTime = TrainTime;
model.TrainEVAL = TrainEVAL;
