function [pred, TestTime, TestEVAL] = kernel_approximation_predict(model, Xt, Yt)

%     load rank_elm_model.mat;
    kernelsampler = model.kernelsampler;
    OutputWeight = model.OutputWeight;

    %%%%%%%%%%% Calculate the output of testing input
    t1=clock;

    H_test = transform(kernelsampler, Xt);

    pred = (H_test * OutputWeight);
    
    TestEVAL = 0;
    if ~isempty(Yt)
        TestEVAL = evaluation_preds(pred, Yt, model.learn_type);
    end

    t2 = clock;
    TestTime = etime(t2,t1);
