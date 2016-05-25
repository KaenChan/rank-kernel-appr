function load_data_to_mat()

% dataset = 'data\TD2004';
dataset = 'data\OHSUMED';
% dataset = 'data\MQ2007';
% dataset = 'data\MQ2008';

t1=clock;
for i=1:5, % Loop over the folds
    % Read the training and validation data
    dname = [dataset '/Fold' num2str(i) '/']
    [X_train, Y_train, Q_train] = read_letor([dname '/train.txt']);
    [X_vali,Y_vali, Q_vali] = read_letor([dname '/vali.txt']);
    [X_test,Y_test, Q_test] = read_letor([dname '/test.txt']);
    
    fprintf(['save data/letor3_ohsumed_fold' num2str(i) '\n']);
    save(['data/letor3_ohsumed_fold' num2str(i)], 'X_train', 'Y_train', 'Q_train', ...
          'X_vali', 'Y_vali', 'Q_vali', 'X_test', 'Y_test', 'Q_test');

    % fprintf(['save data/letor4_mq2007_fold' num2str(i) '\n']);
    % save(['data/letor4_mq2007_fold' num2str(i)], 'X_train', 'Y_train', 'Q_train', ...
    %       'X_vali', 'Y_vali', 'Q_vali', 'X_test', 'Y_test', 'Q_test');

    % fprintf(['save data/letor4_mq2008_fold' num2str(i) '\n']);
    % save(['data/letor4_mq2008_fold' num2str(i)], 'X_train', 'Y_train', 'Q_train', ...
    %       'X_vali', 'Y_vali', 'Q_vali', 'X_test', 'Y_test', 'Q_test');

    
    % [X_train, Y_train] = read_letor([dname '/trainingset.txt']);
    % [X_vali,Y_vali] = read_letor([dname '/validationset.txt']);
    % [X_test,Y_test] = read_letor([dname '/testset.txt']);
    %
    % fprintf(['save data/letor2_td2004_fold' num2str(i) '\n']);
    % save(['data/letor2_td2004_fold' num2str(i)], 'X_train', 'Y_train', 'Q_train', ...
    %       'X_vali', 'Y_vali', 'Q_vali', 'X_test', 'Y_test', 'Q_test');
end;
t2=clock;
fprintf('Total time: %.2f s\n', etime(t2,t1));
