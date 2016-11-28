clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];

%% Logistic Regression
addpath('liblinear/');
log_ori = train(full(Y), sparse(X1), ['-s 0', 'col']);
y_unlabeled_est = predict(ones(4500, 1), sparse(X2), log_ori, ['-q', 'col']);
Y = [full(Y); y_unlabeled_est];
log_ori_full = train(full(Y), sparse(X), ['-s 0', 'col']);
save('./models/log_ori_full.mat', 'log_ori_full', '-v7.3');

precision_ori_log = zeros(9, 1);
ind = crossvalind('Kfold', 4500, 10);
for i = 1: 10
    idx = 1: 9000;
    idx_test = find(ind == i);
    idx_train = idx;
    idx_train(idx_test) = [];
    
    model = train(full(Y(idx_train)), sparse(X(idx_train, :)),...
        ['-s 0', 'col']);
    Yhat = predict(ones(450, 1), sparse(X(idx_test, :)),...
        model, ['-q', 'col']);
    precision_ori_log(i) = mean(Yhat == Y(idx_test));
end
precision_ori_log_ave = mean(precision_ori_log);
