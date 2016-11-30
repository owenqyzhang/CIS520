clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);

%% Logistic Regression
addpath('liblinear/');

log_ori = train(full(Y), sparse(X1), ['-s 7', 'col']);

[y_unlabeled_est, ~, prob_estimates] = predict(ones(4500, 1), sparse(X2), log_ori, ['-q', 'col']);

idx = 1: 4500;
ind_unlabeled_inlier = idx(abs(prob_estimates) > 1);
X2_inlier = X2(ind_unlabeled_inlier, :);
Y_unlabeled_inlier = y_unlabeled_est(ind_unlabeled_inlier);

X = [X1; X2_inlier];
Y = [full(Y); Y_unlabeled_inlier];

log_ori_full = train(full(Y), sparse(X), ['-s 7', 'col']);
save('./models/log_ori_full.mat', 'log_ori_full', '-v7.3');

% precision_ori_log = zeros(9, 1);
% ind = crossvalind('Kfold', 4500, 10);
% for i = 1: 10
%     idx = 1: 9000;
%     idx_test = find(ind == i);
%     idx_train = idx;
%     idx_train(idx_test) = [];
%     
%     model = train(full(Y(idx_train)), sparse(X(idx_train, :)),...
%         ['-s 0', 'col']);
%     Yhat = predict(ones(450, 1), sparse(X(idx_test, :)),...
%         model, ['-q', 'col']);
%     precision_ori_log(i) = mean(Yhat == Y(idx_test));
% end
% precision_ori_log_ave = mean(precision_ori_log);
