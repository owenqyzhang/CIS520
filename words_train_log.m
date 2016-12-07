clear
clc

load ./data/train_set/words_train.mat
% X1 = full(X);
% X1 = X1(:, 5: end);
% load ./data/train_set_unlabeled/words_train_unlabeled.mat
% X2 = full(X);

%% Logistic Regression
% addpath('liblinear/');

% log_ori = train(full(Y), sparse(X1), ['-s 0', 'col']);

% [y_unlabeled_est, ~, prob_estimates] = predict(ones(4500, 1), sparse(X2), log_ori, ['-q', 'col']);

% idx = 1: 4500;
% ind_unlabeled_inlier = idx(abs(prob_estimates) > 1);
% X2_inlier = X2(ind_unlabeled_inlier, :);
% Y_unlabeled_inlier = y_unlabeled_est(ind_unlabeled_inlier);

% X = [X1; X2];
% Y = [full(Y); y_unlabeled_est];

% log_ori_full = train(full(Y), sparse(X), ['-s 0', 'col']);
% save('./models/log_ori_full.mat', 'log_ori_full', '-v7.3');

% ind = crossvalind('Kfold', 4500, 10);
% acc = 0;
% for i = 1: 10
%     idx = 1: 4500;
%     idx_test = find(ind == i);
%     idx_train = find(ind ~= i);
%     
%     model = train(full(Y(idx_train)), sparse(X(idx_train, :)),...
%         ['-s 0', 'col']);
%     Yhat = predict(ones(450, 1), sparse(X(idx_test, :)),...
%         model, ['-q', 'col']);
%     acc = acc + mean(Yhat == Y(idx_test));
% end
Yhat = predict_labels(X, [], [], [], [], []);