clear
clc

load ./data/train_set/words_train.mat

Yhat = predict_labels(X, [], [], [], [], []);
acc = mean(Yhat == Y);
% X = full(X);
% Y = full(Y);

% load ./models/coeff.mat
% X_pca = X * coeff;

% ind = crossvalind('Kfold', 4500, 10);
% idx = 1: 4500;
% idx_test = find(ind == 1);
% idx_train = idx;
% idx_train(idx_test) = [];

% X_train = X(idx_train, :);
% X_pca_train = X_pca(idx_train, :);
% Y_train = Y(idx_train);

% X_test = X(idx_test, :);
% X_pca_test = X_pca(idx_test, :);
% Y_test = Y(idx_test);

% t = templateTree('MinLeafSize', 1155, 'MaxNumSplits', 12);
% Boost = fitcensemble(X_train, Y_train, 'Method', 'LogitBoost', 'Learners', 'Tree', 'NumLearningCycles', 490, 'CrossVal', 'off', 'LearnRate', 0.17);

% Yhat_train = predict(Boost, X_train);
% acc_train = mean(Yhat_train == Y_train);

% Yhat = predict(ada_compact, X_pca_test);
% acc_test = mean(Yhat == Y_test);