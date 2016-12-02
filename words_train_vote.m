clear
clc

load ./data/train_set/words_train.mat

Yhat = predict_labels(X, [], [], [], [], []);
acc = mean(Yhat == Y);

X = full(X);
Xh = [ones(4500, 1), X];
% load ./data/train_set/train_cnn_feat.mat
% X_cnn = train_cnn_feat;
Y = full(Y);

%% Randomly divide into training set and test set.
% ind = crossvalind('Kfold', 4500, 10);
% idx = 1: 4500;
% idx_test = find(ind == 1);
% idx_train = idx;
% idx_train(idx_test) = [];

% X_train = X(idx_train, :);
% Xh_train = [ones(4050, 1), X_train];
% Y_train = Y(idx_train);

% X_test = X(idx_test, :);
% Xh_test = [ones(450, 1), X_test];
% Y_test = Y(idx_test);

%% LogitBoost
% boost = fitcensemble(X_pca_train, Y_train, 'Method', 'LogitBoost',...
%     'NumLearningCycles', 500, 'LearnRate', 0.69396);

% Yhat_train_boost = predict(boost, X_pca_train);
% acc_train_boost = mean(Yhat_train_boost == Y_train);

% Yhat_boost = predict(boost, X_pca_test);
% acc_boost = mean(Yhat_boost == Y_test);

%% K-Nearest Neighbors
% KNN = fitcknn(Xh_train, Y_train, 'NumNeighbors', 18);

% Yhat_train_knn = predict(KNN, Xh_train);
% acc_train_knn = mean(Yhat_train_knn == Y_train);

% Yhat_knn = predict(KNN, Xh_test);
% acc_knn = mean(Yhat_knn == Y_test);

%% Naive Bayes
% NB = fitcnb(Xh_train, Y_train, 'Distribution', 'mn');

% Yhat_train_NB = predict(NB, Xh_train);
% acc_train_NB = mean(Yhat_train_NB == Y_train);

% Yhat_NB = predict(NB, Xh_test);
% acc_NB = mean(Yhat_NB == Y_test);

%% SVM Word Count
% SVM_W = fitcsvm(Xh_train, Y_train, 'KernelFunction', 'rbf',...
%     'Standardize', true, 'KernelScale', 100);

% Yhat_train_SVM_W = predict(SVM_W, Xh_train);
% acc_train_SVM_W = mean(Yhat_train_SVM_W == Y_train);

% Yhat_SVM_W = predict(SVM_W, Xh_test);
% acc_SVM_W = mean(Yhat_SVM_W == Y_test);

%% SVM CNN
% SVM_CNN = fitcsvm(X_cnn_train, Y_train, 'KernelFunction', 'rbf',...
%     'Standardize', true, 'KernelScale', 250);

% Yhat_train_SVM_CNN = predict(SVM_CNN, X_cnn_train);
% acc_train_SVM_CNN = mean(Yhat_train_SVM_CNN == Y_train);

% Yhat_SVM_CNN = predict(SVM_CNN, X_cnn_test);
% acc_SVM_CNN = mean(Yhat_SVM_CNN == Y_test);

%% Logistic Regression Liblinear
% addpath('liblinear/');

% logistic = train(Y_train, sparse(Xh_train), ['-s 7', 'col']);

% Yhat_train_logistic = predict(ones(4050, 1), sparse(Xh_train),...
%     logistic, ['-q', 'col']);
% acc_train_logistic = mean(Yhat_train_logistic == Y_train);

% Yhat_logistic = predict(ones(450, 1), sparse(Xh_test), logistic,...
%     ['-q', 'col']);
% acc_logistic = mean(Yhat_logistic == Y_test);

%% Logistic Regression Gradient Descent
% initial_w = zeros(size(Xh_train, 2), 1);
% w = gradientDescent(Xh_train, Y_train, initial_w, 0.07, 1000, 2);

% Yhat_train_log_gd = predict_log(w, Xh_train);
% acc_train_log_gd = mean(Yhat_train_log_gd == Y_train);

% Yhat_log_gd = predict_log(w, Xh_test);
% acc_log_gd = mean(Yhat_log_gd == Y_test);

%% Majority Vote
% Yhat_train = [Yhat_train_logistic, Yhat_train_NB, Yhat_train_log_gd,...
%     Yhat_train_SVM_W, Yhat_train_boost, Yhat_train_knn];
% Y_train_vote_est = mode(Yhat_train, 2);
% Vote_log = train(Y_train, sparse(Yhat_train), ['-s 0', 'col']);
% Y_train_vote_log = predict(ones(4050, 1), sparse(Yhat_train),...
%     Vote_log, ['-q', 'col']);
% acc_train_vote = mean(Y_train_vote_est == Y_train);
% acc_train_vote_log = mean(Y_train_vote_log == Y_train);

% Yhat = [Yhat_logistic, Yhat_NB, Yhat_SVM_W, Yhat_boost, Yhat_log_gd,...
%     Yhat_knn];
% Y_vote_log = predict(ones(450, 1), sparse(Yhat),...
%     Vote_log, ['-q', 'col']);
% Y_est = mode(Yhat, 2);
% acc_vote = mean(Y_est == Y_test);
% acc_vote_log = mean(Y_vote_log == Y_test);
%% Train models with all the data
% boost = fitcensemble(X_pca, Y, 'Method', 'LogitBoost',...
%     'NumLearningCycles', 500, 'LearnRate', 0.69396);

% KNN = fitcknn(Xh, Y, 'NumNeighbors', 18);

% NB = fitcnb(Xh, Y, 'Distribution', 'mn');

% SVM_W = fitcsvm(Xh, Y, 'KernelFunction', 'rbf',...
%     'Standardize', true, 'KernelScale', 100);

% SVM_CNN = fitcsvm(X_cnn, Y, 'KernelFunction', 'rbf',...
%     'Standardize', true, 'KernelScale', 250);

% addpath('liblinear/');
% logistic = train(Y, sparse(Xh), ['-s 7', 'col']);

% initial_w = zeros(size(Xh, 2), 1);
% w = gradientDescent(Xh, Y, initial_w, 0.07, 1000, 2);
