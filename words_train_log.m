clear
clc

load ./data/train_set/words_train.mat
X = full(X);
load ./data/train_set/train_cnn_feat.mat
X_cnn = train_cnn_feat;
Y = full(Y);

ind = crossvalind('Kfold', 4500, 10);
idx = 1: 4500;
idx_test = find(ind == 1);
idx_train = idx;
idx_train(idx_test) = [];

X_train = X(idx_train, :);
X_cnn_train = X_cnn(idx_train, :);
Y_train = Y(idx_train);

X_test = X(idx_test, :);
X_cnn_test = X_cnn(idx_test, :);
Y_test = Y(idx_test);

%% Naive Bayes
NB = fitcnb(X_train, Y_train, 'Distribution', 'mn');

Yhat_train_NB = predict(NB, X_train);
acc_train_NB = mean(Yhat_train_NB == Y_train);

Yhat_NB = predict(NB, X_test);
acc_NB = mean(Yhat_NB == Y_test);

%% SVM Word Count
SVM_W = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf',...
    'Standardize', true, 'KernelScale', 100);

Yhat_train_SVM_W = predict(SVM_W, X_train);
acc_train_SVM_W = mean(Yhat_train_SVM_W == Y_train);

Yhat_SVM_W = predict(SVM_W, X_test);
acc_SVM_W = mean(Yhat_SVM_W == Y_test);

%% SVM CNN
SVM_CNN = fitcsvm(X_cnn_train, Y_train, 'KernelFunction', 'rbf',...
    'Standardize', true, 'KernelScale', 250);

Yhat_train_SVM_CNN = predict(SVM_CNN, X_cnn_train);
acc_train_SVM_CNN = mean(Yhat_train_SVM_CNN == Y_train);

Yhat_SVM_CNN = predict(SVM_CNN, X_cnn_test);
acc_SVM_CNN = mean(Yhat_SVM_CNN == Y_test);

%% Logistic Regression
addpath('liblinear/');

logistic = train(Y_train, sparse(X_train), ['-s 0', 'col']);

Yhat_train_logistic = predict(ones(4050, 1), sparse(X_train),...
    logistic, ['-q', 'col']);
acc_train_logistic = mean(Yhat_train_logistic == Y_train);

Yhat_logistic = predict(ones(450, 1), sparse(X_test), logistic,...
    ['-q', 'col']);
acc_logistic = mean(Yhat_logistic == Y_test);

%% Majority Vote
Yhat_train = [Yhat_train_logistic, Yhat_train_NB, Yhat_train_SVM_CNN, Yhat_train_SVM_W];
Y_train_est = mode(Yhat_train, 2);
acc_train_vote = mean(Y_train_est == Y_train);

Yhat = [Yhat_logistic, Yhat_NB, Yhat_SVM_CNN, Yhat_SVM_W];
Y_est = mode(Yhat, 2);
acc_vote = mean(Y_est == Y_test);
