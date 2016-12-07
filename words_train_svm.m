clear
clc

load ./data/train_set/words_train.mat
Y = full(Y);

%% Logistic Regression
% addpath('libsvm/');

% K = kernel_gaussian(X, X, 30);
% svm = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', 150));

% idx = 1: 4500;

% err_35_cv = 0;
% err_25_cv = 0;
% err_30_cv = 0;
% err_40_cv = 0;
% err_20_cv = 0;
% ind = crossvalind('Kfold', 4500, 10);
% for i = 1: 10
%     idx_test = find(ind == i);
%     idx_train = find(ind ~= i);
%     
%     X_test = X(idx_test, :);
%     X_train = X(idx_train, :);
%     
%     Y_test = Y(idx_test);
%     Y_train = Y(idx_train);
%     
%     k_gauss_35 = @(x, x2) kernel_gaussian(x, x2, 35);
%     [err_gauss_35, ~] = kernel_libsvm(X_train, Y_train, X_test, Y_test, k_gauss_35);
%     
%     k_gauss_25 = @(x, x2) kernel_gaussian(x, x2, 25);
%     [err_gauss_25, ~] = kernel_libsvm(X_train, Y_train, X_test, Y_test, k_gauss_25);
% 
%     k_gauss_30 = @(x, x2) kernel_gaussian(x, x2, 30);
%     [err_gauss_30, ~] = kernel_libsvm(X_train, Y_train, X_test, Y_test, k_gauss_30);
% 
%     k_gauss_40 = @(x, x2) kernel_gaussian(x, x2, 40);
%     [err_gauss_40, ~] = kernel_libsvm(X_train, Y_train, X_test, Y_test, k_gauss_40);    
% 
%     k_gauss_20 = @(x, x2) kernel_gaussian(x, x2, 20);
%     [err_gauss_20, ~] = kernel_libsvm(X_train, Y_train, X_test, Y_test, k_gauss_20);    
% 
%     err_35_cv = err_35_cv + err_gauss_35;
%     err_25_cv = err_25_cv + err_gauss_25;
%     err_30_cv = err_30_cv + err_gauss_30;
%     err_40_cv = err_40_cv + err_gauss_40;
%     err_20_cv = err_20_cv + err_gauss_20;
% end

% err_35_cv = err_35_cv / 10;
% err_25_cv = err_25_cv / 10;
% err_30_cv = err_30_cv / 10;
% err_40_cv = err_40_cv / 10;
% err_20_cv = err_20_cv / 10;

Yhat = predict_labels(X, [], [], [], [], []);