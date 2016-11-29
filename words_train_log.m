clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
X1 = [ones(4500, 1), X1];
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X2 = [ones(4500, 1), X2];
X = [X1; X2];

%% Logistic Regression
initial_w = zeros(size(X, 2), 1);

lambda = 2;

learn_rate = linspace(0.06, 0.09, 10);

ind = crossvalind('Kfold', 4500, 10);
precision = zeros(10, 1);
for i = 1: 10
    for j = 1
        X_train = X1(ind ~= j, :);
        X_test = X1(ind == j, :);
        Y_train = Y(ind ~= j);
        Y_test = Y(ind == j);
        w = gradientDescent(X_train, Y_train, initial_w, learn_rate(i), 1000, lambda);
        Yhat = predict_log(w, X_test);
        precision(i) = precision(i) + mean(Yhat == Y_test);
    end
end
% precision = precision ./ 10;

% w_labeled = gradientDescent(X1, Y, initial_w, 0.01, 1000, 2);
