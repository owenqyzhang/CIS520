clear
clc

load ./data/train_set/words_train.mat
% X = full(X);
% Y = full(Y);

% robust_boost = fitensemble(X, Y, 'RobustBoost', 4500, 'Tree','RobustErrorGoal',0.15,'RobustMaxMargin',1);

tic;
Yhat = predict_labels(X, [], [], [], [], []);
toc;
acc = mean(Yhat == Y);