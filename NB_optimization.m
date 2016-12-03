clear
clc
close all

load ./data/train_set/words_train.mat
% X = full(X);
% Y=full(Y);

% mu = mean(X);
% sigma = std(X);

% X = (X - mu) ./ sigma;

% X(isnan(X)) = 0;

% rng default
% Mdl = fitcnb(X(:, 1: 3000),Y,'Distribution','kernel',...
%     'OptimizeHyperparameters',{'Width', 'Kernel'},...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus', 'MaxObjectiveEvaluations', 50,...
%     'Holdout', 0.1));

tic;
Yhat = predict_labels(X, [], [], [], [], []);
toc;
acc = mean(Yhat == Y);
