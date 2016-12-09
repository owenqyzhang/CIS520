clear
clc

load ./data/train_set/words_train.mat
X = full(X);
X = X(:, 2: end);
X = double(X > 0);
Y = full(Y);

%% Decision Tree

model = fitctree(X, Y, 'OptimizeHyperParameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch',...
    'AcquisitionFunctionName', 'expected-improvement-plus', 'NumGridDivisions', 20));
% idx = 1: 4500;
% ind = crossvalind('Kfold', 4500, 5);

% train_acc = zeros(91, 1);
% test_acc = zeros(91, 1);

% for i = 1: 5
%     idx_test = find(ind == i);
%     idx_train = find(ind ~= i);
%     
%     X_test = X(idx_test, :);
%     X_train = X(idx_train, :);
%     
%     Y_test = Y(idx_test);
%     Y_train = Y(idx_train);

%     for depth = 10: 100
%         dt = dt_train(X_train, Y_train, depth);
%         
%         Y_train_hat = zeros(size(Y_train));
%         Y_hat = zeros(size(Y_test));
%         
%         for p = 1: size(X_train, 1)
%             [~, Y_train_hat(p)] = max(dt_value(dt, X_train(p, :)));
%         end
%         
%         for p = 1:size(X_test,1)
%             [~, Y_hat(p)] = max(dt_value(dt, X_test(p, :)));
%         end
%         
%         train_acc(depth - 9) = mean(Y_train_hat == Y_train);
%         test_acc(depth - 9) = mean(Y_hat ~= Y_test);
%         
%     end
% end

% train_acc = train_acc / 5;
% test_acc = test_acc / 5;