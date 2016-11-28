% clear
% clc
% 
% load ./data/train_set/words_train.mat
% X1 = full(X);
% X1 = [ones(4500, 1), X1];
% load ./data/train_set_unlabeled/words_train_unlabeled.mat
% X2 = full(X);
% X2 = [ones(4500, 1), X2];
% X = [X1; X2];

%% AdaBoostM1
% ada = fitensemble(X1,Y,'AdaBoostM1', 300,'Tree','LearnRate',0.1);
% err_train_ada = mean(predict(ada, X1) == Y);

%% Robust Boost
rb1 = fitensemble(X1, Y, 'RobustBoost', 4500, 'Tree','RobustErrorGoal',0.15,'RobustMaxMargin',1);
% ind = crossvalind('Kfold', 4500, 10);
% num_models = [4500, 5000, 5500, 6000, 6500, 7000];
% precision_rb11 = zeros(1, 6);
% precision_rb21 = zeros(1, 6);
% for j = 1: 6
%     X_train = X1(ind ~= 1, :);
%     X_test = X1(ind == 1, :);
%     Y_train = Y(ind ~= 1);
%     Y_test = Y(ind == 1);
%     rb1 = fitensemble(X1,Y,'RobustBoost',num_models(j),...
%         'Tree','RobustErrorGoal',0.15,'RobustMaxMargin',1);
%     Yhat_rb1 = predict(rb1, X_test);
%     precision_rb11(j) = mean(Yhat_rb1 == Y_test);
%     rb2 = fitensemble(X1,Y,'RobustBoost',num_models(j),...
%         'Tree','RobustErrorGoal',0.01);
%     Yhat_rb2 = predict(rb2, X_test);
%     precision_rb21(j) = mean(Yhat_rb2 == Y_test);
% end
% precision_rb1 = precision_rb1 ./ 10;
% precision_rb2 = precision_rb2 ./ 10;
