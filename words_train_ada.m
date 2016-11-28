clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];
load ./models/coeff.mat
X1_pca = X1 * coeff;
X2_pca = X2 * coeff;

%%
% [coeff, score] = pca(X, 'NumComponents', 750);

%% AdaBoostM1
% ada = fitensemble(X1,Y,'AdaBoostM1', 300,'Tree','LearnRate',0.1);
% err_train_ada = mean(predict(ada, X1) == Y);

%% Robust Boost
ind = crossvalind('Kfold', 4500, 10);
num_models = [2000, 2500, 3000, 3500, 4000];
precision_rb = zeros(1, 5);
precision_ada = zeros(1, 5);
for j = 1: 5
    X_train = X1_pca(ind ~= 1, :);
    X_test = X1_pca(ind == 1, :);
    Y_train = Y(ind ~= 1);
    Y_test = Y(ind == 1);
    rb = fitcensemble(X_train,Y_train,'Method','RobustBoost',...
        'NumLearningCycles',num_models(j),'Learners','tree');
    Yhat_rb1 = predict(rb, X_test);
    precision_rb(j) = mean(Yhat_rb1 == Y_test);
    ada = fitcensemble(X_test,Y_train,'RobustBoost',num_models(j),...
        'Tree','RobustErrorGoal',0.01);
    Yhat_rb2 = predict(ada, X_test);
    precision_ada(j) = mean(Yhat_rb2 == Y_test);
end
