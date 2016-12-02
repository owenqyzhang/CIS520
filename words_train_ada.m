clear
clc

load ./data/train_set/words_train.mat
X = full(X);

Y = full(Y);

% Yhat = predict_labels(sparse(X), [], [], [], [], []);
% acc = mean(Yhat == Y);
%%
idx = 1: 4500;
ind = crossvalind('Kfold', 4500, 10);

acc_train = 0;
acc_test = 0;
for i = 1: 10
    ind_train = idx(ind == i);
    ind_test = idx(ind ~= i);
    
    X_train = X(ind_train, :);
    X_test = X(ind_test, :);
    
    Y_train = Y(ind_train);
    Y_test = Y(ind_test);
    
    rb = fitensemble(X_train, Y_train, 'RobustBoost', 4000, 'Tree',...
        'RobustErrorGoal', 0.15, 'RobustMaxMargin', 1);
    
    Yhat_train = predict(rb, X_train);
    acc_train = acc_train + mean(Yhat_train == Y_train);
    
    Yhat_test = predict(rb, X_test);
    acc_test = acc_test + mean(Yhat_test == Y_test);
end