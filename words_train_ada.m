clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];
load ./models/coeff.mat
X1_pca = (X1 - mean(X1)) * coeff;
X2_pca = (X2 - mean(X2)) * coeff;
Y = full(Y);

%%
% [coeff, score] = pca(X, 'NumComponents', 750);

%% AdaBoostM1
ada = fitcensemble(X1_pca,Y,'OptimizeHyperparameters','all');
save('./models/ada.mat','ada','-v7.3')
Yhat = predict_labels(X1_pca, [], [], [], [], []);
acc = mean(Yhat == Y);