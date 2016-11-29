clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];

unzip('./models.zip')
load ./models/coeff.mat
X1_pca = (X1 - mean(X1)) * coeff;
X2_pca = (X2 - mean(X2)) * coeff;
Y = full(Y);

%%
% [coeff, score] = pca(X, 'NumComponents', 750);

%% AdaBoostM1
% optimization = fitcensemble(X1_pca,Y,'OptimizeHyperparameters','all');
% ada = fitcensemble(X1_pca, Y, 'Method', 'LogitBoost', 'NumLearningCycles', 500,...
%     'LearnRate', 0.69396);
% save('./models/ada.mat','ada','-v7.3')
Yhat = predict_labels(sparse(X1), [], [], [], [], []);
acc = mean(Yhat == Y);