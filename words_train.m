clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];
load ./models/coeff.mat

%% PCA Dimensioin Reduction
% cov_train = cov(X);
% [coeff_train, latent] = pcacov(cov_train);
score_train = X * coeff_train(:, 1: 750);

%% Naive Bayes
Mdl_pca = fitcnb(score_train(1: 4500, :), Y);
% save('./models/nb_pca.mat', 'Mdl_pca', '-v7.3');
y_unlabeled_est = predict(Mdl_pca, score_train(4501: 9000, :));
Y = [full(Y); y_unlabeled_est];
Mdl_pca_full = fitcnb(score_train(:, :), Y);
save('./models/nb_pca_full.mat', 'Mdl_pca', '-v7.3');

precision_pca_nb = zeros(9, 1);
ind = crossvalind('Kfold', 4500, 10);
for i = 1: 10
    idx = 1: 9000;
    idx_test = find(ind == i);
    idx_train = idx;
    idx_train(idx_test) = [];
    
    X_pca_train = score_train(idx_train, :);
    Y_train = Y(idx_train);
    X_pca_test = score_train(idx_test, :);
    Y_test = Y(idx_test);
    Mdl = fitcnb(X_pca_train, Y_train);
    Yhat_pca = predict(Mdl, X_pca_test);
    precision_pca_nb(i) = mean(Yhat_pca == Y_test);
end
precision_pca_nb_ave = mean(precision_pca_nb);
