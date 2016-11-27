clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];
load ./models/coeff.mat
X_pca = X * coeff;
% load ./train_set/words_train_pca.mat
% load ./train_set/words_train_rbm.mat

%%
% idx = 1: 9000;
% idx_test = randperm(9000, 900);
% idx_train = idx;
% idx_train(idx_test) = [];
% [coeff, score_train, score_test, numpc] = ...
%     pca_getpc(X(idx_train, :), X(idx_test, :));


%% Bayes GMM
% mu_test = mean(test_pca);
% 
% uni = [1, 1];
% alt = [sum(full(Y(1: 4000)) == 0), sum(full(Y(1: 4000)) == 0)];
% 
% clusters = [50, 100, 200];
% bayes_gmm_acc = zeros(2, 3);
% for i = 1: 3
%     bayes_gmm_acc(1, i) = bayes_gmm_err(train_pca, trainY, test_pca, testY, uni, clusters(i));
%     bayes_gmm_acc(2, i) = bayes_gmm_err(train_pca, trainY, test_pca, testY, alt, clusters(i));
% end

%% Semisupervised
% load ./models/logistic_1.mat
% y_unlabeled_est = 
precision_ori_log = zeros(9, 1);
precision_pca_log = zeros(9, 1);
for i = 1: 9
    idx = 1: 4500;
    idx_test = randperm(4500, 500);
    idx_train = idx;
    idx_train(idx_test) = [];
    Y = full(Y) + 1;
    [B_ori, dev_ori, stats_ori] = mnrfit(X(idx_train, :),...
        Y(idx_train));
    ori_pred = mnrval(B_ori, X(idx_test, :));
    Yhat_ori = ori_pred(:, 2) >= 0.5;
    precision_ori_log(i) = mean(Yhat_ori == Y(idx_test));
    [B_pca, dev_pca, stats_pca] = mnrfit(X_pca(idx_train, :),...
        Y(idx_train));
    pca_pred = mnrval(B_pca, X_pca(idx_test, :));
    Yhat_pca = pca_pred(:, 2) >= 0.5;
    precision_pca_log(i) = mean(Yhat_pca == Y(idx_test));
end
precision_ori_log_ave = mean(precision_ori_log);
precision_pca_log_ave = mean(precision_pca_log);
% log_ori = logistic(full(X), full(Y), [], []);
% log_pca = logistic(X_pca, full(Y), [], []);
