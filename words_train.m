clear
clc

load ./data/train_set/words_train.mat
X1 = full(X);
load ./data/train_set_unlabeled/words_train_unlabeled.mat
X2 = full(X);
X = [X1; X2];
load ./models/coeff.mat
X_pca = X * coeff(:, 1: 750);
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
addpath('liblinear/');
log_ori = train(full(Y), sparse(X1), ['-s 0', 'col']);
% log_ori = logistic(X1, full(Y), [], []);
% Yhat = predict_labels(X2, [], [], [], [], []);
y_unlabeled_est = predict(ones(4500, 1), sparse(X2), log_ori, ['-q', 'col']);
Y = [full(Y); y_unlabeled_est];
log_ori_full = train(full(Y), sparse(X), ['-s 0', 'col']);
save('./models/log_ori.mat', 'log_ori', '-v7.3');

% precision_ori_log = zeros(9, 1);
% precision_pca_log = zeros(9, 1);
% ind = crossvalind('Kfold', 4500, 10);
% for i = 1: 10
%     idx = 1: 9000;
%     idx_test = find(ind == i);
%     idx_train = idx;
%     idx_train(idx_test) = [];
% 
%     [~, precision_ori_log(i)] = logistic(X(idx_train, :), Y(idx_train),...
%         X(idx_test, :), Y(idx_test));
%     [~, precision_pca_log(i)] = logistic(X_pca(idx_train, :), Y(idx_train),...
%         X_pca(idx_test, :), Y(idx_test));
% end
% precision_ori_log_ave = mean(precision_ori_log);
% precision_pca_log_ave = mean(precision_pca_log);

%% Naive Bayes
% Mdl_pca = fitcnb(X_pca(1: 4500, 1: 750), Y);
% save('./models/nb_pca.mat', 'Mdl_pca', '-v7.3');
% precision_ori_nb = zeros(9, 1);
% precision_pca_nb = zeros(9, 1);
% ind = crossvalind('Kfold', 4500, 10);
% for i = 1: 10
%     X_ori_train = X_pca(ind ~= i, :);
%     X_pca_train = X_pca(ind ~= i, 1: 750);
%     Y_train = Y(ind ~= i);
%     X_ori_test = X_pca(ind == i, :);
%     X_pca_test = X_pca(ind == i, 1: 750);
%     Y_test = Y(ind == i); 
%     Mdl_ori = fitcnb(X_ori_train, Y_train);
%     Mdl_pca = fitcnb(X_pca_train, Y_train);
%     Yhat_ori = predict(Mdl_ori, X_ori_test);
%     Yhat_pca = predict(Mdl_pca, X_pca_test);
%     precision_ori_nb(i) = mean(Yhat_ori == Y_test);
%     precision_pca_nb(i) = mean(Yhat_pca == Y_test);
% end
% precision_ori_nb_ave = mean(precision_ori_nb);
% precision_pca_nb_ave = mean(precision_pca_nb);
