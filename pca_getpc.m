function [ coeff_train, score_train, score_test, numpc ] = pca_getpc( train_x, test_x )

%   input: original X for training and testing
%   output: PCAed X for training and testing, number of PCs that you
%   selected

cov_train = cov(train_x);
[coeff_train, latent] = pcacov(cov_train);
score_train = train_x * coeff_train;
score_test = test_x * coeff_train;

figure, plot(cumsum(latent)/sum(latent));
%% Set you numpc here, you should acheive 90% reconstruction accuracy
numpc = 750;

end

