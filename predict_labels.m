function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)

load ./models/majority_vote_homo.mat
addpath('liblinear/');

X = full(word_counts);
Xh = [ones(4500, 1), X];

Yhat_knn = predict(KNN, Xh);
Yhat_NB = predict(NB, Xh);
Yhat_SVM_W = predict(SVM_W, Xh);
Yhat_logistic = predict(ones(4500, 1), sparse(Xh), logistic, ['-q', 'col']);
Yhat_log_gd = predict_log(w, Xh);

Y_est = [Yhat_logistic, Yhat_NB, Yhat_SVM_W, Yhat_log_gd, Yhat_knn];
Y_hat = mode(Y_est, 2);

end
