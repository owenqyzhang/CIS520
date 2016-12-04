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

load ./models/log_ori_full.mat
addpath('./liblinear/')

n = size(word_counts, 1);
[~, ~, prob_estimates] = predict(ones(n, 1), sparse(word_counts), log_ori_full, ['-q', 'col']);

Y_hat = zeros(4500, 1);
[~, ind] = sort(prob_estimates, 'descend');
Y_hat(ind(1: 2500)) = 1;

end
