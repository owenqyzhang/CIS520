clear
clc

load ./data/train_set/words_train.mat

tic;
Yhat = predict_labels(X, [], [], [], [], []);
toc;
acc = mean(Yhat == Y);