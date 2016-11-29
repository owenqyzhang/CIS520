load train_cnn_feat.mat
load train_raw_img.mat
load words_train.mat
load raw_tweets_train.mat
load train_color.mat
load train_img_prob.mat
load train_tweet_id_img.mat
load ordered_tweets_train.mat
X_label=X;
label_id=tweet_ids;
load raw_tweets_train_unlabeled.mat
load train_unlabeled_cnn_feat.mat
load train_unlabeled_color.mat
load train_unlabeled_img_prob.mat
load train_unlabeled_raw_img.mat
load train_unlabeled_tweet_id_img.mat
load words_train_unlabeled.mat
load ordered_tweets_train_unlabel.mat
X_unlabel=X;
unlabel_id=tweet_ids;

X=X_label;
Y=full(Y);

rng default
Mdl = fitcnb(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
