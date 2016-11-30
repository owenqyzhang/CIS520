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

X=full(X_label);
Y=full(Y);

rand_ind=randperm(4500);
X_rand=X(rand_ind,:);
Y_rand=Y(rand_ind,:);
X_train=X_rand(1:4000,:);
Y_train=Y_rand(1:4000);
X_test=X_rand(4001:4500,:);
Y_test=Y_rand(4001:4500);

%rng(1); % For reproducibility
Mdl = TreeBagger(200,X_train,Y_train,'OOBPrediction','On',...
      'Surrogate','on','PredictorSelection','curvature',...
      'OOBPredictorImportance','on','Method','classification');

predicted_label = predict(Mdl,X_test);
predicted_label= cell2mat(predicted_label);
predicted_label=double(predicted_label)-48;
acc=mean(predicted_label==Y_test);
view(Mdl.Trees{1},'Mode','graph');
figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
