function [ Yhat, prob ] = predict_log( w, X )
%PREDICT_LOG Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT_LOG(w, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

prob = sigmoid(X * w);
Yhat = double(prob >= 0.5);

end

