function [ J, grad ] = costFunctionReg( w, X, y, lambda )
%UCOSTFUNCTIONREG Compute cost and gradient for logistic regression with
%regularization
%   J = COSTFUNCTIONREG(w, X, y, lambda) computes the cost of using
%   w as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);

J = mean(-y .* log(sigmoid(X * w)) - ...
    (1 - y) .* log(1 - sigmoid(X * w))) + ...
    lambda * (sum(w .^ 2) - w(1) ^ 2) / 2 / m;

grad = mean((sigmoid(X * w) - y) .* X)';
reg = lambda * w ./ m;
reg(1) = 0;
grad = grad + reg;

end

