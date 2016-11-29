function [ w ] = gradientDescent( X, y, w, alpha, num_iters, lambda )
%GRADIENTDESCENT Performs gradient descent to learn w
%   w = GRADIENTDESCENTMULTI(x, y, w, alpha, num_iters) updates w by
%   taking num_iters gradient steps with learning rate alpha

for iter = 1: num_iters
    [~, grad] = costFunctionReg(w, X, y, lambda);
    w = w - alpha * grad;
end

end

