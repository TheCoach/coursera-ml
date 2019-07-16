function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

theta = reshape(theta, [], 1);
y = reshape(y, [], 1);

m = length(y);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J = (sum((X * theta - y) .^ 2) + sum(theta(2:end, :) .^ 2 .* lambda)) / (2 * m);
grad = X' * (X * theta - y) / m;
grad(2:end, :) = grad(2:end, :) + theta(2:end, :) .* lambda / m;



% =========================================================================

grad = grad(:);

end
