function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% linear hypothesis
h = X*theta;

% unregularized cost
J = 1/(2*m)*(norm(h-y,'fro'))^2;

% regularization term
regularization_term = lambda/(2*m)*(norm(theta(2:end),'fro'))^2;

% regularized cost
J = J + regularization_term;

% unregularized gradient
grad = 1/m.*(X'*(h-y));

% regularization term
regularization_term = lambda/m.*(theta(2:end));

% regularized gradient
grad = grad + [0; regularization_term];




% =========================================================================

grad = grad(:);

end
