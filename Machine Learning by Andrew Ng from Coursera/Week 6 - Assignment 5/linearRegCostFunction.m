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
% size(theta)
% size(X)
% size(y)
% Model/Hypothesis
hx = X*theta;
% Cost Function
J = (1/(2*m))*(sum(((hx-y).^2)));
% Regularization
theta(1) = 0;
J_reg = (lambda/(2*m))*(sum((theta).^2));
% Cost Function with Regularization
J = J + J_reg;
% Gradient
grad = (1/m)*(X'*(hx-y));
% size(grad)
% Regularization
grad_reg = (lambda/m)*theta;
% size(grad_reg)
% Gradient with Regularization
grad = grad + grad_reg;
% =========================================================================

grad = grad(:);

end
