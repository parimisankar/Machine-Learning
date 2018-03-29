function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% Hypothesis
z = X*theta;
hx = sigmoid(z);
% No regularization for first term. Set theta(1,1) to zero
theta(1,1) = 0;
% Cost Function
J_Reg = (lambda/(2*m))*(sum(sum(theta.^2)));
J = ((-1/m)*((y'*log(hx))+(1-y)'*(log(1-hx)))) + J_Reg;
% Gradient i.e., partial derivative of cost function wrt each theta
grad_Reg = (lambda/m)*theta;
grad = (X'*(hx-y))/m + grad_Reg;
% =============================================================

end