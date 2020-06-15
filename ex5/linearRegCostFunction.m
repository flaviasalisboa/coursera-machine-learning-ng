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

predictions = X * theta; %predictions on all m examples
sqrerrors = (predictions-y).^2; % squared errors
h = predictions;

semTheta0 = theta(2:size(theta));
regularTheta = [0;semTheta0];

J = 1/(2*m) * sum(sqrerrors) + (lambda/(2*m))*regularTheta'*regularTheta;

grad = (1/m)*(X'*(h-y)) + (lambda/m)*regularTheta;

%grad = (1/m)*(X'*(h-y)+lambda*regularTheta);

% =========================================================================

grad = grad(:);

end
