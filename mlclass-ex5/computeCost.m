function [J, grad] = computeCost(X, y, theta, lambda)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
ho = (theta'*X')'; % 3x1' * 1x2'
J = sum((ho-y) .^2 )/( 2*m );
nt = [0;theta(2:end)];
J+=lambda*sum(nt.^2)/(2*m);

grad = ((ho-y)' * X / m)' + lambda*nt/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
