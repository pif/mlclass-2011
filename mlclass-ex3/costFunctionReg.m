function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m=length(y);

[J, grad] = costFunction(theta, X, y);
nt = [0;theta(2:end)];

J+=lambda*sum(nt.^2)/(2*m);
grad+=(lambda*nt/m)';

% =============================================================

end
