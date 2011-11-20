function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('my variable sizes\n');
size(nn_params)		;%   10285       1
size(input_layer_size)	;%   1   1
size(hidden_layer_size)	;%   1   1
size(num_labels)	;%   1   1
size(X)			;%   5000    400
size(y)			;%   5000      1
size(lambda)		;%   1   1
size(Theta1)		%   25   401
size(Theta2)		%   10   26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare x,y
y = eye(num_labels)(y,:);
y = y';

ho = zeros(m, 1);
X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forward prop
ho = ho';
X = [ones(1,m); X];
X = X';
z2 = Theta1*X';
size(Theta1); 	% 25  401
size(X);	% 5000  401
size(z2);	% 25  5000
a2 = sigmoid(z2);
a2 = [ones(size(a2,2),1)'; a2];
z3 = Theta2*a2;
size(Theta2);	% 10  26
size(z3);	% 10  5000
a3 = sigmoid(z3);
ho = a3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ho = ho';
size(ho);		%   10   5000
J = sum(sum(-y.*log(ho)-(1-y).*log(1-ho))')/m;

l = 1;
regt1 = Theta1(:, 2:end);
regt2 = Theta2(:, 2:end);
reg = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);

J += reg;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = a3-y';
szt2 = Theta2'(2:end,:);
size(szt2);
size(d3);
sz2 = sigmoidGradient(z2);
size(sz2);
%   26   10
%   10   5000
%   25   5000
% z2 = [ones(1,5000);z2];
d2 = szt2*d3.*sz2;

D1 = d2*a2(2:end,:)';
D2 = d3*a3(2:end,:)';

Theta1_grad = D1/m;
Theta2_grad = D2/m;
% for t = 1:m
%   a = X(t);
% end;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%disp(Theta1_grad)
%disp(Theta2_grad)
grad = [Theta1_grad(:) ; Theta2_grad(:)];
size(grad);
size(J);

end
