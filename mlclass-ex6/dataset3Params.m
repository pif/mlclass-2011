function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

pC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
pS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
LpC = length(pC);
LpS = length(pS);
% pS = [0.1];
%yval
errors = zeros(LpC, LpS);
minerr = 10000;
for i = 1:LpC
	for j = 1:LpS
		err = trainModel(pC(i), pS(j), X,y, Xval, yval);

		errors(i,j) = err;
		fprintf('Error: %f\t MinError: %f\t',err, minerr);
		if (err<minerr) 
			minerr = err;
			C = pC(i);
			sigma = pS(j);
		endif 
		
	end
end

meshc(linspace(1,LpC,LpC),linspace(1,LpS,LpS),100*errors);
pause;
% =========================================================================
C
sigma
end
