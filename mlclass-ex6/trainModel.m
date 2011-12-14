function [err, model, predictions] = trainModel(pC, pS, X,y, Xval, yval)
 	fprintf('\nC=%f\tsigma=%f\n',pC,pS);
	model= svmTrain(X, y, pC, @(x1, x2) gaussianKernel(x1, x2, pS)); 
	% visualizeBoundary(X, y, model);
	% pause;
	predictions = svmPredict(model, Xval);
	err = mean(double(predictions ~= yval));
end
