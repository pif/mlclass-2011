function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

size(centroids);
size(X);
for i = 1:m
    % size(centroids(idx(i)))
    % size(X(i,:))
	centroids(idx(i),:)+=X(i,:);
end
%centroids
%size(idx)
acc = accumarray(idx,1);
cns = repmat(acc,1,n);
diff = -size(cns,1)+ size(centroids,1);

if (diff>0) 
	cns = [cns; zeros(diff,n)];
end

size(cns);
centroids = centroids./cns;

% =============================================================


end

