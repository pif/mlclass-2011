function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%
%num_users
%num_movies
%num_features
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%X
%Theta
%X*Theta'-Y
%R
%Y

ho = X*Theta'-Y;

J = sum(sum((R.*ho).^2))/2 + sum(sum(Theta.^2))*lambda/2 + sum(sum(X.^2))*lambda/2;

X_grad = R.*ho*Theta + lambda*X;
Theta_grad = (X'*(R.*ho))' + lambda*Theta;

% =============================================================
% Without regularization
% Training collaborative filtering...
% Iteration   100 | Cost: 2.656553e+04
% Recommender system learning completed.
% Top recommendations for you:
% Predicting rating 14.3 for movie Lost in Space (1998)
% Predicting rating 14.3 for movie Two Friends (1986)
% Predicting rating 14.0 for movie A Chef in Love (1996)
% Predicting rating 13.6 for movie Prefontaine (1997)
% Predicting rating 13.5 for movie Until the End of the World (Bis ans Ende der Welt) (1991)
% Predicting rating 13.5 for movie Heaven & Earth (1993)
% Predicting rating 13.2 for movie Saint of Fort Washington, The (1993)
% Predicting rating 13.0 for movie Some Mother's Son (1996)
% Predicting rating 12.8 for movie Clean Slate (Coup de Torchon) (1981)
% Predicting rating 12.7 for movie Audrey Rose (1977)
% Original ratings provided:
% Rated 4 for Toy Story (1995)
% Rated 3 for Twelve Monkeys (1995)
% Rated 5 for Usual Suspects, The (1995)
% Rated 4 for Outbreak (1995)
% Rated 5 for Shawshank Redemption, The (1994)
% Rated 3 for While You Were Sleeping (1995)
% Rated 5 for Forrest Gump (1994)
% Rated 2 for Silence of the Lambs, The (1991)
% Rated 4 for Alien (1979)
% Rated 5 for Die Hard 2 (1990)
% Rated 5 for Sphere (1998)

% regularized J
% Top recommendations for you:
% Predicting rating 15.8 for movie Victor/Victoria (1982)
% Predicting rating 15.0 for movie Love! Valour! Compassion! (1997)
% Predicting rating 14.6 for movie Reluctant Debutante, The (1958)
% Predicting rating 14.5 for movie Secret Garden, The (1993)
% Predicting rating 14.4 for movie Game, The (1997)
% Predicting rating 13.8 for movie Time to Kill, A (1996)
% Predicting rating 13.7 for movie From Dusk Till Dawn (1996)
% Predicting rating 13.5 for movie Double vie de Véronique, La (Double Life of Veronique, The) (1991)
% Predicting rating 13.3 for movie Bad Boys (1995)
% Predicting rating 13.1 for movie Phantoms (1998)


grad = [X_grad(:); Theta_grad(:)];

end
