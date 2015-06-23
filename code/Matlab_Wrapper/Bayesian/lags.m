function Y = lags(X,k)

% Function that returns a matrix with the first k time lags of the input vector.

Y = nan(size(X));
for i = 1:k
    Y(:,i) = lag(X,i);
end