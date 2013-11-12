function X=lag(Y,k)

% Function that returns the k:th time lag of the input vector.

X=[ones(k,size(Y,2))*NaN;Y(1:size(Y,1)-k,:)];