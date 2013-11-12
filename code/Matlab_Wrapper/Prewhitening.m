function [y, X] = Prewhitening(y, X, rhos)

nLags = length(rhos);
yOrig = y;
XOrig = X;
for j = 1:nLags
    y = y - lag(yOrig,j)*rhos(j);
    X = X - lag(XOrig,j)*rhos(j);
end
y = y(nLags+1:end,:);
X = X(nLags+1:end,:);

