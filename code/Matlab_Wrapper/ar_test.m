clear all
close all
clc

load design_matrix

beta(1) = 1*randn(1,1);
beta(2) = 5*randn(1,1);
beta(3) = 1*randn(1,1);
beta(3) = abs(beta(3))*1000;

X = zeros(160,2);
X(:,1) = design_matrix(:,1);
X(:,2) = design_matrix(:,3);
X(:,3) = ones(160,1);

Y = beta(1) * X(:,1) + beta(2) * X(:,2) + beta(3) * X(:,3);

plot(Y,'b')
hold on

ar = 0.65;
noise = 0.7*randn(160,1);
for t = 2:160
    noise(t) = 0.7*randn + ar * noise(t-1);
end

Y = Y + noise;
plot(Y,'r')
hold on

beta'
r = 0;

%Y = Y - mean(Y);
Ywhitened = Y;
Xwhitened = X;

N = 1;

for i = 1:3
    
   %betahat = pinv(X(N:end,:)) * Ywhitened(N:end,:)
   %betahat = pinv(Xwhitened(N:end,:)) * Ywhitened(N:end)   
   X_ = pinv(Xwhitened);
   betahat = X_(:,N:end) * Ywhitened(N:end)   
   eps = Y(N:end) - X(N:end,:)*betahat;
   meann = mean(eps)
   
   t = betahat(1)/sqrt(var(eps)*[1 0 0]*inv(X'*X)*[1 0 0]')
   %t = betahat(1)/sqrt(var(eps)*[1 0 ]*inv(X'*X)*[1 0 ]')
   
   c0 = eps(1)*eps(1);
   c1 = 0;
   for t = 2+N:160-(N-1)
       c0 = c0 + eps(t)*eps(t);
       c1 = c1 + eps(t)*eps(t-1);       
   end
   c0 = c0 / 160;
   c1 = c1 / 159;
   
   r = c1/c0
       
   %Ywhitened(1) = Y(1);
   %Xwhitened(1,1) = X(1,1);
   %Xwhitened(1,2) = X(1,2);
   
   %Ywhitened(1,1) = Y(1);
   %Xwhitened(1,1) = X(1,1);
   %Xwhitened(1,2) = X(1,2);
   
   Ywhitened(1) = sqrt(1-r*r)*Y(1);
   %Xwhitened(1,1) = sqrt(1-r*r)*X(1,1);
   %Xwhitened(1,2) = sqrt(1-r*r)*X(1,2);
   %Xwhitened(1,3) = 1-r;
   %Xwhitened(1,3) = sqrt(1-r*r)*X(1,3);
   
   Xwhitened(1,1) = 0;
   Xwhitened(1,2) = 0;
   Xwhitened(1,3) = 0;
   
   for t = 2:160
        Ywhitened(t) = Y(t) - r * Y(t-1);
        Xwhitened(t,1) = X(t,1) - r * X(t-1,1);
        Xwhitened(t,2) = X(t,2) - r * X(t-1,2);
        Xwhitened(t,3) = X(t,3) - r * X(t-1,3);
   end
   
   Xwhitened(:,1) = Xwhitened(:,1) - mean(Xwhitened(:,1));
   Xwhitened(:,2) = Xwhitened(:,2) - mean(Xwhitened(:,2));
   
   %Y = Ywhitened;
   %X = Xwhitened;
   
   N = 2;
end

%plot(X*betahat1,'c')
%hold on
plot(X*betahat,'g')
%hold on
%plot(Ywhitened,'c')
hold off

close all

% tic
% for i = 1:10000
%    %X = randn(160,3);
%    a = pinv(X); 
% end
% toc



