

function [PrActive, paramDraws] = GibbsDynGLM(y, X, modelOpt, priorOpt, algoOpt)

% Unpacking model and data options, prior hyperparameters, and algorithmic options
[beta0, tau, iota, r, c, a0, b0] = v2struct(priorOpt);
[nIter, prcBurnin] = v2struct(algoOpt);
[k, stimulusPos] = v2struct(modelOpt);
[T, p] = size(X);

% Allocating space for storage of posterior draws
paramDraws.beta = zeros(nIter,p);
paramDraws.noise = zeros(nIter,1+k); % Stores sigma2 followed by rho'

if k>0
    rho0 = [r zeros(1,k-1)]';
    [ytilde, Xtilde] = Prewhitening(y, X, rho0);   
else % White noise
    rho0 = nan;
    ytilde = y;
    Xtilde = X;
end

Omega0 = tau^2*( X'*X \ eye(p) );
invOmega0 = Omega0 \ eye(p);
A0 = diag( c^2./( (1:k).^iota) );
if length(beta0) == 1 && p>1
    beta0 = repmat(beta0,p,1);
end
invA0 = A0 \ eye(k);


%% Initial values
rho = rho0; % Starting Gibbs sampling with rho at its prior mean

%% Gibbs sampling
nBurnin = round(nIter*(prcBurnin/100));
for i = 1:(nBurnin + nIter)
    
    % Block 1 - Step 1a. Update sigma2
    invOmegaT = invOmega0 + Xtilde'*Xtilde;
    OmegaT = invOmegaT \ eye(p);
    betaT = OmegaT*(invOmega0*beta0 + Xtilde'*ytilde);
    aT = a0 + T/2;
    bT = b0 + 0.5*(ytilde'*ytilde + beta0'*invOmega0*beta0 - betaT'*invOmegaT*betaT);
    sigma2 = 1/gamrnd(aT,1/bT);
    % sigma2 = randIGMV(aT,bT);
    
    
    % Block 1 - Step 1b. Update beta | sigma2
    beta = mvnrnd(betaT,sigma2*OmegaT)';
    %beta = randmvnMV(betaT,sigma2*OmegaT)';
    
    % Block 2. Update rho
    if k>0
        u = y - X*beta;
        Z = lags(u,k);
        u = u(k+1:end);
        Z = Z(k+1:end,:);
        invAT = invA0 + Z'*Z/sigma2;
        AT = invAT \ eye(k);
        rhoT = AT*(invA0*rho0 + Z'*u/sigma2);
        rhoProp = mvnrnd(rhoT,sigma2*AT)';
        if abs(rhoProp)<1 % TODO: Generalize to stationarity
        	rho = rhoProp;
        end
        %rho = randmvnMV(rhoT,sigma2*AT)';
        [ytilde, Xtilde] = Prewhitening(y, X, rho);
    end
    
    
    % Storing results
    if i > nBurnin
        paramDraws.noise(i-nBurnin,1) = sigma2;
        paramDraws.noise(i-nBurnin,2:(k+1)) = rho';
        paramDraws.beta(i-nBurnin,:) = beta';
    end
end

PrActive = sum(paramDraws.beta(:,stimulusPos)>0)/nIter;


