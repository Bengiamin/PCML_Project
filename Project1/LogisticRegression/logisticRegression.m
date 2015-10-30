function [ beta ] = logisticRegression(y,tX,alpha)

%Algorithm parameter
maxIters = 1000;

% initialize
si = size(tX(1,:));
beta = zeros(si(2),1);

for k = 1:maxIters
    % Compute gradient
    g = computeGradientLR(y,tX,beta);
    
    % Compute cost
    L = logLikelihoodCost(y,tX,beta);
    
    % Update beta according to gradient
    beta = beta - alpha.*g;
    
    % Convergence check
    if g'*g < 1e-2; break; end;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
end

end

