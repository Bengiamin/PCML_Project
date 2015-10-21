function [ beta ] = logisticRegression(y,tX,alpha)

%Algorithm parameter
maxIters = 1000;

% initialize
beta = [0; 0; 0];

for k = 1:maxIters
    % Compute gradient
    g = computeGradient(y,tX,beta);
    
    % Compute cost
    L = computeCost(y,tX,beta)
    
    % Update beta according to gradient
    beta = beta - alpha.*g;
    
    % Convergence check
    if g'*g < 1e-2; break; end;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
end

%plot
h = [min(tX(:,1)):.01:max(tX(:,1))];
w = [min(tX(:,2)):1:max(tX(:,2))];
[hx, wx] = meshgrid(h,w);

pred = beta(1)*ones(size(hx)) + beta(2)*hx + beta(3)*wx;

contourf(hx, wx, pred, 1);
end

