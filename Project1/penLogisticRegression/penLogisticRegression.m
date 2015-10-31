function [ beta ] = penLogisticRegression(y,tX,alpha,lambda)

%Algorithm parameter
maxIters = 1000;

% initialize
si = size(tX(1,:));
beta = zeros(si(2),1);

for k = 1:maxIters
    
    %Comput cost, gradient and hessian
    [L, g, H] = logisticRegLoss(beta, y, tX, lambda);

    % Update beta according to Newton's method
    d = H \ g;
    beta = beta - alpha.*d;
    
    % Convergence check
    if g'*g < 1e-2; break; end;
    
end




end

