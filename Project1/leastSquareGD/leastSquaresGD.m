function [ beta ] = leastSquaresGD(y,tX,alpha)

% algorithm parametes
maxIters = 1000;

% initialize
si = size(tX(1,:));
beta = zeros(si(2),1);

% iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1\n');

for k = 1:maxIters
    %Compute gradent
    g = computeGradient(y,tX,beta);
    
    % Update beta according to gradient
    beta = beta - alpha.*g;
    
    disp 'beta'
    disp(beta)
    disp 'g'
    disp(g)
    
    pause;
    
    % Convergence limitation
    if g'*g < 1e-5; break; end;
end

end

