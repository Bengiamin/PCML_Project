function [ beta ] = leastSquaresGD(y,tX,alpha)

% algorithm parametes
maxIters = 1000;

% initialize
si = size(tX(1,:));
beta = zeros(si(2),1);

for k = 1:maxIters
    %Compute gradent
    g = computeGradientLS(y,tX,beta);
 
    % Update beta according to gradient
    beta = beta - alpha.*g;
    
   % disp 'beta'
    %disp(beta)
    %disp 'g'
    %disp(g)
    
    %pause;
    
    % Convergence limitation
    if g'*g < 1e-5; break; end;
end
end
    
function [g] = computeGradientLS(y,tX,beta)
    e = y - tX*beta;    
    g = tX'*e/(-length(y));
end





