function [ beta ] = leastSquaresGD(y,tX,alpha)

% algorithm parametes
maxIters = 1000;

% initialize
beta = [0; 0];

% iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1\n');

for k = 1:maxIters
    %Compute gradent
    g = computeGradient(y,tX,beta);
    
    %Compute cost using RMSE
    L = computeCost(y,tX,beta);
    
    % Update beta according to gradient
    beta = beta - alpha.*g;
    
    % Convergence limitation
    if g'*g < 1e-5; break; end;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    %Plot the advancement off gradient descent
    
    %    subplot(121);
    %    plot(beta(1), beta(2), 'o', 'color', 0.7*[1 1 1], 'markersize', 12);
    %    pause(.5) % wait half a second
    %
    %     visualize function f on the data
    %    subplot(122);
    %     x = [1.2:.01:2]; % height from 1m to 2m
    %     x_normalized = (x - meanX)./stdX;
    %     f = beta(1) + beta(2).*x_normalized;
    %     plot(height, weight,'.');
    %     hold on;
    %     plot(x,f,'r-');
    %     hx = xlabel('x');
    %     hy = ylabel('y');
    %     hold off;
end

end

