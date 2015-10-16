% Load data
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
y = gender;
X = [height(:) weight(:)];
% randomly permute data
N = length(y);
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% subsample
y = y(1:200);
X = X(1:200,:);

meanX = mean(X);
X(:,1) = X(:,1) - meanX(1);
X(:,2) = X(:,2) - meanX(2);

X(:,1) = X(:,1)./std(X(:,1));
X(:,2) = X(:,2)./std(X(:,2));

male = X(y, :);
female = X(not(y), :);

% algorithm parametes
tX = [ones(size(X,1),1) X];
maxIters = 1000;
alpha = 0.1;
converged = 0;

% initialize
beta = [0; 0; 0];

fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1\n');
for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    g = computeGradient(y,tX,beta);
    
    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    L = computeCost(y,tX,beta)
    
    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta - alpha.*g;
    
    % INSERT CODE FOR CONVERGENCE
    gre = g'*g;
    if g'*g < 1e-5; break; end;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
    
    % print
    %fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    
    h = [min(X(:,1)):.01:max(X(:,1))];
    w = [min(X(:,2)):1:max(X(:,2))];
    [hx, wx] = meshgrid(h,w);
    % predict for each pair, i.e. create tX for each [hx,wx]
    % and then predict the value. After that you should
    % reshape `pred` so that you can use `contourf`.
    % For this you need to understand how `meshgrid` works.
    
    pred = beta(1)*ones(size(hx)) + beta(2)*hx + beta(3)*wx;
    
    % plot the decision surface
    contourf(hx, wx, pred, 1);
    pause(.5) % wait half a second
    
    % plot indiviual data points
    hold on
    myBlue = [0.06 0.06 1];
    myRed = [1 0.06 0.06];
    plot(male(:,1),male(:,2),'xr','color',myRed,'linewidth', 2, 'markerfacecolor', myRed);
    hold on
    plot(female(:,1),female(:,2),'or','color', myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
    xlabel('height');
    ylabel('weight');
    xlim([min(h) max(h)]);
    ylim([min(w) max(w)]);
    grid on;
end
