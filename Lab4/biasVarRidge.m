% load data
clear all;
load('dataEx3.mat');


for s = 1:50 % # of seeds
    setSeed(s);
    % generate data
    
    N = 50;
    X = linspace(0.1,2*pi,N)';
    y = sin(X(:)) + 0.3*randn(N,1);
    
    % randomly permute data
    idx = randperm(N);
    y = y(idx);
    X = X(idx);
    
    [XTr, yTr, XTe, yTe] = split(y,X,0.9);
    
    
    
    % for degree k
    degrees = [1:10];
    for k = 1:length(degrees)
        
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degrees(k))];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degrees(k))];
        
        % get beta using least squares
        % compute train and test RMSE
        beta = ridge_bestLambda(yTr,XTr,k,4);
        
        rmseTr(s,k) = computeCost(yTr,tXTr,beta);
        rmseTe(s,k) = computeCost(yTe,tXTe,beta);
    end
end
% compute expected train and test error
rmseTr_mean = mean(rmseTr);
rmseTe_mean = mean(rmseTe);
% plot
plot(degrees, rmseTe,'r','color',[1 0.7 0.7]);
hold on
plot(degrees, rmseTr,'b','color',[0.7 0.7 1]);
plot(degrees, rmseTe_mean,'r','linewidth', 3);
hold on
plot(degrees, rmseTr_mean,'b','linewidth', 3);
xlabel('degree');
ylabel('error');

