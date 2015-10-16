function [ betaStar ] = ridge_bestLambda( y, X, degree, K )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


% split data in K fold (we will only create indices)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values (INSERT CODE)
lambda = logspace(-2,2,100);

% K-fold cross validation
for i = 1:length(lambda)
	for k = 1:K
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = y(idxTe);
		XTe = X(idxTe,:);
		yTr = y(idxTr);
		XTr = X(idxTr,:);
		% form tX (INSERT CODE)
     
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];
        
		% least squares (INSERT CODE)
        %beta = leastSquares(yTr,tXTr);
        beta = ridgeRegression(yTr, tXTr, lambda(i));
        
		% training and test MSE(INSERT CODE)
		mseTrSub(k) = computeCost(yTr,tXTr,beta); 

		% testing MSE using least squares
		mseTeSub(k) = computeCost(yTe,tXTe,beta);  

	end
	mseTr(i) = mean(mseTrSub);
	mseTe(i) = mean(mseTeSub);
end

[errStar, star] = min(mseTe);

lambdaStar = lambda(star);

betaStar = ridgeRegression(yTr, tXTr, lambdaStar);




end

