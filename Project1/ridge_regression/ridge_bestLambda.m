function [ betaStar, lambdaStar ] = ridge_bestLambda( y, X, K )
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
lambda = logspace(-2,2,200);

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
     
        tXTr = [ones(length(yTr), 1) XTr];
        tXTe = [ones(length(yTe), 1) XTe]; 
        
		% least squares (INSERT CODE)
        %beta = leastSquares(yTr,tXTr);
        beta = ridgeRegression(yTr, tXTr, lambda(i));
        
		% training and test MSE(INSERT CODE)
		mseTrSub(k) = sqrt(2*MSE(yTr,tXTr,beta)); 

		% testing MSE using least squares
		mseTeSub(k) = sqrt(2* MSE(yTe,tXTe,beta));  

	end
	mseTr(i) = mean(mseTrSub);
	mseTe(i) = mean(mseTeSub);
end

figure;
semilogx(lambda, mseTr)
hold on; 
semilogx(lambda, mseTe)


[errStar, star] = min(mseTe);

lambdaStar = lambda(star)

SP=lambdaStar; %your point goes here
line([SP SP], [0 max(mseTe)]);

betaStar = ridgeRegression(yTr, tXTr, lambdaStar);




end

