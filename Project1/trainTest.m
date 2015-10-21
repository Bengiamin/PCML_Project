function [ output_args ] = trainTest(y, tX, prop)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    
    %split with seeding
    % get train and test data
	[tXTr, yTr, tXTe, yTe] = split(y,tX,prop);
    

    % CV ?
    
    
    % train set
    beta = leastSquaresGD(yTr, tXTr, 0.1);
    
    % test error for each set
    rmseTr = sqrt(2*computeCost(yTr,tXTr,beta)); 
	rmseTe = sqrt(2*computeCost(yTe,tXTe,beta)); 

	% print 
	fprintf('Proportion %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', prop, rmseTr, rmseTe);


end

