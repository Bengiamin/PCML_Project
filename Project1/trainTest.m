
function [ beta ] = trainTest(y, X, prop, degree)
%split the data and train it. output train error and test error
%   Detailed explanation goes here
    


    %split with seeding
    % get train and test data
	[XTr, yTr, XTe, yTe] = split(y,X,prop);
    
    % CV ?

    
    if degree > 0
        XTr = myPoly(XTr, degree);
        XTe = myPoly(XTe, degree); 
    end
    
    tXTr = [ones(length(yTr), 1) XTr];
    tXTe = [ones(length(yTe), 1) XTe]; 


    % train set
    [beta, lambdaS] = ridge_bestLambda(yTr, XTr, 5);
    
    % test error for each set
    rmseTr = sqrt(2*MSE(yTr,tXTr,beta)); 
	rmseTe = sqrt(2*MSE(yTe,tXTe,beta)); 

    
    
	% print 
	fprintf('Proportion %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', prop, rmseTr, rmseTe);

    

end

