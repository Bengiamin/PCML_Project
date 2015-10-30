function [ rmseTr, rmseTe] = compute_score(y, X,model, idx, alpha , lambda, K, degree)

%Model available: ls, lsgd, rr, lr, plr
if idx ~= 0
    X = X(:,idx);
end

% split data in K fold (we will only create indices)
setSeed(1);
K = 4;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    XTe = X(idxTe,:);
    yTr = y(idxTr);
    XTr = X(idxTr,:);
    
    if degree == 0
        tXTr = [ones(length(yTr), 1) XTr];
        tXTe = [ones(length(yTe), 1) XTe];
    else
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];
    end
    
    switch model
        case 'lsgd'
            beta = leastSquaresGD(yTr,tXTr,alpha);
        case 'ls'
            beta = leastSquares(yTr,tXTr);
        case 'rr'
           beta = ridgeRegression(yTr,tXTr,lambda);
        case 'lr'
            beta = logisticRegression(yTr,tXTr,alpha);
        case 'plr'
            beta = penLogisticRegression(yTr,tXTr,alpha,lambda);
        otherwise
            Disp('Unknown model! use one of those: ls, lsgd, rr, lr, plr');
    end
    
    rmseTrSub(k) = sqrt(2*MSE(yTr,tXTr,beta));
    rmseTeSub(k) = sqrt(2*MSE(yTe,tXTe,beta));
end

    rmseTr = mean(rmseTrSub);
	rmseTe = mean(rmseTeSub);

end

