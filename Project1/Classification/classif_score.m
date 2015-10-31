function [ errorTr, errorTe] = compute_score(y, X, idx, alpha , lambda, K, degree)

%Model available: ls, lsgd, rr, lr, plr
if idx ~= 0
    X = X(:,idx);
end

% split data in K fold (we will only create indices)
setSeed(1);
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
    
   beta = penLogisticRegression(yTr,tXTr,alpha,lambda);
   
   [y_hatTr, probTr] = predictY(tXTr, beta);
   [y_hatTe, probTe] = predictY(tXTe, beta);
   
   errTr(k) = zeroOneLoss(y_hatTr, yTr);
   errTe(k) = zeroOneLoss(y_hatTe, yTe);
   
end

errorTr = mean(errTr);
errorTe = mean(errTe);

end

