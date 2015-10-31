
%Lambda are computed using cross validation
lambda_large = 18.5896;
lambda_small = 27.8898;

K = 110;
degree = 5;

% split data in K fold (we will only create indices)
setSeed(1);
X_train = noCategX; 

N = size(y_train,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y_train(idxTe);
    XTe = X_train(idxTe,:);
    yTr = y_train(idxTr);
    XTr = X_train(idxTr,:);
    
    idx = find(yTr > 3400);
    y_large = yTr(idx);
    X_large = XTr(idx,:);
    
    idx = find(yTr <= 3400);
    y_small = yTr(idx);
    X_small = XTr(idx,:);
    
    %Here we are adding polynomial row of the most relevant row of X
    tXTr_large = [ones(length(y_large), 1) X_large myPoly(X_large(:,[11 12]),degree)];
    tXTr_small = [ones(length(y_small), 1) X_small myPoly(X_small(:,[38 14 40]),degree)];
    
    tXTe = [ones(length(yTe), 1) XTe];
    
    beta_large = ridgeRegression(y_large,tXTr_large,lambda_large);
    beta_small = ridgeRegression(y_small,tXTr_small,lambda_small);
    
    %Extract Test value for model large
    idx = find(XTe(:,10) > 0.471);
    yTe_large = yTe(idx);
    tXTe_large = [ones(length(yTe_large), 1) XTe(idx,:) myPoly(XTe(idx,[11 12]),degree)];
    
    idx = find(XTe(:,10) <= 0.471);
    yTe_small = yTe(idx);
    tXTe_small = [ones(length(yTe_small), 1) XTe(idx,:) myPoly(XTe(idx,[38 14 40]),degree)];
    
    rmseTr_large(k) = sqrt(2*MSE(y_large,tXTr_large,beta_large));
    rmseTr_small(k) = sqrt(2*MSE(y_small,tXTr_small,beta_small));
    rmseTr_mean(k) = (rmseTr_large(k)+rmseTr_small(k))/2;
  
    rmseTe_large(k) = sqrt(2*MSE(yTe_large,tXTe_large,beta_large));
    rmseTe_small(k) = sqrt(2*MSE(yTe_small,tXTe_small,beta_small));
    rmseTe_mean(k) = (rmseTe_large(k)+rmseTe_small(k))/2;
end

rmseTrs = mean(rmseTr_small);
rmseTrl = mean(rmseTr_large);
rmseTr = mean (rmseTr_mean)

rmseTes = mean(rmseTe_small);
rmseTel = mean(rmseTe_large);
rmseTe = mean (rmseTe_mean)

%data_preprocessing;
