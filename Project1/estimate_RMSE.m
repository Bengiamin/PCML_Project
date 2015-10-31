data_preprocessing;

%Lambda are computed using cross validation
lambda_large = 18.5896;
lambda_small = 27.8898;

%K-fold parameter
K = 20;

%Degree of polynomial for more significant feature
degree = 5;

% split data in K fold (we will only create indices)
setSeed(1);

%Split the data into k subset
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
    
    %%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%
    
    %Split the training data into two set large and small
    idx = find(yTr > 3400);
    y_large = yTr(idx);
    X_large = XTr(idx,:);
    
    idx = find(yTr <= 3400);
    y_small = yTr(idx);
    X_small = XTr(idx,:);
    
    %Creating the tXtr matrices some relevant row of X are added as
    %polynomial into the matrix.
    tXTr_large = [ones(length(y_large), 1) X_large myPoly(X_large(:,[14 15]),degree)];
    tXTr_small = [ones(length(y_small), 1) X_small myPoly(X_small(:,[48 17 50]),degree)];
    
    %Compute the betas using ridge regression
    beta_large = ridgeRegression(y_large,tXTr_large,lambda_large);
    beta_small = ridgeRegression(y_small,tXTr_small,lambda_small);
    
    %%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%
    
    %Spling the test value into two set according to the 13th row of X
    idx = find(X_eval(:,13) > 0.471);
    y_eval_large = y_eval(idx);
    tX_eval_large = [ones(length(y_eval_large), 1) X_eval(idx,:) myPoly(X_eval(idx,[14 15]),degree)];
    
    idx = find(X_eval(:,13) <= 0.471);
    y_eval_small = y_eval(idx);
    tX_eval_small = [ones(length(y_eval_small), 1) X_eval(idx,:) myPoly(X_eval(idx,[48 17 50]),degree)];
    
    
    %Compute the RMSE for training and test set
    rmseTr_large(k) = sqrt(2*MSE(y_large,tXTr_large,beta_large));
    rmseTr_small(k) = sqrt(2*MSE(y_small,tXTr_small,beta_small));
    rmseTr_mean(k) = (rmseTr_large(k)+rmseTr_small(k))/2;
  
    rmseTe_large(k) = sqrt(2*MSE(y_eval_large,tX_eval_large,beta_large));
    rmseTe_small(k) = sqrt(2*MSE(y_eval_small,tX_eval_small,beta_small));
    rmseTe_mean(k) = (rmseTe_large(k)+rmseTe_small(k))/2;
end

%Here depending on k there is a trade of between negative bias and variance
rmseTrs = mean(rmseTr_small);
rmseTrl = mean(rmseTr_large);
rmseTr = mean (rmseTr_mean)
stdTr = std(rmseTr_mean)

rmseTes = mean(rmseTe_small);
rmseTel = mean(rmseTe_large);
rmseTe = mean (rmseTe_mean)
stdTe = std(rmseTe_mean)
