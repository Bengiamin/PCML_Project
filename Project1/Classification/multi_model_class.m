data_preprocessing_class;

%Lambda are computed using cross validation
lambda_1 = 10;
lambda_2 = 10;

%K-fold parameter
K = 6;

%alpha
alpha = 0.2;

%Degree of polynomial for more significant feature
degree = 0;

% split data in K fold (we will only create indices)
setSeed(1);

%Split the data into k subset
N = size(y_bal,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y_bal(idxTe);
    XTe = X_bal(idxTe,:);
    yTr = y_bal(idxTr);
    XTr = X_bal(idxTr,:);
    
    %%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%
    
    %Split the training data into two set large and small
    idx = find(XTr(:,2) > 18);
    yTr_1 = yTr(idx);
    XTr_1 = XTr(idx,:);
    
    idx = find(XTr(:,2) <= 18);
    yTr_2 = yTr(idx);
    XTr_2 = XTr(idx,:);
    
    %Creating the tXtr matrices some relevant row of X are added as
    %polynomial into the matrix.
    tXTr_1 = [ones(length(yTr_1), 1) XTr_1 ];
    tXTr_2 = [ones(length(yTr_2), 1) XTr_2 ];
    
    %Compute the betas using ridge regression
    beta_1 = penLogisticRegression(yTr_1,tXTr_1,alpha,lambda_1);
    beta_2 = penLogisticRegression(yTr_2,tXTr_2,alpha,lambda_2);
    
    %%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%
    
    %Spliting the test value into two set according to the 13th row of X
    idx = find(XTe(:,2) > 18);
    yTe_1 = yTe(idx);
    tXTe_1 = [ones(length(yTe_1), 1) XTe(idx,:)];
    
    idx = find(XTe(:,2) <= 18);
    yTe_2 = yTe(idx);
    tXTe_2 = [ones(length(yTe_2), 1) XTe(idx,:) ];
    
    
    %Compute the RMSE for training and test set
    [y_hatTr1, probTr1] = predictY(tXTr_1, beta_1);
    [y_hatTr2, probTr2] = predictY(tXTr_2, beta_2);
   
    errTr1(k) = zeroOneLoss(y_hatTr1, yTr_1);
    errTr2(k) = zeroOneLoss(y_hatTr2, yTr_2);
    errMeanTr(k) = (errTr1(k) + errTr2(k)) /2;
    
    [y_hatTe1, probTe1] = predictY(tXTe_1, beta_1);
    [y_hatTe2, probTe2] = predictY(tXTe_2, beta_2);
   
    errTe1(k) = zeroOneLoss(y_hatTe1, yTe_1);
    errTe2(k) = zeroOneLoss(y_hatTe2, yTe_2);
    errMeanTe(k) = (errTe1(k) + errTe2(k)) /2;
    
end

%Here depending on k there is a trade of between negative bias and variance
meanTR_1 = mean(errTr1);
meanTR_2 = mean(errTr2);
meanTR = mean (errMeanTr)
stdTr = std(errMeanTr)

meanTE_1 = mean(errTe1);
meanTE_2 = mean(errTe2);
meanTE = mean (errMeanTe)
stdTr = std(errMeanTe)
