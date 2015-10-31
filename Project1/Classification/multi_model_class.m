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
    yTe = y_train(idxTe);
    XTe = X_train(idxTe,:);
    yTr = y_train(idxTr);
    XTr = X_train(idxTr,:);
    
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
    
    %Spliting the test value into two set according to the 2nd row of X
    idx = find(XTe(:,2) > 18);
    yTe_1 = yTe(idx);
    tXTe_1 = [ones(length(yTe_1), 1) XTe(idx,:)];
    
    idx = find(XTe(:,2) <= 18);
    yTe_2 = yTe(idx);
    tXTe_2 = [ones(length(yTe_2), 1) XTe(idx,:) ];
    
    
    %Compute the prediction for training and test set
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
    
    
    %%%%%%%%%%%%%%%%%% Testing on unbalanced (evaluation) test %%%%%%%%%%%%%%%%%%%
     %Spliting the test value into two set according to the 2nd row of X
    idx = find(X_eval(:,2) > 18);
    y_eval_1 = y_eval(idx);
    tX_eval_1 = [ones(length(y_eval_1), 1) X_eval(idx,:)];
    
    idx = find(X_eval(:,2) <= 18);
    y_eval_2 = y_eval(idx);
    tX_eval_2 = [ones(length(y_eval_2), 1) X_eval(idx,:) ];
    
     %Compute the prediction for evaluation set
    [y_hat_eval_1, probE1] = predictY(tX_eval_1, beta_1);
    [y_hat_eval_2, probE2] = predictY(tX_eval_2, beta_2);
   
    err_eval_1(k) = zeroOneLoss(y_hat_eval_1, y_eval_1);
    err_eval_2(k) = zeroOneLoss(y_hat_eval_2, y_eval_2);
    errMean_eval(k) = (err_eval_1(k) + err_eval_2(k)) /2;
    
    
    
end

%Here depending on k there is a trade of between negative bias and variance
meanTR_1 = mean(errTr1);
meanTR_2 = mean(errTr2);
meanTR = mean (errMeanTr)
stdTr = std(errMeanTr)

meanTE_1 = mean(errTe1);
meanTE_2 = mean(errTe2);
meanTE = mean (errMeanTe)
stdTe = std(errMeanTe)

mean_EVAL_1 = mean(err_eval_1);
mean_EVAL_2 = mean(err_eval_2);
mean_EVAL = mean (errMean_eval)
std_EVAL = std(errMean_eval)