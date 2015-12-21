


%Training our model on the data
forest = TreeBagger(91, X_cnn_pca(:,1:21), y,'NVarToSample', 9);


K = 6;

%Split the evaluation data into k subset
N = size(y_eval,1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K,Nk);

for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    idxTe = idxCV([1:k-1 k+1:end],:);
    idxTe = idxTe(:);
    
    X_eval_T = X_eval_cnn(idxTe,1:21);
    y_eval_T = y2_eval(idxTe);
    
    %Predict on subset of evaluation
    yhat = predict(forest, X_eval_T);
    yhat = str2num(cell2mat(yhat));
    
    %Aggregate to binary classification
    classVoteBin = yhat;
    
    classVoteBin(classVoteBin ~= 4) = 1;
    classVoteBin(classVoteBin == 4) = 0;
    
    berTeBin(k) = compute_ber(classVoteBin, y_eval_T, [1,0]);
    
end

fprintf('\n   BER Testing error binary: %.2f%%\n\n', mean(berTeBin) * 100 );