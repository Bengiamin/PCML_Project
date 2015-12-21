

training = [X_cnn_pca(:,1:21)];

% split data in K fold (we will only create indices)
setSeed(1);

K = 6;

%Split the data into k subset
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K,Nk);

for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    
    
    Tr = [];
    Te = [];
    
    %Here we can subsample to the class 4 if we want
    [Tr.X, Tr.y] = subsample(training(idxTr,:), y(idxTr), 1) ;
    
    Te.X = training(idxTe,:);
    Te.y = y(idxTe);
    
    Te.y2 = y2(idxTe);
    
    
    
    
    
    fprintf('Training using random forests (treebagger) ...\n');
    
    %Training random forest
    forest = TreeBagger(91, Tr.X, Tr.y,'NVarToSample', 9);
 
    yhat = [];
    
    %Predicting 
    yhat.Te = predict(forest, Te.X);
    yhat.Te = str2num(cell2mat(yhat.Te));
    
    yhat.Tr =  predict(forest, Tr.X);
    yhat.Tr = str2num(cell2mat(yhat.Tr));
    
    %Aggregating to binary
    classVoteBin = yhat.Te;
    
    classVoteBin(classVoteBin ~= 4) = 1;
    classVoteBin(classVoteBin == 4) = 0;
    
    %computer ber for this model
    berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
    berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);
    
    berTeBin(k) = compute_ber(classVoteBin, Te.y2, [1,0]);
    
    
end


fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );
fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
fprintf('\n   BER Testing error binary: %.2f%%\n\n', mean(berTeBin) * 100 );


