%Execute preprocessing to load the data 
%To run this script you need to have the Piotr toolbox added to the path

% split data in K fold (we will only create indices)
setSeed(1);

K = 6;
min = 1;

for nbFea = 1:1:30
    for depth = 1:1:20
        for tree = 1:1:30
            
    %Randomly generate K folds
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
        
        training =  [X_hog_pca(:,1:5) X_cnn_pca(:,1:9)];
        
        Tr = [];
        Te = [];
        
        %Split into training an testing
        Tr.X = training(idxTr,:);
        Tr.y = y(idxTr);
        Tr.y2 = y2(idxTr);
        
        Te.X = training(idxTe,:);
        Te.y = y(idxTe);
        Te.y2 = y2(idxTe);
        
        %Delete useless variable for memory
        clearvars training
        
        %%
        fprintf('Training Random forest model..\n');
        
        %Specify the parameter of the random forest
        pTrain={'maxDepth',depth,'M',tree,'H',4,'F1',nbFea,'split','gini'};
        
        forest = forestTrain( Tr.X, Tr.y, pTrain);
        
        yhat = [];
        
        %Predict class
        yhat.Te = forestApply( Te.X, forest );        
        yhat.Tr =  forestApply( Tr.X, forest );
        
        
        %Compute the BER
        berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
        berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);
        
    end
    
    %Save only the best combination of parameter
    if(mean(berTe) < min)
        bTree = tree;
        bDepth = depth;
        bFea = nbFea;
        
        min = mean(berTe)
    end
    
       end
    end
end

   