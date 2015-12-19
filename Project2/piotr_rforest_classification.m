%clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
%load train/train.mat;

%change to have two classes

y2 = y;
y2(y2 ~= 4) = 1;
y2(y2 == 4) = 0;

% %% --browse through the images and look at labels
% for i=1:10
%     clf();
%
%     % load img
%     img = imread( sprintf('train/imgs/train%05d.jpg', i) );
%
%     % show img
%     imshow(img);
%
%     title(sprintf('Label %d', train.y(i)));
%
%     pause;  % wait for key, 
% end

%% -- Example: split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];


% split data in K fold (we will only create indices)
setSeed(1);

K = 4;

%for nbFea = 1:100:5000
    %Split the data into k subset
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
        
        % NOTE: you should do this randomly! and k-fold!
        Tr.X = training(idxTr,:);
        Tr.y = y(idxTr);
        Tr.y2 = y2(idxTr);
        
        Te.X = training(idxTe,:);
        Te.y = y(idxTe);
        Te.y2 = y2(idxTe);
        
        clearvars training
        
        %%
        fprintf('Training Random forest model..\n');
        
        pTrain={'maxDepth',13,'M',22,'H',4,'F1',8};
        
        forest = forestTrain( Tr.X, Tr.y, pTrain);
        
        pTrain2={'maxDepth',13,'M',22,'H',2,'F1',8};
        
        forest2 = forestTrain( Tr.X, Tr.y2, pTrain);
        
        yhat = [];
        
        yhat.Te = forestApply( Te.X, forest );
        
        %yhat.Te = str2num(cell2mat(yhat.Te));
        
        yhat.Tr =  forestApply( Tr.X, forest );
        
        %yhat.Tr = str2num(cell2mat(yhat.Tr));
        
        
        berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
        berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);
        
        
        % get overall error [NOTE!! this is not the BER, you have to write the code
        %                    to compute the BER!]
        
    end
    
    
    berTeTree = mean(berTe)
    berTrTree = mean(berTr)
    
%end

    %idxss = 1:100:5000;
    %resTr = berTeTree(nbFea);
    %resTe = berTrTree(nbFea);
% %% visualize samples and their predictions (test set)
% figure;
% for i=20:30  % just 10 of them, though there are thousands
%     clf();
%
%     img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
%     imshow(img);
%
%
%     % show if it is classified as pos or neg, and true label
%     title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));
%
%     pause;  % wait for keydo that then, 
% end
