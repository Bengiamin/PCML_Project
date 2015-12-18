%clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
%load train/train.mat;

%change to have two classes

%y2 = y;
%train.y2(train.y2 ~= 4) = 1;
%train.y2(train.y2 == 4) = 0;

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

K = 3;
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
    
    training =  [X_hog, X_cnn];
    
    Tr = [];
    Te = [];
    
    % NOTE: you should do this randomly! and k-fold!
    Tr.X = training(idxTr,:);
    Tr.y = y(idxTr);
    
    Te.X = training(idxTe,:);
    Te.y = y(idxTe);
    
    clearvars training
    
    %%
    fprintf('Training Random forest model..\n');
    
    pTrain={'maxDepth',100,'M',50,'H',4,'F1',1500};
    
    forest = forestTrain( Tr.X, Tr.y, pTrain);
    
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

fprintf('\n BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

fprintf('\n BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
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
