clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
% load train/train.mat;

%change to have two classes

y2 = y;
train.y2(y2 ~= 4) = 1;
train.y2(y2 == 4) = 0;

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

% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X_cnn,1);
Tr.X = X_cnn(Tr.idxs,:);
Tr.y = y2(Tr.idxs);

Te.idxs = 2:2:size(X_cnn,1);
Te.X = X_cnn(Te.idxs,:);
Te.y = y2(Te.idxs);

%%
fprintf('Training SVM model..\n');

%
svmModel = fitcsvm(Tr.normX, Tr.y);

[yhat, scores] = predict(svmModel, Te.normX);

figure;
histogram(yhat);

figure;
histogram(Te.y);

ber = ber(yhat, Te.y, [0,1]);


% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]





fprintf('\n BER Testing error: %.2f%%\n\n', ber * 100 );


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
