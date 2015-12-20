
%script used to load and prepare our data 

clearvars;


load train/train.mat;

y = train.y;

%%%% Process CNN
X_cnn = train.X_cnn;
X_hog = train.X_hog;

clearvars train

% normalize cnn
[X_cnn, mu_cnn, sigma_cnn] = zscore(X_cnn); % train, get mu and std

%%% PRocess HOG

% normalize cnn
[X_hog, mu_hog, sigma_hog] = zscore(X_hog); % train, get mu and std



% load train/test.mat;
% X_test_cnn = normalize(test.X_cnn, mu_cnn, sigma_cnn);
% X_test_hog = normalize(test.X_hog, mu_hog, sigma_hog);
% clearvars test


%Extract evaluation data didn't use during training
[X_cnn, X_hog, y, X_eval_cnn, X_eval_hog, y_eval] = split(y,X_cnn, X_hog,0.8);


%load the PCA transformed version of our data.
load pca_cnn.mat;
X_cnn_pca = single(mapped_data);
load pca_hog.mat;
X_hog_pca = single(mapped_data);

%clear unnecessary variables
clearvars mapped_data mu_cnn mu_hog sigma_cnn sigma_hog;


