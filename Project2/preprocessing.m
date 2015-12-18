clearvars;


load train/train.mat;

y = train.y;

%%%% Process CNN
X_cnn = train.X_cnn;

% normalize cnn
[X_cnn, mu_cnn, sigma_cnn] = zscore(X_cnn); % train, get mu and std

%%% PRocess HOG
X_hog = train.X_hog;

% normalize cnn
[X_hog, mu_hog, sigma_hog] = zscore(X_hog); % train, get mu and std

clearvars train

% load train/test.mat;
% X_test_cnn = normalize(test.X_cnn, mu_cnn, sigma_cnn);
% X_test_hog = normalize(test.X_hog, mu_hog, sigma_hog);
% clearvars test


%Extract evaluation data didn't use during training
[X_cnn, X_hog, y, X_eval_cnn, X_eval_hog, y_eval] = split(y,X_cnn, X_hog,0.8);


coeff_hog = pca(X_hog);



%coeff_cnn = pca(X_cnn,'NumComponents',1000);
