%predict the test results from the binary and multi class SVM models
%assume the mondels stored int .mat files.
%assume the pca mapping matrix stored in .mat file as well

clearvars

load('bestBinarySVM.mat');
load('bestMulticlassSVM.mat');
load('test.mat');
load('pca_cnn_map.mat');
load('normalizedParam');

%normalize test data
%X_test_cnn = normalize(double(test.X_cnn), mu_cnn, sigma_cnn);
[X_test_cnn, mu_cnn, sigma_cnn] = zscore(test.X_cnn);

clearvars test;

X_cnn_pca = X_test_cnn * mapping.M(:,1:100);

yhat_bin = predict(svmBinModel, X_cnn_pca);

yhat_mc = predict(svmClassModel, X_cnn_pca);