clear all;

addpath(genpath('leastSquareGD/'));
addpath(genpath('LogisticRegression/'));
addpath(genpath('penLogisticRegression/'));
addpath(genpath('Data/'));

% load regression data
load('Chennai_regression');

%Split data in two y < 3400 and y > 3400 
idx = find(y_train > 3400);
y_large = y_train(idx);
X_large = X_train(idx,:);

idx = find(y_train <= 3400);
y_small = y_train(idx);
X_small = X_train(idx,:);


tX_train = [ones(size(y_train)) X_train];
tX_large = [ones(size(y_large)) X_large];
tX_small = [ones(size(y_small)) X_small];