clear all;

addpath(genpath('leastSquareGD/'));
addpath(genpath('LogisticRegression/'));
addpath(genpath('penLogisticRegression/'));
addpath(genpath('Data/'));

% load regression data
load('Chennai_classification');

%Extract evaluation data didn't use during training
[X_train, y_train, X_eval,y_eval] = split(y_train,X_train,0.8);

%X_train = normalize(X_train);
%X_test = normalize(X_test);
%X_eval = normalize(X_eval);

%Split data in two y < 3400 and y > 3400 
% idx = find(y_train == 1);
% X_true = X_train(idx,:);
% 
% %split x in 300 values.
% [X_true, a,b,c] =  split(ones(size(X_true,1)), X_true, 1/3);
% 
% idx = find(y_train == -1);
% X_false = X_train(idx,:);

y_train(y_train == -1) = 0;

y_eval(y_eval == -1) = 0;
% 
% X_bal = [X_true; X_false];
% y_bal = [ones(length(X_true), 1); zeros(length(X_false), 1) ];
% 
% ordering = randperm(length(y_bal));
% X_bal = X_bal(ordering, :);
% y_bal = y_bal(ordering, :);

tX_train = [ones(size(y_train)) X_train];

%good indices 
indices = [6,12,23,24,2,5,7,21];
% others [2,5,7,21, 23,24]


% tX_large = [ones(size(y_large)) X_large];
% tX_small = [ones(size(y_small)) X_small];
% 
% categIndex = [1 3 11 20 25 39 41 42 45 47 69];
% noCategIndex = [2 4:10 12:19 21:24 26:38 40 43 44 46 48:68];
% 
% categX = X_train(:, categIndex);
% noCategX = X_train(:, noCategIndex);
