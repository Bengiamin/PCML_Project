clear all;

addpath(genpath('leastSquareGD/'));
addpath(genpath('LogisticRegression/'));
addpath(genpath('penLogisticRegression/'));
addpath(genpath('Data/'));

% load regression data
load('Chennai_classification');

%Extract evaluation data didn't use during training
[X_train, y_train, X_eval,y_eval] = split(y_train,X_train,0.8);

X_train_u = X_train;
X_eval_u = X_eval;
X_test_u = X_test;

X_train = normalize(X_train);
X_test = normalize(X_test);
X_eval = normalize(X_eval);

 categ = [1,4,15,23];
 X_train(:,23) = X_train_u(:,23);
 X_eval(:,23) = X_eval_u(:,23);
 X_test(:,23) = X_test_u(:,23);

%Dummy coding on feature 15
X_train = dummy_coding(X_train_u, 15);
X_eval = dummy_coding(X_eval_u, 15);
X_test = dummy_coding(X_test_u, 15);

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
