%load features

training = [X_cnn_pca X_hog_pca];

%transform labels into binary
y2 = y;
y2(y2 ~= 4) = 1;
y2(y2 == 4) = 0;

%binary fit 
svmModel = fitcsvm(training, y2);

yhat = predict(svmModel, X_test);

