%load features

training = X_cnn_pca(:,1:100);

%transform labels into binary
y2 = y;
y2(y2 ~= 4) = 1;
y2(y2 == 4) = 0;

%binary fit 
svmBinModel = fitcsvm(training, y2);

testing = X_eval_cnn * mapping_cnn.M(:,1:100);


yhat_bin = predict(svmBinModel, testing );

