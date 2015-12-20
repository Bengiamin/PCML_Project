%load features

training = [X_cnn_pca(:,1:100)];


%binary fit 
svmModel = fitcecoc(training, y);

testing = X_eval_cnn * mapping_cnn.M(:,1:100);

yhat = predict(svmModel, testing);

