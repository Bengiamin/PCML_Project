%Compute the score for ermse_xvery featue of x
data_preprocessing_class;

lambda = 0.5;
K= 4;
alpha = 0.1;

D = size(X_train,2);

rmse_x_Tr = zeros(D,1);
rmse_x_Te = zeros(D,1);
min = 10000;
best_x = 0;
for i = 1:size(X_train,2)
    [ rmseTr, rmseTe] = compute_score(y_train, X_train, 'plr', i, alpha , lambda , K, 0);
    
    rmse_x(i) = (rmseTr + rmseTe)/2;
    
    if rmse_x(i) < min
        min = rmse_x(i);
        best_x = i;
    end
    
    disp 'x finish'
    i
end

disp 'Most useful feature of X:'
best_x
figure
scatter(1:D,rmse_x,'+')
title('Log error of model using individual feature of X')
xlabel('Indices of X')
ylabel('Log error')