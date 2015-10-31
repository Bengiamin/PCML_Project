%Compute the score for every featue of x

lambda = 0.5;
K= 10;

D = size(X_small,2);

rmse_x_Tr = zeros(D,1);
rmse_x_Te = zeros(D,1);
min = 10000;
best_x = 0;
for i = 1:size(X_train,2)
    [ rmseTr, rmseTe] = compute_score(y_small, X_small, 'rr', i, 0 , lambda , K, 0);
    
    rmse_x(i) = (rmseTr + rmseTe)/2;
    
    if rmse_x(i) < min
        min = rmse_x(i);
        best_x = i;
    end
    
end

disp 'Most useful feature of X:'
best_x
figure
scatter(1:D,rmse_x,'+')
title('RMSE of model using individual feature of X')
xlabel('Indices of X')
ylabel('RMSE')