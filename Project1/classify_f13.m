

figure
scatter(X_small(:,13), y_small );
hold on;
scatter(X_large(:,13), y_large);


beta = logisticRegression(binY, tX_train(:,[1 14]), 0.00001)

lim = - beta(1)/beta(2);


plot([lim, lim],get(gca,'ylim'));