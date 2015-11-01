function [ output_args ] = visualize(X_train, y_train)
%VISUALIZE Summary of this function goes here
%   Detailed explanation goes here
figure;

idx = find(X_train(:,2) > 18);
yTr_1 = y_train(idx);
XTr_1 = X_train(idx,:);

idx = find(X_train(:,2) <= 18);
yTr_2 = y_train(idx);
XTr_2 = X_train(idx,:);

idx = find(yTr_2 == 1);
X_true = XTr_2(idx,:);

idx = find(yTr_2 == 0);
X_false = XTr_2(idx,:);

for i=1:size(X_true,2)
    %scatter(X_train(:,i),y_train, '.');
    histogram(X_true(:,i))
    hold on
    histogram(X_false(:,i))
    xlabel(i);
    pause;
    hold off
end


end

