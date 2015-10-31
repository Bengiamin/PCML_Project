function [ output_args ] = visualize( X_true, X_false )
%VISUALIZE Summary of this function goes here
%   Detailed explanation goes here
figure;

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

