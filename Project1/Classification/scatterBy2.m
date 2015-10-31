function [ output_args ] = scatterBy2( X_true, X_false, i1, i2  )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
    figure;
    scatter(X_true(:,i1), X_true(:,i2), '+');
    hold on;
    scatter(X_false(:,i1), X_false(:,i2), '.    ');
    xlabel(i1);
    ylabel(i2);
end

