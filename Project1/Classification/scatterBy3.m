function [ output_args ] = scatterBy3( X_true, X_false, i1, i2, i3  )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
    figure;
    scatter3(X_true(:,i1), X_true(:,i2), X_true(:,i3), '+');
    hold on;
    scatter3(X_false(:,i1), X_false(:,i2), X_false(:,i3), '.');

end

