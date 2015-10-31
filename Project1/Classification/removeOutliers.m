function [ X, y ] = removeOutliers( X, y, idx, threshold)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    if(idx == 0)
        idx = 1:size(X,2);
    end

    for i = idx
        notOutliers = find(abs(X(:,i)) < threshold);
        X = X(notOutliers,:);
        if(y ~= 0)
            y = y(notOutliers);
        end
    end
end

