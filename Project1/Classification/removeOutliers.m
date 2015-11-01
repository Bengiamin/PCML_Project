function [ X, y ] = removeOutliers( X, y, idx)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    if(idx == 0)
        idx = 1:size(X,2);
    end
    
    
    threshold = 5;
    for i = idx
        Xn = normalize(X);
        %threshold = 0.5*std(X(:,i));
        notOutliers = find(abs(Xn(:,i)) < threshold);
        X = X(notOutliers,:);
        if(y ~= 0)
            y = y(notOutliers);
        end
    end
end

