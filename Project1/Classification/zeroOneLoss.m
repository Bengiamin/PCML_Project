function [ error ] = zeroOneLoss( y_pred, y )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

   error = sum(y_pred == y) / length(y);
    

end

