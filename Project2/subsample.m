function [ X_ret, y_ret ] = subsample( X, y, prop )
%subsample the data for the class 4 to a certain proportion
%   Detailed explanation goes here

    setSeed(1);
    
    index4 = find(y == 4);
 
    %get a random index 
    N = size(index4,1);
    idx = randperm(N);
    limit = floor((1-prop) * N);
    idx = idx(1:limit);
    
   
    
    y_ret = y;
    y_ret(index4(idx)) = [];
    
    X_ret = X;
    X_ret(index4(idx), :) = [];
    


end

