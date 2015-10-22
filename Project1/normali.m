function [ Xn ] = normali( X )

size(X,2)
Xn = X(:,1) - meanX;

end

