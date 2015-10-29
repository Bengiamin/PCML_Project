function [ S ] = sigma( X )
    

     for i = 1:size(X, 1)
         S(i) = exp(X(i)) / (1 + exp(X(i)));
     end

     S = S';
end

