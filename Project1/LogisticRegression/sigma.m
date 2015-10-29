function [ S ] = sigma( X )

     for i = 1:size(X)
         if X(i) > 0
             S(i) = 1 / (1 + exp(-X(i)));
         else
             S(i) = exp(X(i)) / (1 + exp(X(i)));
     end

     S = S';
end

