function [ S ] = sigma( X )

     for i = 1:size(X)
         S(i) = exp(X(i)) ./ (1 + exp(X(i)));
     end

     S = S';
end

