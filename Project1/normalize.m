function [ Xn ] = normalize( X )

size(X,2);

meanX = mean(X);
stdX = std(X);
Xn = zeros(size(X));

for i = 1:size(X, 2)
    Xn(:, i) = ( X(:,i) - meanX(i) ) / stdX(i) ;
end;


end

