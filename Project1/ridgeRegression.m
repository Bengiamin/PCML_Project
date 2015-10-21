function [ beta ] = ridgeRegression(y,tX, lambda)
lm = lambda.*eye(size(tX,2));
lm(1,1) = 0;
beta = (tX'*tX + lm)\(tX'*y);
end

