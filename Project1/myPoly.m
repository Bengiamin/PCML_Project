function Xpoly = myPoly(X,degree)
% build matrix Phi for polynomial regression of a given degree
for i = 1:size(X,2)
    for k = 1:degree
        Xpoly(:,(i-1)*degree+k) = X(:,i).^k;
    end
end
end

