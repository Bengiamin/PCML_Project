function [L, g, H] = logisticRegLoss(beta, y, tX, lambda)

%Compute cost
L = 0;
S = zeros(size(y));
for i = 1:size(y)
    L = L + y(i)*tX(i,:)*beta - log(1+exp(tX(i,:)*beta));
    S(i,i) = sigma(tX(i,:)*beta)*(1-sigma(tX(i,:)*beta));
end

L = - L + lambda * dot(beta, beta);

lm = lambda.*eye(size(tX,2));
lm(1,1) = 0;

%Compute hession
H = tX'*S*tX + lm;

%Compute gradient
S = sigma(tX*beta);
g = tX'*(S - y);

end

