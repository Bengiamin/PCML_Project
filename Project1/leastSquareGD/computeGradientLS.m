function [g] = computeGradientLS(y,tX,beta)
    e = y - tX*beta;    
    g = tX'*e/(-length(y));
end

