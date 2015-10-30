function [g] = computeGradientLR(y,tX,beta)   
    S = sigma(tX*beta);
    g = tX'*(S - y);
end

