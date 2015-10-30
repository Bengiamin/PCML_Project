function [ L ] = logLikelihoodCost(Y, tX, bet)
    L = 0;
    for i = 1:size(Y)
        L = L + Y(i)*tX(i,:)*bet - log(1+exp(tX(i,:)*bet));
    end
    
    L = -L;
end

