function [ L ] = computeCost(Y, t, bet)
    L = 0;
    for i = 1:size(Y)
        L = L + Y(i)*t(1,:)*bet - log(1+exp(t(1,:)*bet));
    end
    
    L = -L;
end

