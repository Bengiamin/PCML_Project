function [ L ] = RMSE(Y, t, bet)
    e = Y - t*bet;
    L = e'*e/(2*length(Y));
end

