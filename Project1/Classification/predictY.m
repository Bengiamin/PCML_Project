function [ y_hat, prob ] = predictY( tX, beta )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    

    pred = tX * beta;
    prob = sigma(pred);
    
    y_hat = round(prob);
    



end

