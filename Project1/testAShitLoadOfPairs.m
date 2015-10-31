errTR = zeros(27);
errTE = zeros(27);

for i = 1:27
    for j = 1:27
        [errTR(i,j), errTE(i,j)] = classif_score(y_train, X_train, [i,j], 0.5, 10, 4, 0);
        fprintf('For features %d, %d : err TR = %0.4f , err TE %0.4f\n', i,j,errTR(i, j), errTE(i,j));

    end
    
end

meanTR = mean(errTR)
meanTE = mean(errTE)

[minTR, minITR] = min(min(errTR))
[minTE, minITE] = min(min(errTE))
