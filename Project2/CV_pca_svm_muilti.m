
for i = 1:20

        mapped_data_svm = single(mapped_data(:,1:i));


        % split data in K fold (we will only create indices)
        setSeed(1);

        K = 4;
        %Split the data into k subset
        N = size(y,1);
        idx = randperm(N);
        Nk = floor(N/K);
        idxCV = zeros(K,Nk);
        
        berTe = zeros(K, 1);
        berTr = zeros(K,1);
        
        for k = 1:K
            idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
        end

        for k = 1:K
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);


            Tr = [];
            Te = [];

            % NOTE: you should do this randomly! and k-fold!
            Tr.X = mapped_data(idxTr,:);
            Tr.y = y(idxTr);

            Te.X = mapped_data(idxTe,:);
            Te.y = y(idxTe);




            fprintf('Training using multi svm ...\n');

            %pTrain={'maxDepth',100,'M',50,'H',4,'F1',1500};

             svmModel = fitcecoc(Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(svmModel, Te.X);
            
            yhat.Tr =  predict(svmModel, Tr.X);

            
            berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);


        end

        berMeanTrain(i) = mean(berTr);
        berMeanTest(i) = mean(berTe);

        
        fprintf('\n Testing with dimension %.f', i );
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
end


