
% For multi-class svm classification,
% test the principal components to see what error they give us and keep
% only the most useful ones.
for i = 1:1

        mapped_data_svm = single(X_cnn_pca(:,1:100) );


        % split data in K fold (we will only create indices)
        setSeed(1);

        K = 5;
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
            Tr.X = mapped_data_svm(idxTr,:);
            Tr.y = y(idxTr);

            Te.X = mapped_data_svm(idxTe,:);
            Te.y = y(idxTe);


        


            fprintf('Training using multi svm ...\n');

            %multi class fit using svm from matlab
             svmModel = fitcecoc(Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(svmModel, Te.X);
            
            yhat.Tr =  predict(svmModel, Tr.X);

            yhat.Te(yhat.Te ~= 4) = 1;
            yhat.Te(yhat.Te == 4) = 0;

            yhat.Tr(yhat.Tr ~= 4) = 1;
            yhat.Tr(yhat.Tr == 4) = 0;

            Tr.y(Tr.y ~= 4) = 1;
            Tr.y(Tr.y == 4) = 0;

            Te.y(Te.y ~= 4) = 1;
            Te.y(Te.y == 4) = 0;
            
                        %computer ber for this model
            berTe(k) = compute_ber(yhat.Te, Te.y, [0,1]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [0,1]);


        end

              %compute the average ber
        berMeanTrain(i) = mean(berTr);
        berMeanTest(i) = mean(berTe);

        
        fprintf('\n Testing with dimension %.f', i );
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
end


