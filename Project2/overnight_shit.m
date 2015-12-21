
% For multi-class svm classification using CNN
% test the principal components to see what error they give us and keep
% only the most useful ones.
for i = 1:35

        mapped_data_svm = single(X_cnn_pca(:,1:25+i*5) );


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
            Tr.X = mapped_data_svm(idxTr,:);
            Tr.y = y(idxTr);

            Te.X = mapped_data_svm(idxTe,:);
            Te.y = y(idxTe);




            fprintf('Training using multi svm with CNN, %d features ...\n', 25+i*5);

            %multi class fit using svm from matlab
             svmModel = fitcecoc(Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(svmModel, Te.X);
            
            yhat.Tr =  predict(svmModel, Tr.X);

                        %computer ber for this model
            berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);


        end

              %compute the average ber
        berMeanTrainCnn(1,i) = mean(berTr);
        berMeanTrainCnn(2,i) = 25+i*5;
        berMeanTestCnn(1,i) = mean(berTe);
        berMeanTestCnn(2,i) = 25+i*5;

        
        fprintf('\n Testing CNN with dimension %.f', i );
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
end


% For multi-class svm classification using HOG
% test the principal components to see what error they give us and keep
% only the most useful ones.
for i = 1:52
    
        j = i;
        if (i > 15) 
            j = 15 + (i-15)*5;
        end

        mapped_data_svm = single(X_hog_pca(:,j) );


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
            Tr.X = mapped_data_svm(idxTr,:);
            Tr.y = y(idxTr);

            Te.X = mapped_data_svm(idxTe,:);
            Te.y = y(idxTe);




            fprintf('Training using multi svm Hog, %d features...\n', j);

            %multi class fit using svm from matlab
             svmModel = fitcecoc(Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(svmModel, Te.X);
            
            yhat.Tr =  predict(svmModel, Tr.X);

                        %computer ber for this model
            berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);


        end

              %compute the average ber
        berMeanTrainHog(1,i) = mean(berTr);
        berMeanTrainHog(2,i) = 25+i*5;
        berMeanTestHog(1,i) = mean(berTe);
        berMeanTestHog(2,i) = 25+i*5;
        
        fprintf('\n Testing CNN with dimension %.f', i );
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
end


