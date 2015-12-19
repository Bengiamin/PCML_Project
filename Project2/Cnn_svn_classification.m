%svm baseline using a k-fold


%change to have two classes

y2 = y;
y2(y2 ~= 4) = 1;
y2(y2 == 4) = 0;

training = [X_hog X_cnn];


% split data in K fold (we will only create indices)
        setSeed(1);

        K = 4;
        %Split the data into k subset<
        N = size(y,1);
        idx = randperm(N);
        Nk = floor(N/K);
        idxCV = zeros(K,Nk);
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
            Tr.X = training(idxTr,:);
            %Tr.X = X_cnn(idxTr,:);
            Tr.y = y2(idxTr);

            Te.X = training(idxTe,:);
            %Te.X = X_cnn(idxTe,:);
            Te.y = y2(idxTe);




            fprintf('Training using svm ...\n');


            svmModel = fitcsvm(Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(svmModel, Te.X);
           
            yhat.Tr =  predict(svmModel, Tr.X);


            berTe(k) = compute_ber(yhat.Te, Te.y, [0,1]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [0,1]);


        end

 
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );

