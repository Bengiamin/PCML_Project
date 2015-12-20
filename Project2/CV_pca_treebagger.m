
% test the principal components to see what error they give us and keep
% only the most useful ones.
for i = 1:25

    mapped_data_tree = single(mapped_data(:,1:i));
            
        % split data in K fold (we will only create indices)
        setSeed(1);

        K = 8;
        %Split the data into k subset
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
            Tr.X = mapped_data_tree(idxTr,:);
            Tr.y = y(idxTr);

            Te.X = mapped_data_tree(idxTe,:);
            Te.y = y(idxTe);




            fprintf('Training using random forests (treebagger) ...\n');

            %use treebagger from matlab 
            forest = TreeBagger(50, Tr.X, Tr.y);

            yhat = [];

            yhat.Te = predict(forest, Te.X);
            yhat.Te = str2num(cell2mat(yhat.Te));
            
            yhat.Tr =  predict(forest, Tr.X);
            yhat.Tr = str2num(cell2mat(yhat.Tr));

            %computer ber for this model
            berTe(k) = compute_ber(yhat.Te, Te.y, [1,2,3,4]);
            berTr(k) = compute_ber(yhat.Tr, Tr.y, [1,2,3,4]);


        end
        
        %compute the average ber
        berMeanTrain(i) = mean(berTr);
        berMeanTest(i) = mean(berTe);

        
        fprintf('\n Testing with dimension %.f', i );
        fprintf('\n   BER Testing error: %.2f%%\n\n', mean(berTe) * 100 );

        fprintf('\n   BER Training error: %.2f%%\n\n',mean(berTr) * 100 );
end


