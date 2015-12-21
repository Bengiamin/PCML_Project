%You should run the preprocessing script to load variable

K = 6;

%computing the k Fold
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    
    training =  [X_cnn_pca(:,1:100)];
    
    Tr = [];
    Te = [];
    
    % Spliting training and testing
    Tr.X = training(idxTr,:);
    Tr.y = y2(idxTr);
    
    Te.X = training(idxTe,:);
    Te.y = y2(idxTe);
    
    clearvars training
    
    %%
    fprintf('Training simple neural network..\n');
    
    rng('default');
    rng(8339);  % fix seed, this    NN may be very sensitive to initialization
    
    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 10 2]);
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 0;
    
    nn.learningRate = 2;
    
    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);
    
    % prepare labels for NN
    LL = [1*(Tr.y == 1), ...
        1*(Tr.y == 0),]; 
    % first column, p(y=1)
    % second column, p(y=2), etc
    
    [nn, L] = nntrain(nn, Tr.X, LL, opts);
    
    
    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.X, zeros(size(Te.X,1), nn.size(end)));
    nn.testing = 0;
    
    
    % predict on the test set
    nnPred = nn.a{end};
    
    % get the most likely class
    [~,classVote] = max(nnPred,[],2);

    classVote(classVote == 2) = 0;

    berTe(k) = compute_ber(classVote, Te.y, [1,0]);
    
end

fprintf('\nTesting error: %.2f%%\n\n', mean(berTe) * 100 );
