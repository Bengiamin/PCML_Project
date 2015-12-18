function [X_cnn, X_hog, y, X_eval_cnn, X_eval_hog, y_eval] = split(y_in, X_cnn_in, X_hog_in, prop)

% split the data into train and test given a proportion
		setSeed(1);
    N = size(y_in,1);
		% generate random indices
		idx = randperm(N);
    Ntr = floor(prop * N);
		% select few as training and others as testing
		idxTr = idx(1:Ntr);
		idxTe = idx(Ntr+1:end);
		% create train-test split
    X_cnn = X_cnn_in(idxTr,:);
    X_hog = X_hog_in(idxTr,:);
    y = y_in(idxTr);
    X_eval_cnn = X_cnn_in(idxTe,:);
    X_eval_hog = X_hog_in(idxTe,:);
    y_eval = y_in(idxTe);

end

function setSeed(seed)
% set seed
	global RNDN_STATE  RND_STATE
	RNDN_STATE = randn('state');
	randn('state',seed);
	RND_STATE = rand('state');
	%rand('state',seed);
	rand('twister',seed);
end
