function [ X ] = dummy_coding( X_train, idx )
    notIdx=setdiff(1:size(X_train,2),idx);
    X = X_train(:,notIdx); 
    
    for i = idx
       X = [ X  dummyvar(X_train(:,i))];
    end

end

