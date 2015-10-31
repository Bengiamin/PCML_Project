
% plot data
for i=1:size(X_train,2)
    scatter(X_train(:,i),y_train, '.');
    hold on;
end

% me = mean(y_train);
% 
% plot(get(gca,'xlim'), [me me]); 
% 
% %Split data in two y < 3400 and y > 3400 
% idx = find(y_train > 3400);
% y_large = y_train(idx);
% X_large = X(idx,:);
% 
% idx = find(y_train <= 3400);
% y_small = y_train(idx);
% X_small = X(idx,:);
% 
% Sme = mean(y_small);
% 
% plot(get(gca,'xlim'), [Sme Sme]);
% 
% Lme = mean(y_large);
% 
% plot(get(gca,'xlim'), [Lme Lme]);

figure;
for i=1:size(X_train,2)
    for j=1:size(X_train,2)
        scatterBy2( X_true, X_false, i, j)
        pause
    end
end




%plot data
% for i=1:69
%     scatter(X_small(:,i),y_small);
%     hold on;
% end

% for i=1:69
%     scatter(X_large(:,i),y_large);
%     hold on;
% end