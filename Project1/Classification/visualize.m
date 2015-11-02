function [ output_args ] = visualize(X_train, y_train)
%VISUALIZE Summary of this function goes here
%   Detailed explanation goes here
figure;

idx = find(X_train(:,2) > 18);
yTr_1 = y_train(idx);
XTr_1 = X_train(idx,:);

idx = find(X_train(:,2) < 18);
yTr_2 = y_train(idx);
XTr_2 = X_train(idx,:);

idx = find(y_train == 1);
X_true = X_train(idx,:);

idx = find(y_train == 0);
X_false = X_train(idx,:);

for i=1:size(X_true,2)
    %scatter(X_train(:,i),y_train, '.');
    histogram(X_true(:,i))
    hold on
    histogram(X_false(:,i))
    
    hx = xlabel('Value taken by a single feature');
    hy = ylabel('Number of value');
    
    % the following code makes the plot look nice and increase font size etc.
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;
    
    i
    pause;
    hold off
end


end

