
%Compute the score for every featue of x

lambda = 0.5;
K= 10;

D = size(X_small,2);

rmse_x_Tr = zeros(D,1);
rmse_x_Te = zeros(D,1);
min = 10000;
best_x = 0;
for i = 1:size(X_small,2)
    [ rmseTr, rmseTe] = compute_score(y_small, X_small, 'rr', i, 0 , lambda , K, 0);
    
    rmse_x(i) = (rmseTr + rmseTe)/2;
    
    if rmse_x(i) < min
        min = rmse_x(i);
        best_x = i;
    end
    
end

D = size(X_large,2);

min = 10000;
best_x = 0;
for i = 1:size(X_large,2)
    [ rmseTr, rmseTe] = compute_score(y_large, X_large, 'rr', i, 0 , lambda , K, 0);
    
    rmse_xl(i) = (rmseTr + rmseTe)/2;
    
    if rmse_xl(i) < min
        min = rmse_xl(i);
        best_x = i;
    end
    
end

% visualize
figure(1);
bar([rmse_x' rmse_xl'])
hx = xlabel('Indices of X');
hy = ylabel('RMSE');

% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

% print the file
print -dpdf histY.pdf

% Next you should CROP PDF using pdfcrop in linux and mac. Windows - not sure of a solution.