
for i=1:10
    clf();

    % load img
    img = imread( sprintf('train/imgs/train%05d.jpg', i) );

    % show img
    subplot(131);
    imshow(img);

    subplot(132);
    
    feature = hog( single(img)/255, 17, 8);
    featureTrain = train.X_hog(i,:);
    featureTrain = reshape(featureTrain, size(feature));
    
%     size(feature)
%     size(featureTrain)
    
    im( hogDraw(feature) ); colormap gray;
    axis off; colorbar off;
    
    subplot(133);
    
    im( hogDraw(featureTrain) ); colormap gray;
    axis off; colorbar off;
    
    %title(sprintf('Label %d', train.y(i)));

    pause;  % wait for key, 
end

