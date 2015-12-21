	

    hx = xlabel('First x principal components');
	hy = ylabel('BER');
	ht = title(sprintf('BER for SVM with the first x PC'));
    
    plot(berMeanTest(2,:), berMeanTest(1, :), 'r-','linewidth',2);
    hold on
    plot(berMeanTrain(2, :), berMeanTrain(1, :),'b-','linewidth',2);
    
    
    legend('BER test', 'BER train');
    
	ylim([0.05, 0.2]);
    xlim([0, 50]);
	set([hx, hy, ht], 'fontsize', 14);
