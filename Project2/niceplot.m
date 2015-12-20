	
    values = xvals;
    hx = xlabel('x');
	hy = ylabel('y');
	ht = title(sprintf('Polynomial degree %d',degree));
    
    plot(values, f,'r-','linewidth',2);
	xlim([min(xvals), max(xvals)]);
	set([hx, hy, ht], 'fontsize', 14);
