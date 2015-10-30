function [ output_args ] = plot_polyBeta(y, X, degree, beta )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% plot fit
	plot(X, y, 'ob', 'markersize', 4); % plot data
	hold on;
	xvals = [min(X)-0.1:.1:max(X)+.1];
	tX = [ones(length(xvals),1) myPoly(xvals(:), degree)];
	f = tX*beta;
	plot(xvals, f,'r-','linewidth',2);
	xlim([min(xvals), max(xvals)]);
	hx = xlabel('x');
	hy = ylabel('y');
	ht = title(sprintf('Polynomial degree %d',degree));
	set([hx, hy, ht], 'fontsize', 14);

end

