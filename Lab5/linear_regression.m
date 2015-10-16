% Load data
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
y = gender;
X = [height(:) weight(:)];
% randomly permute data
N = length(y);
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% subsample
y = y(1:200);
X = X(1:200,:);

male = X(y, :);
female = X(not(y), :);

%figure
%scatter(male(:,1),male(:,2))
%hold on
%scatter(female(:,1),female(:,2))

% create a 2−D meshgrid of values of heights and weights
h = [min(X(:,1)):.01:max(X(:,1))];
w = [min(X(:,2)):1:max(X(:,2))];
[hx, wx] = meshgrid(h,w);
% predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

tX = [ones(size(X,1),1) X];

beta = leastSquares(y,tX);

L = computeCost(y,tX,beta)

pred = beta(1)*ones(size(hx)) + beta(2)*hx + beta(3)*wx;

% plot the decision surface
contourf(hx, wx, pred, 1);
% plot indiviual data points
hold on
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(male(:,1),male(:,2),'xr','color',myRed,'linewidth', 2, 'markerfacecolor', myRed);
hold on
plot(female(:,1),female(:,2),'or','color', myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
xlabel('height');
ylabel('weight');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;

