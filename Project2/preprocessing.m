clearvars;


load train/train.mat;

X_hog = train.X_hog;
%X_cnn = train.X_cnn;
y = train.y;

clearvars train

coeff_hog = pca(X_hog);
%coeff_cnn = pca(X_cnn,'NumComponents',10000);