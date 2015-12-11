function [ ber ] = compute_ber( yhat, y, classes )
%Compute BER error
%   classes should be [N1,N2] is binary [N1,N2,N3,N4] for all classes.
%   

c = length(classes);
ber = 0;

for i = 1:c
   
   indices = find(y == classes(i));
   %disp(indices)
   ni = length(indices);
   tmp = 0;
   
   for j = 1:length(indices)
       if y(indices(j)) ~= yhat(indices(j))
%           disp( y(j))
%           disp( yhat(j))
%           disp( y(j) == yhat(j))
        tmp = tmp + 1;
       end
   end
   ber = ber + tmp/ni;
    
end

ber = ber / c;

end

