close all
clear all
clc

sx = 64; sy = 64; sz = 20;

temps = abs(100*randn(sy,sx,sz));

myrand = zeros(size(temps));

a = 16807.0;
m = 2147483647.0; 
reciprocal_m = 1.0/m;

for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            
            seed = temps(y,x,z)*1;
            
            seed = (x + y*64 + z*64*64)*1000;
            %seed = x * reciprocal_m;
            
            temp = seed * a;
            seed = (temp - m * floor(temp * reciprocal_m));
            
            myrand(y,x,z) = seed * reciprocal_m;
            
        end
    end
end

hist(myrand(:),50)