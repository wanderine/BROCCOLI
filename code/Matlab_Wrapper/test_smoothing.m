%  	 BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
%    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

%mex Smoothing.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  

mex -g Smoothing.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  

test_data = randn(69,123,33,100);
filter_x = randn(9,1);
filter_x = filter_x / sum(abs(filter_x));
filter_y = randn(9,1);
filter_y = filter_y / sum(abs(filter_y));
filter_z = randn(9,1);
filter_z = filter_z / sum(abs(filter_z));
smoothed_volumes_opencl = Smoothing(test_data,filter_x,filter_y,filter_z);

temp = zeros(1,9,1);
temp(1,:,1) = filter_x;
filter_x = temp;

temp = zeros(9,1,1);
temp(:,1,1) = filter_y;
filter_y = temp;

temp = zeros(1,1,9);
temp(1,1,:) = filter_z;
filter_z = temp;

smoothed_volumes_cpu = zeros(size(test_data));
for t = 1:size(test_data,4)
   volume = test_data(:,:,:,t);
   smoothed_volume = convn(volume,filter_x,'same');
   smoothed_volume = convn(smoothed_volume,filter_y,'same');   
   smoothed_volume = convn(smoothed_volume,filter_z,'same');
   smoothed_volumes_cpu(:,:,:,t) = smoothed_volume;
end

%smoothed_volumes_cpu = test_data;

figure
plot(squeeze(smoothed_volumes_cpu(25,25,25,:)),'r')
hold on
plot(squeeze(smoothed_volumes_opencl(25,25,25,:)+0.0),'b')
hold off

figure
imagesc([smoothed_volumes_cpu(:,:,20,1)  smoothed_volumes_opencl(:,:,20,1)])

smoothed_volumes_cpu(1,5,1,1)
smoothed_volumes_opencl(1,5,1,1)

tot_error = sum(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))
max_error = max(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))



