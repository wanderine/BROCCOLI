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

% Compile the Matlab wrapper to get a mex-file
if ispc
    %mex -g SmoothingMex.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib      
    mex SmoothingMex.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen 
   
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\BROCCOLI_test_data\';
   
    opencl_platform = 0;
    opencl_device = 0;
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';
    
    %mex -g SmoothingMex.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex SmoothingMex.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release    -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    
    opencl_platform = 2; % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 0;
end

% Set smoothing amount
smoothing_FWHM = 8.0;
smoothing_type = 0; % 0 = Gaussian smoothing, 1 = Random smoothing

study = 'Cambridge';

% Load fMRI data
s = 5;
dirs = dir([basepath study]);
subject = dirs(s+2).name % Skip . and .. 'folders'
%EPI_nii = load_nii([basepath study '/' subject '/func/rest.nii.gz']);
EPI_nii = load_nii([basepath study '/rest1.nii.gz']);
fMRI_volumes = double(EPI_nii.img);
[sy sx sz st] = size(fMRI_volumes)

voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
    
% Create smoothing filters for Matlab

% Gaussian lowpass filters
if smoothing_type == 0
    % Convert smoothing amount in FWHM to sigma
    sigma_x = smoothing_FWHM / 2.354 / voxel_size_x;
    sigma_y = smoothing_FWHM / 2.354 / voxel_size_y;
    sigma_z = smoothing_FWHM / 2.354 / voxel_size_z;

    filter_x = fspecial('gaussian',9,sigma_x);
    filter_x = filter_x(:,5);
    filter_x = filter_x / sum(abs(filter_x));

    filter_y = fspecial('gaussian',9,sigma_y);
    filter_y = filter_y(:,5);
    filter_y = filter_y / sum(abs(filter_y));

    filter_z = fspecial('gaussian',9,sigma_z);
    filter_z = filter_z(:,5);
    filter_z = filter_z / sum(abs(filter_z));
% Random filters
elseif smoothing_type == 1
    filter_x = randn(9,1);
    filter_y = randn(9,1);
    filter_z = randn(9,1);
    
    % Normalize
    filter_x = filter_x / sum(abs(filter_x));
    filter_y = filter_y / sum(abs(filter_y));
    filter_z = filter_z / sum(abs(filter_z));
end

% Put 1D filters into 3D arrays
temp = zeros(1,9,1);
temp(1,:,1) = filter_x;
filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = filter_y;
filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = filter_z;
filter_zz = temp;

% Do separable smoothing in Matlab, x, y, z
smoothed_volumes_cpu = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
   volume = fMRI_volumes(:,:,:,t);
   smoothed_volume = convn(volume,filter_xx,'same');
   smoothed_volume = convn(smoothed_volume,filter_yy,'same');   
   smoothed_volume = convn(smoothed_volume,filter_zz,'same');
   smoothed_volumes_cpu(:,:,:,t) = smoothed_volume;
end

% Do smoothing with OpenCL
tic
smoothed_volumes_opencl = SmoothingMex(fMRI_volumes,filter_x,filter_y,filter_z,voxel_size_x,voxel_size_y,voxel_size_z,smoothing_FWHM,smoothing_type,opencl_platform,opencl_device);
toc

% Compare smoothing results, as plots
figure
plot(squeeze(smoothed_volumes_cpu(25,25,15,:)),'r')
hold on
plot(squeeze(smoothed_volumes_opencl(25,25,15,:)),'b')
hold off
legend('Matlab','BROCCOLI')

% Compare smoothing results, as images
slice = 25;
figure
imagesc([fMRI_volumes(:,:,slice,1)]); colormap gray
title('Original')
figure
imagesc([smoothed_volumes_cpu(:,:,slice,1)  smoothed_volumes_opencl(:,:,slice,1)]); colormap gray
title('Matlab smoothed                             BROCCOLI smoothed')

% Calculate error between Matlab and OpenCL
total_error = sum(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))
max_error = max(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))
mean_error = mean(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))



