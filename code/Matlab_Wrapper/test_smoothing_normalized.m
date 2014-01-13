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
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\OpenfMRI\';
    
    %mex -g SmoothingNormalized.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex SmoothingNormalized.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    
    opencl_platform = 0;
    opencl_device = 0;
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';
    
    %mex -g SmoothingNormalized.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex SmoothingNormalized.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    
    opencl_platform = 2; % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 0;
end

study = 'RhymeJudgment/ds003';

subject = 1;

if subject < 10
    subject = ['/sub00' num2str(subject)];
else
    subject = ['/sub0' num2str(subject)];
end

%-----------------------------------------------------------------------
% Settings
%-----------------------------------------------------------------------

% Set smoothing amount
smoothing_FWHM = 8.0;
smoothing_type = 1; % 0 = Gaussian smoothing, 1 = Random smoothing

%-----------------------------------------------------------------------
% Load data
%-----------------------------------------------------------------------

EPI_nii = load_nii([basepath study subject '/BOLD/task001_run001/bold.nii.gz']);

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
TR = 2;

fMRI_volumes = double(EPI_nii.img);
[sy sx sz st] = size(fMRI_volumes);
[sy sx sz st]

%-----------------------------------------------------------------------
% Create mask
%-----------------------------------------------------------------------

% Smooth with 4 mm filter
sigma_x = 4 / 2.354 / EPI_voxel_size_x;
sigma_y = 4 / 2.354 / EPI_voxel_size_y;
sigma_z = 4 / 2.354 / EPI_voxel_size_z;

smoothing_filter_x = fspecial('gaussian',9,sigma_x);
smoothing_filter_x = smoothing_filter_x(:,5);
smoothing_filter_x = smoothing_filter_x / sum(abs(smoothing_filter_x));

smoothing_filter_y = fspecial('gaussian',9,sigma_y);
smoothing_filter_y = smoothing_filter_y(:,5);
smoothing_filter_y = smoothing_filter_y / sum(abs(smoothing_filter_y));

smoothing_filter_z = fspecial('gaussian',9,sigma_z);
smoothing_filter_z = smoothing_filter_z(:,5);
smoothing_filter_z = smoothing_filter_z / sum(abs(smoothing_filter_z));

temp = zeros(1,9,1);
temp(1,:,1) = smoothing_filter_x;
filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
filter_zz = temp;

volume = fMRI_volumes(:,:,:,1);
smoothed_volume = convn(volume,filter_xx,'same');
smoothed_volume = convn(smoothed_volume,filter_yy,'same');
smoothed_volume = convn(smoothed_volume,filter_zz,'same');

threshold = 0.9 * mean(smoothed_volume(:));
brain_mask = double(volume > threshold);
brain_mask(brain_mask == 0) = 0.01;

%-----------------------------------------------------------------------
% Create smoothing filters for Matlab
%-----------------------------------------------------------------------

% Gaussian lowpass filters
if smoothing_type == 0
    % Convert smoothing amount in FWHM to sigma
    sigma_x = smoothing_FWHM / 2.354 / EPI_voxel_size_x;
    sigma_y = smoothing_FWHM / 2.354 / EPI_voxel_size_y;
    sigma_z = smoothing_FWHM / 2.354 / EPI_voxel_size_z;

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

% Smooth mask
volume = brain_mask;
smoothed_volume = convn(volume,filter_xx,'same');
smoothed_volume = convn(smoothed_volume,filter_yy,'same');
smoothed_volume = convn(smoothed_volume,filter_zz,'same');
smoothed_brain_mask = smoothed_volume;

% Do separable smoothing in Matlab, x, y, z
smoothed_volumes_cpu_unnormalized = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
   volume = fMRI_volumes(:,:,:,t);
   smoothed_volume = convn(volume,filter_xx,'same');
   smoothed_volume = convn(smoothed_volume,filter_yy,'same');   
   smoothed_volume = convn(smoothed_volume,filter_zz,'same');
   smoothed_volumes_cpu_unnormalized(:,:,:,t) = smoothed_volume .* brain_mask;
end

% Do normalized separable smoothing in Matlab, x, y, z
smoothed_volumes_cpu = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
   volume = fMRI_volumes(:,:,:,t) .* brain_mask;
   smoothed_volume = convn(volume,filter_xx,'same');
   smoothed_volume = convn(smoothed_volume,filter_yy,'same');   
   smoothed_volume = convn(smoothed_volume,filter_zz,'same');
   smoothed_volumes_cpu(:,:,:,t) = smoothed_volume ./ smoothed_brain_mask .* brain_mask;
end


% Do smoothing with OpenCL
tic
smoothed_volumes_opencl = SmoothingNormalized(fMRI_volumes,brain_mask,smoothed_brain_mask,filter_x,filter_y,filter_z,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,smoothing_FWHM,smoothing_type,opencl_platform,opencl_device);
toc

% Compare smoothing results, as plots
figure
plot(squeeze(smoothed_volumes_cpu(25,25,15,:)),'r')
hold on
plot(squeeze(smoothed_volumes_opencl(25,25,15,:)),'b')
hold off
legend('Matlab','BROCCOLI')

% Compare smoothing results, as images
slice = 20;
figure
imagesc([fMRI_volumes(:,:,slice,1)]); colormap gray
title('Original')
figure
imagesc([smoothed_volumes_cpu(:,:,slice,1) smoothed_volumes_opencl(:,:,slice,1)]); colormap gray; colorbar
title('Matlab smoothed                             BROCCOLI smoothed')

% Calculate error between Matlab and OpenCL
total_error = sum(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))
max_error = max(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))
mean_error = mean(abs(smoothed_volumes_cpu(:) - smoothed_volumes_opencl(:)))



