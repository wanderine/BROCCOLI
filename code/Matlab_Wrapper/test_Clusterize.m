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

% Make a mex-file
if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\OpenfMRI\';
    opencl_platform = 0;
    opencl_device = 0;
    
    %mex -g Clusterize.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex Clusterize.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab/')
    basepath = '/data/andek/OpenfMRI/';
    opencl_platform = 2;
    opencl_device = 0;
    
    %mex -g Clusterize.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex Clusterize.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
end


%-----------------------------------------------------------------------
% Settings
%-----------------------------------------------------------------------

voxel_size = 2;
smoothing_amount = 5.5;
cluster_defining_threshold = 0.5;

%--------------------------------------------------------------------------------------
% Load MNI templates
%--------------------------------------------------------------------------------------

MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

MNI_brain_mask = MNI_brain_mask(1:1:end,1:1:end,1:1:end);

MNI_brain_mask(:,:,1) = 0;
MNI_brain_mask(:,:,end) = 0;

[MNI_sy MNI_sx MNI_sz] = size(MNI_brain_mask);
[MNI_sy MNI_sx MNI_sz]

%-----------------------------------------------------------------------
% Create smoothing filters for data
%-----------------------------------------------------------------------

sigma_x = smoothing_amount / 2.354 / voxel_size;
sigma_y = smoothing_amount / 2.354 / voxel_size;
sigma_z = smoothing_amount / 2.354 / voxel_size;

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
smoothing_filter_data_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_data_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_data_zz = temp;

%-----------------------------------------------------------------------
% Smooth data
%-----------------------------------------------------------------------

data = 10*randn(MNI_sy,MNI_sx,MNI_sz);

smoothed_volume = convn(data,smoothing_filter_data_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_data_yy,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_data_zz,'same');
data = smoothed_volume .* MNI_brain_mask;

load randomdata.mat

%tic
%[cluster_indices,largest_cluster] = Clusterize(data,MNI_brain_mask,cluster_defining_threshold,opencl_platform,opencl_device,1);
%toc

tic
[cluster_indices,largest_cluster] = Clusterize(data,MNI_brain_mask,cluster_defining_threshold,opencl_platform,opencl_device,2);
toc

tic
[cluster_indices,largest_cluster] = Clusterize(data,MNI_brain_mask,cluster_defining_threshold,opencl_platform,opencl_device,3);
toc


[labels,N] = bwlabeln(data > cluster_defining_threshold);
cluster_extents = zeros(N,1);
for i = 1:N
    cluster_extents(i) = sum(labels(:) == i);
end
N

% cluster_indices(cluster_indices == MNI_sx * MNI_sy * MNI_sz * 3) = 0;
% 
% j = 1;
% cluster_sizes = 0;
% for i = 1:max(cluster_indices(:))
%     a = sum(cluster_indices(:) == i);
%     if a > 0
%         i;
%         cluster_sizes(j) = a;        
%         j = j + 1;
%     end
% end
% 
% [sort(cluster_sizes') sort(cluster_extents)]


