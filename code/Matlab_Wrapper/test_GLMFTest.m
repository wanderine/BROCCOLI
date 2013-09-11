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

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\BROCCOLI_test_data\';
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab/')
    basepath = '/data/andek/BROCCOLI_test_data/';
end

if ispc
    %mex -g GLMFTest.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  
    mex GLMFTest.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib      
    opencl_platform = 0;
    opencl_device = 0;
elseif isunix
    %mex -g GLMFTest.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug 
    mex GLMFTest.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release    
    opencl_platform = 2;
    opencl_device = 0;
end



%study = 'Oulu';
%study = 'ICBM';
study = 'Cambridge';
%study = 'Beijing';
%study = 'OpenfMRI';
substudy = 'Mixed';
subject = 2;


dirs = dir([basepath study]);
s = subject + 2;
subject = dirs(s).name
    

EPI_smoothing_amount = 5.5;
AR_smoothing_amount = 7.0;

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
    %EPI_nii = load_nii([basepath study '/rest' num2str(subject) '.nii.gz']);
    EPI_nii = load_nii([basepath study '/' subject '/func/rest'  '.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(subject) '.nii.gz']);
end

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

fMRI_volumes = double(EPI_nii.img);
[sy sx sz st] = size(fMRI_volumes);
[sy sx sz st]


% Create regressors
[sy sx sz st] = size(fMRI_volumes)
mask = randn(sy,sx,sz);

X_GLM = zeros(st,3);
X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+10),1) =   0;  % Activity
    X_GLM_((NN+11):(NN+20),1) =  1;  % Rest
    NN = NN + 20;
end
X_GLM(:,1) = X_GLM_(1:st) - mean(X_GLM_(1:st));
X_GLM(:,2) = ones(st,1);
X_GLM(:,3) = randn(st,1);

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

% Create contrasts
%contrasts = zeros(size(X_GLM,2),3);
%contrasts = [1 0; 0 -1];
contrasts = [0 0 -3; 1 0 2];
contrasts = [1 3 0; 0 1 -2; -1 0 1];
contrasts = [1 3 0; 0 1 -2];


ctxtxc_GLM = inv(contrasts*inv(X_GLM'*X_GLM)*contrasts')

statistical_maps_cpu = zeros(sy,sx,sz);
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
residuals_cpu = zeros(sy,sx,sz,st);
residual_variances_cpu = zeros(sy,sx,sz);


for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            timeseries = squeeze(fMRI_volumes(y,x,z,:));
            %timeseries = timeseries - mean(timeseries);
            %fMRI_volumes(y,x,z,:) = timeseries;
            beta = xtxxt_GLM*timeseries;
            betas_cpu(y,x,z,:) = beta;
            eps = timeseries - X_GLM*beta;
            residuals_cpu(y,x,z,:) = eps;
            residual_variances_cpu(y,x,z) = sum((eps-mean(eps)).^2)/(st-size(X_GLM,2) - 1);
            %residual_variances_cpu(y,x,z) = var(eps);
            
            %F-test
            statistical_maps_cpu(y,x,z) = (contrasts*beta)' * 1/residual_variances_cpu(y,x,z) * ctxtxc_GLM * (contrasts*beta) / size(contrasts,1);
            
        end
    end
end


tic
[betas_opencl, residuals_opencl, residual_variances_opencl, statistical_maps_opencl, ...
 ar1_estimates_opencl, ar2_estimates_opencl, ar3_estimates_opencl, ar4_estimates_opencl, ...
 design_matrix, design_matrix2] = ...
GLMFTest(fMRI_volumes,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,EPI_smoothing_amount,AR_smoothing_amount,... 
    EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,opencl_platform,opencl_device);
toc

slice = 30;

figure
imagesc([betas_cpu(:,:,slice,1) betas_opencl(:,:,slice,1)]); colorbar
title('Beta')

figure
imagesc([residual_variances_cpu(:,:,slice) residual_variances_opencl(:,:,slice)]); colorbar
title('Residual variances')

figure
%imagesc([statistical_maps_cpu(:,:,slice,1) statistical_maps_opencl(:,:,slice,1)]); colorbar
%imagesc([statistical_maps_cpu(:,:,slice,1)/119/2 statistical_maps_opencl(:,:,slice,1)*1000]); colorbar
imagesc([statistical_maps_cpu(:,:,slice,1) statistical_maps_opencl(:,:,slice,1)]); colorbar
title('Statistical map')

%figure
%imagesc([ar1_estimates_opencl(:,:,slice) ]); colorbar

%figure
%imagesc([ar2_estimates_opencl(:,:,slice) ]); colorbar

%figure
%imagesc([ar3_estimates_opencl(:,:,slice) ]); colorbar

%figure
%imagesc([ar4_estimates_opencl(:,:,slice) ]); colorbar

beta_tot_error = sum(abs(betas_cpu(:) - betas_opencl(:)))
beta_max_error = max(abs(betas_cpu(:) - betas_opencl(:)))

%for slice = 1:sz
%    slice
%    a = betas_cpu(:,:,slice,1);
%    b = betas_opencl(:,:,slice,1);
%    max(a(:) - b(:))
%end

residual_tot_error = sum(abs(residuals_cpu(:) - residuals_opencl(:)))
residual_max_error = max(abs(residuals_cpu(:) - residuals_opencl(:)))

residual_variances_tot_error = sum(abs(residual_variances_cpu(:) - residual_variances_opencl(:)))
residual_variances_max_error = max(abs(residual_variances_cpu(:) - residual_variances_opencl(:)))

stat_tot_error = sum(abs(statistical_maps_cpu(:) - statistical_maps_opencl(:)))
stat_max_error = max(abs(statistical_maps_cpu(:) - statistical_maps_opencl(:)))

% for slice = 1:sz
%     slice
%     a = statistical_map_cpu(:,:,slice,1);
%     b = statistical_map_opencl(:,:,slice,1);
%     max(a(:) - b(:))
% end


