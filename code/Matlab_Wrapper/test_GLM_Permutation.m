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

addpath('D:\nifti_matlab')
addpath('D:\BROCCOLI_test_data')

mex GLM_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  
%mex -g GLM_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  

basepath = 'D:\BROCCOLI_test_data\';
%study = 'Oulu';
%study = 'ICBM';
study = 'Cambridge';
%study = 'Beijing';
%study = 'OpenfMRI';
substudy = 'Mixed';
subject = 1;
opencl_platform = 0;
opencl_device = 0;

NUMBER_OF_PERMUTATIONS = 1000;
EPI_smoothing_amount = 5.5;
AR_smoothing_amount = 7.0;

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
    EPI_nii = load_nii([basepath study '/rest' num2str(subject) '.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(subject) '.nii.gz']);
end

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

fMRI_volumes = double(EPI_nii.img);

%fMRI_volumes = randn(91,109,91,40);
[sy sx sz st] = size(fMRI_volumes);
[sy sx sz st]

%MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(2) 'mm_brain_mask.nii']);
%MNI_brain_mask = double(MNI_brain_mask_nii.img);

%for t = 1:st
%    fMRI_volumes(:,:,:,t) = fMRI_volumes(:,:,:,t) .* MNI_brain_mask;
%end

% Create regressors
[sy sx sz st] = size(fMRI_volumes)
mask = randn(sy,sx,sz);

X_GLM_ = zeros(st,5);
X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+10),1) =   0;  % Activity
    X_GLM_((NN+11):(NN+20),1) =  1;  % Rest
    NN = NN + 20;
end
X_GLM(:,1) = X_GLM_(1:st) - mean(X_GLM_(1:st));
a = ones(st,1)/st;
X_GLM(:,2) = a/norm(a(:));
a = -(st-1)/2:(st-1)/2;
b = a.*a;
c = a.*a.*a;
X_GLM(:,3) = a/norm(a(:));
X_GLM(:,4) = b/norm(b(:));
X_GLM(:,5) = c/norm(c(:));

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

% Create contrasts
%contrasts = zeros(size(X_GLM,2),3);
contrasts = [1 0 0 0 0]';
%contrasts = [1 0]';
%contrasts(:,1) = [1 0 0 0 0 0 0 0]';
%contrasts(:,2) = [0 1 0 0 0 0 0 0]';
%contrasts(:,3) = [0 0 0 0 1 0 0 0]';
for i = 1:size(contrasts,2)
    contrast = contrasts(:,i);
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

tic
[betas_opencl, residuals_opencl, residual_variances_opencl, statistical_maps_opencl, ar1_estimates_opencl, ar2_estimates_opencl, ar3_estimates_opencl, ar4_estimates_opencl, permutation_distribution, detrended_fMRI_volumes_opencl, whitened_fMRI_volumes_opencl, permuted_fMRI_volumes_opencl] = ... 
    GLM_Permutation(fMRI_volumes,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,EPI_smoothing_amount,AR_smoothing_amount,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,NUMBER_OF_PERMUTATIONS,opencl_platform,opencl_device);
toc

hist(permutation_distribution,50)



