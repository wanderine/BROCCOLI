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

%mex FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
mex -g FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

subject = 29;
voxel_size = 1;
opencl_platform = 0;

T1_nii = load_nii(['mprage_anonymized' num2str(subject) '.nii.gz']);
T1 = double(T1_nii.img);
T1 = T1/max(T1(:));
MNI_nii = load_nii(['../../test_data/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));
MNI_brain_mask_nii = load_nii(['../../test_data/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));
EPI_nii = load_nii(['rest' num2str(subject) '.nii.gz']);
fMRI_volumes = double(EPI_nii.img);
%fMRI_volumes = fMRI_volumes(:,:,1:22,:);
fMRI_volumes = fMRI_volumes(:,:,:,5:end);
%fMRI_volumes = fMRI_volumes/max(fMRI_volumes(:));

EPI = fMRI_volumes(:,:,:,1);
EPI = EPI/max(EPI(:));

[sy sx sz] = size(T1)

T1_voxel_size_x = T1_nii.hdr.dime.pixdim(2);
T1_voxel_size_y = T1_nii.hdr.dime.pixdim(3);
T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);

MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

%%
% Settings for image registration
number_of_iterations_for_image_registration = 30;
number_of_iterations_for_motion_correction = 3;
coarsest_scale_T1_MNI = 8/voxel_size;
coarsest_scale_EPI_T1 = 4/voxel_size;
MM_T1_Z_CUT = 10/voxel_size;
MM_EPI_Z_CUT = 20/voxel_size;
load filters.mat

%%
% Create smoothing filters
smoothing_filter_x = fspecial('gaussian',9,0.3);
smoothing_filter_x = smoothing_filter_x(:,5);
smoothing_filter_x = smoothing_filter_x / sum(abs(smoothing_filter_x));
smoothing_filter_y = smoothing_filter_x;
smoothing_filter_z = smoothing_filter_x;

%%
% Create regressors
[sy sx sz st] = size(fMRI_volumes)
mask = randn(sy,sx,sz);

X_GLM_ = zeros(st,5);
X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+10),1) =   1;  % Activity
    X_GLM_((NN+11):(NN+20),1) =  0;  % Rest
    NN = NN + 20;
end
a = X_GLM_(1:st) - mean(X_GLM_(1:st));
X_GLM(:,1) = a/norm(a(:));
my_ones = ones(st,1);
X_GLM(:,2) = my_ones/norm(my_ones);
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
%contrasts(:,1) = [1 0 0 0 0 0 0 0]';
%contrasts(:,2) = [0 1 0 0 0 0 0 0]';
%contrasts(:,3) = [0 0 0 0 1 0 0 0]';
for i = 1:size(contrasts,2)
    contrast = contrasts(:,i);
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

%%

tic
[beta_volumes, residuals, residual_variances, statistical_maps, T1_MNI_registration_parameters, EPI_T1_registration_parameters, ...
 EPI_MNI_registration_parameters, motion_parameters, motion_corrected_volumes_opencl, smoothed_volumes_opencl ...
 ar1_estimates, ar2_estimates, ar3_estimates, ar4_estimates] = ... 
FirstLevelAnalysis(fMRI_volumes,T1,MNI,MNI_brain_mask,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,T1_voxel_size_x,T1_voxel_size_y, ... 
T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale_T1_MNI, ...
coarsest_scale_EPI_T1,MM_T1_Z_CUT,MM_EPI_Z_CUT,number_of_iterations_for_motion_correction,smoothing_filter_x,smoothing_filter_y,smoothing_filter_z, ...
X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,opencl_platform);
toc

T1_MNI_registration_parameters

EPI_T1_registration_parameters

EPI_MNI_registration_parameters

figure
plot(motion_parameters(:,1),'g')
hold on
plot(motion_parameters(:,2),'r')
hold on
plot(motion_parameters(:,3),'b')
hold off
title('Translation')
legend('X','Y','Z')

figure
plot(motion_parameters(:,4),'g')
hold on
plot(motion_parameters(:,5),'r')
hold on
plot(motion_parameters(:,6),'b')
hold off
title('Rotation')
legend('X','Y','Z')


slice = 100/voxel_size;

%figure
%imagesc(motion_corrected_volumes_opencl(:,:,20,10)); colorbar


%figure
%imagesc(smoothed_volumes_opencl(:,:,20,10)); colorbar


figure
imagesc(MNI(:,:,slice)); colorbar

%figure
%imagesc(beta_volumes(:,:,slice,1)); colorbar
%title('Beta 1')

figure
imagesc(beta_volumes(:,:,slice,2)); colorbar
title('Beta 2')

%figure
%imagesc(beta_volumes(:,:,slice,3)); colorbar
%title('Beta 3')

%figure
%imagesc(residual_variances(:,:,slice)); colorbar
%title('Residual variances')

figure
imagesc(statistical_maps(:,:,slice,1)); colorbar
title('t-values')

figure
imagesc(ar1_estimates(:,:,30)); colorbar

figure
imagesc(ar2_estimates(:,:,30)); colorbar

figure
imagesc(ar3_estimates(:,:,30)); colorbar

figure
imagesc(ar4_estimates(:,:,30)); colorbar
title('Residual timeserie')


figure
imagesc(residuals(:,:,30)); colorbar
title('Residuals')


means = mean(residuals,4);
figure
imagesc(means(:,:,30)); colorbar
title('Means of residuals')

means = squeeze(mean(mean(mean(residuals))));
figure
plot(squeeze(abs(fftshift(fft(means))))); colorbar
title('Spectra of mean residual')

figure
plot(squeeze(means)); colorbar
title('Mean residual timeseries')

%%


