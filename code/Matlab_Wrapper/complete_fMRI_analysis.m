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

mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
mex -g RegisterEPIT1.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

load filters.mat

subject = 22;
voxel_size = 1;

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

EPI = fMRI_volumes(:,:,:,1);
EPI = EPI/max(EPI(:));

[sy sx sz] = size(T1)


opencl_platform = 0;

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
number_of_iterations_for_image_registration = 30;
coarsest_scale = 8/voxel_size;
MM_T1_Z_CUT = 10;

tic
[aligned_T1_opencl, skullstripped_T1_opencl, interpolated_T1_opencl, registration_parameters_T1_MNI, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl, downsampled_volume_opencl] = ... 
RegisterT1MNI(T1,MNI,MNI_brain_mask,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_T1_Z_CUT,opencl_platform);
toc

%%
number_of_iterations_for_image_registration = 60;
coarsest_scale = 4/voxel_size;
MM_EPI_Z_CUT = 20;

tic
[aligned_EPI_opencl, interpolated_EPI_opencl, registration_parameters_EPI_T1, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ... 
RegisterEPIT1(EPI,skullstripped_T1_opencl,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_EPI_Z_CUT,opencl_platform);
toc

%%
mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  
number_of_iterations_for_motion_correction = 10;
[motion_corrected_fMRI_volumes, motion_parameters, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = MotionCorrection(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction,opencl_platform);

figure
plot(motion_parameters(:,1),'g')
hold on
plot(motion_parameters(:,2),'r')
hold on
plot(motion_parameters(:,3),'b')
hold off
title('Translation')

figure
plot(motion_parameters(:,4),'g')
hold on
plot(motion_parameters(:,5),'r')
hold on
plot(motion_parameters(:,6),'b')
hold off
title('Rotation')

%%
filter_x = fspecial('gaussian',9,1);
filter_x = filter_x(:,5);
filter_x = filter_x / sum(abs(filter_x));
filter_y = filter_x;
filter_z = filter_x;

temp = zeros(1,9,1);
temp(1,:,1) = filter_x;
filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = filter_y;
filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = filter_z;
filter_zz = temp;

mex -g Smoothing.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  
tic
smoothed_fMRI_volumes = Smoothing(motion_corrected_fMRI_volumes,filter_x,filter_y,filter_z,opencl_platform);
toc
%%
% Create regressors
[sy sx sz st] = size(fMRI_volumes)
mask = randn(sy,sx,sz);

X_GLM_ = zeros(st,8);
X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+10),1) =   1;  % Activity
    X_GLM_((NN+11):(NN+20),1) =  0;  % Rest
    NN = NN + 20;
end
X_GLM(:,1) = X_GLM_(1:st);

% Add motion regressors
X_GLM(:,2) = motion_parameters(:,1);
X_GLM(:,3) = motion_parameters(:,2);
X_GLM(:,4) = motion_parameters(:,3);
X_GLM(:,5) = motion_parameters(:,4);
X_GLM(:,6) = motion_parameters(:,5);
X_GLM(:,7) = motion_parameters(:,6);
X_GLM(:,8) = ones(st,1);

xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

% Create contrasts
contrasts = zeros(size(X_GLM,2),3);
contrasts(:,1) = [1 0 0 0 0 0 0 0]';
contrasts(:,2) = [0 1 0 0 0 0 0 0]';
contrasts(:,3) = [0 0 0 0 1 0 0 0]';
for i = 1:size(contrasts,2)
    contrast = contrasts(:,i);
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

mex -g GLM.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib  
tic
%[betas, residuals, residual_variances, statistical_maps] = GLM(smoothed_fMRI_volumes,mask,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,registration_parameters_T1_MNI,registration_parameters_EPI_T1,MNI_DATA_W,MNI_DATA_H,MNI_DATA_D,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,MM_EPI_Z_CUT,opencl_platform);
[betas, residuals, residual_variances, statistical_maps] = GLM(smoothed_fMRI_volumes,mask,X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,opencl_platform);
toc


slice = 30;
figure; imagesc(statistical_maps(:,:,slice,1)); colorbar
figure; imagesc(statistical_maps(:,:,slice,2)); colorbar
figure; imagesc(statistical_maps(:,:,slice,3)); colorbar

%%


