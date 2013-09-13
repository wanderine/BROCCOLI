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
    %basepath = 'D:\BROCCOLI_test_data\';
    basepath = 'D:\';
    %mex -g FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
    mex FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    
    
    opencl_platform = 0;
    opencl_device = 0;
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';    
    %mex -g FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug 
    mex FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release
    
    opencl_platform = 2;
    opencl_device = 0;
end

%study = 'Oulu';
%study = 'ICBM';
%study = 'Cambridge';
study = 'Beijing';
%study = 'OpenfMRI';
%substudy = 'Mixed';

subject = 3;
dirs = dir([basepath study]);
subject = dirs(subject+2).name
voxel_size = 1;
beta_space = 1; % 0 = EPI, 1 = MNI

% Statistical settings
USE_TEMPORAL_DERIVATIVES = 1;
REGRESS_MOTION = 0;
REGRESS_CONFOUNDS = 0;
confounds = 1;

% Settings for image registration
number_of_iterations_for_parametric_image_registration = 5;
number_of_iterations_for_nonparametric_image_registration = 5;
number_of_iterations_for_motion_correction = 3;
coarsest_scale_T1_MNI = 8/voxel_size;
coarsest_scale_EPI_T1 = 8/voxel_size;
MM_T1_Z_CUT = 30;
MM_EPI_Z_CUT = 20;

EPI_smoothing_amount = 0.5;
AR_smoothing_amount = 7.0;

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu'))  )
    %T1_nii = load_nii([basepath study '\mprage_anonymized' num2str(subject) '.nii.gz']);
    T1_nii = load_nii([basepath study '/' subject '/anat/mprage_skullstripped.nii.gz']);
    %T1_nii = load_nii([basepath study '/' subject '/anat/mprage_anonymized.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    T1_nii = load_nii([basepath study '\' substudy '\highres' num2str(subject) '.nii.gz']);
end

T1 = double(T1_nii.img);
T1 = T1/max(T1(:));

MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));

MNI_brain_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain.nii']);
MNI_brain = double(MNI_brain_nii.img);
MNI_brain = MNI_brain/max(MNI_brain(:));

MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
    %EPI_nii = load_nii([basepath study '/rest' num2str(subject) '.nii.gz']);
    EPI_nii = load_nii([basepath study '/' subject '/func/rest.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(subject) '.nii.gz']);
end

fMRI_volumes = double(EPI_nii.img);
%fMRI_volumes = fMRI_volumes/max(fMRI_volumes(:));
[sy sx sz st] = size(fMRI_volumes);

EPI = fMRI_volumes(:,:,:,1);
EPI = EPI/max(EPI(:));

[T1_sy T1_sx T1_sz] = size(T1);
[T1_sy T1_sx T1_sz]
[MNI_sy MNI_sx MNI_sz] = size(MNI);
[MNI_sy MNI_sx MNI_sz]
[EPI_sy EPI_sx EPI_sz] = size(EPI);
[EPI_sy EPI_sx EPI_sz]

if (strcmp(study,'Beijing'))
    T1_voxel_size_x = T1_nii.hdr.dime.pixdim(1);
    T1_voxel_size_y = T1_nii.hdr.dime.pixdim(2);
    T1_voxel_size_z = T1_nii.hdr.dime.pixdim(3);
elseif (strcmp(study,'OpenfMRI'))
    T1_voxel_size_x = T1_nii.hdr.dime.pixdim(3);
    T1_voxel_size_y = T1_nii.hdr.dime.pixdim(2);
    T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);
else    
    T1_voxel_size_x = T1_nii.hdr.dime.pixdim(2);
    T1_voxel_size_y = T1_nii.hdr.dime.pixdim(3);
    T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);
end

MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

%%

load filters_for_parametric_registration.mat
load filters_for_nonparametric_registration.mat

%%
% Create regressors
[sy sx sz st] = size(fMRI_volumes);
mask = randn(sy,sx,sz);

X_GLM_ = zeros(st,3);
X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+10),1) =   0;  % Activity
    X_GLM_((NN+11):(NN+20),1) =  1;  % Rest
    NN = NN + 20;
end
a = X_GLM_(1:st) - mean(X_GLM_(1:st));
X_GLM(:,1) = a/norm(a(:));


X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+5),1) =   0;  % Activity
    X_GLM_((NN+6):(NN+10),1) =  1;  % Rest
    NN = NN + 10;
end
a = X_GLM_(1:st) - mean(X_GLM_(1:st));
X_GLM(:,2) = a/norm(a(:));

X_GLM_ = zeros(st,1);
NN = 0;
while NN < st
    X_GLM_((NN+1):(NN+15),1) =   0;  % Activity
    X_GLM_((NN+16):(NN+30),1) =  1;  % Rest
    NN = NN + 30;
end
a = X_GLM_(1:st) - mean(X_GLM_(1:st));
X_GLM(:,3) = a/norm(a(:));



xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';

% Create contrasts

contrasts = [1 0 0];

for i = 1:size(contrasts,1)
    contrast = contrasts(i,:)';
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

%%

tic
[beta_volumes, residuals, residual_variances, statistical_maps, T1_MNI_registration_parameters, EPI_T1_registration_parameters, ...
 EPI_MNI_registration_parameters, motion_parameters, motion_corrected_volumes_opencl, smoothed_volumes_opencl ...
 ar1_estimates, ar2_estimates, ar3_estimates, ar4_estimates, design_matrix1, design_matrix2, aligned_t1, aligned_t1_nonparametric,aligned_epi] = ... 
FirstLevelAnalysis(fMRI_volumes,T1,MNI,MNI_brain,MNI_brain_mask,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z, ...
f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
f1_nonparametric_registration, f2_nonparametric_registration, f3_nonparametric_registration, f4_nonparametric_registration, f5_nonparametric_registration, f6_nonparametric_registration, ...
m1, m2, m3, m4, m5, m6, ...
filter_directions_x, filter_directions_y, filter_directions_z, ...
number_of_iterations_for_parametric_image_registration, number_of_iterations_for_nonparametric_image_registration, ...
coarsest_scale_T1_MNI, coarsest_scale_EPI_T1,MM_T1_Z_CUT,MM_EPI_Z_CUT,number_of_iterations_for_motion_correction,REGRESS_MOTION,EPI_smoothing_amount,AR_smoothing_amount, ...
X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,USE_TEMPORAL_DERIVATIVES,beta_space,confounds,REGRESS_CONFOUNDS,opencl_platform,opencl_device);
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
title('Translation (mm)')
legend('X','Y','Z')

figure
plot(motion_parameters(:,4),'g')
hold on
plot(motion_parameters(:,5),'r')
hold on
plot(motion_parameters(:,6),'b')
hold off
title('Rotation (degrees)')
legend('X','Y','Z')

if beta_space == 1
    slice = round(0.5*MNI_sz);
else
   slice = round(EPI_sz/2); 
end


%figure
%imagesc(motion_corrected_volumes_opencl(:,:,20,10)); colorbar


figure
imagesc(smoothed_volumes_opencl(:,:,20,10)); colorbar


figure
imagesc(MNI(:,:,slice)); colormap gray

figure
imagesc(beta_volumes(:,:,slice,2)); colormap gray; 
title('Beta 2')

figure
%imagesc(statistical_maps(20:end-19,20:end-19,slice,1)); colorbar
imagesc(statistical_maps(10:end-10,10:end-10,slice,1)); colorbar
title('t-values')

if beta_space == 1
    slice = round(0.45*MNI_sz);
else
   slice = round(EPI_sz/2); 
end

figure
imagesc(flipud(squeeze(MNI(slice,:,:))')); colormap gray

figure
imagesc(flipud(squeeze(beta_volumes(slice,:,:,2))')); colormap gray
title('Beta 2')

figure
imagesc(flipud(squeeze(statistical_maps(slice,:,:,1))')); colorbar
title('t-values')

% 
% %figure
% %imagesc(beta_volumes(:,:,slice,1)); colorbar
% %title('Beta 1')
% 

%figure
%imagesc(beta_volumes(:,:,slice,3)); colorbar
%title('Beta 3')

figure
imagesc(residual_variances(:,:,slice)); colorbar
title('Residual variances')


 
figure
imagesc(ar1_estimates(:,:,32)); colorbar

figure
imagesc(flipud(squeeze(ar1_estimates(35,:,:,1))')); colorbar

%figure
%imagesc(ar2_estimates(:,:,30)); colorbar
 
%figure
%imagesc(ar3_estimates(:,:,30)); colorbar
 
%figure
%imagesc(ar4_estimates(:,:,30)); colorbar

%title('Residual timeserie')
 
% 
% figure
% imagesc(residuals(:,:,30)); colorbar
% title('Residuals')
% 
% 
% means = mean(residuals,4);
% figure
% imagesc(means(:,:,30)); colorbar
% title('Means of residuals')
% 
% means = squeeze(mean(mean(mean(residuals))));
% figure
% plot(squeeze(abs(fftshift(fft(means))))); colorbar
% title('Spectra of mean residual')
% 
% figure
% plot(squeeze(means)); colorbar
% title('Mean residual timeseries')

%%


