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

mex -g FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

basepath = 'D:\BROCCOLI_test_data\';
%study = 'OpenfMRI';
%study = 'Beijing';
study = 'Cambridge';
substudy = 'Mixed'
subject = 9;
voxel_size = 2;
beta_space = 1; % 0 = EPI, 1 = MNI
opencl_platform = 0;
opencl_device = 0;


EPI_smoothing_amount = 6.0;
AR_smoothing_amount = 7.0;

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu'))  )
    T1_nii = load_nii([basepath study '\mprage_anonymized' num2str(subject) '.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    T1_nii = load_nii([basepath study '\' substudy '\highres' num2str(subject) '.nii.gz']);
end

T1 = double(T1_nii.img);
T1 = T1/max(T1(:));
MNI_nii = load_nii(['../../test_data/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
MNI = double(MNI_nii.img);
MNI = MNI/max(MNI(:));
MNI_brain_mask_nii = load_nii(['../../test_data/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
    EPI_nii = load_nii([basepath study '/rest' num2str(subject) '.nii.gz']);
elseif ( strcmp(study,'OpenfMRI'))
    EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(subject) '.nii.gz']);
end

fMRI_volumes = double(EPI_nii.img);
%fMRI_volumes = fMRI_volumes(:,:,1:22,:);
%fMRI_volumes = fMRI_volumes(:,:,:,5:end);
[sy sx sz st] = size(fMRI_volumes)
%fMRI_volumes = fMRI_volumes/max(fMRI_volumes(:));
means = zeros(size(fMRI_volumes));
mean_volume = mean(fMRI_volumes,4);
for t = 1:st
    means(:,:,:,t) = mean_volume;
end
fMRI_volumes = fMRI_volumes - means/1.25;

EPI = fMRI_volumes(:,:,:,1);
EPI = EPI/max(EPI(:));

[T1_sy T1_sx T1_sz] = size(T1)
[MNI_sy MNI_sx MNI_sz] = size(MNI)
[EPI_sy EPI_sx EPI_sz] = size(EPI)

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
number_of_iterations_for_image_registration = 50;
number_of_iterations_for_motion_correction = 3;
coarsest_scale_T1_MNI = 8/voxel_size;
coarsest_scale_EPI_T1 = 4/voxel_size;
MM_T1_Z_CUT = 10;
MM_EPI_Z_CUT = 20;
load filters.mat

[motion_corrected_volumes_cpu,motion_parameters_cpu, rotations_cpu, scalings_cpu, quadrature_filter_response_reference_1_cpu, quadrature_filter_response_reference_2_cpu, quadrature_filter_response_reference_3_cpu] = perform_fMRI_registration_CPU(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction);


%%
% Create smoothing filters

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
smoothing_filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_zz = temp;

volume = fMRI_volumes(:,:,:,1);
smoothed_volume = convn(volume,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');   
smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');
   
threshold = 0.9 * mean(smoothed_volume(:));
brain_mask = double(smoothed_volume >= threshold) + 0.001;
   
smoothed_volume = convn(brain_mask,smoothing_filter_xx,'same');
smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');   
smoothed_brain_mask = convn(smoothed_volume,smoothing_filter_zz,'same');


sigma_x = EPI_smoothing_amount / 2.354 / EPI_voxel_size_x;
sigma_y = EPI_smoothing_amount / 2.354 / EPI_voxel_size_y;
sigma_z = EPI_smoothing_amount / 2.354 / EPI_voxel_size_z;

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
smoothing_filter_xx = temp;

temp = zeros(9,1,1);
temp(:,1,1) = smoothing_filter_y;
smoothing_filter_yy = temp;

temp = zeros(1,1,9);
temp(1,1,:) = smoothing_filter_z;
smoothing_filter_zz = temp;

smoothed_volumes_cpu = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
   volume = motion_corrected_volumes_cpu(:,:,:,t);
   %volume = fMRI_volumes(:,:,:,t);
   smoothed_volume = convn(volume .* brain_mask,smoothing_filter_xx,'same');
   smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');   
   smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');
   smoothed_volumes_cpu(:,:,:,t) = smoothed_volume ./ smoothed_brain_mask;
end

smoothed_volumes_cpu2 = zeros(size(fMRI_volumes));
for t = 1:size(fMRI_volumes,4)
   volume = motion_corrected_volumes_cpu(:,:,:,t);   
   smoothed_volume = convn(volume,smoothing_filter_xx,'same');
   smoothed_volume = convn(smoothed_volume,smoothing_filter_yy,'same');   
   smoothed_volume = convn(smoothed_volume,smoothing_filter_zz,'same');
   smoothed_volumes_cpu2(:,:,:,t) = smoothed_volume;
end


%%
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
%contrasts(:,1) = [1 0 0 0 0 0 0 0]';
%contrasts(:,2) = [0 1 0 0 0 0 0 0]';
%contrasts(:,3) = [0 0 0 0 1 0 0 0]';
for i = 1:size(contrasts,2)
    contrast = contrasts(:,i);
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

mask = ones(sy,sx,sz,st);
statistical_maps_cpu = zeros(sy,sx,sz,size(contrasts,2));
betas_cpu = zeros(sy,sx,sz,size(X_GLM,2));
residuals_cpu = zeros(sy,sx,sz,st);
residual_variances_cpu = zeros(sy,sx,sz);

for x = 1:sx
    for y = 1:sy
        for z = 1:sz
            timeseries = squeeze(smoothed_volumes_cpu(y,x,z,:));
            %timeseries = timeseries - mean(timeseries);            
            beta = xtxxt_GLM*timeseries;
            betas_cpu(y,x,z,:) = beta;
            eps = timeseries - X_GLM*beta;
            residuals_cpu(y,x,z,:) = eps;
            %residual_variances_cpu(y,x,z) = sum((eps-mean(eps)).^2)/(st-size(X_GLM,2));
            residual_variances_cpu(y,x,z) = var(eps);
            for i = 1:size(contrasts,2)
                contrast = contrasts(:,i);
                statistical_maps_cpu(y,x,z,i) = contrast'*beta / sqrt( residual_variances_cpu(y,x,z) * ctxtxc_GLM(i));
            end
        end
    end
end


%%



tic
[beta_volumes, residuals, residual_variances, statistical_maps, T1_MNI_registration_parameters, EPI_T1_registration_parameters, ...
 EPI_MNI_registration_parameters, motion_parameters, motion_corrected_volumes_opencl, smoothed_volumes_opencl ...
 ar1_estimates, ar2_estimates, ar3_estimates, ar4_estimates] = ... 
FirstLevelAnalysis(fMRI_volumes,T1,MNI,MNI_brain_mask,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,T1_voxel_size_x,T1_voxel_size_y, ... 
T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale_T1_MNI, ...
coarsest_scale_EPI_T1,MM_T1_Z_CUT,MM_EPI_Z_CUT,number_of_iterations_for_motion_correction,EPI_smoothing_amount, AR_smoothing_amount, ...
X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,beta_space,opencl_platform,opencl_device);
toc


T1_MNI_registration_parameters

EPI_T1_registration_parameters

EPI_MNI_registration_parameters

%%


figure
plot(motion_parameters(:,1),'g')
hold on
plot(motion_parameters(:,2),'r')
hold on
plot(motion_parameters(:,3),'b')
hold off
title('Translation')

figure
plot(motion_parameters_cpu(:,1),'g')
hold on
plot(motion_parameters_cpu(:,2),'r')
hold on
plot(motion_parameters_cpu(:,3),'b')
hold off
title('Translation CPU')

figure
plot(motion_parameters(:,4),'g')
hold on
plot(motion_parameters(:,5),'r')
hold on
plot(motion_parameters(:,6),'b')
hold off
title('Rotation')

figure
plot(rotations_cpu(:,1),'g')
hold on
plot(rotations_cpu(:,2),'r')
hold on
plot(rotations_cpu(:,3),'b')
hold off
title('Rotation CPU')

slice = 13;

figure
imagesc(motion_corrected_volumes_opencl(:,:,slice,1)); colorbar
title('MC')

figure
imagesc(motion_corrected_volumes_cpu(:,:,slice,1)); colorbar
title('MC cpu')

figure
imagesc([motion_corrected_volumes_cpu(:,:,slice,2) - motion_corrected_volumes_opencl(:,:,slice,2)]); colorbar
title('MC cpu - gpu')

figure
imagesc(smoothed_volumes_opencl(:,:,slice,1) .* brain_mask(:,:,slice)); colorbar
title('SM')

figure
imagesc(smoothed_volumes_cpu(:,:,slice,1) .* brain_mask(:,:,slice)); colorbar
title('SM cpu')

figure
imagesc(smoothed_volumes_cpu(:,:,slice,1)); colorbar
title('SM cpu')

figure
imagesc(smoothed_volumes_cpu2(:,:,slice,1)); colorbar
title('SM cpu')

figure
imagesc([smoothed_volumes_cpu(:,:,slice,2) - smoothed_volumes_opencl(:,:,slice,2)]); colorbar
title('SM cpu - gpu')

figure
imagesc(beta_volumes(:,:,slice,1)); colorbar
title('Beta')

figure
imagesc(betas_cpu(:,:,slice,1)); colorbar
title('Beta cpu')

%figure
%imagesc([betas_cpu(:,:,slice,1) - beta_volumes(:,:,slice,1)]); colorbar
%title('Beta cpu - gpu')

figure
imagesc(residual_variances(:,:,slice)); colorbar
title('Residual variances')

figure
imagesc(residual_variances_cpu(:,:,slice)); colorbar
title('Residual variances cpu')

figure
imagesc(statistical_maps(:,:,slice)); colorbar
title('t-values')

figure
imagesc(statistical_maps_cpu(:,:,slice)); colorbar
title('t-values cpu')



% for t = 1:st    
%     figure(100)
%     imagesc([fMRI_volumes(:,:,38,t) motion_corrected_volumes_cpu(:,:,38,t)  motion_corrected_volumes_opencl(:,:,38,t) ])
%     pause(0.1)
% end

