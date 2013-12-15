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

ismiha = 1

if ismiha
    addpath('/home/miha/Delo/BROCCOLI/nifti')
    basepath = '/data/miha/OpenfMRI/';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/normalization';
    
    mex ../code/Matlab_Wrapper/FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/opt/cuda/include/ -I/opt/cuda/include/CL -L/usr/lib -I/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB -L/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB    -I/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB/Eigen
    
    opencl_platform = 0;  % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 0;
    
    
    %% Only used in Octave for compatibility with Matlab
    if exist('do_braindead_shortcircuit_evaluation', 'builtin')
      do_braindead_shortcircuit_evaluation(1);
      warning('off', 'Octave:possible-matlab-short-circuit-operator');
      
      % We need this for 'decimate'
      pkg load signal
    end
    
    test_data_dir = '/data/miha/BROCCOLI/motion_correction'
else
    if ispc
        %addpath('D:\nifti_matlab')
        addpath('D:\nifti_matlab')
        basepath = 'D:\OpenfMRI\';    
        opencl_platform = 2; % 0 Nvidia, 1 Intel, 2 AMD
        opencl_device = 1;
        %mex -g ../code/Matlab_Wrapper/FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
        mex ../code/Matlab_Wrapper/FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib    -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    elseif isunix
        addpath('/home/andek/Research_projects/nifti_matlab')
        basepath = '/data/andek/OpenfMRI/';    
        opencl_platform = 2; 
        opencl_device = 0;
        %mex -g ../code/Matlab_Wrapper/FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
        mex ../code/Matlab_Wrapper/FirstLevelAnalysis.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    end
end

study = 'RhymeJudgment/ds003';

subject = 1; %5 has bad skullstrip

if subject < 10
    subject = ['/sub00' num2str(subject)];
else
    subject = ['/sub0' num2str(subject)];
end

voxel_size = 2;
beta_space = 0; % 0 = EPI, 1 = MNI
save_as_nifti = 1; % Save statistical map as nifti file or not

%--------------------------------------------------------------------------------------
% Statistical settings
%--------------------------------------------------------------------------------------

use_temporal_derivatives = 1;
regress_motion = 1;
regress_confounds = 0;

EPI_smoothing_amount = 6.0;
AR_smoothing_amount = 7.0;

%--------------------------------------------------------------------------------------
% Settings for image registration
%--------------------------------------------------------------------------------------

number_of_iterations_for_parametric_image_registration = 15;
number_of_iterations_for_nonparametric_image_registration = 10;
number_of_iterations_for_motion_correction = 3;
coarsest_scale_T1_MNI = 8/voxel_size;
coarsest_scale_EPI_T1 = 8/voxel_size;
MM_T1_Z_CUT = 0;
MM_EPI_Z_CUT = 0;

load filters_for_parametric_registration.mat
load filters_for_nonparametric_registration.mat

%--------------------------------------------------------------------------------------
% Load T1 volume
%--------------------------------------------------------------------------------------

T1_nii = load_nii([basepath study subject '/anatomy/highres001_brain.nii']);
T1 = double(T1_nii.img);

[T1_sy T1_sx T1_sz] = size(T1);
[T1_sy T1_sx T1_sz]

T1_voxel_size_x = T1_nii.hdr.dime.pixdim(3);
T1_voxel_size_y = T1_nii.hdr.dime.pixdim(2);
T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);

%--------------------------------------------------------------------------------------
% Load fMRI data
%--------------------------------------------------------------------------------------

EPI_nii = load_nii([basepath study subject '/BOLD/task001_run001/bold.nii']);
fMRI_volumes = double(EPI_nii.img);
[sy sx sz st] = size(fMRI_volumes);

EPI = fMRI_volumes(:,:,:,1);

[EPI_sy EPI_sx EPI_sz] = size(EPI);
[EPI_sy EPI_sx EPI_sz st]

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
%TR = EPI_nii.hdr.dime.pixdim(5);
TR = 2;

%--------------------------------------------------------------------------------------
% Load MNI templates
%--------------------------------------------------------------------------------------

MNI_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
MNI = double(MNI_nii.img);

MNI_brain_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain.nii']);
MNI_brain = double(MNI_brain_nii.img);
MNI_brain = MNI_brain/max(MNI_brain(:));

MNI_brain_mask_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);
MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));

[MNI_sy MNI_sx MNI_sz] = size(MNI);
[MNI_sy MNI_sx MNI_sz]

MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);

%--------------------------------------------------------------------------------------
% Create GLM regressors
%--------------------------------------------------------------------------------------

% Make high resolution regressors first

highres_factor = 10
X_GLM_highres = zeros(st*highres_factor,2);
X_GLM = zeros(st,2);

for regressor = 1:2
    
    fid = fopen([basepath study subject '/model/model001/onsets/task001_run001/cond00' num2str(regressor) '.txt']);
    text = textscan(fid,'%f%f%f');
    fclose(fid);

    onsets = text{1};
    durations = text{2};
    values = text{3};

    for i = 1:length(onsets)    
        start = round(onsets(i)*highres_factor/TR); 
        activity_length = round(durations(i)*highres_factor/TR);
        value = values(i);

        for j = 1:activity_length
            X_GLM_highres(start+j,regressor) = value; 
        end                    
    end
    
    % Downsample
    temp = decimate(X_GLM_highres(:,regressor),highres_factor);
    X_GLM(:,regressor) = temp(1:st);
end
    
xtxxt_GLM = inv(X_GLM'*X_GLM)*X_GLM';


%--------------------------------------------------------------------------------------
% Load confound regressors
%--------------------------------------------------------------------------------------

% Calculate number of confounds

confounds = 1;

if (regress_confounds == 1)
    
    fid = fopen([basepath study '/sub00' subject '/BOLD/task001_run001/QA/confound.txt']);
    text = textscan(fid,'%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
    fclose(fid);

    confounds = zeros(st,14);

    for i = 1:14
        confounds(:,i) = text{i};
    end
    
end


%--------------------------------------------------------------------------------------
% Setup contrasts
%--------------------------------------------------------------------------------------

% Contrasts for confounding regressors are automatically set to zeros by BROCCOLI 
%contrasts = [1  0];
    
contrasts = [1  0;
             1  0;
             1  0;
             1  0];

for i = 1:size(contrasts,1)
    contrast = contrasts(i,:)';
    ctxtxc_GLM(i) = contrast'*inv(X_GLM'*X_GLM)*contrast;
end
ctxtxc_GLM

%--------------------------------------------------------------------------------------
% Run first level analysis
%--------------------------------------------------------------------------------------


tic
[beta_volumes, residuals, residual_variances, statistical_maps, T1_MNI_registration_parameters, EPI_T1_registration_parameters, ...
 EPI_MNI_registration_parameters, motion_parameters, motion_corrected_volumes_opencl, smoothed_volumes_opencl ...
 ar1_estimates, ar2_estimates, ar3_estimates, ar4_estimates, design_matrix1, design_matrix2, aligned_T1, aligned_T1_nonparametric, aligned_EPI, cluster_indices, whitened_models] = ... 
FirstLevelAnalysis(fMRI_volumes,T1,MNI,MNI_brain,MNI_brain_mask,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z, ...
f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
f1_nonparametric_registration, f2_nonparametric_registration, f3_nonparametric_registration, f4_nonparametric_registration, f5_nonparametric_registration, f6_nonparametric_registration, ...
m1, m2, m3, m4, m5, m6, ...
filter_directions_x, filter_directions_y, filter_directions_z, ...
number_of_iterations_for_parametric_image_registration, number_of_iterations_for_nonparametric_image_registration, ...
coarsest_scale_T1_MNI, coarsest_scale_EPI_T1,MM_T1_Z_CUT,MM_EPI_Z_CUT,number_of_iterations_for_motion_correction,regress_motion,EPI_smoothing_amount,AR_smoothing_amount, ...
X_GLM,xtxxt_GLM',contrasts,ctxtxc_GLM,use_temporal_derivatives,beta_space,confounds,regress_confounds,opencl_platform,opencl_device);
toc

T1_MNI_registration_parameters

EPI_T1_registration_parameters

EPI_MNI_registration_parameters

slice = round(0.55*MNI_sy);
%figure; imagesc(flipud(squeeze(aligned_T1(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(aligned_T1_nonparametric(slice,:,:))')); colormap gray
title('Aligned T1')
figure; imagesc(flipud(squeeze(aligned_EPI(slice,:,:))')); colormap gray
title('Aligned EPI')
figure; imagesc(flipud(squeeze(MNI_brain(slice,:,:))')); colormap gray
title('MNI template')

slice = round(0.47*MNI_sz);
%figure; imagesc(aligned_T1(:,:,slice)); colormap gray
figure; imagesc(aligned_T1_nonparametric(:,:,slice)); colormap gray
title('Aligned T1')
figure; imagesc(aligned_EPI(:,:,slice)); colormap gray
title('Aligned EPI')
figure; imagesc(MNI_brain(:,:,slice)); colormap gray
title('MNI template')

%figure; imagesc(flipud(squeeze(EPI(35,:,:))')); colormap gray

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
   slice = round(EPI_sz/2);  -6.046
end


figure
imagesc(MNI_brain(:,:,slice)); colormap gray; colorbar
title('MNI template')

%figure
%imagesc(beta_volumes(:,:,slice,1)); colormap gray;  colorbar
%title('Beta 1')

figure
%imagesc(statistical_maps(20:end-19,20:end-19,slice,1)); colorbar
%imagesc(statistical_maps(10:end-10,10:end-10,slice,4)); colorbar
imagesc(statistical_maps(:,:,slice,1)); colorbar
title('t-values')

%figure
%imagesc(ar1_estimates(:,:,20)); colorbar

if beta_space == 1
    slice = round(0.45*MNI_sz);
else
   slice = round(EPI_sz/2); 
end

figure
imagesc(flipud(squeeze(MNI_brain(slice,:,:))')); colormap gray; colorbar
title('MNI template')

%figure
%imagesc(flipud(squeeze(fMRI_volumes(slice,:,:,2))')); colormap gray
%title('fMRI')

%figure
%imagesc(flipud(squeeze(beta_volumes(slice,:,:,1))')); colorbar; colormap gray
%title('Beta 1')


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

%figure
%imagesc(flipud(squeeze(ar1_estimates(30,:,:,1))')); colorbar

%figure
%imagesc(residual_variances(:,:,slice)); colorbar
%title('Residual variances')

figure
imagesc(design_matrix1); colormap gray
title('Design matrix')

figure
imagesc(ar1_estimates(:,:,slice)); colorbar

figure
imagesc(ar2_estimates(:,:,slice)); colorbar
 
%figure
%imagesc(ar3_estimates(:,:,30)); colorbar
 
%figure
%imagesc(ar4_estimates(:,:,30)); colorbar

if save_as_nifti == 1

	% Flip upside down
	for slice = 1:33
	   statistical_maps_flipped(:,:,slice) = flipud(statistical_maps(:,:,slice,1));
	end

	a = statistical_maps(:,:,:,1);

	new_file.hdr = EPI_nii.hdr;
	new_file.hdr.dime.dim = [3 64 64 33 1 1 1 1];
	new_file.hdr.dime.vox_offset = 352;
	new_file.hdr.scl_slope = 1;
	new_file.hdr.dime.cal_max = max(a(:));
	new_file.hdr.dime.cal_min = min(a(:));
	new_file.hdr.dime.gl_max = max(a(:));
	new_file.hdr.dime.gl_min = min(a(:));
	new_file.hdr.dime.datatype = 16;
	new_file.hdr.dime.bitpix = 32;

	new_file.original.hdr.dime.dim = [3 64 64 33 1 1 1 1];
	new_file.original.hdr.dime.vox_offset = 352;
	new_file.original.hdr.scl_slope = 1;
	new_file.original.hdr.dime.cal_max = max(a(:));
	new_file.original.hdr.dime.cal_min = min(a(:));
	new_file.original.hdr.dime.gl_max = max(a(:));
	new_file.original.hdr.dime.gl_min = min(a(:));
	new_file.original.hdr.dime.datatype = 16;
	new_file.original.hdr.dime.bitpix = 32;

	new_file.img = single(statistical_maps_flipped);    
	filename = ['BROCCOLI_statistical_map_10co_iterations.nii'];
    %filename = ['BROCCOLI_statistical_map_no_whitening.nii'];
	save_nii(new_file,filename);

end

k = waitforbuttonpress


