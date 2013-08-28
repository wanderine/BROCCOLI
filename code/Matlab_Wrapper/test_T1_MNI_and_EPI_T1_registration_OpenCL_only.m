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
    %basepath = '../../test_data/fcon1000/classic/';
    %mex RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
    mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
    
    %mex RegisterEPIT1.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
    %mex -g RegisterEPIT1.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';
    %basepath = '../../test_data/fcon1000/classic/';
    mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug
    
end




%study = 'Cambridge';
%subject = 'sub00156'
%subject = 'sub00294'
%subject = 'sub01361'
study = 'Beijing';
%subject = 'sub00440'
%subject = 'sub01018'
%subject = 'sub01244'
%study = 'ICBM';
%subject = 'sub00448'
%subject = 'sub00623'
%subject = 'sub02382'
%study = 'Oulu';
%study = 'OpenfMRI';
substudy = 'Mixed';
%substudy = 'Balloon';

%subject = 18;

voxel_size = 1;
opencl_platform = 2;
opencl_device = 0;

number_of_iterations_for_image_registration = 10;
coarsest_scale = 8/voxel_size;
MM_T1_Z_CUT = 100;

dirs = dir([basepath study]);

for s = 3:length(dirs)
    subject = dirs(s).name
    
    close all
    
    if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu'))  )
        T1_nii = load_nii([basepath study '/'  subject '/anat/mprage_anonymized.nii.gz']);
        %T1_nii = load_nii([basepath study '/mprage_anonymized' num2str(subject) '.nii.gz']);
    elseif ( strcmp(study,'OpenfMRI'))
        T1_nii = load_nii([basepath study '/' substudy '/highres' subject '.nii.gz']);
    end
    T1 = double(T1_nii.img);
    T1 = T1/max(T1(:)); 
    
    MNI_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
    MNI = double(MNI_nii.img);
    MNI = MNI/max(MNI(:));
    
    MNI_brain_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain.nii']);
    MNI_brain = double(MNI_brain_nii.img);
    MNI_brain = MNI_brain/max(MNI_brain(:));
    
    %temp = zeros(256,256,256);
    %temp(40:40+181,15:15+217,40:40+181) = MNI;
    %MNI = temp;
    
    
    %T1 = zeros(size(MNI));
    %T1(2:end,2:end,2:end) = MNI(1:end-1,1:end-1,1:end-1);
    
    MNI_brain_mask_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
    MNI_brain_mask = double(MNI_brain_mask_nii.img);
    MNI_brain_mask = MNI_brain_mask/max(MNI_brain_mask(:));
    
    %temp = zeros(256,256,256);
    %temp(40:40+181,15:15+217,40:40+181) = MNI_brain_mask;
    %MNI_brain_mask = temp;
    
    if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
        %EPI_nii = load_nii([basepath study '/rest' num2str(subject) '.nii.gz']);
    elseif ( strcmp(study,'OpenfMRI'))
        %EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(subject) '.nii.gz']);
    end
    
    %fMRI_volumes = double(EPI_nii.img);
    %fMRI_volumes = fMRI_volumes(:,:,1:22,:);
    %fMRI_volumes = fMRI_volumes(:,:,:,5:end);
    %[sy sx sz st] = size(fMRI_volumes)
    %fMRI_volumes = fMRI_volumes/max(fMRI_volumes(:));
    
    %EPI = fMRI_volumes(:,:,:,1);
    %EPI = EPI/max(EPI(:));
    
    [T1_sy T1_sx T1_sz] = size(T1)
    [MNI_sy MNI_sx MNI_sz] = size(MNI)
    %[EPI_sy EPI_sx EPI_sz] = size(EPI)
    
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
    
    %EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
    %EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
    %EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);
    
    
    %%
    load filters_for_parametric_registration.mat
    load filters_for_nonparametric_registration.mat
    
    
    
    tic
    [aligned_T1_opencl, aligned_T1_nonparametric_opencl, skullstripped_T1_opencl, interpolated_T1_opencl, registration_parameters_opencl, ...
        quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, ...
        quadrature_filter_response_4_opencl, quadrature_filter_response_5_opencl, quadrature_filter_response_6_opencl, ...
        phase_differences_opencl, phase_certainties_opencl, phase_gradients_opencl, downsampled_volume_opencl, ...
        slice_sums, top_slice, ...
        t11_opencl, t12_opencl, t13_opencl, t22_opencl, t23_opencl, t33_opencl, ...
        displacement_x_opencl, displacement_y_opencl, displacement_z_opencl, A_matrix, h_vector] = ...
        RegisterT1MNI(T1,MNI,MNI_brain,MNI_brain_mask,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z, ...
        f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
        f1_nonparametric_registration,f2_nonparametric_registration,f3_nonparametric_registration,f4_nonparametric_registration,f5_nonparametric_registration,f6_nonparametric_registration, ...
        m1, m2, m3, m4, m5, m6, ...
        filter_directions_x, filter_directions_y, filter_directions_z, ...
        number_of_iterations_for_image_registration,coarsest_scale,MM_T1_Z_CUT,opencl_platform, opencl_device);
    toc
    
    %fr = convn(interpolated_T1_opencl,f3,'same');
    
    
    registration_parameters_opencl
    
    slice = round(0.55*MNI_sy);
    
    figure; imagesc(flipud(squeeze(interpolated_T1_opencl(slice,:,:))')); colormap gray
    figure; imagesc(flipud(squeeze(skullstripped_T1_opencl(slice,:,:))')); colormap gray
    figure; imagesc(flipud(squeeze(aligned_T1_opencl(slice,:,:))')); colormap gray
    %hold on
    %quiver(flipud(squeeze(displacement_x_opencl(slice,:,:))'),flipud(squeeze(displacement_z_opencl(slice,:,:))'))
    %hold off
    figure; imagesc(flipud(squeeze(aligned_T1_nonparametric_opencl(slice,:,:))')); colormap gray
    figure; imagesc(flipud(squeeze(MNI_brain(slice,:,:))')); colormap gray
    %figure; imagesc( [ flipud(squeeze((MNI(slice,:,:) - aligned_T1_opencl(slice,:,:) ))') flipud(squeeze(abs(MNI(slice,:,:) - aligned_T1_nonparametric_opencl(slice,:,:) ))')]  ); colormap gray
    
    slice = round(0.47*MNI_sz);
    figure; imagesc(squeeze(interpolated_T1_opencl(:,:,slice))); colormap gray
    figure; imagesc(squeeze(skullstripped_T1_opencl(:,:,slice))); colormap gray
    figure; imagesc(squeeze(aligned_T1_opencl(:,:,slice))); colormap gray
    %hold on
    %quiver(displacement_x_opencl(:,:,slice),displacement_y_opencl(:,:,slice))
    %hold off
    figure; imagesc(squeeze(aligned_T1_nonparametric_opencl(:,:,slice))); colormap gray
    figure; imagesc(squeeze((MNI_brain(:,:,slice)))); colormap gray
    
    pause
    
end
%%

% number_of_iterations_for_image_registration = 40;
% coarsest_scale = 4/voxel_size;
% MM_EPI_Z_CUT = 20;
%
%
% filter_x = fspecial('gaussian',9,2);
% filter_x = filter_x(:,5);
% filter_x = filter_x / sum(abs(filter_x));
% filter_y = filter_x;
% filter_z = filter_x;
%
% temp = zeros(1,9,1);
% temp(1,:,1) = filter_x;
% filter_xx = temp;
%
% temp = zeros(9,1,1);
% temp(:,1,1) = filter_y;
% filter_yy = temp;
%
% temp = zeros(1,1,9);
% temp(1,1,:) = filter_z;
% filter_zz = temp;
%
% smoothed_volume = convn(skullstripped_T1_opencl,filter_xx,'same');
% smoothed_volume = convn(smoothed_volume,filter_yy,'same');
% smoothed_skullstripped_T1_opencl = convn(smoothed_volume,filter_zz,'same');
%
%
%
% tic
% [aligned_EPI_opencl, interpolated_EPI_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ...
% RegisterEPIT1(EPI,skullstripped_T1_opencl,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_EPI_Z_CUT,opencl_platform, opencl_device);
% %[aligned_EPI_opencl, interpolated_EPI_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ...
% %RegisterEPIT1(smoothed_skullstripped_T1_opencl,EPI,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,f1_parametric_registration,f2_parametric_registration,f3_parametric_registration,number_of_iterations_for_image_registration,coarsest_scale,MM_EPI_Z_CUT,opencl_platform);
% toc
%
%
% %close all
% slice = round(0.6*MNI_sy);
% figure; imagesc(flipud(squeeze(interpolated_EPI_opencl(slice,:,:))')); colormap gray
% figure; imagesc(flipud(squeeze(aligned_EPI_opencl(slice,:,:))')); colormap gray
% figure; imagesc(flipud(squeeze(skullstripped_T1_opencl(slice,:,:))')); colormap gray
% %figure; imagesc(flipud(squeeze(EPI(slice,:,:))')); colormap gray
% %figure; imagesc(flipud(squeeze(MNI(slice,:,:))')); colormap gray
%
% slice = round(0.6*MNI_sz);
% figure; imagesc(squeeze(interpolated_EPI_opencl(:,:,slice))); colormap gray
% figure; imagesc(squeeze(aligned_EPI_opencl(:,:,slice))); colormap gray
% figure; imagesc(squeeze(skullstripped_T1_opencl(:,:,slice))); colormap gray
% %figure; imagesc(squeeze(EPI(:,:,slice))); colormap gray
% %figure; imagesc(squeeze(MNI(:,:,slice))); colormap gray
%
% %figure; imagesc(flipud(squeeze(interpolated_EPI_opencl(:,slice,:))')); colormap gray
% %figure; imagesc(flipud(squeeze(aligned_EPI_opencl(:,slice,:))')); colormap gray
% %figure; imagesc(flipud(squeeze(smoothed_skullstripped_T1_opencl(:,slice,:))')); colormap gray
% %figure; imagesc(flipud(squeeze(MNI(:,slice,:))')); colormap gray
%
%
% registration_parameters_opencl
%
