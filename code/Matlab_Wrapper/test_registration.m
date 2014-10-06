%  	 BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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
        
    mex RegisterTwoVolumesMex.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    %mex -g RegisterTwoVolumesMex.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
      
    opencl_platform = 0; % 0 Nvidia, 1 Intel, 2 AMD
    opencl_device = 0;
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/normalization';
    
    %mex -g RegisterTwoVolumesMex.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    mex RegisterTwoVolumesMex.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    
    opencl_platform = 1; % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 1;
end

%------------------------------------

show_results = 1;                   % Show resulting registration or not
save_warped_volume_matlab = 0;      % Save warped volume as Matlab file or not
save_warped_volume_nifti = 0;       % Save warped volume as nifti file or not

%------------------------------------

%study = 'Baltimore';
study = 'Cambridge'; N = 10;

voxel_size = 1;
SIGMA = 5;

number_of_iterations_for_linear_image_registration = 10;
number_of_iterations_for_nonlinear_image_registration = 10;
MM_T1_Z_CUT = 0;

% Load MNI template, without skull
MNI_brain_nii = load_nii(['../../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain.nii']);
MNI_brain = double(MNI_brain_nii.img);
[MNI_sy MNI_sx MNI_sz] = size(MNI_brain);    

% Load quadrature filters
load filters_for_parametric_registration.mat
load filters_for_nonparametric_registration.mat
    
dirs = dir([basepath study]);

normalization_times = zeros(N,1);

% Loop over subjects
%for s = 10:N+10
for s = 2:N+1

    subject = dirs(s+2).name % Skip . and .. 'folders'
    
    s
    
    %close all
    
    if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu'))  || (strcmp(study,'Baltimore')) ) 
        %T1_nii = load_nii([basepath study '/' subject '/anat/mprage_skullstripped.nii.gz']);
        T1_nii = load_nii([basepath study '/' subject]);
    elseif ( strcmp(study,'OpenfMRI'))
        T1_nii = load_nii([basepath study '/' substudy '/highres' subject '.nii']);
    end
    T1 = double(T1_nii.img);       
    
    [T1_sy T1_sx T1_sz] = size(T1);
    [T1_sy T1_sx T1_sz]
        
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
        
    MNI_voxel_size_x = MNI_brain_nii.hdr.dime.pixdim(2);
    MNI_voxel_size_y = MNI_brain_nii.hdr.dime.pixdim(3);
    MNI_voxel_size_z = MNI_brain_nii.hdr.dime.pixdim(4);    
    
    % Run the registration with OpenCL           
    start = clock;
    [aligned_T1_linear, aligned_T1_nonlinear, interpolated_T1, registration_parameters, ...
        displacement_x, displacement_y, displacement_z] = ...
        RegisterTwoVolumesMex(T1,MNI_brain,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z, ...
        f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
        f1_nonparametric_registration,f2_nonparametric_registration,f3_nonparametric_registration,f4_nonparametric_registration,f5_nonparametric_registration,f6_nonparametric_registration, ...
        m1, m2, m3, m4, m5, m6, ...
        filter_directions_x, filter_directions_y, filter_directions_z, ...
        number_of_iterations_for_nonlinear_image_registration,number_of_iterations_for_nonlinear_image_registration,MM_T1_Z_CUT, SIGMA, opencl_platform, opencl_device);
    normalization_times(s) = etime(clock,start);
    elapsed_time = etime(clock,start)
            
    registration_parameters
            
    % Show some nice results
    if show_results == 1
        slice = round(0.55*MNI_sy);
        figure(1); imagesc(flipud(squeeze(interpolated_T1(slice,:,:))')); colormap gray
        figure(2); imagesc(flipud(squeeze(aligned_T1_linear(slice,:,:))')); colormap gray    
        figure(3); imagesc(flipud(squeeze(MNI_brain(slice,:,:))')); colormap gray
        figure(4); imagesc(flipud(squeeze(aligned_T1_nonlinear(slice,:,:))')); colormap gray
    
        slice = round(0.47*MNI_sz);
        figure(5); imagesc(squeeze(interpolated_T1(:,:,slice))); colormap gray
        figure(6); imagesc(squeeze(aligned_T1_linear(:,:,slice))); colormap gray
        figure(7); imagesc(squeeze((MNI_brain(:,:,slice)))); colormap gray
        figure(8); imagesc(squeeze(aligned_T1_nonlinear(:,:,slice))); colormap gray
    end
    
    %k = waitforbuttonpress 
        
    % Save normalized volume as a Matlab file
    if save_warped_volume_matlab == 1
        filename = [basepath_BROCCOLI '/BROCCOLI_warped_subject' num2str(s) '.mat'];
        save(filename,'aligned_T1_nonlinear')
    end
    
    % Save normalized volume as a nifti file
    if save_warped_volume_nifti == 1
        new_file.hdr = MNI_nii.hdr;
        new_file.hdr.dime.datatype = 16;
        new_file.hdr.dime.bitpix = 16;
        new_file.img = single(aligned_T1_nonlinear);    
        
        filename = [basepath_BROCCOLI '/BROCCOLI_warped_subject' num2str(s) '.nii'];
            
        save_nii(new_file,filename);
    end
    
    pause(1)
    
end



