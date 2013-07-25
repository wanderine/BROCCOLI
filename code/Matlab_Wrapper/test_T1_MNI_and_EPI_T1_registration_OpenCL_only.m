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

%mex RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

%mex RegisterEPIT1.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib
mex -g RegisterEPIT1.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

load filters.mat

subject = 23;
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
EPI = double(EPI_nii.img);
EPI = EPI(:,:,:,5);
EPI = EPI/max(EPI(:));

[sy sx sz] = size(T1)

number_of_iterations_for_image_registration = 30;
coarsest_scale = 8/voxel_size;
MM_T1_Z_CUT = 10;

opencl_platform = 0;

% Make sure T1 has same voxel size as MNI
T1_voxel_size_x = T1_nii.hdr.dime.pixdim(2);
T1_voxel_size_y = T1_nii.hdr.dime.pixdim(3);
T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);

MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);

EPI_voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
EPI_voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
EPI_voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

tic
[aligned_T1_opencl, skullstripped_T1_opencl, interpolated_T1_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl, downsampled_volume_opencl] = ... 
RegisterT1MNI(T1,MNI,MNI_brain_mask,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_T1_Z_CUT,opencl_platform);
toc

registration_parameters_opencl

slice = 100/voxel_size;
%figure; imagesc(flipud(squeeze(T1(slice,:,:))'))
figure; imagesc(flipud(squeeze(interpolated_T1_opencl(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(skullstripped_T1_opencl(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(aligned_T1_opencl(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(MNI(slice,:,:))')); colormap gray

figure; imagesc(squeeze(interpolated_T1_opencl(:,:,slice))); colormap gray
figure; imagesc(squeeze(skullstripped_T1_opencl(:,:,slice))); colormap gray
figure; imagesc(squeeze(aligned_T1_opencl(:,:,slice))); colormap gray
figure; imagesc(squeeze(MNI(:,:,slice))); colormap gray

%%

number_of_iterations_for_image_registration = 60;
coarsest_scale = 4/voxel_size;
MM_EPI_Z_CUT = 20;


filter_x = fspecial('gaussian',9,2);
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

smoothed_volume = convn(skullstripped_T1_opencl,filter_xx,'same');
smoothed_volume = convn(smoothed_volume,filter_yy,'same');   
smoothed_skullstripped_T1_opencl = convn(smoothed_volume,filter_zz,'same');



tic
[aligned_EPI_opencl, interpolated_EPI_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ... 
RegisterEPIT1(EPI,skullstripped_T1_opencl,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_EPI_Z_CUT,opencl_platform);
%[aligned_EPI_opencl, interpolated_EPI_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ... 
%RegisterEPIT1(smoothed_skullstripped_T1_opencl,EPI,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,EPI_voxel_size_x,EPI_voxel_size_y,EPI_voxel_size_z,f1,f2,f3,number_of_iterations_for_image_registration,coarsest_scale,MM_EPI_Z_CUT,opencl_platform);
toc


%close all
slice = 60/voxel_size;
figure; imagesc(flipud(squeeze(interpolated_EPI_opencl(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(aligned_EPI_opencl(slice,:,:))')); colormap gray
figure; imagesc(flipud(squeeze(skullstripped_T1_opencl(slice,:,:))')); colormap gray
%figure; imagesc(flipud(squeeze(EPI(slice,:,:))')); colormap gray
%figure; imagesc(flipud(squeeze(MNI(slice,:,:))')); colormap gray

slice = 80/voxel_size;
figure; imagesc(squeeze(interpolated_EPI_opencl(:,:,slice))); colormap gray
figure; imagesc(squeeze(aligned_EPI_opencl(:,:,slice))); colormap gray
figure; imagesc(squeeze(skullstripped_T1_opencl(:,:,slice))); colormap gray
%figure; imagesc(squeeze(EPI(:,:,slice))); colormap gray
%figure; imagesc(squeeze(MNI(:,:,slice))); colormap gray

%figure; imagesc(flipud(squeeze(interpolated_EPI_opencl(:,slice,:))')); colormap gray
%figure; imagesc(flipud(squeeze(aligned_EPI_opencl(:,slice,:))')); colormap gray
%figure; imagesc(flipud(squeeze(smoothed_skullstripped_T1_opencl(:,slice,:))')); colormap gray
%figure; imagesc(flipud(squeeze(MNI(:,slice,:))')); colormap gray


registration_parameters_opencl

