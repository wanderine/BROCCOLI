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

% Set some paths, depending on Windows or Linux
if ispc
    addpath('D:\nifti_matlab') % Change to your folder for nifti matlab
    basepath = '../test_data/fcon1000/classic/';
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab') % Change to your folder for nifti matlab
    basepath = '../test_data/fcon1000/classic/';
end


% Select study and subject

study = 'Cambridge';
subject = 'sub00156'
%subject = 'sub00294'
%subject = 'sub01361'

%study = 'Beijing';
%subject = 'sub00440'
%subject = 'sub01018'
%subject = 'sub01244'

%study = 'ICBM';
%subject = 'sub00448'
%subject = 'sub00623'
%subject = 'sub02382'

% Set voxel size to work with (1 mm or 2 mm)
voxel_size = 1;

% Select OpenCL platform and OpenCL device
% (use GetOpenCLInfo to see your platforms and devices)
opencl_platform = 0;
opencl_device = 0;

% Set parameters for registration
number_of_iterations_for_parametric_image_registration = 10;
number_of_iterations_for_nonparametric_image_registration = 10;
coarsest_scale = 8/voxel_size;

% Number of slices to cut from the initial T1 volume
if (strcmp(study,'Cambridge')) 
    MM_T1_Z_CUT = 30; 
elseif (strcmp(study,'Beijing'))
    MM_T1_Z_CUT = 50; 
elseif (strcmp(study,'ICBM')) 
    MM_T1_Z_CUT = 10;    
end
    

% Load MNI template with skull
MNI_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm.nii']);
MNI = double(MNI_nii.img);
[MNI_sy MNI_sx MNI_sz] = size(MNI);
MNI_size = size(MNI);
MNI_size
MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);

% Load MNI template without skull
MNI_brain_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain.nii']);
MNI_brain = double(MNI_brain_nii.img);

% Load MNI brain mask
MNI_brain_mask_nii = load_nii(['../brain_templates/MNI152_T1_' num2str(voxel_size) 'mm_brain_mask.nii']);
MNI_brain_mask = double(MNI_brain_mask_nii.img);

load filters_for_parametric_registration.mat
load filters_for_nonparametric_registration.mat

% Load T1 volume
T1_nii = load_nii([basepath study '/'  subject '/anat/mprage_skullstripped.nii.gz']);
T1 = double(T1_nii.img);
[T1_sy T1_sx T1_sz] = size(T1);
T1_size = size(T1);
T1_size

if (strcmp(study,'Beijing'))
    T1_voxel_size_x = T1_nii.hdr.dime.pixdim(1);
    T1_voxel_size_y = T1_nii.hdr.dime.pixdim(2);
    T1_voxel_size_z = T1_nii.hdr.dime.pixdim(3);
else
    T1_voxel_size_x = T1_nii.hdr.dime.pixdim(2);
    T1_voxel_size_y = T1_nii.hdr.dime.pixdim(3);
    T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);
end

% Run registration (and get a lot of debugging outputs)
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
    number_of_iterations_for_parametric_image_registration,number_of_iterations_for_nonparametric_image_registration,coarsest_scale,MM_T1_Z_CUT,opencl_platform, opencl_device);
toc

% Put the registration parameters into a 4 x 4 matrix
affine_registration_parameters = zeros(4,4);
affine_registration_parameters(1,4) = registration_parameters_opencl(1);
affine_registration_parameters(2,4) = registration_parameters_opencl(2);
affine_registration_parameters(3,4) = registration_parameters_opencl(3);
affine_registration_parameters(4,4) = 1;
affine_registration_parameters(1,1) = registration_parameters_opencl(4) + 1;
affine_registration_parameters(2,1) = registration_parameters_opencl(5);
affine_registration_parameters(3,1) = registration_parameters_opencl(6);
affine_registration_parameters(4,1) = 0;
affine_registration_parameters(1,2) = registration_parameters_opencl(7);
affine_registration_parameters(2,2) = registration_parameters_opencl(8) + 1;
affine_registration_parameters(3,2) = registration_parameters_opencl(9);
affine_registration_parameters(4,2) = 0;
affine_registration_parameters(1,3) = registration_parameters_opencl(10);
affine_registration_parameters(2,3) = registration_parameters_opencl(11);
affine_registration_parameters(3,3) = registration_parameters_opencl(12) + 1;
affine_registration_parameters(4,3) = 0;

affine_registration_parameters

% Look at sagittal results
slice = round(0.55*MNI_sy);
figure; imagesc(flipud(squeeze(interpolated_T1_opencl(slice,:,:))')); colormap gray; title('Original T1 volume, after interpolation to MNI voxel size and z cut')
figure; imagesc(flipud(squeeze(aligned_T1_opencl(slice,:,:))')); colormap gray; title('Aligned T1 volume, linear registration')
figure; imagesc(flipud(squeeze(aligned_T1_nonparametric_opencl(slice,:,:))')); colormap gray; title('Aligned T1 volume, linear + non-linear registration')
figure; imagesc(flipud(squeeze(MNI_brain(slice,:,:))')); colormap gray; title('MNI volume')

% Look at axial results
slice = round(0.47*MNI_sz);
figure; imagesc(squeeze(interpolated_T1_opencl(:,:,slice))); colormap gray; title('Original T1 volume, after interpolation to MNI voxel size and z cut')
figure; imagesc(squeeze(aligned_T1_opencl(:,:,slice))); colormap gray; title('Aligned T1 volume, linear registration')
figure; imagesc(squeeze(aligned_T1_nonparametric_opencl(:,:,slice))); colormap gray; title('Aligned T1 volume, linear + non-linear registration')
figure; imagesc(squeeze((MNI_brain(:,:,slice)))); colormap gray; title('MNI volume')

