function [aligned_T1_parametric, aligned_T1_nonparametric, interpolated_T1, registration_parameters, displacement_x, displacement_y, displacement_z,reference] = RegisterTwoVolumes(filename_volume,filename_reference_volume,number_of_iterations_for_parametric_image_registration, number_of_iterations_for_nonparametric_image_registration, coarsest_scale, MM_T1_Z_CUT,opencl_platform,opencl_device)

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
  
% Load volume to transform
T1_nii = load_nii(filename_volume);
T1 = double(T1_nii.img);
[T1_sy T1_sx T1_sz] = size(T1);

T1_voxel_size_x = T1_nii.hdr.dime.pixdim(2);
T1_voxel_size_y = T1_nii.hdr.dime.pixdim(3);
T1_voxel_size_z = T1_nii.hdr.dime.pixdim(4);
  
% Load reference volume
MNI_nii = load_nii(filename_reference_volume);
MNI = double(MNI_nii.img);
[MNI_sy MNI_sx MNI_sz] = size(MNI);

MNI_voxel_size_x = MNI_nii.hdr.dime.pixdim(2);
MNI_voxel_size_y = MNI_nii.hdr.dime.pixdim(3);
MNI_voxel_size_z = MNI_nii.hdr.dime.pixdim(4);   

% Load quadrature filters
load filters_for_parametric_registration.mat
load filters_for_nonparametric_registration.mat 
    
% Run the registration with OpenCL           
start = clock;
[aligned_T1_parametric, aligned_T1_nonparametric, interpolated_T1, registration_parameters, displacement_x, displacement_y, displacement_z] = ...
        RegisterTwoVolumesMex(T1,MNI,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z, ...
        f1_parametric_registration,f2_parametric_registration,f3_parametric_registration, ...
        f1_nonparametric_registration,f2_nonparametric_registration,f3_nonparametric_registration,f4_nonparametric_registration,f5_nonparametric_registration,f6_nonparametric_registration, ...
        m1, m2, m3, m4, m5, m6, ...
        filter_directions_x, filter_directions_y, filter_directions_z, ...
        number_of_iterations_for_parametric_image_registration,number_of_iterations_for_nonparametric_image_registration,coarsest_scale,MM_T1_Z_CUT,opencl_platform, opencl_device);

reference = MNI;    

